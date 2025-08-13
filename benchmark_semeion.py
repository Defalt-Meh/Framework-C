#!/usr/bin/env python3
"""
benchmark_semeion.py – FRAMEWORK-C vs. PyTorch vs. scikit-learn on the Semeion digits task

- FRAMEWORK-C: mini-batch SGD + cosine LR (with warmup)
- PyTorch:     AdamW + CE
- scikit-learn: MLPClassifier('adam') trained via partial_fit over all mini-batches

Fairness notes:
- Inputs standardized (fit on train)
- All math on CPU, float32
- Threads pinned to 1 for NumPy/BLAS and Torch
- Evaluation not included in per-epoch training times

Extras:
- Avoids PyTorch "non-writable NumPy array" warning by copying small batches
- OAT calibration (if frameworkc exposes fc.calibrate / fc.oat_calibrate) runs before training
"""

import os, time, math, pathlib, warnings
# ---- pin threads BEFORE importing numpy/torch ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# Silence sklearn's tail-batch clipping warning (it's expected when last batch is smaller)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Got `batch_size` less than 1 or larger than sample size"
)

import torch
import torch.nn as nn
import torch.optim as optim

import frameworkc as fc

# sklearn is optional
try:
    from sklearn.neural_network import MLPClassifier
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False

# ───── hyper-parameters ─────────────────────────────────────────────
SEED   = 1
EPOCHS = 128
BATCH  = 128
LR_C   = 0.08            # base LR for C (cosine-decayed)
LR_T   = 5e-3            # AdamW lr
WARMUP_EPOCHS = 3
DATA   = pathlib.Path("data/semeion.data")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

# ───── load & split data ────────────────────────────────────────────
raw = np.loadtxt(DATA, dtype=np.float32)
X, T = raw[:, :256], raw[:, 256:]
y    = T.argmax(axis=1).astype(np.int64)

idx = rng.permutation(len(X))
split = int(0.8 * len(X))
tr, te = idx[:split], idx[split:]

X_tr, T_tr, y_tr = X[tr], T[tr], y[tr]
X_te, T_te, y_te = X[te], T[te], y[te]

# ───── standardize inputs (fit on train only) ───────────────────────
mu  = X_tr.mean(axis=0, keepdims=True)
std = X_tr.std(axis=0, keepdims=True) + 1e-6
X_tr = (X_tr - mu) / std
X_te = (X_te - mu) / std

# ensure float32 contiguous & writable
X_tr = np.ascontiguousarray(X_tr, dtype=np.float32); X_tr.setflags(write=True)
T_tr = np.ascontiguousarray(T_tr, dtype=np.float32); T_tr.setflags(write=True)
X_te = np.ascontiguousarray(X_te, dtype=np.float32); X_te.setflags(write=True)
T_te = np.ascontiguousarray(T_te, dtype=np.float32); T_te.setflags(write=True)

nips, nhid, nops = 256, 64, 10

# ───── build nets ───────────────────────────────────────────────────
cnet = fc.build(nips, nhid, nops, SEED)

# OAT calibration (run NOW so we can adopt B*)
info = None
try:
    if hasattr(fc, "calibrate"):
        info = fc.calibrate(cnet)            # throughput mode (no latency cap)
    elif hasattr(fc, "oat_calibrate"):
        info = fc.oat_calibrate(cnet)
except Exception:
    info = None

if info is not None:
    BATCH = int(info.get("B_star", BATCH))
    a_ms  = float(info.get("alpha_ms", 0.0))
    b_ms  = float(info.get("beta_ms", 0.0))
    step_ms = a_ms + b_ms * BATCH
    thr = (BATCH / (step_ms / 1e3)) if step_ms > 0 else float("inf")
    print(f"[OAT] α={a_ms*1e3:.3f}µs β={b_ms*1e3:.3f}µs/sample  "
          f"B*={BATCH} → step={step_ms:.3f} ms, throughput≈{thr:,.0f} samp/s")
else:
    print("[OAT] not available (module built without FWC_OAT)")

torch_net = nn.Sequential(
    nn.Linear(nips, nhid),
    nn.ReLU(),
    nn.Linear(nhid, nops),     # logits
)
opt     = optim.AdamW(torch_net.parameters(), lr=LR_T, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss()

sk_model = None
if SK_AVAILABLE:
    # We'll drive epochs/batches via partial_fit.
    sk_model = MLPClassifier(
        hidden_layer_sizes=(nhid,),
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2
        batch_size=BATCH,           # OK if tail < batch (we silence the warning)
        learning_rate="constant",
        learning_rate_init=LR_T,    # same scale as Torch for fairness
        max_iter=1,                 # unused with partial_fit
        shuffle=False,              # we control shuffling
        warm_start=True,
        random_state=SEED,
        n_iter_no_change=EPOCHS+1,  # disable early stopping
        verbose=False
    )
    classes = np.arange(nops, dtype=np.int64)

# ───── helpers ──────────────────────────────────────────────────────
def cosine_lr(base_lr, t, T, warmup=WARMUP_EPOCHS):
    if t < warmup:  # linear warmup
        return base_lr * (t + 1) / max(1, warmup)
    t_adj = min(max(t - warmup, 0), max(T - warmup, 1))
    return 0.5 * base_lr * (1 + math.cos(math.pi * t_adj / max(T - warmup, 1)))

def cross_entropy_onehot(probs: np.ndarray, onehot: np.ndarray, eps=1e-9) -> float:
    return float(-(onehot * np.log(probs + eps)).sum() / probs.shape[0])

def accuracy_from_labels(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    return (pred_labels == true_labels).mean() * 100.0

def c_batch_predict(x_arr: np.ndarray) -> np.ndarray:
    # frameworkc returns probabilities for multi-class
    return fc.predict_batch(cnet, x_arr)

@torch.no_grad()
def torch_probs(x_arr: np.ndarray) -> np.ndarray:
    torch_net.eval()
    logits = torch_net(torch.from_numpy(x_arr.copy()))
    return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)

def sk_probs(x_arr: np.ndarray) -> np.ndarray:
    return sk_model.predict_proba(x_arr).astype(np.float32)

# ───── training loop ────────────────────────────────────────────────
print("Epoch |        C: CE  Acc  Δt  |     Torch: CE  Acc  Δt |    SK: CE   Acc   Δt")
print("------+------------------------+------------------------+------------------------")

N = len(X_tr)
total_c = total_t = total_s = 0.0

for ep in range(1, EPOCHS + 1):
    # shuffle once per epoch
    perm = rng.permutation(N)
    lr_c = cosine_lr(LR_C, ep - 1, EPOCHS)

    # ---- FRAMEWORK-C training (timed) ----
    t0 = time.perf_counter()
    for i in range(0, N, BATCH):
        sel = perm[i:i+BATCH]
        xb  = X_tr[sel]
        tb  = T_tr[sel]
        fc.train_batch(cnet, xb, tb, lr_c)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    # ---- Torch training (timed) ----
    torch_net.train()
    t0 = time.perf_counter()
    for i in range(0, N, BATCH):
        sel = perm[i:i+BATCH]
        # .copy() keeps things simple & silences the "non-writable numpy array" warning safely
        xb = torch.from_numpy(X_tr[sel].copy())
        yb = torch.from_numpy(y_tr[sel].copy())
        opt.zero_grad(set_to_none=True)
        logits = torch_net(xb)
        loss   = ce_loss(logits, yb)
        loss.backward()
        opt.step()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    # ---- scikit-learn training (timed) ----
    if SK_AVAILABLE and sk_model is not None:
        t0 = time.perf_counter()
        first = True
        for i in range(0, N, BATCH):
            sel = perm[i:i+BATCH]
            xb = X_tr[sel]
            yb = y_tr[sel]
            if first:
                sk_model.partial_fit(xb, yb, classes=classes)
                first = False
            else:
                sk_model.partial_fit(xb, yb)
        dt_s = time.perf_counter() - t0
        total_s += dt_s
    else:
        dt_s = 0.0

    # ---- compute CE/Acc on FULL TRAIN (not timed) ----
    # C
    probs_c_tr = c_batch_predict(X_tr)
    ce_c  = cross_entropy_onehot(probs_c_tr, T_tr)
    acc_c = accuracy_from_labels(probs_c_tr.argmax(1), y_tr)

    # Torch
    probs_t_tr = torch_probs(X_tr)
    ce_t  = cross_entropy_onehot(probs_t_tr, T_tr)
    acc_t = accuracy_from_labels(probs_t_tr.argmax(1), y_tr)

    # SK
    if SK_AVAILABLE and sk_model is not None:
        probs_s_tr = sk_probs(X_tr)
        ce_s  = cross_entropy_onehot(probs_s_tr, T_tr)
        acc_s = accuracy_from_labels(probs_s_tr.argmax(1), y_tr)
    else:
        ce_s, acc_s = float("nan"), float("nan")

    print(f"{ep:5d} | {ce_c:7.4f} {acc_c:6.2f} {dt_c:5.3f} |"
          f" {ce_t:7.4f} {acc_t:6.2f} {dt_t:5.3f} |"
          f" {ce_s:7.4f} {acc_s:6.2f} {dt_s:5.3f}")

# ───── totals ───────────────────────────────────────────────────────
print("\n===== Total Training Time =====")
print(f" FRAMEWORK-C : {total_c:.3f} s")
print(f" PyTorch     : {total_t:.3f} s")
if SK_AVAILABLE and sk_model is not None:
    print(f" scikit-learn: {total_s:.3f} s")
print("===============================")

# ───── final test quality ───────────────────────────────────────────
probs_c_te = c_batch_predict(X_te)
probs_t_te = torch_probs(X_te)

acc_c = accuracy_from_labels(probs_c_te.argmax(1), y_te)
acc_t = accuracy_from_labels(probs_t_te.argmax(1), y_te)
ce_c  = cross_entropy_onehot(probs_c_te, T_te)
ce_t  = cross_entropy_onehot(probs_t_te, T_te)

print("\n===== Test Set Quality =====")
print("              CE        Acc(%)")
print("---------------------------------")
print(f" C-Net     : {ce_c:9.4f} {acc_c:9.2f}")
print(f" Torch     : {ce_t:9.4f} {acc_t:9.2f}")

if SK_AVAILABLE and sk_model is not None:
    probs_s_te = sk_probs(X_te)
    ce_s  = cross_entropy_onehot(probs_s_te, T_te)
    acc_s = accuracy_from_labels(probs_s_te.argmax(1), y_te)
    print(f" sklearn   : {ce_s:9.4f} {acc_s:9.2f}")

# ───── raw inference timing (best-of) ───────────────────────────────
REPEAT = 30

def best(fn, repeat=REPEAT, warmups=3):
    # quick warm-up to stabilize Python/BLAS dispatch
    for _ in range(warmups):
        fn()
    best_t = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t * 1e3  # ms

torch_net.eval()

print("\n===== Inference Times (best of {:d}) =====".format(REPEAT))
print(" Path              Time (ms)")
print("-------------------------------")
print(f" C-single      : {best(lambda: [fc.predict(cnet, row) for row in X_te]):9.3f}")
print(f" C-batch       : {best(lambda: c_batch_predict(X_te)):9.3f}")
print(f" Torch-single  : {best(lambda: [torch_net(torch.from_numpy(r.copy())) for r in X_te]):9.3f}")
print(f" Torch-batch   : {best(lambda: torch_net(torch.from_numpy(X_te.copy()))):9.3f}")
if SK_AVAILABLE and sk_model is not None:
    print(f" SK-single     : {best(lambda: [sk_model.predict_proba(row.reshape(1,-1)) for row in X_te]):9.3f}")
    print(f" SK-batch      : {best(lambda: sk_model.predict_proba(X_te)):9.3f}")
print("==========================================")
