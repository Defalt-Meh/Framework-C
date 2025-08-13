#!/usr/bin/env python3
"""
benchmark_mnist.py
==================
MNIST benchmark: FRAMEWORK-C (via frameworkc) vs. PyTorch vs. scikit-learn.

• Epoch-0 (untrained) validation accuracy for C & Torch (sklearn needs fitting → n/a)
• Per-epoch: validation accuracy & training time for C / Torch / sklearn
• Final test accuracy for all three
• Batch-inference latency (best-of-20), in seconds, for all three

Fairness:
- CPU-only, float32 everywhere
- Inputs standardized only by scaling to [0,1] (MNIST)
- Threads pinned to 1 for NumPy/BLAS and Torch
- Evaluation not counted in per-epoch training times
"""

import os, time, gzip, struct, warnings
from pathlib import Path

# ---- pin threads BEFORE importing numpy/torch ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import frameworkc as fc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# sklearn is optional
try:
    from sklearn.neural_network import MLPClassifier
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path("data/MNIST")
NIPS, NHID, NOPS = 28 * 28, 128, 10
EPOCHS = 128
LR_C, LR_T = 0.10, 0.10              # keep same LR scale for rough parity
BATCH_C = 256                         # will be overwritten by OAT if available
BATCH_T = 256
BATCH_S = 512                         # sklearn prefers a bit larger batch on CPU
DEVICE = "cpu"

# Torch threads pinned to 1 for apples-to-apples CPU timings
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

# Silence sklearn tail-batch clipping warning (last chunk < batch_size)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Got `batch_size` less than 1 or larger than sample size"
)

# ─── Helpers to read IDX files ─────────────────────────────────────
def _open_idx(fn: Path):
    if fn.exists():
        return fn.open("rb")
    gz = fn.with_suffix(fn.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rb")
    raise FileNotFoundError(f"Missing {fn}(.gz)")

def load_images(name):
    with _open_idx(DATA_DIR / name) as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(n * r * c), dtype=np.uint8)
    arr = data.reshape(n, r * c).astype(np.float32) / 255.0
    return np.ascontiguousarray(arr, dtype=np.float32)

def load_labels(name):
    with _open_idx(DATA_DIR / name) as f:
        _, n = struct.unpack(">II", f.read(8))
        lbl = np.frombuffer(f.read(n), dtype=np.uint8)
    return np.ascontiguousarray(lbl, dtype=np.int64)

# ─── 1) Load full dataset ──────────────────────────────────────────
X_all = load_images("train-images.idx3-ubyte")
Y_all = load_labels("train-labels.idx1-ubyte")
X_te  = load_images("t10k-images.idx3-ubyte")
Y_te  = load_labels("t10k-labels.idx1-ubyte")

# ─── 2) Split train/val ───────────────────────────────────────────
rng   = np.random.default_rng(0)
perm  = rng.permutation(len(X_all))
split = int(0.8 * len(X_all))
tr, va = perm[:split], perm[split:]

X_tr, Y_tr = X_all[tr], Y_all[tr]
X_va, Y_va = X_all[va], Y_all[va]

# One-hot labels for FRAMEWORK-C (C-contiguous)
I = np.eye(NOPS, dtype=np.float32)
Y_tr_c = I[Y_tr]
Y_va_c = I[Y_va]

# Torch tensors (copy() to silence "non-writable NumPy" warning)
X_tr_t = torch.from_numpy(X_tr.copy()).to(DEVICE)
Y_tr_t = torch.from_numpy(Y_tr.copy()).long().to(DEVICE)
X_va_t = torch.from_numpy(X_va.copy()).to(DEVICE)
Y_va_t = torch.from_numpy(Y_va.copy()).long().to(DEVICE)
X_te_t = torch.from_numpy(X_te.copy()).to(DEVICE)
Y_te_t = torch.from_numpy(Y_te.copy()).long().to(DEVICE)

# ─── 3) Build networks ─────────────────────────────────────────────
net_c = fc.build(NIPS, NHID, NOPS, 42)

# OAT calibration (run now; adopt B*)
info = None
try:
    if hasattr(fc, "calibrate"):
        info = fc.calibrate(net_c)          # throughput mode (no latency cap)
    elif hasattr(fc, "oat_calibrate"):
        info = fc.oat_calibrate(net_c)
except Exception:
    info = None

if info is not None:
    BATCH_C = int(info.get("B_star", BATCH_C))
    a_ms = float(info.get("alpha_ms", 0.0))
    b_ms = float(info.get("beta_ms", 0.0))
    step_ms = a_ms + b_ms * BATCH_C
    thr = (BATCH_C / (step_ms / 1e3)) if step_ms > 0 else float("inf")
    print(f"[OAT] α={a_ms*1e3:.3f}µs β={b_ms*1e3:.3f}µs/sample  "
          f"B*={BATCH_C} → step={step_ms:.3f} ms, throughput≈{thr:,.0f} samp/s")
else:
    print("[OAT] not available (module built without FWC_OAT)")

torch.manual_seed(42)
net_t = nn.Sequential(
    nn.Linear(NIPS, NHID),
    nn.ReLU(),
    nn.Linear(NHID, NOPS),
).to(DEVICE)

# Try PyTorch 2.x compile for CPU. Falls back if unavailable.
try:
    net_t = torch.compile(net_t, mode="reduce-overhead", fullgraph=False)
except Exception:
    pass

opt_t  = optim.SGD(net_t.parameters(), lr=LR_T, momentum=0.9)
loss_t = nn.CrossEntropyLoss()

loader = DataLoader(
    TensorDataset(X_tr_t, Y_tr_t),
    batch_size=BATCH_T, shuffle=True,
    num_workers=0,              # data already in RAM; extra workers add overhead
    persistent_workers=False
)

# scikit-learn model (trained via partial_fit over all mini-batches)
sk = None
if SK_AVAILABLE:
    sk = MLPClassifier(
        hidden_layer_sizes=(NHID,),
        activation="relu",
        solver="adam",
        alpha=1e-4,                    # L2
        batch_size=BATCH_S,
        learning_rate="constant",
        learning_rate_init=LR_T,       # same scale as Torch
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        max_iter=1,
        shuffle=False,                 # we control shuffling
        warm_start=True,
        random_state=42,
        n_iter_no_change=EPOCHS+1,     # disable early stop
        verbose=False,
        tol=0.0                        # avoid extra convergence checks
    )
    SK_CLASSES = np.arange(NOPS, dtype=np.int64)

# ─── 4) Epoch 0 baseline ───────────────────────────────────────────
print("Epoch |  C-val(%) | Torch-val(%) | SK-val(%) | Δt-C(s) | Δt-T(s) | Δt-S(s)")
print("------+-----------+-------------+----------+---------+---------+---------")

out_va_c = fc.predict_batch(net_c, X_va)
acc_c0   = (out_va_c.argmax(1) == Y_va).mean() * 100

net_t.eval()
with torch.inference_mode():
    logits0 = net_t(X_va_t)
acc_t0 = (logits0.argmax(1).eq(Y_va_t)).float().mean().item() * 100

print(f"{0:5d} | {acc_c0:9.2f} | {acc_t0:12.2f} | {'   n/a':>8} | {0.0:7.3f} | {0.0:7.3f} | {0.0:7.3f}")

# ─── 5) Training loop ─────────────────────────────────────────────
total_c = total_t = total_s = 0.0
n_train = X_tr.shape[0]

for ep in range(1, EPOCHS + 1):
    # Simple shared decay for both C and Torch
    lr_c = LR_C * (0.95 ** (ep - 1))
    lr_t = LR_T * (0.95 ** (ep - 1))
    for g in opt_t.param_groups:
        g['lr'] = lr_t

    # Shuffle indices once per epoch (drives C and sklearn)
    perm = rng.permutation(n_train)

    # FRAMEWORK-C: mini-batch SGD (timed)
    t0 = time.perf_counter()
    for i in range(0, n_train, BATCH_C):
        sel = perm[i : i + BATCH_C]
        xb = X_tr[sel]
        yb = Y_tr_c[sel]
        fc.train_batch(net_c, xb, yb, lr_c)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    out_va_c = fc.predict_batch(net_c, X_va)
    acc_c   = (out_va_c.argmax(1) == Y_va).mean() * 100

    # PyTorch (timed)
    t0 = time.perf_counter()
    net_t.train()
    for xb_t, yb_t in loader:  # already batches/shuffles on CPU
        opt_t.zero_grad(set_to_none=True)
        out = net_t(xb_t)
        loss = loss_t(out, yb_t)
        loss.backward()
        opt_t.step()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    net_t.eval()
    with torch.inference_mode():
        logits = net_t(X_va_t)
    acc_t = (logits.argmax(1).eq(Y_va_t)).float().mean().item() * 100

    # scikit-learn (timed, partial_fit over *all* mini-batches)
    if SK_AVAILABLE and sk is not None:
        t0 = time.perf_counter()
        first = True
        for i in range(0, n_train, BATCH_S):
            sel = perm[i : i + BATCH_S]
            xb = X_tr[sel].astype(np.float32, copy=False)
            yb = Y_tr[sel]
            if first:
                sk.partial_fit(xb, yb, classes=SK_CLASSES)
                first = False
            else:
                sk.partial_fit(xb, yb)
        dt_s = time.perf_counter() - t0
        total_s += dt_s
        acc_s = (sk.predict(X_va) == Y_va).mean() * 100
    else:
        dt_s = 0.0
        acc_s = float("nan")

    print(f"{ep:5d} | {acc_c:9.2f} | {acc_t:12.2f} | {acc_s:8.2f} | {dt_c:7.3f} | {dt_t:7.3f} | {dt_s:7.3f}")

# ─── 6) Test accuracy & inference ─────────────────────────────────
out_te_c = fc.predict_batch(net_c, X_te)
acc_c_te = (out_te_c.argmax(1) == Y_te).mean() * 100

net_t.eval()
with torch.inference_mode():
    logits_te = net_t(X_te_t)
acc_t_te = (logits_te.argmax(1).eq(Y_te_t)).float().mean().item() * 100

if SK_AVAILABLE and sk is not None:
    acc_s_te = (sk.predict(X_te) == Y_te).mean() * 100
else:
    acc_s_te = float("nan")

print(f"\nTest Accuracies: C={acc_c_te:.2f}%  Torch={acc_t_te:.2f}%"
      + (f"  sklearn={acc_s_te:.2f}%" if SK_AVAILABLE and sk is not None else "  sklearn=n/a"))

def bench(fn, reps=20, warmup=3):
    # Warmup (JIT/compile/cache)
    for _ in range(warmup):
        fn()
    best_t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t

# Torch inference-optimized copy (if available)
net_t_inf = net_t
try:
    net_t_inf = torch.jit.script(net_t.eval())
    net_t_inf = torch.jit.optimize_for_inference(net_t_inf)
except Exception:
    pass

inf_c = bench(lambda: fc.predict_batch(net_c, X_te))
with torch.inference_mode():
    inf_t = bench(lambda: net_t_inf(X_te_t))

if SK_AVAILABLE and sk is not None:
    # predict_proba includes the output layer like our prob path
    inf_s = bench(lambda: sk.predict_proba(X_te))
else:
    inf_s = float("nan")

print(f"\nBatch inference (best-of-20):")
print(f"  C       : {inf_c:7.4f} s /10k samples")
print(f"  Torch   : {inf_t:7.4f} s /10k samples")
print(f"  sklearn : {inf_s:7.4f} s /10k samples" if SK_AVAILABLE and sk is not None else "  sklearn :    n/a")

print(f"\nTotal training time:")
print(f"  C       : {total_c:.3f} s")
print(f"  Torch   : {total_t:.3f} s")
print(f"  sklearn : {total_s:.3f} s" if SK_AVAILABLE and sk is not None else "  sklearn :    n/a")

print(f"\nAvg time/epoch:")
print(f"  C       : {total_c/EPOCHS:.3f} s")
print(f"  Torch   : {total_t/EPOCHS:.3f} s")
print(f"  sklearn : {total_s/EPOCHS:.3f} s" if SK_AVAILABLE and sk is not None else "  sklearn :    n/a")
