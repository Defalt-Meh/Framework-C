#!/usr/bin/env python3
"""
benchmark_semeion.py – FRAMEWORK-C vs. PyTorch on the Semeion digits task
Fixes:
- Mini-batch SGD for FRAMEWORK-C (was full-batch)
- Cosine LR schedule (+ tiny warmup)
- Standardize features (fit on train)
- Cross-entropy reporting (matches C's softmax+CE path)
"""

import time, math, pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import frameworkc as fc

# ───── hyper-parameters ─────────────────────────────────────────────
SEED   = 1
EPOCHS = 128
BATCH  = 128             # use same order for both C and Torch
LR_C   = 0.08            # base LR for C (cosine-decayed)
LR_T   = 5e-3            # AdamW LR often stable here
WARMUP_EPOCHS = 3
DATA   = pathlib.Path("data/semeion.data")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ───── load & split data ────────────────────────────────────────────
raw = np.loadtxt(DATA, dtype=np.float32)
X, T = raw[:, :256], raw[:, 256:]
y    = T.argmax(axis=1)

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

# ensure float32 contiguous for C calls
X_tr = np.ascontiguousarray(X_tr, dtype=np.float32)
T_tr = np.ascontiguousarray(T_tr, dtype=np.float32)
X_te = np.ascontiguousarray(X_te, dtype=np.float32)
T_te = np.ascontiguousarray(T_te, dtype=np.float32)

nips, nhid, nops = 256, 64, 10  # a bit wider helps with CE

# ───── build nets ───────────────────────────────────────────────────
cnet = fc.build(nips, nhid, nops, SEED)

torch_net = nn.Sequential(
    nn.Linear(nips, nhid),
    nn.ReLU(),                 # match C’s ReLU-ready path if compiled that way
    nn.Linear(nhid, nops),     # logits
)
opt     = optim.AdamW(torch_net.parameters(), lr=LR_T, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss()   # expects class indices

# ───── helpers ──────────────────────────────────────────────────────
def cosine_lr(base_lr, t, T, warmup=WARMUP_EPOCHS):
    if t < warmup:  # linear warmup
        return base_lr * (t + 1) / max(1, warmup)
    t_adj = min(max(t - warmup, 0), max(T - warmup, 1))
    return 0.5 * base_lr * (1 + math.cos(math.pi * t_adj / max(T - warmup, 1)))

def accuracy_from_onehot(net_out: np.ndarray, target_onehot: np.ndarray) -> float:
    return (net_out.argmax(1) == target_onehot.argmax(1)).mean() * 100.0

def cross_entropy_onehot(probs: np.ndarray, onehot: np.ndarray, eps=1e-9) -> float:
    # probs should be softmax outputs (FRAMEWORK-C already returns probs for nops>1)
    return float(-(onehot * np.log(probs + eps)).sum() / probs.shape[0])

def c_batch_predict(x_arr: np.ndarray) -> np.ndarray:
    return fc.predict_batch(cnet, x_arr)

# ───── training loop ────────────────────────────────────────────────
print("Epoch |   C-CE    C-acc   | Torch-CE  Torch-acc | Δt-C(s) | Δt-T(s)")
print("------+--------------------+---------------------+---------+---------")

total_c = total_t = 0.0
N = len(X_tr)

for ep in range(1, EPOCHS + 1):
    # shuffle indices
    perm = rng.permutation(N)
    lr_c = cosine_lr(LR_C, ep - 1, EPOCHS)

    # ---- FRAMEWORK-C mini-batch training ----
    t0 = time.perf_counter()
    for i in range(0, N, BATCH):
        sel = perm[i:i+BATCH]
        xb  = X_tr[sel]
        tb  = T_tr[sel]
        # mini-batch SGD step in C
        fc.train_batch(cnet, xb, tb, lr_c)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    # compute C CE & acc on training set
    out_c_tr = c_batch_predict(X_tr)
    ce_c     = cross_entropy_onehot(out_c_tr, T_tr)
    acc_c    = accuracy_from_onehot(out_c_tr, T_tr)

    # ---- PyTorch mini-batch training (AdamW + CE) ----
    torch_net.train()
    t0 = time.perf_counter()
    loss_sum = 0.0
    hits_sum = 0
    for i in range(0, N, BATCH):
        sel = perm[i:i+BATCH]
        xb = torch.from_numpy(X_tr[sel])
        yb = torch.from_numpy(y_tr[sel])
        opt.zero_grad(set_to_none=True)
        logits = torch_net(xb)
        loss   = ce_loss(logits, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            loss_sum += loss.item() * xb.size(0)
            hits_sum += (logits.argmax(1) == yb).sum().item()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    ce_t  = loss_sum / N
    acc_t = hits_sum / N * 100.0

    print(f"{ep:5d} | {ce_c:8.4f} {acc_c:7.2f} | {ce_t:8.4f} {acc_t:7.2f} |"
          f" {dt_c:7.3f} | {dt_t:7.3f}")

print("\n===== Total Training Time =====")
print(f" FRAMEWORK-C : {total_c:.3f} s")
print(f" PyTorch     : {total_t:.3f} s")
print("===============================")

# ───── final test quality ───────────────────────────────────────────
out_c_te = c_batch_predict(X_te)                                      # probs
logits_t = torch_net(torch.from_numpy(X_te)).detach().numpy()
probs_t  = torch.softmax(torch.from_numpy(logits_t), dim=1).numpy()

acc_c = (out_c_te.argmax(1) == y_te).mean() * 100.0
acc_t = (probs_t .argmax(1) == y_te).mean() * 100.0
ce_c  = cross_entropy_onehot(out_c_te, T_te)
ce_t  = cross_entropy_onehot(probs_t , T_te)

print("\n===== Test Set Quality =====")
print("          CE        Acc(%)")
print("-----------------------------")
print(f" C-Net : {ce_c:9.4f} {acc_c:9.2f}")
print(f" Torch : {ce_t:9.4f} {acc_t:9.2f}")

# ───── raw inference timing (best-of) ───────────────────────────────
REPEAT = 30
def best(fn, repeat=REPEAT):
    best_t = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t * 1e3  # ms

print("\n===== Inference Times (best of {:d}) =====".format(REPEAT))
print(" Path            Time (ms)")
print("-----------------------------")
print(f" C-single    : {best(lambda: [fc.predict(cnet, row.tolist()) for row in X_te]):9.3f}")
print(f" C-batch     : {best(lambda: c_batch_predict(X_te)):9.3f}")
print(f" Torch-single: {best(lambda: [torch_net(torch.from_numpy(r)) for r in X_te]):9.3f}")
print(f" Torch-batch : {best(lambda: torch_net(torch.from_numpy(X_te))):9.3f}")
print("==========================================")
