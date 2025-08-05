#!/usr/bin/env python3
"""
benchmark_mnist.py
==================
MNIST benchmark: FRAMEWORK-C (via my_module.c API) vs. PyTorch.

• Reports epoch 0 (untrained) validation accuracy  
• Runs 10 training epochs, printing per-epoch val accuracy & times  
• Final test classification accuracy  
• Batch-inference latency (best of 20) for both models, shown in seconds
"""

import gzip
import struct
import time

import numpy as np
from pathlib import Path
import frameworkc as fc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path("data/MNIST")
NIPS, NHID, NOPS = 28 * 28, 128, 10
EPOCHS = 10
LR_C, LR_T = 0.1, 0.1
BATCH_T = 256
DEVICE = "cpu"  # force CPU

# ─── IDX loader ────────────────────────────────────────────────────
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
    return np.ascontiguousarray(arr)

def load_labels(name):
    with _open_idx(DATA_DIR / name) as f:
        _, n = struct.unpack(">II", f.read(8))
        lbl = np.frombuffer(f.read(n), dtype=np.uint8)
    return np.ascontiguousarray(lbl)

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

# C-contiguous one-hot labels for FRAMEWORK-C
Y_tr_c = np.eye(NOPS, dtype=np.float32)[Y_tr]
Y_va_c = np.eye(NOPS, dtype=np.float32)[Y_va]

# ─── 3) Build networks ─────────────────────────────────────────────
net_c = fc.build(NIPS, NHID, NOPS, 42)

torch.manual_seed(42)
net_t = nn.Sequential(
    nn.Linear(NIPS, NHID),
    nn.ReLU(),
    nn.Linear(NHID, NOPS),
).to(DEVICE)
opt_t  = optim.SGD(net_t.parameters(), lr=LR_T, momentum=0.9)
loss_t = nn.CrossEntropyLoss()

# DataLoader (single-process) for PyTorch
X_tr_t = torch.from_numpy(X_tr.copy())
Y_tr_t = torch.from_numpy(Y_tr.copy()).long()
loader = DataLoader(
    TensorDataset(X_tr_t, Y_tr_t),
    batch_size=BATCH_T, shuffle=True, num_workers=0
)

# ─── 4) Epoch 0 baseline ───────────────────────────────────────────
print("Epoch |  C-val(%) | Torch-val(%) | Δt-C(s) | Δt-T(s)")
print("------+-----------+-------------+--------+--------")

out_va_c = fc.predict_batch(net_c, X_va)
acc_c0   = (out_va_c.argmax(1) == Y_va).mean() * 100

net_t.eval()
with torch.no_grad():
    logits0 = net_t(torch.from_numpy(X_va).to(DEVICE)).cpu()
acc_t0 = (logits0.argmax(1).numpy() == Y_va).mean() * 100

print(f"{0:5d} | {acc_c0:9.2f} | {acc_t0:12.2f} | {0.0:6.3f} | {0.0:6.3f}")

# ─── 5) Training loop ─────────────────────────────────────────────
total_c = total_t = 0.0
n_train = X_tr.shape[0]
for ep in range(1, EPOCHS + 1):
    # FRAMEWORK-C (normalize full-batch learning rate)
    t0 = time.perf_counter()
    fc.train_batch(net_c, X_tr, Y_tr_c, LR_C)
    dt_c = time.perf_counter() - t0
    total_c += dt_c
    out_va_c = fc.predict_batch(net_c, X_va)
    acc_c = (out_va_c.argmax(1) == Y_va).mean() * 100

    # PyTorch
    t0 = time.perf_counter()
    net_t.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_t.zero_grad()
        loss_t(net_t(xb), yb).backward()
        opt_t.step()
    dt_t = time.perf_counter() - t0
    total_t += dt_t
    net_t.eval()
    with torch.no_grad():
        logits = net_t(torch.from_numpy(X_va).to(DEVICE)).cpu()
    acc_t = (logits.argmax(1).numpy() == Y_va).mean() * 100

    print(f"{ep:5d} | {acc_c:9.2f} | {acc_t:12.2f} | {dt_c:6.3f} | {dt_t:6.3f}")

# ─── 6) Test accuracy & inference ─────────────────────────────────
out_te_c  = fc.predict_batch(net_c, X_te)
acc_c_te  = (out_te_c.argmax(1) == Y_te).mean() * 100

net_t.eval()
with torch.no_grad():
    logits_te = net_t(torch.from_numpy(X_te).to(DEVICE)).cpu()
acc_t_te  = (logits_te.argmax(1).numpy() == Y_te).mean() * 100

print(f"\nTest Accuracies: C={acc_c_te:.2f}%  Torch={acc_t_te:.2f}%")

def bench(fn, reps=20):
    """Return best time in seconds."""
    best_t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t

# measure batch inference in seconds
inf_c = bench(lambda: fc.predict_batch(net_c, X_te))
inf_t = bench(lambda: net_t(torch.from_numpy(X_te).to(DEVICE)).cpu())

print(f"\nBatch inference (best-of-20):")
print(f"  C    : {inf_c:6.4f} s /10k samples")
print(f"  Torch: {inf_t:6.4f} s /10k samples")

print(f"\nTotal training time:")
print(f"  C    : {total_c:.3f} s")
print(f"  Torch: {total_t:.3f} s")

print(f"\nTraining time (avg per epoch):")
print(f"  C    : {total_c / EPOCHS:.3f} s")
print(f"  Torch: {total_t / EPOCHS:.3f} s")
