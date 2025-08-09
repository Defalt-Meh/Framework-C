#!/usr/bin/env python3
"""
benchmark_semeion.py – FRAMEWORK-C vs. PyTorch on the Semeion digits task
-----------------------------------------------------------------
* loads semeion.data
* 80 / 20 stratified split
* trains both models for a few epochs (using batch APIs)
* prints loss, accuracy, and training times per epoch
* shows final test quality
* shows raw inference speed (best-of timing)
- vectorized C training via `train_batch`
- x1.3 times slower in training compared to torch
"""

import time
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import frameworkc as fc


# ───── hyper-parameters ─────────────────────────────────────────────
EPOCHS = 128            # enough to hit ≈92 % on both nets
BATCH  = 256            # mini-batch size for torch
LR_C   = 0.05           # learning-rate C   (plain SGD)
LR_T   = 0.01           # learning-rate torch (Adam)
DATA   = pathlib.Path("data/semeion.data")

# ───── load & split data ────────────────────────────────────────────
raw = np.loadtxt(DATA, dtype=np.float32)
x, t = raw[:, :256], raw[:, 256:]
labels = t.argmax(axis=1)

rng = np.random.default_rng(0)
idx = rng.permutation(len(x))
split = int(0.8 * len(x))
tr, te = idx[:split], idx[split:]

x_tr, t_tr, lbl_tr = x[tr], t[tr], labels[tr]
x_te, t_te, lbl_te = x[te], t[te], labels[te]

# Pre-convert to float32 contiguous arrays for batch C calls
X_tr = x_tr.astype(np.float32)
Y_tr = t_tr.astype(np.float32)
X_te = x_te.astype(np.float32)
Y_te = t_te.astype(np.float32)

nips, nhid, nops = 256, 28, 10

# ───── build networks ───────────────────────────────────────────────
cnet = fc.build(nips, nhid, nops, 1)

torch_net = nn.Sequential(
    nn.Linear(nips, nhid),
    nn.Sigmoid(),
    nn.Linear(nhid, nops),
    nn.Sigmoid(),
)
opt     = optim.Adam(torch_net.parameters(), lr=LR_T)
loss_fn = nn.MSELoss()

# ───── helper functions ─────────────────────────────────────────────
def accuracy_from_onehot(net_out: np.ndarray, target_onehot: np.ndarray) -> float:
    pred = net_out.argmax(1)
    true = target_onehot .argmax(1)
    return (pred == true).mean() * 100.0

def c_batch_predict(x_arr: np.ndarray) -> np.ndarray:
    """Vectorised C inference returning an (N × nops) array."""
    return fc.predict_batch(cnet, x_arr)

# ───── training loop with timing ────────────────────────────────────
total_c = total_t = 0.0

print("Epoch |   C-loss  C-acc   | Torch-loss Torch-acc | Δt-C(s) | Δt-T(s)")
print("------+--------------------+-----------------------+---------+---------")
for ep in range(1, EPOCHS + 1):
    lr_decay = LR_C * (0.97 ** (ep - 1))

    # ---- FRAMEWORK-C batch training ----
    t0 = time.perf_counter()
    fc.train_batch(cnet, X_tr, Y_tr, lr_decay)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    # compute C loss & acc on training set
    out_c_tr = c_batch_predict(X_tr)
    loss_c   = float(np.mean((out_c_tr - Y_tr) ** 2))
    acc_c    = accuracy_from_onehot(out_c_tr, Y_tr)

    # ---- PyTorch (Adam) batch training ----
    torch_net.train()
    loss_t = 0.0
    hits_t = 0

    t0 = time.perf_counter()
    for i in range(0, len(X_tr), BATCH):
        xb = torch.from_numpy(X_tr[i : i + BATCH])
        tb = torch.from_numpy(Y_tr[i : i + BATCH])
        opt.zero_grad()
        out = torch_net(xb)
        l   = loss_fn(out, tb)
        l.backward()
        opt.step()

        loss_t += l.item() * xb.size(0)
        hits_t += (out.argmax(1) == tb.argmax(1)).sum().item()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    loss_t /= len(X_tr)
    acc_t   = hits_t / len(X_tr) * 100.0

    print(f"{ep:5d} | {loss_c:8.4f} {acc_c:7.2f} |"
          f" {loss_t:8.4f} {acc_t:7.2f} |"
          f" {dt_c:7.3f} | {dt_t:7.3f}")

print("\n===== Total Training Time =====")
print(f" FRAMEWORK-C : {total_c:.3f} s")
print(f" PyTorch     : {total_t:.3f} s")
print("===============================")

# ───── final test accuracy & quality ───────────────────────────────
out_c_te = c_batch_predict(X_te)
out_t_te = torch_net(torch.from_numpy(X_te)).detach().numpy()

acc_c = (out_c_te .argmax(1) == lbl_te).mean() * 100.0
acc_t = (out_t_te.argmax(1) == lbl_te).mean() * 100.0
mse_c = float(np.mean((out_c_te - Y_te) ** 2))
mse_t = float(np.mean((out_t_te - Y_te) ** 2))

print("\n===== Test Set Quality =====")
print("          MSE       Acc(%)")
print("-----------------------------")
print(f" C-Net : {mse_c:9.4f} {acc_c:9.2f}")
print(f" Torch : {mse_t:9.4f} {acc_t:9.2f}")

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
