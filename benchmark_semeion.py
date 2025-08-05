#!/usr/bin/env python3
"""
benchmark_semeion.py – FRAMEWORK-C vs. PyTorch on the Semeion digits task
-----------------------------------------------------------------
* loads semeion.data
* 80 / 20 stratified split
* trains both models for a few epochs
* prints loss, accuracy, and training times per epoch
* shows final test quality
* shows raw inference speed (best-of timing)
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

nips, nhid, nops = 256, 28, 10

# ───── build networks ───────────────────────────────────────────────
cnet = fc.build(nips, nhid, nops, 1)

torch_net = nn.Sequential(
    nn.Linear(nips, nhid),
    nn.Sigmoid(),
    nn.Linear(nhid, nops),
    nn.Sigmoid(),
)
opt = optim.Adam(torch_net.parameters(), lr=LR_T)
loss_fn = nn.MSELoss()

# ───── helper functions ─────────────────────────────────────────────
def accuracy_from_onehot(net_out, target_onehot):
    pred = net_out.argmax(1)
    true = target_onehot.argmax(1)
    return (pred == true).float().mean().item() * 100.0

def c_batch_predict(x_arr: np.ndarray) -> np.ndarray:
    """Vectorised C inference returning an (N × nops) array."""
    return fc.predict_batch(cnet, x_arr.astype(np.float32))

# ───── training loop with timing ────────────────────────────────────
total_c = total_t = 0.0

print("Epoch  |  C-loss   C-acc   |  Torch-loss  Torch-acc  | Δt-C(s) | Δt-T(s)")
print("-------+-------------------+--------------------------+---------+---------")
for ep in range(1, EPOCHS + 1):
    # ---- FRAMEWORK-C (plain SGD) ----
    lr_decay = LR_C * (0.97 ** (ep - 1))
    loss_c = 0.0
    hits_c = 0

    t0 = time.perf_counter()
    for i in range(len(x_tr)):
        loss_c += fc.train_one(
            cnet, x_tr[i].tolist(), t_tr[i].tolist(), lr_decay
        )
        if np.argmax(fc.predict(cnet, x_tr[i].tolist())) == lbl_tr[i]:
            hits_c += 1
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    loss_c /= len(x_tr)
    acc_c = hits_c / len(x_tr) * 100.0

    # ---- PyTorch (Adam) ----
    torch_net.train()
    loss_t = 0.0
    hits_t = 0

    t0 = time.perf_counter()
    for i in range(0, len(x_tr), BATCH):
        xb = torch.from_numpy(x_tr[i : i + BATCH])
        tb = torch.from_numpy(t_tr[i : i + BATCH])
        opt.zero_grad()
        out = torch_net(xb)
        l = loss_fn(out, tb)
        l.backward()
        opt.step()
        loss_t += l.item() * len(xb)
        hits_t += (out.argmax(1) == tb.argmax(1)).sum().item()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    loss_t /= len(x_tr)
    acc_t = hits_t / len(x_tr) * 100.0

    print(f"{ep:5d}  | {loss_c:7.4f} {acc_c:7.2f} |"
          f"  {loss_t:7.4f}   {acc_t:7.2f}   |"
          f"{dt_c:8.3f} | {dt_t:8.3f}")

# ───── total training time ──────────────────────────────────────────
print("\n========== Total Training Time ==========")
print(f"   FRAMEWORK-C : {total_c:.3f} s")
print(f"   PyTorch     : {total_t:.3f} s")
print("==========================================")

# ───── final test accuracy & quality ───────────────────────────────
c_out = c_batch_predict(x_te)
t_out = torch_net(torch.from_numpy(x_te)).detach().numpy()

acc_c = (c_out.argmax(1) == lbl_te).mean() * 100.0
acc_t = (t_out.argmax(1) == lbl_te).mean() * 100.0
mse_c = np.mean((c_out - t_te) ** 2)
mse_t = np.mean((t_out - t_te) ** 2)

print("\n========== Test Set Quality ==========")
print("                     MSE        Acc(%)")
print("   -------------------------------------")
print(f"   FRAMEWORK-C : {mse_c:9.4f} {acc_c:9.2f}")
print(f"   PyTorch     : {mse_t:9.4f} {acc_t:9.2f}")

# ───── raw inference timing (best-of) ───────────────────────────────
batch = x_te.copy()  # 1k samples → decent timing
REPEAT = 30

def best(fn, repeat=REPEAT):
    best_t = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t * 1e3  # → ms

print("\n========== Inference Times (best of {:d}) ==========".format(REPEAT))
print("   Path            Time (ms)")
print("   --------------------------")
print(f"   C-single    : {best(lambda: [fc.predict(cnet, row.tolist()) for row in batch]):9.3f}")
print(f"   C-batch     : {best(lambda: c_batch_predict(batch)):9.3f}")
print(f"   Torch-single: {best(lambda: [torch_net(torch.from_numpy(r)) for r in batch]):9.3f}")
print(f"   Torch-batch : {best(lambda: torch_net(torch.from_numpy(batch))):9.3f}")
print("=============================================================")

