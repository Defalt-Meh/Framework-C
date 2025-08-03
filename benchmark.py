#!/usr/bin/env python3
"""
benchmark.py – FRAMEWORK-C vs. PyTorch on the Semeion digits task
-----------------------------------------------------------------
* loads semeion.data
* 80 / 20 stratified split
* trains both models for a few epochs
* prints loss  and accuracy on both splits
* shows speed (inference) afterwards
"""

import sys, time, math, pathlib, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import frameworkc as fc

# ───── hyper-parameters ─────────────────────────────────────────────
EPOCHS = 128            # enough to hit ≈92 % on both nets
BATCH  = 256           # mini-batch size for torch
LR_C   = 0.05           # learning-rate C   (plain SGD)
LR_T   = 0.01          # learning-rate torch (Adam)
DATA   = pathlib.Path("semeion.data")

# ───── load & split data ────────────────────────────────────────────
raw = np.loadtxt(DATA, dtype=np.float32)
x, t = raw[:, :256], raw[:, 256:]

# one hot → class index (for accuracy calc only)
labels = t.argmax(axis=1)

rng = np.random.default_rng(0)
idx  = rng.permutation(len(x))
split = int(0.8 * len(x))
tr, te = idx[:split], idx[split:]

x_tr, t_tr, lbl_tr = x[tr], t[tr], labels[tr]
x_te, t_te, lbl_te = x[te], t[te], labels[te]

nips, nops = 256, 10
nhid = 28

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

# ───── training loop ────────────────────────────────────────────────
def accuracy(net_out, target_onehot):
    pred = net_out.argmax(1)
    true = target_onehot.argmax(1)
    return (pred == true).float().mean().item() * 100.0

print("Epoch  |  C-loss   C-acc   |  Torch-loss  Torch-acc")
print("-------+-------------------+------------------------")
for ep in range(1, EPOCHS+1):
    # ---- FRAMEWORK-C (plain SGD) ----
    lr_decay = LR_C * (0.97 ** (ep-1))
    loss_c = 0.0; hit = 0
    for i in range(len(x_tr)):
        loss_c += fc.train_one(cnet, x_tr[i].tolist(), t_tr[i].tolist(), lr_decay)
        if np.argmax(fc.predict(cnet, x_tr[i].tolist())) == lbl_tr[i]:
            hit += 1
    acc_c = hit / len(x_tr) * 100.0
    loss_c /= len(x_tr)

    # ---- PyTorch ----
    torch_net.train()
    loss_t = 0.0; hit_t = 0
    for i in range(0, len(x_tr), BATCH):
        xb = torch.from_numpy(x_tr[i:i+BATCH])
        tb = torch.from_numpy(t_tr[i:i+BATCH])
        opt.zero_grad()
        out = torch_net(xb)
        loss = loss_fn(out, tb)
        loss.backward()
        opt.step()
        loss_t += loss.item() * len(xb)
        hit_t  += (out.argmax(1) == tb.argmax(1)).sum().item()
    loss_t /= len(x_tr)
    acc_t   = hit_t / len(x_tr) * 100.0

    print(f"{ep:5d}  | {loss_c:7.4f} {acc_c:7.2f} |"
          f"  {loss_t:7.4f}   {acc_t:7.2f}")

# ───── final test accuracy ──────────────────────────────────────────
def c_batch_predict(x_arr):
    out = np.empty((len(x_arr), nops), np.float32)
    fc.predict_batch(cnet, x_arr.astype(np.float32), out)
    return out

# helper ----------------------------------------------------------
def c_batch_predict(x_arr: np.ndarray) -> np.ndarray:
    """vectorised C inference"""
    return fc.predict_batch(cnet, x_arr.astype(np.float32))

c_out  = c_batch_predict(x_te)
t_out  = torch_net(torch.from_numpy(x_te)).detach().numpy()

acc_c = (c_out.argmax(1) == lbl_te).mean() * 100.0
acc_t = (t_out.argmax(1) == lbl_te).mean() * 100.0
mse_c = np.mean((c_out - t_te)**2)
mse_t = np.mean((t_out - t_te)**2)

print("\n========== Test set quality ==========\n")
print("                     MSE        Acc(%)")
print("   -------------------------------------")
print(f"   FRAMEWORK-C : {mse_c:9.4f} {acc_c:9.2f}")
print(f"   PyTorch     : {mse_t:9.4f} {acc_t:9.2f}")

# ───── raw speed (inference only) ───────────────────────────────────
batch = x_te.copy()          # 1 k samples → decent timing

REPEAT = 30                 # “best of” measurements

def best(fn, repeat=REPEAT):
    best_t = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t * 1e3      # → ms

c_single = lambda: [fc.predict(cnet, row.tolist()) for row in batch]
c_batch  = lambda: c_batch_predict(batch)
t_single = lambda: [torch_net(torch.from_numpy(r)) for r in batch]
t_batch  = lambda: torch_net(torch.from_numpy(batch))

print(f"\n========== Inference times (best of {REPEAT}) ==========\n")
print("   Path            Time (ms)")
print("   --------------------------")
print(f"   C-single    : {best(c_single):9.3f}")
print(f"   C-batch     : {best(c_batch ):9.3f}")
print(f"   Torch-single: {best(t_single):9.3f}")
print(f"   Torch-batch : {best(t_batch ):9.3f}")
print("=============================================================")

