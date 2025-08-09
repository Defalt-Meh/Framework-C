#!/usr/bin/env python3
"""
benchmark_semeion_improved.py – Improved Framework-C training with:
 * label sanitation for Semeion (-1/1 -> 0/1, optional 1..9,0 -> 0..9)
 * manual input normalization
 * mini-batch SGD (true batch, averaged gradients on C side)
 * per-epoch shuffling
 * constant learning rate (tiny dataset)
 * per-epoch test accuracy monitoring
 * post-training inference timing (best-of)
"""

import time
import math
import pathlib
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
import frameworkc as fc

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Hyperparameters ───────────────────────────────────────────────
EPOCHS       = 128
LR_C_INIT    = 0.05        # keep it simple & constant for Semeion
LR_DECAY     = 1.00        # no decay (dataset is tiny)
BATCH_C      = 64          # more updates/epoch than 256
LR_S         = 0.01
BATCH_S      = 256
DATA_PATH    = pathlib.Path("data/semeion.data")
RNG          = np.random.default_rng(0)
REPEAT       = 30

# ── Helpers: sanitize Semeion labels ─────────────────────────────
def sanitize_labels(T_raw: np.ndarray) -> np.ndarray:
    """
    Ensures one-hot in {0,1} and fixes common Semeion column-order quirk.
    - Some dumps use -1/1 -> map to 0/1
    - Some order labels as [1..9,0] -> rotate last column to index 0
    """
    T = T_raw.astype(np.float32, copy=True)

    # Map -1/1 -> 0/1 if needed
    if T.min() < 0.0:
        T = (T > 0.0).astype(np.float32)

    # If rows aren't one-hot, try rotating (1..9,0 -> 0..9)
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        # try a single right-rotation (last column becomes first)
        T_rot = np.concatenate([T[:, -1:], T[:, :-1]], axis=1)
        row_sums_rot = T_rot.sum(axis=1)
        if np.allclose(row_sums_rot, 1.0):
            T = T_rot
        else:
            # Still not one-hot? fall back to hard one-hot via argmax
            idx = T.argmax(axis=1)
            T = np.eye(T.shape[1], dtype=np.float32)[idx]

    return T

# ── Load and split data ─────────────────────────────────────────
raw = np.loadtxt(DATA_PATH, dtype=np.float32)
X, T = raw[:, :256], raw[:, 256:]
T = sanitize_labels(T)                 # <<< important for Semeion
y = T.argmax(axis=1)

perm  = RNG.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = perm[:split], perm[split:]

X_tr, T_tr, y_tr = X[train_idx], T[train_idx], y[train_idx]
X_te, T_te, y_te = X[test_idx],  T[test_idx],  y[test_idx]

# ── Manual normalization ─────────────────────────────────────────
mu      = X_tr.mean(axis=0, keepdims=True)
sigma   = X_tr.std(axis=0, keepdims=True) + 1e-8
X_tr_n  = ((X_tr - mu) / sigma).astype(np.float32)
X_te_n  = ((X_te - mu) / sigma).astype(np.float32)

# ── Build Framework-C network ────────────────────────────────────
nips, nhid, nops = 256, 28, 10
cnet = fc.build(nips, nhid, nops, 1)  # seed=1 for reproducibility

# ── Build sklearn MLP baseline ───────────────────────────────────
mlp = MLPClassifier(
    hidden_layer_sizes=(nhid,), activation='relu', solver='sgd',
    learning_rate_init=LR_S, momentum=0.9, batch_size=BATCH_S,
    learning_rate='constant', max_iter=1, warm_start=True,
    random_state=0, verbose=False
)

# ── Training loop ────────────────────────────────────────────────
total_c = total_s = 0.0

print("Epoch |   C-loss  C-acc   |   S-logloss  S-acc  | Δt-C(s) | Δt-S(s) |  C-te-acc |  S-te-acc")
print("------+--------------------+--------------------------+---------+---------+-----------+-----------")

for ep in range(1, EPOCHS + 1):
    lr_c = LR_C_INIT * (LR_DECAY ** (ep - 1))

    # Framework-C: shuffle and mini-batch training
    idx = RNG.permutation(len(X_tr_n))
    X_sh, T_sh = X_tr_n[idx], T_tr[idx]

    t0 = time.perf_counter()
    for i in range(0, len(X_sh), BATCH_C):
        xb = X_sh[i:i+BATCH_C]
        tb = T_sh[i:i+BATCH_C]
        fc.train_batch(cnet, xb, tb, lr_c)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    # metrics on training set
    out_c_tr = fc.predict_batch(cnet, X_tr_n)
    loss_c   = float(np.mean((out_c_tr - T_tr) ** 2))
    acc_c    = (out_c_tr.argmax(axis=1) == y_tr).mean() * 100

    # sklearn: shuffle + one-iteration fit
    idx_s      = RNG.permutation(len(X_tr_n))
    X_ep, y_ep = X_tr_n[idx_s], y_tr[idx_s]

    t0 = time.perf_counter()
    mlp.fit(X_ep, y_ep)
    dt_s = time.perf_counter() - t0
    total_s += dt_s

    proba_s_tr = mlp.predict_proba(X_ep)
    loss_s     = log_loss(y_ep, proba_s_tr)
    acc_s      = mlp.score(X_ep, y_ep) * 100

    # test-set evaluations
    out_c_te = fc.predict_batch(cnet, X_te_n)
    acc_c_te = (out_c_te.argmax(axis=1) == y_te).mean() * 100
    acc_s_te = mlp.score(X_te_n, y_te) * 100

    print(f"{ep:5d} | {loss_c:8.4f} {acc_c:7.2f} |"
          f" {loss_s:12.4f} {acc_s:7.2f} |"
          f" {dt_c:7.3f} | {dt_s:7.3f} |"
          f" {acc_c_te:9.2f} | {acc_s_te:9.2f}")

# ── Totals & final accuracies ─────────────────────────────────────
print("\n===== Total Training Time =====")
print(f" FRAMEWORK-C : {total_c:.3f} s")
print(f" sklearn MLP : {total_s:.3f} s")

out_c_te = fc.predict_batch(cnet, X_te_n)
acc_c_te = (out_c_te.argmax(axis=1) == y_te).mean() * 100
acc_s_te = mlp.score(X_te_n, y_te) * 100

print("\n===== Final Test Accuracies =====")
print(f" Framework-C Test Accuracy      : {acc_c_te:6.2f}%")
print(f" scikit-learn MLP Test Accuracy : {acc_s_te:6.2f}%")

# ── Inference timing (best-of REPEAT) ─────────────────────────────
def best(fn, repeat=REPEAT):
    best_t = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best_t = min(best_t, time.perf_counter() - t0)
    return best_t * 1e3  # ms

print(f"\n===== Inference Times (best of {REPEAT}) =====")
print(" Path            Time (ms)")
print("-----------------------------")
print(f" C-single    : {best(lambda: [fc.predict(cnet, row.tolist()) for row in X_te_n]):9.3f}")
print(f" C-batch     : {best(lambda: fc.predict_batch(cnet, X_te_n)):9.3f}")
print(f" S-single    : {best(lambda: [mlp.predict([row]) for row in X_te_n]):9.3f}")
print(f" S-batch     : {best(lambda: mlp.predict_proba(X_te_n)):9.3f}")
print("==========================================")
