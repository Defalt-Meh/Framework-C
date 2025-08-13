#!/usr/bin/env python3
"""
benchmark_afd.py — APS Failure (Scania)
FRAMEWORK-C (frameworkc) vs. PyTorch vs. scikit-learn.

Optimizations:
- Torch: AdamW + cosine LR, label smoothing, WeightedRandomSampler (balanced),
  early stopping on val AUROC (patience=6)
- sklearn: Adam (adaptive) with tuned LR, batch_size=256, one partial_fit per epoch,
  early stopping on val AUROC (patience=6)

Everything runs CPU, single-thread for fair timing.
"""

import os, csv, time, copy
from pathlib import Path
import numpy as np

# Repro + single-thread fairness
SEED = 42
np.random.seed(SEED)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_NUM_THREADS", "1")

# ─── Imports ───────────────────────────────────────────────────────
try:
    import frameworkc as fc
except Exception as e:
    raise SystemExit("Build `frameworkc` first via setup.py. Error: %s" % e)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import roc_auc_score, average_precision_score
try:
    from sklearn.neural_network import MLPClassifier
    SK_AVAILABLE = True
except Exception:
    SK_AVAILABLE = False

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path("data/aps_failure_test_set")
TRAIN_CSV = DATA_DIR / "aps_failure_training_set.csv"
TEST_CSV  = DATA_DIR / "aps_failure_test_set.csv"

EPOCHS = 128           # early stopping usually halts before this
NHID1, NHID2 = 256, 64
LR_C = 0.02
LR_T = 0.0035         # tuned for AdamW + cosine
WD_T = 1e-4

BATCH_C = 256         # may be overwritten by OAT
BATCH_T = 256         # torch batch size
BATCH_S = 256         # sklearn batch size

DEVICE = "cpu"
torch.manual_seed(SEED)
torch.set_num_threads(1)

FP_COST, FN_COST = 10, 500   # APS official cost metric
PATIENCE = 6                 # early stopping patience (epochs)

# ─── Data loading / preprocessing ──────────────────────────────────
def load_aps_csv(path: Path):
    """Return X (float32), y (int64), and feature names. 'class' is 'pos'/'neg'."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            y_idx = header.index("class")
        except ValueError:
            y_idx = 0
        feat_idx = [i for i in range(len(header)) if i != y_idx]
        feat_names = [header[i] for i in feat_idx]

        X_rows, y_rows = [], []
        for row in reader:
            if not row:
                continue
            lab = row[y_idx].strip()
            y_rows.append(1 if lab == "pos" else 0)
            vals = []
            for i in feat_idx:
                s = row[i].strip()
                if s == "na" or s == "" or s.lower() == "nan":
                    vals.append(np.nan)
                else:
                    try: vals.append(float(s))
                    except: vals.append(np.nan)
            X_rows.append(vals)

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y, feat_names

def fit_imputer_and_scaler(X):
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X_imp = np.where(np.isnan(X), med, X)
    mean = X_imp.mean(axis=0)
    std  = X_imp.std(axis=0)
    std  = np.where(std > 0, std, 1.0)
    return med, mean, std

def transform_with(med, mean, std, X):
    X = np.where(np.isnan(X), med, X)
    X = (X - mean) / std
    return X.astype(np.float32, copy=False)

def stratified_split(y, val_frac=0.10, seed=0):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]; rng.shuffle(idx0)
    idx1 = np.where(y == 1)[0]; rng.shuffle(idx1)
    n0v = int(len(idx0) * val_frac)
    n1v = int(len(idx1) * val_frac)
    va = np.concatenate([idx0[:n0v], idx1[:n1v]])
    tr = np.concatenate([idx0[n0v:], idx1[n1v:]])
    rng.shuffle(va); rng.shuffle(tr)
    return tr, va

def probs_from_logits_2way(logits_np):
    m = logits_np.max(axis=1, keepdims=True)
    e = np.exp(logits_np - m)
    return e[:, 1] / e.sum(axis=1, keepdims=True)[:, 0]

def find_best_threshold(y_true, pos_scores, fp_cost=10, fn_cost=500):
    order = np.argsort(pos_scores)[::-1]
    y_sorted = y_true[order]
    scores_sorted = pos_scores[order]
    P = int(y_true.sum())
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    costs = [fp_cost * 0 + fn_cost * P]
    thrs  = [1.01]  # predict all neg
    for k in range(1, len(y_true) + 1):
        tp = int(tps[k - 1]); fp = int(fps[k - 1]); fn = P - tp
        costs.append(fp_cost * fp + fn_cost * fn)
        thr = 0.5 * (scores_sorted[k - 1] + scores_sorted[k]) if k < len(y_true) else -1.0
        thrs.append(thr)

    i = int(np.argmin(costs))
    return float(thrs[i]), int(costs[i])

# ─── Load & prep ───────────────────────────────────────────────────
X_tr_full, y_tr_full, feat_names = load_aps_csv(TRAIN_CSV)
X_te,        y_te,    _          = load_aps_csv(TEST_CSV)

med, mean, std = fit_imputer_and_scaler(X_tr_full)
X_tr_full = transform_with(med, mean, std, X_tr_full)
X_te      = transform_with(med, mean, std, X_te)

tr_idx, va_idx = stratified_split(y_tr_full, val_frac=0.10, seed=SEED)
X_tr, y_tr = X_tr_full[tr_idx], y_tr_full[tr_idx]
X_va, y_va = X_tr_full[va_idx], y_tr_full[va_idx]

NIPS = X_tr.shape[1]
NOPS = 2

# one-hot for C backend
I = np.eye(NOPS, dtype=np.float32)

# Torch tensors
X_tr_t = torch.from_numpy(X_tr.copy()).to(DEVICE)
y_tr_t = torch.from_numpy(y_tr.copy()).long().to(DEVICE)
X_va_t = torch.from_numpy(X_va.copy()).to(DEVICE)
y_va_t = torch.from_numpy(y_va.copy()).long().to(DEVICE)
X_te_t = torch.from_numpy(X_te.copy()).to(DEVICE)
y_te_t = torch.from_numpy(y_te.copy()).long().to(DEVICE)

# ─── Build models ──────────────────────────────────────────────────
# FRAMEWORK-C (use 2-layer if available)
try:
    net_c = fc.build2(NIPS, NHID1, NHID2, NOPS, SEED)
    print("[FWC] Using 2-layer build (NHID1, NHID2) =", NHID1, NHID2)
except Exception:
    net_c = fc.build(NIPS, NHID1, NOPS, SEED)
    print("[FWC] Using 1-layer build (NHID1) =", NHID1)

# OAT calibration
info = None
try:
    if hasattr(fc, "calibrate"):          info = fc.calibrate(net_c)
    elif hasattr(fc, "oat_calibrate"):     info = fc.oat_calibrate(net_c)
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
    print("[OAT] not available (FWC_OAT off)")

# ─── PyTorch model (optimized) ─────────────────────────────────────
class Net(nn.Module):
    def __init__(self, d, h1, h2, k):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, k)
        )
    def forward(self, x): return self.seq(x)

net_t = Net(NIPS, NHID1, NHID2, NOPS).to(DEVICE)

# Balanced sampling to fight class imbalance (50/50 expectation)
n_pos = int((y_tr == 1).sum()); n_neg = int((y_tr == 0).sum())
w_pos = 0.5 / max(n_pos, 1)
w_neg = 0.5 / max(n_neg, 1)
weights = np.where(y_tr == 1, w_pos, w_neg).astype(np.float64)
sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                    batch_size=BATCH_T, sampler=sampler,
                    num_workers=0, persistent_workers=False)

# Loss: class-weighted + label smoothing for calibration
cw = torch.tensor([1.0, max(n_neg / max(n_pos,1), 1.0)], dtype=torch.float32, device=DEVICE)
loss_t = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

# Optimizer + LR schedule
opt_t  = optim.AdamW(net_t.parameters(), lr=LR_T, weight_decay=WD_T)
sched_t = optim.lr_scheduler.CosineAnnealingLR(opt_t, T_max=EPOCHS, eta_min=LR_T*0.1)

# ─── scikit-learn model (optimized) ────────────────────────────────
if SK_AVAILABLE:
    sk = MLPClassifier(
        hidden_layer_sizes=(NHID1, NHID2),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=BATCH_S,
        learning_rate="adaptive",      # decays on plateaus
        learning_rate_init=0.003,      # saner for APS
        beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        max_iter=1, warm_start=True, shuffle=True,
        random_state=SEED, verbose=False, tol=0.0
    )
    SK_CLASSES = np.arange(NOPS, dtype=np.int64)
else:
    sk = None

# ─── Epoch 0 baseline ─────────────────────────────────────────────
def epoch0_auc():
    out_va_c = fc.predict_batch(net_c, X_va)
    p_va_c   = probs_from_logits_2way(out_va_c)
    net_t.eval()
    with torch.inference_mode():
        p_va_t = probs_from_logits_2way(net_t(X_va_t).cpu().numpy())
    return roc_auc_score(y_va, p_va_c), roc_auc_score(y_va, p_va_t)

auc_c0, auc_t0 = epoch0_auc()
print("Epoch |   C-AUROC |  T-AUROC | SK-AUROC |   C-Cost |   T-Cost |  SK-Cost | Δt-C(s) | Δt-T(s) | Δt-S(s)")
print("------+-----------+----------+----------+----------+----------+----------+---------+---------+--------")
print(f"{0:5d} | {auc_c0:9.4f} | {auc_t0:8.4f} | {'   n/a' if sk is None else '  n/a':>8} | "
      f"{0:8d} | {0:8d} | {'   n/a':>8} | {0.0:7.3f} | {0.0:7.3f} | {0.0:6.3f}")

# ─── Training loop with early stopping ────────────────────────────
total_c = total_t = total_s = 0.0
n_train = X_tr.shape[0]
best_t = -np.inf; best_t_state = None; t_bad = 0
best_sk = -np.inf; best_sk_state = None; s_bad = 0

for ep in range(1, EPOCHS + 1):
    lr_c = LR_C * (0.95 ** (ep - 1))

    # FRAMEWORK-C
    t0 = time.perf_counter()
    perm = np.random.permutation(n_train)
    for i in range(0, n_train, BATCH_C):
        sel = perm[i:i+BATCH_C]
        fc.train_batch(net_c, X_tr[sel], I[y_tr[sel]], lr_c)
    dt_c = time.perf_counter() - t0
    total_c += dt_c

    # Torch (balanced batches)
    t0 = time.perf_counter()
    net_t.train()
    for xb_t, yb_t in loader:
        opt_t.zero_grad(set_to_none=True)
        out = net_t(xb_t)
        loss = loss_t(out, yb_t)
        loss.backward()
        opt_t.step()
    sched_t.step()
    dt_t = time.perf_counter() - t0
    total_t += dt_t

    # sklearn: one partial_fit per epoch
    if sk is not None:
        t0 = time.perf_counter()
        if ep == 1:
            sk.partial_fit(X_tr, y_tr, classes=SK_CLASSES)
        else:
            sk.partial_fit(X_tr, y_tr)
        dt_s = time.perf_counter() - t0
        total_s += dt_s
    else:
        dt_s = 0.0

    # Validation metrics + early stopping
    p_va_c = probs_from_logits_2way(fc.predict_batch(net_c, X_va))
    auc_c  = roc_auc_score(y_va, p_va_c)
    thr_c, cost_c = find_best_threshold(y_va, p_va_c, FP_COST, FN_COST)

    net_t.eval()
    with torch.inference_mode():
        p_va_t = probs_from_logits_2way(net_t(X_va_t).cpu().numpy())
    auc_t = roc_auc_score(y_va, p_va_t)
    thr_t, cost_t = find_best_threshold(y_va, p_va_t, FP_COST, FN_COST)

    if sk is not None:
        p_va_s = sk.predict_proba(X_va)[:, 1]
        auc_s  = roc_auc_score(y_va, p_va_s)
        thr_s, cost_s = find_best_threshold(y_va, p_va_s, FP_COST, FN_COST)
        auc_s_str, cost_s_str = f"{auc_s:8.4f}", f"{cost_s:8d}"
    else:
        auc_s_str, cost_s_str = f"{'   n/a':>8}", f"{'   n/a':>8}"
        auc_s = -np.inf

    print(f"{ep:5d} | {auc_c:9.4f} | {auc_t:8.4f} | {auc_s_str} | "
          f"{cost_c:8d} | {cost_t:8d} | {cost_s_str} | "
          f"{dt_c:7.3f} | {dt_t:7.3f} | {dt_s:6.3f}")

    # Track best Torch
    if auc_t > best_t + 1e-4:
        best_t = auc_t
        best_t_state = copy.deepcopy(net_t.state_dict())
        t_bad = 0
    else:
        t_bad += 1

    # Track best sklearn
    if sk is not None:
        if auc_s > best_sk + 1e-4:
            best_sk = auc_s
            # store coefs_ / intercepts_ (lighter than pickling whole estimator)
            best_sk_state = ( [c.copy() for c in sk.coefs_],
                              [b.copy() for b in sk.intercepts_] )
            s_bad = 0
        else:
            s_bad += 1

    # Early stop when both (Torch, sklearn) have plateaued; C is fast anyway
    if t_bad >= PATIENCE + 14 and (sk is None or s_bad >= PATIENCE + 14):
        print(f"[Early stop] No AUROC improvement for {PATIENCE + 14} epochs.")
        break

# Restore best Torch / sklearn states (if improved)
if best_t_state is not None:
    net_t.load_state_dict(best_t_state)
if sk is not None and best_sk_state is not None:
    sk.coefs_, sk.intercepts_ = best_sk_state

# ─── Final test metrics ───────────────────────────────────────────
# Thresholds from validation
p_va_c = probs_from_logits_2way(fc.predict_batch(net_c, X_va))
thr_c, _ = find_best_threshold(y_va, p_va_c, FP_COST, FN_COST)
p_te_c = probs_from_logits_2way(fc.predict_batch(net_c, X_te))
pred_c = (p_te_c >= thr_c).astype(np.int64)
cost_c_te = FP_COST * ((pred_c==1)&(y_te==0)).sum() + FN_COST * ((pred_c==0)&(y_te==1)).sum()
acc_c_te  = (pred_c == y_te).mean() * 100
auc_c_te  = roc_auc_score(y_te, p_te_c)
apr_c_te  = average_precision_score(y_te, p_te_c)

net_t.eval()
with torch.inference_mode():
    p_va_t = probs_from_logits_2way(net_t(X_va_t).cpu().numpy())
thr_t, _ = find_best_threshold(y_va, p_va_t, FP_COST, FN_COST)
with torch.inference_mode():
    p_te_t = probs_from_logits_2way(net_t(X_te_t).cpu().numpy())
pred_t = (p_te_t >= thr_t).astype(np.int64)
cost_t_te = FP_COST * ((pred_t==1)&(y_te==0)).sum() + FN_COST * ((pred_t==0)&(y_te==1)).sum()
acc_t_te  = (pred_t == y_te).mean() * 100
auc_t_te  = roc_auc_score(y_te, p_te_t)
apr_t_te  = average_precision_score(y_te, p_te_t)

if SK_AVAILABLE and sk is not None:
    p_va_s = sk.predict_proba(X_va)[:,1]
    thr_s, _ = find_best_threshold(y_va, p_va_s, FP_COST, FN_COST)
    p_te_s  = sk.predict_proba(X_te)[:,1]
    pred_s  = (p_te_s >= thr_s).astype(np.int64)
    cost_s_te = FP_COST * ((pred_s==1)&(y_te==0)).sum() + FN_COST * ((pred_s==0)&(y_te==1)).sum()
    acc_s_te  = (pred_s == y_te).mean() * 100
    auc_s_te  = roc_auc_score(y_te, p_te_s)
    apr_s_te  = average_precision_score(y_te, p_te_s)
else:
    cost_s_te = float("nan"); acc_s_te = float("nan"); auc_s_te = float("nan"); apr_s_te = float("nan")

print("\n===== Test Metrics (val-tuned threshold) =====")
print(f"{'Path':<12} {'Acc(%)':>8} {'AUROC':>8} {'PR-AUC':>8} {'APS-Cost':>10}")
print(f"{'C-Net':<12} {acc_c_te:8.2f} {auc_c_te:8.4f} {apr_c_te:8.4f} {cost_c_te:10d}")
print(f"{'Torch':<12} {acc_t_te:8.2f} {auc_t_te:8.4f} {apr_t_te:8.4f} {cost_t_te:10d}")
print(f"{'sklearn':<12} "
      f"{('   n/a' if not np.isfinite(acc_s_te) else f'{acc_s_te:8.2f}')} "
      f"{('   n/a' if not np.isfinite(auc_s_te) else f'{auc_s_te:8.4f}')} "
      f"{('   n/a' if not np.isfinite(apr_s_te) else f'{apr_s_te:8.4f}')} "
      f"{('      n/a' if not np.isfinite(cost_s_te) else f'{int(cost_s_te):10d}')}")

# ─── Inference timings (best-of-20) ───────────────────────────────
def bench(fn, reps=20, warmup=3):
    for _ in range(warmup): fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        best = min(best, time.perf_counter() - t0)
    return best

def bench_torch(forward, reps=20, warmup=3):
    with torch.inference_mode():
        for _ in range(warmup): forward()
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter(); forward()
            best = min(best, time.perf_counter() - t0)
    return best

Ntest = X_te.shape[0]
# Torch inference-optimized copy (optional JIT; safe to skip if it errors)
net_t_inf = net_t
try:
    net_t_inf = torch.jit.script(net_t.eval())
    net_t_inf = torch.jit.optimize_for_inference(net_t_inf)
except Exception:
    net_t_inf = net_t.eval()

inf_c = bench(lambda: fc.predict_batch(net_c, X_te))
inf_t = bench_torch(lambda: net_t_inf(X_te_t))
if SK_AVAILABLE and sk is not None:
    inf_s = bench(lambda: sk.predict_proba(X_te))
else:
    inf_s = float("nan")

scale = 10000.0 / Ntest
print(f"\n===== Inference Times (best of 20) =====")
print(f"{'Path':<12} {'Time (ms) /10k':>16}")
print(f"{'C-batch':<12} {inf_c*1e3*scale:16.3f}")
print(f"{'Torch-batch':<12} {inf_t*1e3*scale:16.3f}")
print(f"{'SK-batch':<12} {('      n/a' if not np.isfinite(inf_s) else f'{inf_s*1e3*scale:16.3f}')}")
print(f"\n===== Training Time =====")
print(f"{'C (total)':<12} {total_c:8.3f} s   (avg/epoch {total_c/max(1,ep):.3f} s)")
print(f"{'Torch':<12} {total_t:8.3f} s   (avg/epoch {total_t/max(1,ep):.3f} s)")
print(f"{'sklearn':<12} {('    n/a' if not np.isfinite(inf_s) else f'{total_s:8.3f}')} s   "
      f"(avg/epoch {('n/a' if not np.isfinite(inf_s) else f'{total_s/max(1,ep):.3f}')} s)")
