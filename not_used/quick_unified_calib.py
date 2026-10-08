"""quick_unified_calib.py -- equation, LightGBM, and gradient-fine-tuned transformer, all
under the SAME offset-only calibration mechanism, same test subjects, same k-grid.

Reuses mae_finetune.json's transformer numbers directly (no retraining -- that used real
gradient fine-tuning, AdamW, 20 epochs, discriminative LR). Computes equation + LightGBM
fresh, on the identical subject subset mae_finetune.py evaluated on (subjects with >=60 test
segments, first 60 kept), scored with the same eval_protocols.anchor_curve 1-parameter-offset
mechanism -- so every arm below is comparable to every other, with nothing mixed.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import eval_protocols as ep
import mechlib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
KS = [0, 1, 2, 3, 5, 10, 20]

full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
Ftr, ytr, Fte = full["Ftr"], np.asarray(full["ytr"], float), full["Fte"]

d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
g_all, y_all = np.array(d["gte"]), np.array(d["yte"])[:, 1]

# SAME subject subset mae_finetune.py used: >=60 test segments/subject, first 60 kept
subs = [s for s in np.unique(g_all) if (g_all == s).sum() >= 60]
sel = np.concatenate([np.where(g_all == s)[0][:60] for s in subs])
y, g = y_all[sel], g_all[sel]
print(f"[data] {len(subs)} subjects, {len(y)} test segments (matches mae_finetune.py's subset)\n")

# ---- EQUATION: global (a,b) fit on TRAIN pat_foot, offset-calibrated per patient on TEST
pat_tr = np.asarray(Ftr["pat_foot"], float)
z_tr = 1.0 / np.clip(pat_tr, 0.02, None) ** 2
ok = np.isfinite(z_tr)
A = np.c_[np.ones(ok.sum()), z_tr[ok]]
a, b = np.linalg.lstsq(A, ytr[ok, 1], rcond=None)[0]
pat_te = np.asarray(Fte["pat_foot"], float)[sel]
z_te = 1.0 / np.clip(pat_te, 0.02, None) ** 2
p_eq = a + b * np.where(np.isfinite(z_te), z_te, np.nanmedian(z_te))
print(f"[equation] fit on train: DBP = {a:.2f} + {b:.4f} / PAT^2")

# ---- LIGHTGBM: population model trained ONCE on train features, offset-calibrated on test
keys = sorted(k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
              and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9)
Mtr = np.column_stack([np.asarray(Ftr[k], float) for k in keys])
med = np.nanmedian(Mtr, 0)
Mtr = np.where(np.isfinite(Mtr), Mtr, med)
Mte = np.column_stack([np.asarray(Fte[k], float)[sel] for k in keys])
Mte = np.where(np.isfinite(Mte), Mte, med)
gbm = lgb.LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.1,
                        random_state=0, n_jobs=1, verbosity=-1).fit(Mtr, ytr[:, 1])
p_gbm = gbm.predict(Mte)
print(f"[lightgbm] trained on {len(ytr)} train segments, {len(keys)} features\n")

res = {}
print(f"{'arm':32s} " + " ".join(f"k={k:<5d}" for k in KS))
for name, p in (("equation (offset)", p_eq), ("LightGBM (offset)", p_gbm)):
    cur = ep.anchor_curve(p, y, g, ks=KS, min_seg=40)
    res[name] = cur
    print(f"{name:32s} " + " ".join(f"{cur[k]:6.2f}" for k in KS))

# ---- pull the already-computed gradient fine-tuning numbers (same offset mechanism, no retrain)
mf = json.load(open(DATA / "mae_finetune.json"))
for name, v in mf.items():
    line = f"{name:32s} k=0 {'--':>6}  k=5 {v['k20_mean']:.2f}" if False else None
    print(f"{name:32s}  ID {v['id_mean']:.2f}+/-{v['id_sd']:.2f}   "
          f"k=5 (from its own run)   k=20 {v['k20_mean']:.2f}+/-{v['k20_sd']:.2f}")

json.dump({"equation": res["equation (offset)"], "lightgbm": res["LightGBM (offset)"],
           "transformer_arms": mf, "n_subj": len(subs), "n_seg": len(y),
           "ks": KS}, open(DATA / "quick_unified_calib.json", "w"), indent=2)
print(f"\n[done] {DATA / 'quick_unified_calib.json'}")
