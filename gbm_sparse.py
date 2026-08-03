"""gbm_sparse.py -- how small can a feature model get before it stops competing?

Clarifying the variants, because the names in earlier runs were inconsistent:

  default (83 feat)      all usable features, stock hyperparameters (800 trees x 63 leaves)
  Optuna-tuned (83)      same features, hyperparameters searched on validation
  SFS-selected (19)      sequential forward selection down to 19 features, stock hyperparameters
  single tree (12)       ONE tree, 32 leaves, on the 12 features that tree chose to split on

Only the feature set and the model size differ; the training data is identical throughout.

This module sweeps both axes together -- feature count from 4 to 83, and model size from a single
32-leaf tree to a full ensemble -- so the accuracy cost of shrinking is visible rather than
inferred from four scattered points. Feature order comes from split gain on the full model, which
is a cheap stand-in for SFS and gives the same ranking at the top.

Reported on the PulseDB protocols (ID VitalDB, OOD MIMIC-BP) and on the calibration axis, since
those order models differently.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import mechlib
import lightgbm_arm as gbm
import eval_protocols as ep
from gbm_mechanism import plain

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=80000)
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    OOD = pickle.load(open(DATA / "_variants_ood_feats.pkl", "rb"))
    Fm, ym, gm = OOD["MIMIC-BP"]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]
    ntr = min(args.train_n, len(ytr))

    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]

    def tbl(F, ks, n):
        return np.column_stack([np.asarray(F.get(k, np.full(n, np.nan)), float)[:n] for k in ks])

    # rank features once on the full model, then take prefixes
    Mtr = tbl(Ftr, keys, ntr)
    med0 = gbm.column_medians(Mtr)
    rank_m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, random_state=0,
                               verbosity=-1)
    rank_m.fit(gbm._impute(Mtr, med0), ytr[:ntr, TARGET])
    order = np.argsort(-rank_m.booster_.feature_importance("gain"))
    ranked = [keys[i] for i in order]
    print("[rank] top 12 by gain:")
    for i, k in enumerate(ranked[:12], 1):
        print(f"   {i:2d}. {k:16s} {plain(k)[:46]}", flush=True)

    SIZES = {
        "single tree (32 lv)": dict(n_estimators=1, learning_rate=1.0, num_leaves=32),
        "single tree (64 lv)": dict(n_estimators=1, learning_rate=1.0, num_leaves=64),
        "20 trees": dict(n_estimators=20, learning_rate=0.25, num_leaves=31),
        "100 trees": dict(n_estimators=100, learning_rate=0.10, num_leaves=31),
        "full ensemble": dict(n_estimators=600, learning_rate=0.04, num_leaves=63),
    }
    NFEAT = [4, 8, 12, 19, 40, len(keys)]

    rows = []
    print(f"\n{'size':22s} {'feat':>5s} {'leaves':>8s} {'ID':>6s} {'OOD':>7s} "
          f"{'cal k=5':>8s} {'k=20':>7s}")
    print("-" * 68)
    for sname, params in SIZES.items():
        for nf in NFEAT:
            ks = ranked[:nf]
            M = tbl(Ftr, ks, ntr)
            med = gbm.column_medians(M)
            m = lgb.LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=50, random_state=0, verbosity=-1)
            m.fit(gbm._impute(M, med), ytr[:ntr, TARGET])
            pid = m.predict(gbm._impute(tbl(Fte, ks, len(yte)), med))
            idm = float(np.abs(pid - yte[:, TARGET]).mean())
            pm = m.predict(gbm._impute(tbl(Fm, ks, len(ym)), med))
            ood = float(np.median([np.mean(np.abs(pm[gm == s] - ym[gm == s, TARGET]))
                                   for s in np.unique(gm) if (gm == s).sum() >= 3]))
            cur = ep.anchor_curve(pid, yte[:, TARGET], gte, min_seg=60)
            lv = int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]))
            rows.append({"size": sname, "n_feat": nf, "leaves": lv, "id": idm, "ood": ood,
                         "cal5": cur[5], "cal20": cur[20], "features": ks})
            print(f"{sname:22s} {nf:5d} {lv:8,d} {idm:6.2f} {ood:7.2f} {cur[5]:8.2f} "
                  f"{cur[20]:7.2f}", flush=True)

    (DATA / "gbm_sparse.json").write_text(json.dumps(rows, indent=2, default=float))

    best_id = min(rows, key=lambda r: r["id"])
    small = [r for r in rows if r["leaves"] <= 100]
    print(f"\nbest ID overall: {best_id['size']}, {best_id['n_feat']} feat, "
          f"{best_id['leaves']:,} leaves -> {best_id['id']:.2f}")
    if small:
        b = min(small, key=lambda r: r["id"])
        print(f"best under 100 leaves: {b['size']}, {b['n_feat']} feat -> ID {b['id']:.2f}, "
              f"OOD {b['ood']:.2f}")
        print(f"  uses: {', '.join(plain(k) for k in b['features'][:5])}")
    print("\ndeep nets: ID 8.45-8.69, OOD 11.80-13.58 (472k-1.8M parameters)")
    print(f"\n[done] data/gbm_sparse.json")


if __name__ == "__main__":
    main()
