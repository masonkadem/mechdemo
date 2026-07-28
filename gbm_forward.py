"""gbm_forward.py -- feature-selection study for the interpretable LightGBM, benchmarked
against the paper's CalibFree deep models.

Three models, one figure/table:
  1. paper deep models    : the published CalibFree Vital MAE (fixed reference numbers)
  2. LightGBM (all cues)  : kitchen-sink -- every physiological cue + demographics
  3. LightGBM (forward)   : greedy forward selection -- start empty, add the single feature
                            that most improves held-out DBP MAE, stop when adding the best
                            remaining feature no longer helps (patience). Yields the minimal
                            feature set and the accuracy-vs-#features curve.

Uses the cached cue features from gbm_families.py (data/_fam_cues_*.pkl) so it runs fast.

    python gbm_forward.py --cache data/_fam_cues_vitaldb_full_calfree_3000.pkl
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent

# paper Table 3, CalibFree Vital (in-distribution), MAE mmHg -- fixed reference
PAPER_CALFREE = {
    "Baseline (median)": (14.88, 9.44),
    "LeNet1d":           (12.37, 7.89),
    "XResNet1d50":       (12.40, 7.85),
    "XResNet1d101":      (12.70, 8.05),
    "Inception1d":       (14.54, 10.96),
    "S4":                (12.39, 8.03),
}


def cv_dbp_mae(Xtr, ytr, Xva, yva):
    """DBP-only quick GBM MAE on the held-out split (target index 1)."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    m.fit(Xtr, ytr[:, 1], eval_set=[(Xva, yva[:, 1])],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return float(np.abs(m.predict(Xva) - yva[:, 1]).mean())


def forward_select(sctr, ytr, dmtr, scva, yva, dmva, all_feats, patience=2, min_gain=0.02):
    """Greedy forward selection on DBP MAE. Returns (order, curve) where order is the feature
    added at each step and curve[i] is the val MAE after i+1 features. Stops when the best
    candidate improves MAE by < min_gain for `patience` consecutive steps."""
    chosen, order, curve = [], [], []
    remaining = list(all_feats)
    best_so_far = np.inf
    stale = 0
    ntr, nva = len(ytr), len(yva)
    while remaining:
        scores = []
        for f in remaining:
            trial = chosen + [f]
            # separate cue features from demographics; both flow through build_feature_table
            cues = [c for c in trial if c not in ("age", "sex", "bmi")]
            dkeys = tuple(c for c in trial if c in ("age", "sex", "bmi"))
            Xtr, _ = gbm.build_feature_table(sctr, dmtr, cues, n=ntr, demo_keys=dkeys)
            Xva, _ = gbm.build_feature_table(scva, dmva, cues, n=nva, demo_keys=dkeys)
            scores.append((cv_dbp_mae(gbm._impute(Xtr), ytr, gbm._impute(Xva), yva), f))
        scores.sort()
        best_mae, best_f = scores[0]
        gain = best_so_far - best_mae
        chosen.append(best_f); order.append(best_f); curve.append(best_mae)
        remaining.remove(best_f)
        if gain < min_gain:
            stale += 1
            if stale >= patience:
                break
        else:
            stale = 0
        best_so_far = min(best_so_far, best_mae)
    return order, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/_fam_cues_vitaldb_full_calfree_3000.pkl")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.02)
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        blob = pickle.load(f)
    sctr, ytr, gtr, dmtr = blob["tr"]
    scva, yva, gva, dmva = blob["va"]
    scte, yte, gte, dmte = blob["te"]

    # candidate feature universe: every cue that is finite on enough segments + demographics
    cue_feats = [k for k, v in sctr.items()
                 if np.isfinite(np.asarray(v, float)).mean() > 0.3]
    demo_feats = [k for k in ("age", "sex", "bmi") if dmtr and k in dmtr]
    all_feats = cue_feats + demo_feats
    print(f"[fwd] candidate features ({len(all_feats)}): {all_feats}")

    # ---- model 2: all features
    Xtr, names = gbm.build_feature_table(sctr, dmtr, cue_feats, n=len(ytr),
                                         demo_keys=tuple(demo_feats))
    Xva, _ = gbm.build_feature_table(scva, dmva, cue_feats, n=len(yva), demo_keys=tuple(demo_feats))
    Xte, _ = gbm.build_feature_table(scte, dmte, cue_feats, n=len(yte), demo_keys=tuple(demo_feats))
    models, _ = gbm.train_gbm(gbm._impute(Xtr), ytr, gbm._impute(Xva), yva, names)
    all_mae = np.abs(gbm.predict_gbm(models, gbm._impute(Xte)) - yte).mean(0)
    print(f"[fwd] ALL features ({len(names)}): ID SBP {all_mae[0]:.2f} / DBP {all_mae[1]:.2f}")

    # ---- model 3: forward selection
    print("[fwd] running greedy forward selection on DBP...")
    order, curve = forward_select(sctr, ytr, dmtr, scva, yva, dmva, all_feats,
                                  patience=args.patience, min_gain=args.min_gain)
    # the selected set is where the curve stopped improving (drop the trailing stale adds)
    best_i = int(np.argmin(curve))
    selected = order[:best_i + 1]
    # retrain selected on test
    Xtr, nm = gbm.build_feature_table(sctr, dmtr, [c for c in selected if c not in demo_feats],
                                      n=len(ytr), demo_keys=tuple(f for f in demo_feats if f in selected))
    Xva, _ = gbm.build_feature_table(scva, dmva, [c for c in selected if c not in demo_feats],
                                     n=len(yva), demo_keys=tuple(f for f in demo_feats if f in selected))
    Xte, _ = gbm.build_feature_table(scte, dmte, [c for c in selected if c not in demo_feats],
                                     n=len(yte), demo_keys=tuple(f for f in demo_feats if f in selected))
    smodels, _ = gbm.train_gbm(gbm._impute(Xtr), ytr, gbm._impute(Xva), yva, nm)
    sel_mae = np.abs(gbm.predict_gbm(smodels, gbm._impute(Xte)) - yte).mean(0)
    print(f"[fwd] SELECTED ({len(selected)}): {selected}")
    print(f"[fwd] SELECTED: ID SBP {sel_mae[0]:.2f} / DBP {sel_mae[1]:.2f}")

    print("\n[fwd] forward-selection DBP curve (val):")
    for i, (f, m) in enumerate(zip(order, curve)):
        mark = " <- selected stop" if i == best_i else ""
        print(f"   {i+1:2d}. +{f:<12s} DBP MAE {m:.3f}{mark}")

    out = {
        "paper_calfree": PAPER_CALFREE,
        "gbm_all": {"n": len(names), "sbp": float(all_mae[0]), "dbp": float(all_mae[1]),
                    "features": names},
        "gbm_forward": {"order": order, "curve": curve, "selected": selected,
                        "n": len(selected), "sbp": float(sel_mae[0]), "dbp": float(sel_mae[1])},
    }
    (ROOT / "data" / "gbm_forward.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] data/gbm_forward.json")


if __name__ == "__main__":
    main()
