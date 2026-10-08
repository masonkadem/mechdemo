"""sfs_select.py -- forward sequential feature selection, chosen on TRAIN only.

gbm_sparse.json shows 4 features beating 83 on OOD (12.09 vs 13.49) and on calibrated error
(5.49 vs 5.93). But its subsets are nested -- 4 is a prefix of 8 is a prefix of 12 -- so they
came from ranking features by importance, not from searching. Ranking cannot see interactions:
two features that are each mediocre alone but complementary together are never found, and two
features that are individually strong but redundant are both kept. The non-monotonic OOD
column (12.09, 12.40, 13.70, 12.57) is the signature of that.

Forward selection searches instead: at each step, try every remaining feature and keep the one
that most improves grouped cross-validation error.

The selection NEVER touches MIMIC. Choosing features by out-of-distribution score would use the
test set to build the model, and the generalisation claim would then be circular. Folds are
subject-disjoint, so a patient's own segments cannot leak across the split.

    python sfs_select.py --max-feat 8
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import convert_pulsedb as cp
import run_feature_full_stream as rfs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def cv_mae(M, y, groups, cols, cfg, n_splits=4):
    import lightgbm as lgb
    errs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(M, y, groups=groups):
        m = lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1,
                              **cfg).fit(M[np.ix_(tr, cols)], y[tr])
        errs.append(np.abs(m.predict(M[np.ix_(te, cols)]) - y[te]).mean())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-feat", type=int, default=20,
                    help="hard cap; the search normally stops earlier via --tol/--patience")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="stop once the average CV MAE gain over the last --patience steps "
                         "is below this many mmHg -- i.e. the curve has plateaued")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--sample", type=int, default=40000, help="segments used for the search")
    ap.add_argument("--leaves", type=int, default=32)
    ap.add_argument("--trees", type=int, default=1,
                    help="1 = original minimal-capacity search; e.g. 150 for a regularized "
                         "multi-tree search")
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1.0,
                    help="LightGBM default is 0.1; kept at 1.0 to match --trees 1's original "
                         "single-shot behaviour unless overridden")
    ap.add_argument("--target", choices=["sbp", "dbp"], default="dbp")
    args = ap.parse_args()
    ycol = 0 if args.target == "sbp" else 1

    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = blob["Ftr"], np.asarray(blob["ytr"], float)[:, ycol]
    keys = sorted(k for k in Ftr
                  if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.5
                  and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9)

    _, _, ids_all, _ = rfs.read_meta_cached(cp.DB / "VitalDB_Train_Subset.mat")
    tr_mask, base = rfs.train_mask_like_convert(ids_all)
    sub = base[tr_mask]
    assert len(sub) == len(ytr), f"{len(sub)} ids vs {len(ytr)} rows"

    M = np.column_stack([np.asarray(Ftr[k], float) for k in keys])
    med = np.nanmedian(M, 0)
    M = np.where(np.isfinite(M), M, med)

    # subsample SEGMENTS but keep every patient, so the grouped folds stay meaningful
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ytr))[:args.sample]
    M, y, gsub = M[idx], ytr[idx], sub[idx]
    print(f"[sfs] {len(y)} segments, {len(np.unique(gsub))} patients, {len(keys)} candidates")
    print(f"[sfs] {args.trees} tree(s), {args.leaves} leaves, lr={args.lr}, "
          f"subsample={args.subsample}, reg_alpha={args.reg_alpha}, reg_lambda={args.reg_lambda}, "
          f"4-fold subject-disjoint CV\n")

    cfg = dict(n_estimators=args.trees, learning_rate=args.lr, num_leaves=args.leaves,
               subsample=args.subsample, reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda)
    if args.subsample < 1.0:
        cfg["subsample_freq"] = 1        # LightGBM ignores subsample unless a freq is set
    chosen, remaining, hist = [], list(range(len(keys))), []
    for step in range(args.max_feat):
        best, best_j = np.inf, None
        for j in remaining:
            s = cv_mae(M, y, gsub, chosen + [j], cfg)
            if s < best:
                best, best_j = s, j
        chosen.append(best_j); remaining.remove(best_j)
        hist.append({"n": len(chosen), "added": keys[best_j], "cv_mae": best,
                     "features": [keys[c] for c in chosen]})
        print(f"  {len(chosen)}  +{keys[best_j]:16} CV MAE {best:.4f}", flush=True)

        # Plateau stop: once the average gain per step over the last `patience` steps drops
        # below `tol`, adding more features is no longer buying anything worth its added
        # complexity. Needs at least patience+1 points to measure a trailing average gain.
        if len(hist) > args.patience:
            recent = [hist[-i - 1]["cv_mae"] for i in range(args.patience + 1)][::-1]
            avg_gain = (recent[0] - recent[-1]) / args.patience
            if avg_gain < args.tol:
                print(f"  -- plateau: avg gain over last {args.patience} steps "
                      f"{avg_gain:.4f} < tol {args.tol}, stopping at n={len(chosen)}",
                      flush=True)
                break
        if not remaining:
            print("  -- exhausted all candidate features", flush=True)
            break

    ranked = ["ppg_skew_g", "ppg_p90", "dw10", "pow_hf"]
    have = [i for i, k in enumerate(keys) if k in ranked]
    base_score = cv_mae(M, y, gsub, have, cfg)
    print(f"\n  importance-ranked 4 ({', '.join(ranked)}): CV MAE {base_score:.4f}")
    sfs4 = [h for h in hist if h["n"] == 4]
    if sfs4:
        print(f"  SFS 4              ({', '.join(sfs4[0]['features'])}): "
              f"CV MAE {sfs4[0]['cv_mae']:.4f}")
        print(f"  -> SFS is {base_score - sfs4[0]['cv_mae']:+.4f} mmHg better on train CV")

    out = DATA / f"sfs_selection_trees{args.trees}_{args.target}.json"
    json.dump({"history": hist, "ranked4": ranked, "ranked4_cv": base_score,
               "n_segments": int(len(y)), "n_patients": int(len(np.unique(gsub))),
               "leaves": args.leaves, "trees": args.trees, "lr": args.lr,
               "subsample": args.subsample, "reg_alpha": args.reg_alpha,
               "reg_lambda": args.reg_lambda, "target": args.target,
               "tol": args.tol, "patience": args.patience},
              open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
