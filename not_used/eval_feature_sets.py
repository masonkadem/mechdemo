"""eval_feature_sets.py -- score competing minimal feature sets on MIMIC, once, per target.

Feature sets disagree about what matters, and disagree differently for SBP vs DBP:

  importance-ranked 4   picked by ranking features individually on DBP (gbm_sparse.json);
                        reused as a fixed candidate set for SBP too, so this arm is a control,
                        not something re-derived per target
  SFS (plateau)         forward-selected on TRAIN only, per target, per tree count, stopped
                        automatically once additional features stop paying (sfs_select.py)
  PAT only              the governing-law variable alone

Nothing here is selected on MIMIC. Sets are fixed by sfs_select.py beforehand; MIMIC (a
different hospital, different monitors) is scored once, afterwards.

    python eval_feature_sets.py --target dbp
    python eval_feature_sets.py --target sbp
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

STATIC_SETS = {
    "importance-ranked 4 (morphology, DBP-derived)":
        ["ppg_skew_g", "ppg_p90", "dw10", "pow_hf"],
    "PAT only": ["pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"],
}
CAPS = {"single tree": dict(n_estimators=1, learning_rate=1.0, num_leaves=32),
        "800 trees": dict(n_estimators=800, learning_rate=0.03, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=50)}


def sfs_sets(target):
    """Every completed SFS run for this target: its FULL (plateaued) set and its 4-feature
    prefix, labelled by tree count so single-tree and 150-tree selections are both shown."""
    out = {}
    for path in sorted(DATA.glob(f"sfs_selection_trees*_{target}.json")):
        d = json.loads(path.read_text())
        trees = d.get("trees", 1)
        feats = d["history"][-1]["features"]
        out[f"SFS full ({len(feats)}feat, {trees}t)"] = feats
        if len(feats) > 4:
            out[f"SFS top-4 ({trees}t)"] = d["history"][3]["features"]
    return out


def score(pred, y, rows, k=5, gap=5):
    unc = float(np.median([np.abs(pred[i] - y[i]).mean() for i in rows.values()]))
    cal = [float(np.abs(pred[idx[k + gap:]] + np.mean(y[idx[:k]] - pred[idx[:k]])
                        - y[idx[k + gap:]]).mean())
           for idx in rows.values() if k + gap < len(idx) - 3]
    return unc, (float(np.median(cal)) if cal else np.nan), len(cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sbp", "dbp"], default="dbp")
    args = ap.parse_args()
    ycol = 0 if args.target == "sbp" else 1

    import lightgbm as lgb
    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = blob["Ftr"], np.asarray(blob["ytr"], float)[:, ycol]

    # widest "full" MIMIC cache present (both targets, keyed by patient count)
    caches = sorted(DATA.glob("_mimic_feats_full_*.pkl"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not caches:
        raise SystemExit("no full MIMIC cache yet -- run mimic_ood_calibrated.py first "
                         "(it now caches both SBP and DBP)")
    print(f"[cache] {caches[-1].name}")
    Fm, Ym, gm = pickle.load(open(caches[-1], "rb"))
    ym = np.asarray(Ym, float)[:, ycol]
    rows = {p: np.where(gm == p)[0] for p in np.unique(gm)}
    print(f"[mimic] target={args.target} | {len(ym)} segments, {len(rows)} patients\n")

    SETS = dict(STATIC_SETS)
    SETS.update(sfs_sets(args.target))

    res = {}
    for name, keys in SETS.items():
        keys = [k for k in keys if k in Ftr and k in Fm]
        if len(keys) < 2:
            print(f"  {name}: SKIPPED, features unavailable"); continue
        Mtr = np.column_stack([np.asarray(Ftr[k], float)[:len(ytr)] for k in keys])
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        Mm = np.column_stack([np.asarray(Fm[k], float) for k in keys])
        Mm = np.where(np.isfinite(Mm), Mm, med)
        for cap, cfg in CAPS.items():
            m = lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1, **cfg).fit(Mtr, ytr)
            u, c, n = score(m.predict(Mm), ym, rows)
            res[f"{name} | {cap}"] = {"n_feat": len(keys), "ood_uncal": u, "ood_cal": c,
                                      "n_pat": n, "features": keys}
            print(f"  {name:34} {cap:12} n={len(keys):2d}  OOD {u:6.2f} | calibrated {c:6.2f}")

    out = DATA / f"feature_set_ood_{args.target}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
