"""id_ood_sfs_table.py -- the SAME feature sets, SAME capacities, on ID (CalFree) and OOD
(MIMIC), side by side.

Reuses eval_feature_sets.py's exact SETS/CAPS and its `score()` function unchanged, so the
OOD numbers are identical to feature_set_ood.json and the ID numbers are computed under the
same offset-calibration convention (k=5, gap=5, per-patient offset) -- nothing here mixes
protocols across the two columns the way an earlier ad-hoc check did.

    python id_ood_sfs_table.py --target dbp
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

from eval_feature_sets import STATIC_SETS, CAPS, score, sfs_sets

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["sbp", "dbp"], default="dbp")
    args = ap.parse_args()
    ycol = 0 if args.target == "sbp" else 1

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte = full["Ftr"], np.asarray(full["ytr"], float), full["Fte"]

    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g_id, y_id_all = np.array(d["gte"]), np.array(d["yte"], float)
    rows_id = {p: np.where(g_id == p)[0] for p in np.unique(g_id)}

    caches = sorted(DATA.glob("_mimic_feats_full_*.pkl"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    Fm, Ym, gm = pickle.load(open(caches[-1], "rb"))
    ym = np.asarray(Ym, float)[:, ycol]
    rows_ood = {p: np.where(gm == p)[0] for p in np.unique(gm)}

    SETS = dict(STATIC_SETS); SETS.update(sfs_sets(args.target))
    y_id = y_id_all[:, ycol]

    print(f"target={args.target}\n")
    print(f"{'set':34} {'cap':12} {'ID uncal':>9} {'ID cal(k5)':>11}   "
          f"{'OOD uncal':>9} {'OOD cal(k5)':>11}")
    res = {}
    for name, keys in SETS.items():
        keys = [k for k in keys if k in Ftr and k in Fte and k in Fm]
        if len(keys) < 2:
            continue
        Mtr = np.column_stack([np.asarray(Ftr[k], float)[:len(ytr)] for k in keys])
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        Mid = np.column_stack([np.asarray(Fte[k], float)[:len(y_id)] for k in keys])
        Mid = np.where(np.isfinite(Mid), Mid, med)
        Mood = np.column_stack([np.asarray(Fm[k], float) for k in keys])
        Mood = np.where(np.isfinite(Mood), Mood, med)

        for cap, cfg in CAPS.items():
            m = lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1, **cfg).fit(Mtr, ytr[:, ycol])
            u_id, c_id, _ = score(m.predict(Mid), y_id, rows_id)
            u_ood, c_ood, _ = score(m.predict(Mood), ym, rows_ood)
            res[f"{name} | {cap}"] = {"id_uncal": u_id, "id_cal": c_id,
                                      "ood_uncal": u_ood, "ood_cal": c_ood, "n_feat": len(keys)}
            print(f"{name:34} {cap:12} {u_id:9.2f} {c_id:11.2f}   {u_ood:9.2f} {c_ood:11.2f}")

    out = DATA / f"id_ood_sfs_table_{args.target}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
