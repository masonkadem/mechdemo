"""gbm_capacity_calbased.py -- does PAT-only's OOD advantage also show up as a CalBased edge?

gbm_constrained.json (single tree / 20 trees, id/ood/cal20) has no producer script in this
repo, and it never carried CalBased -- an orphan result, same problem the _calib_*.npy arrays
had. This regenerates it, adds CalBased and AAMI alongside id/ood/cal20, and sweeps capacity
properly instead of two hand-picked settings.

The question: at 800-tree capacity PAT-only is worst on every axis except cross-hospital OOD
(MIMIC). If capacity is really the reason -- models with room to spare fit dataset-specific
structure that does not transfer -- then CalBased and AAMI should move the SAME way as OOD does
as capacity shrinks, since CalBased memorisation is exactly the kind of dataset-specific fit a
constrained model cannot afford. If PAT-only's CalBased score stays bad even at low capacity,
the capacity story is incomplete.

    python gbm_capacity_calbased.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import eval_protocols as ep
import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = 1  # DBP, matching eval_protocols and gbm_mechanism
RATE = ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "r_count", "hrv_lfhf"]
PAT = ["pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"]
CAPACITIES = [
    ("single tree (32 lv)", dict(n_estimators=1, num_leaves=32, learning_rate=1.0)),
    ("20 trees (32 lv)", dict(n_estimators=20, num_leaves=32, learning_rate=0.3)),
    ("150 trees (32 lv)", dict(n_estimators=150, num_leaves=32, learning_rate=0.1)),
    ("800 trees (63 lv)", dict(n_estimators=800, num_leaves=63, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=50)),
]


def tbl(F, ks_, n):
    return np.column_stack([np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_])


def main():
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    allk = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]

    ext = {}
    for name, fn in ep.SUBSETS.items():
        p = ep.BP / fn
        if not p.exists():
            print(f"[skip] {name}: {p} not present"); continue
        print(f"[load] {name} ...", flush=True)
        import mechlib
        import features_full as ff
        sig, y, subj, demo = ep.load_subset(p, max_seg=12000)
        X = mechlib.normalize(sig[:, :, [0, 1]])
        ext[name] = {"F": ff.compute_full(X, ep.FS), "y": y,
                     "g": np.array([abs(hash(s)) % 1000000 for s in subj])}

    pat_in = [k for k in PAT if k in allk]
    rate_in = [k for k in RATE if k in allk]
    morph = [k for k in allk if k not in rate_in and k not in pat_in]
    arms = {"PAT only": pat_in, "no shortcuts (morphology)": morph, "all 83": allk}
    print(f"[arms] PAT {len(pat_in)} | morphology {len(morph)} | all {len(allk)}\n")

    res = {}
    hdr = f"{'capacity':22s} {'arm':28s} {'id':>6s} {'ood(anchor)':>11s} {'CalBased':>9s} {'AAMI':>7s}"
    print(hdr); print("-" * len(hdr))
    for cap_name, params in CAPACITIES:
        params = dict(params, random_state=0, n_jobs=1, verbosity=-1)
        for arm_name, ks_ in arms.items():
            if not ks_:
                continue
            Mtr = tbl(Ftr, ks_, len(ytr))
            med = gbm.column_medians(Mtr)
            m = lgb.LGBMRegressor(**params).fit(gbm._impute(Mtr, med), ytr[:, TARGET])
            pte = m.predict(gbm._impute(tbl(Fte, ks_, len(yte)), med))
            curve = ep.anchor_curve(pte, yte[:, TARGET], gte, min_seg=60)

            row = {"n_feat": len(ks_), "id_k0": curve[0], "id_k20": curve[20]}
            cells = []
            for pname, e in ext.items():
                pe = m.predict(gbm._impute(tbl(e["F"], ks_, len(e["y"])), med))
                mae = float(np.median([
                    np.mean(np.abs(pe[e["g"] == s] - e["y"][e["g"] == s, TARGET]))
                    for s in np.unique(e["g"]) if (e["g"] == s).sum() >= 3]))
                row[pname] = mae
                cells.append(mae)
            res[f"{arm_name} | {cap_name}"] = row
            cb = f"{row.get('CalBased', float('nan')):9.2f}" if "CalBased" in row else "      . "
            aa = f"{row.get('AAMI', float('nan')):7.2f}" if "AAMI" in row else "     . "
            print(f"{cap_name:22s} {arm_name:28s} {curve[0]:6.2f} {curve[20]:11.2f} {cb} {aa}",
                  flush=True)

    print("\n--- PAT-only vs morphology, per capacity ---")
    for cap_name, _ in CAPACITIES:
        p = res.get(f"PAT only | {cap_name}")
        m = res.get(f"no shortcuts (morphology) | {cap_name}")
        if not (p and m):
            continue
        print(f"  {cap_name}")
        for axis in ("id_k20", "CalBased", "AAMI"):
            if axis in p and axis in m:
                print(f"    {axis:10} PAT {p[axis]:6.2f}  morph {m[axis]:6.2f}  "
                      f"-> {'PAT better' if p[axis] < m[axis] else 'morph better'} "
                      f"({m[axis] - p[axis]:+.2f})")

    json.dump(res, open(DATA / "gbm_capacity_calbased.json", "w"), indent=2)
    print(f"\n[done] {DATA / 'gbm_capacity_calbased.json'}")


if __name__ == "__main__":
    main()
