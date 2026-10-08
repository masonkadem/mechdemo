"""gbm_external_calib_burden.py -- calibration burden (k vs benefit), on the paper's own
four external OOD datasets, not just VitalDB.

Every calibration-burden number so far in this project comes from VitalDB (PulseDB CalFree,
shuffled; VitalDB continuous, real time). Both are one hospital's ICU/OR population. This asks
whether "how many cuff readings, and does it help" replicates on data from entirely different
sources, sensors and patient populations -- Sensors and UCI (ICU, MIMIC-derived), PPGBP
(outpatient/cardiovascular clinic), BCG (a small bed-sensor study).

k values are capped by segments-per-subject, which varies enormously by dataset:
    PPGBP   2.8 segs/subj  -> k=1 only
    Sensors 9.3            -> up to k=5
    BCG    76.6 (40 subj)  -> up to k=20, but n=40 subjects so wide CIs
    UCI    38.0 (10793 subj)-> up to k=20, tight CIs

Model: single-tree GBM, morphology only (no PAT possible -- none of the four have ECG),
trained once on PulseDB VitalDB CalibFree, exactly as in gbm_external_ood.py. Single tree
because that capacity generalised best across most of these datasets today, and it lets more
k values run in the time available.

Benefit = paired per-subject |uncalibrated error| - |offset-calibrated error|, bootstrapped
over subjects (not segments) -- the same statistic and the same caution as drift_analyze.py,
because the naive difference-of-medians overstated the VitalDB result earlier this session.

    python gbm_external_calib_burden.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import features_full as ff
import mechlib
from gbm_external_ood import DATASETS, load_external

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
TARGET = {"sbp": 0, "dbp": 1}
CAP = dict(n_estimators=1, num_leaves=32, learning_rate=1.0)   # single tree, matches text above
KS_BY_DATASET = {"PPGBP": [1], "Sensors": [1, 2, 3, 5],
                 "BCG": [1, 2, 5, 10, 20], "UCI": [1, 2, 5, 10, 20]}


def paired_benefit(pred, y, groups, k, gap=0, reps=3000, seed=0):
    """Per-subject: hold out everything after the first k, fit an offset from the first k,
    compare uncalibrated vs calibrated |error| on the SAME held-out segments. Bootstrap the
    per-subject paired difference over subjects."""
    unc_e, cal_e = [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < k + gap + 2:
            continue
        cal_idx, test_idx = idx[:k], idx[k + gap:]
        off = float(np.mean(y[cal_idx] - pred[cal_idx]))
        unc_e.append(float(np.abs(pred[test_idx] - y[test_idx]).mean()))
        cal_e.append(float(np.abs(pred[test_idx] + off - y[test_idx]).mean()))
    if len(unc_e) < 5:
        return None
    unc_e, cal_e = np.array(unc_e), np.array(cal_e)
    diff = unc_e - cal_e
    rng = np.random.default_rng(seed)
    n = len(diff)
    bs = [float(np.median(diff[rng.integers(0, n, n)])) for _ in range(reps)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"n_subj": n, "uncal": float(np.median(unc_e)), "cal": float(np.median(cal_e)),
            "benefit": float(np.median(diff)), "benefit_ci": [lo, hi]}


def main():
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = full["Ftr"], np.asarray(full["ytr"], float)

    print("[1/2] loading + extracting the four external datasets ...\n")
    ext = {}
    for name, path in DATASETS.items():
        sig, sbp, dbp, pat = load_external(path)
        X = mechlib.normalize(sig)[:, :, None]
        F = ff.compute_full(X, FS, ppg_ch=0, ecg_ch=None)
        ext[name] = {"F": F, "sbp": sbp, "dbp": dbp, "g": pat}
        print(f"  {name:8} {len(sig):7d} segments, {len(np.unique(pat)):5d} subjects")

    keys = sorted(set(Ftr) & set.intersection(*[set(e["F"]) for e in ext.values()]))
    print(f"\n[feat] {len(keys)} shared morphology features\n")

    res = {}
    for target, ycol in TARGET.items():
        Mtr = np.column_stack([np.asarray(Ftr[k], float)[:len(ytr)] for k in keys])
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        m = lgb.LGBMRegressor(**CAP, random_state=0, n_jobs=1, verbosity=-1)
        m.fit(Mtr, ytr[:, ycol])

        print(f"=== {target.upper()}, single-tree, morphology only ===")
        for dname, e in ext.items():
            Me = np.column_stack([np.asarray(e["F"][k], float) for k in keys])
            Me = np.where(np.isfinite(Me), Me, med)
            y_e = e["sbp"] if target == "sbp" else e["dbp"]
            pred = m.predict(Me)
            for k in KS_BY_DATASET[dname]:
                r = paired_benefit(pred, y_e, e["g"], k)
                if r is None:
                    print(f"  {dname:8} k={k:<3}  too few subjects with enough segments")
                    continue
                res[f"{dname} | {target} | k={k}"] = r
                sig = "*" if (r["benefit_ci"][0] > 0 or r["benefit_ci"][1] < 0) else " "
                print(f"  {dname:8} k={k:<3} n={r['n_subj']:5d}  uncal {r['uncal']:6.2f} | "
                      f"cal {r['cal']:6.2f} | benefit {r['benefit']:+6.2f} "
                      f"[{r['benefit_ci'][0]:+.2f},{r['benefit_ci'][1]:+.2f}] {sig}")
        print()

    json.dump(res, open(DATA / "gbm_external_calib_burden.json", "w"), indent=2)
    print(f"[done] {DATA / 'gbm_external_calib_burden.json'}")


if __name__ == "__main__":
    main()
