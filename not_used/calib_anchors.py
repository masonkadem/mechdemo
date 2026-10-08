"""calib_anchors.py -- how many calibration anchors does a model actually need?

The question. Cuffless BP devices need periodic cuff calibration; the patent landscape says
calibration-free remains unsolved. So the practical target is not "no anchors" but "FEWER anchors
for the same error". This measures the anchor-count/error curve directly, and asks whether
covariates (demographics, and VitalDB clinical variables) buy anchors back.

Design. For each test subject, take k anchors (the first k segments, i.e. what a device would
collect at fit time), fit a per-subject OFFSET correction on those anchors, and score the rest.
k = 0, 1, 2, 3, 5, 10, 20. Reported against two references that make the numbers interpretable:

  * subject-mean floor : predict each subject's own mean BP. This is what an anchor buys at the
    limit, and it is the baseline our cross-dataset OOD work never beat.
  * k=0                : the uncalibrated model.

Why offset (not scale) calibration: within-subject BP variation here is as large as
between-subject variation, so the dominant per-subject error is a constant bias, and a
one-parameter offset is what a device can realistically fit from a handful of cuff readings.

Feature arms compare what information reduces anchor need:
  waveform            -- morphology only
  waveform + demo     -- + age/sex/BMI
  waveform + demo + clinical (VitalDB-only arm; see below)

NOTE on the clinical arm. VitalDB clinical covariates cannot be joined to PulseDB subjects --
PulseDB anonymized the case IDs, and recovering them would mean matching on age/sex/BMI, a
re-identification attack on a de-identified release. So the clinical arm is reported only as the
DISTRIBUTION of what such covariates could add, using the demographics that ARE carried in the
PulseDB npz. Adding drug state is deliberately excluded: the phenylephrine analysis showed
treatment is a function of the outcome (clinicians dose because BP fell), so a drug feature
encodes reverse causation and would invert out of distribution.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
KS = [0, 1, 2, 3, 5, 10, 20]
TARGET = 1                      # DBP
MIN_SEG = 60                    # a subject needs enough segments to anchor AND score


def anchor_curve(pred, y, groups, ks=KS, min_seg=MIN_SEG):
    """MAE after fitting a per-subject offset on the first k segments, scored on the rest."""
    out, floor = {}, []
    for k in ks:
        errs = []
        for s in np.unique(groups):
            idx = np.where(groups == s)[0]
            if len(idx) < min_seg:
                continue
            hold = idx[max(k, 1):]              # always score on held-out segments
            if len(hold) < 20:
                continue
            off = 0.0 if k == 0 else float(np.mean(y[idx[:k]] - pred[idx[:k]]))
            errs.append(float(np.mean(np.abs(pred[hold] + off - y[hold]))))
        out[k] = float(np.median(errs)) if errs else float("nan")
    for s in np.unique(groups):
        idx = np.where(groups == s)[0]
        if len(idx) >= min_seg:
            floor.append(float(np.mean(np.abs(y[idx] - y[idx].mean()))))
    return out, float(np.median(floor))


def main():
    full = pickle.load(open(ROOT / "data" / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    d = np.load(ROOT / "data" / "vitaldb_full_calfree.npz")
    gte = d["gte"]
    n_te = len(yte)
    gte = gte[:n_te] if len(gte) >= n_te else np.resize(gte, n_te)

    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    demo_tr = {"age": d["age_tr"], "sex": d["sex_tr"], "bmi": d["bmi_tr"]}
    demo_te = {"age": d["age_te"], "sex": d["sex_te"], "bmi": d["bmi_te"]}

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo:
            for dk in ("age", "sex", "bmi"):
                v = np.asarray(demo[dk], float)
                cols.append(v[:n] if len(v) >= n else np.resize(v, n))
        return np.column_stack(cols)

    params = dict(n_estimators=900, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)

    arms = {"waveform": (keys, False), "waveform + demo": (keys, True)}
    res = {}
    for tag, (ks_, use_demo) in arms.items():
        Mtr = tbl(Ftr, ks_, len(ytr), demo_tr if use_demo else None)
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
        Xte = gbm._impute(tbl(Fte, ks_, n_te, demo_te if use_demo else None), med)
        p = m.predict(Xte)
        curve, floor = anchor_curve(p, yte[:, TARGET], gte)
        res[tag] = {"curve": curve, "floor": floor, "n_feat": Mtr.shape[1]}
        print(f"\n{tag}  ({Mtr.shape[1]} features)", flush=True)
        print("   anchors " + "".join(f"{k:>8d}" for k in KS), flush=True)
        print("   MAE     " + "".join(f"{curve[k]:8.2f}" for k in KS), flush=True)
        print(f"   subject-mean floor {floor:.2f} mmHg", flush=True)

    # how many anchors does each arm need to match the 20-anchor waveform result?
    base = res["waveform"]["curve"][20]
    print(f"\n[equivalence] target = waveform @ 20 anchors = {base:.2f} mmHg")
    for tag, r in res.items():
        hit = [k for k in KS if r["curve"][k] <= base]
        print(f"   {tag:18s} reaches it at k = {hit[0] if hit else '>20'}")

    (ROOT / "data" / "calib_anchors.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n[done] data/calib_anchors.json")


if __name__ == "__main__":
    main()
