"""calib_families.py -- which features REDUCE CALIBRATION BURDEN, not which raise accuracy?

The anchor sweep showed demographics cut the anchor requirement 4x (waveform needs 20 anchors to
reach 4.69 mmHg; +age/sex/BMI reaches it at 5). That reframes feature selection: the useful
question for a device is not "which features lower MAE" but "which features let me collect fewer
cuff readings". Those are different questions, and they can have different answers -- demographics
actually HURT at k=0 (7.17 vs 6.88) while helping once anchored, because they inform the
per-subject offset rather than the population mean.

This module measures, per feature family:
  * the full anchor curve (k = 0, 1, 2, 3, 5, 10, 20)
  * `anchors_to_match`: the k at which that family matches the 83-feature waveform model at 20
    anchors -- the direct "how many cuff readings does this feature buy back" number
  * `anchor_value`: MAE(k=0) - MAE(k=5), i.e. how much of the family's usefulness is only
    accessible after calibration

PAT is included as its own arm to settle an open question. The roll-audit reads null on every
model we have, PulseDB PAT is measurable on only 44% of segments, and VitalDB's PAT is an
instrumental constant. If a PAT arm nonetheless reduces anchor need, PAT is doing mechanistic
work that the audit cannot see; if it does not, the null is consistent across instruments.

Permutation importance is computed on the CALIBRATED error (offset fitted per subject on 5
anchors), so it ranks features by their contribution to post-calibration accuracy rather than to
raw accuracy.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
KS = [0, 1, 2, 3, 5, 10, 20]
TARGET = 1
MIN_SEG = 60

FAMILIES = {
    "pat / arrival time": ["pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak",
                           "xcorr_width"],
    "rate / hrv": ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv",
                   "hrv_lf", "hrv_hf", "hrv_lfhf"],
    "reflection / apg": ["aix", "reflect_idx", "notch_depth", "notch_time", "t_b", "t_c",
                         "t_d", "t_e", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a",
                         "takazawa", "ushiro"],
    "complexity / fractal": ["hfd", "katz_fd", "spec_ent", "ppg_skew_g", "ppg_kurt_g"],
    "demographics": [],          # handled via the demo flag
}


def anchor_curve(pred, y, groups, ks=KS, min_seg=MIN_SEG, seed=0):
    """Median per-subject MAE after fitting a one-parameter offset on k random anchors."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        errs = []
        for s in np.unique(groups):
            idx = np.where(groups == s)[0]
            if len(idx) < min_seg:
                continue
            if k == 0:
                hold, off = idx, 0.0
            else:
                a = rng.choice(idx, k, replace=False)
                hold = np.setdiff1d(idx, a)
                if len(hold) < 20:
                    continue
                off = float(np.mean(y[a] - pred[a]))
            errs.append(float(np.mean(np.abs(pred[hold] + off - y[hold]))))
        out[k] = float(np.median(errs)) if errs else float("nan")
    return out


def main():
    full = pickle.load(open(ROOT / "data" / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    d = np.load(ROOT / "data" / "vitaldb_full_calfree.npz")
    n_te = len(yte)
    g = d["gte"][:n_te]
    allk = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    dtr = {"age": d["age_tr"], "sex": d["sex_tr"], "bmi": d["bmi_tr"]}
    dte = {"age": d["age_te"], "sex": d["sex_te"], "bmi": d["bmi_te"]}

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo:
            for dk in ("age", "sex", "bmi"):
                v = np.asarray(demo[dk], float)
                cols.append(v[:n] if len(v) >= n else np.resize(v, n))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    P = dict(n_estimators=900, learning_rate=0.03, num_leaves=63, subsample=0.8,
             colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)

    def fit_pred(ks_, use_demo):
        Mtr = tbl(Ftr, ks_, len(ytr), dtr if use_demo else None)
        if Mtr.shape[1] == 0:
            return None, 0
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**P)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
        Xte = gbm._impute(tbl(Fte, ks_, n_te, dte if use_demo else None), med)
        return m.predict(Xte), Mtr.shape[1]

    have = set(Ftr)
    arms = {"all 83 (waveform)": (allk, False), "all 83 + demo": (allk, True)}
    for tag, ks_ in FAMILIES.items():
        if tag == "demographics":
            arms[tag] = ([], True)
        else:
            sel = [k for k in ks_ if k in have]
            if sel:
                arms[tag] = (sel, False)
                arms[f"{tag} + demo"] = (sel, True)

    res = {}
    print(f"{'arm':26s} {'n':>3s} " + "".join(f"{k:>7d}" for k in KS), flush=True)
    print("-" * (30 + 7 * len(KS)), flush=True)
    for tag, (ks_, ud) in arms.items():
        p, nf = fit_pred(ks_, ud)
        if p is None:
            continue
        c = anchor_curve(p, yte[:, TARGET], g)
        res[tag] = {"n_feat": nf, "curve": c,
                    "anchor_value": float(c[0] - c[5])}
        print(f"{tag:26s} {nf:3d} " + "".join(f"{c[k]:7.2f}" for k in KS), flush=True)

    base = res["all 83 (waveform)"]["curve"][20]
    print(f"\n[equivalence] target = all-83 waveform @ 20 anchors = {base:.2f} mmHg")
    print(f"{'arm':26s} {'anchors needed':>15s} {'anchor value':>14s}")
    for tag, r in sorted(res.items(), key=lambda x: -x[1]["anchor_value"]):
        hit = [k for k in KS if r["curve"][k] <= base]
        r["anchors_to_match"] = hit[0] if hit else None
        print(f"{tag:26s} {str(hit[0]) if hit else '>20':>15s} "
              f"{r['anchor_value']:14.2f}", flush=True)

    # ---- permutation importance on the CALIBRATED error -----------------------
    Mtr = tbl(Ftr, allk, len(ytr), dtr)
    med = gbm.column_medians(Mtr)
    m = lgb.LGBMRegressor(**P)
    m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
    Xte = gbm._impute(tbl(Fte, allk, n_te, dte), med)
    names = allk + ["age", "sex", "bmi"]
    base5 = anchor_curve(m.predict(Xte), yte[:, TARGET], g, ks=[5])[5]
    rng = np.random.default_rng(0)
    imp = []
    for j, nm in enumerate(names):
        Xp = Xte.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        imp.append((nm, anchor_curve(m.predict(Xp), yte[:, TARGET], g, ks=[5])[5] - base5))
    imp.sort(key=lambda x: -x[1])
    print(f"\n[permutation importance on CALIBRATED error, k=5; base {base5:.2f} mmHg]")
    for nm, v in imp[:15]:
        print(f"   {nm:18s} {v:+.3f}")
    res["_perm_importance_k5"] = {nm: float(v) for nm, v in imp}
    res["_base_k5"] = float(base5)

    (ROOT / "data" / "calib_families.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n[done] data/calib_families.json")


if __name__ == "__main__":
    main()
