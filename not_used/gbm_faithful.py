"""gbm_faithful.py -- a small, faithful feature model built on VALIDATED arrival-time proxies.

The previous PAT arm used pat_foot and pat_peak, which score r = 0.026 and 0.208 against invasive
arterial arrival time. Those were the only arrival-time features in the library, so "arrival time
does not help" was partly a statement about two weak estimators.

This arm rebuilds the arrival-time channel from the estimators that actually agree with the
arterial ground truth (pat_groundtruth.py, within subject):

    second_deriv   +0.206     max acceleration of the PPG upstroke
    peak           +0.208     PPG systolic peak
    xcorr_deriv    +0.177     ECG against the PPG first derivative
    max_slope      +0.136     steepest point of the upstroke

plus the morphology that survives within-subject analysis. Features whose apparent association
with BP is pooled-only are excluded: hr retains 8% of its pooled correlation within subject,
rr_mean 3% with a sign flip, rise 1%. Including them inflates in-distribution accuracy without
adding anything a device can use.

The target is a model that is small, uses quantities with a physiological route to blood
pressure, and is competitive with the deep networks rather than merely close.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import mechlib
import lightgbm_arm as gbm
import pat_estimators as PE
import eval_protocols as ep
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
TARGET = 1

# arrival-time estimators that agree with invasive arterial arrival time
GOOD_PAT = ["second_deriv", "peak", "xcorr_deriv", "max_slope"]
# morphology that holds up within subject (|r_within| >= 0.10 in the feature reference)
KEEP_MORPH = ["ppg_p10", "ppg_p90", "ppg_p25", "ppg_p75", "ppg_skew_g", "ppg_kurt_g",
              "decay_slope", "aix", "reflect_idx", "notch_depth", "t_c", "t_d", "t_e",
              "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "takazawa", "dw10", "dw50",
              "vpg_max", "vpg_min", "sys_dia_ratio", "hfd", "spec_ent"]
# pooled-only artifacts, excluded by design
DROP = ["hr", "rr_mean", "rise", "r_count", "period", "rr_sdnn", "rr_rmssd", "rr_pnn50",
        "rr_cv", "hrv_lf", "hrv_hf", "hrv_lfhf", "qrs_amp_mean", "qrs_width"]


def pat_features(X, fs=FS, names=GOOD_PAT):
    """Compute the validated arrival-time estimators for a batch."""
    out = {}
    for n in names:
        out[f"pat_{n}"] = PE.batch(PE.ESTIMATORS[n], X, fs) * 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=60000)
    ap.add_argument("--test-n", type=int, default=20000)
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    gte_full = d["gte"][:len(yte)]

    ntr = min(args.train_n, len(ytr))
    nte = min(args.test_n, len(yte))
    print(f"[pat] computing validated arrival-time features on {ntr}+{nte} segments ...",
          flush=True)
    Xtr = mechlib.normalize(d["Xtr"][:ntr][:, :, [ECG, PPG]])
    Xte = mechlib.normalize(d["Xte"][:nte][:, :, [ECG, PPG]])
    Ptr, Pte = pat_features(Xtr), pat_features(Xte)
    for k, v in Ptr.items():
        print(f"       {k:22s} valid {100*np.isfinite(v).mean():3.0f}%", flush=True)

    have = set(Ftr)
    morph = [k for k in KEEP_MORPH if k in have]
    allk = [k for k in Ftr if k not in DROP
            and np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]

    def tbl(F, P, ks, pk, n):
        cols = [np.asarray(F[k], float)[:n] for k in ks]
        cols += [np.asarray(P[k], float)[:n] for k in pk]
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    pk_all = list(Ptr)
    arms = {
        "validated PAT only": ([], pk_all,
                               dict(n_estimators=300, learning_rate=0.05, num_leaves=15)),
        "morphology only": (morph, [],
                            dict(n_estimators=400, learning_rate=0.04, num_leaves=31)),
        "PAT + morphology (small)": (morph, pk_all,
                                     dict(n_estimators=300, learning_rate=0.05, num_leaves=15)),
        "PAT + morphology": (morph, pk_all,
                             dict(n_estimators=500, learning_rate=0.04, num_leaves=31)),
        "all features (no shortcuts)": (allk, pk_all,
                                        dict(n_estimators=600, learning_rate=0.04,
                                             num_leaves=63)),
    }

    print(f"\n{'arm':30s} {'n':>4s} {'leaves':>8s} {'k=0':>7s} {'k=5':>7s} {'k=20':>7s}")
    print("-" * 70)
    res = {}
    for name, (ks, pk, params) in arms.items():
        Mtr = tbl(Ftr, Ptr, ks, pk, ntr)
        if Mtr.shape[1] == 0:
            continue
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=50, random_state=0, verbosity=-1)
        m.fit(gbm._impute(Mtr, med), ytr[:ntr, TARGET])
        pred = m.predict(gbm._impute(tbl(Fte, Pte, ks, pk, nte), med))
        curve = ep.anchor_curve(pred, yte[:nte, TARGET], gte_full[:nte], min_seg=40)
        leaves = int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]))
        names_all = ks + pk
        gain = m.booster_.feature_importance("gain")
        top = [names_all[i] for i in np.argsort(-gain)[:4] if gain[i] > 0]
        res[name] = {"n_feat": Mtr.shape[1], "leaves": leaves, "curve": curve, "top": top}
        print(f"{name:30s} {Mtr.shape[1]:4d} {leaves:8,d} {curve[0]:7.2f} {curve[5]:7.2f} "
              f"{curve[20]:7.2f}", flush=True)
        print(f"{'':30s} top: {', '.join(top[:3])}", flush=True)

    # deep-net reference on the same anchor protocol
    ref = {"lenet1d": 5.07, "inception1d": 4.92, "xresnet1d50": 4.74,
           "xresnet1d101": 4.93, "transformer": 5.37}
    best_deep = min(ref.values())
    print(f"\nDeep nets at k=20: {min(ref.values()):.2f} to {max(ref.values()):.2f} "
          f"(472k-1.8M parameters)")
    for name, r in sorted(res.items(), key=lambda kv: kv[1]["curve"][20]):
        v = r["curve"][20]
        print(f"  {name:30s} {v:5.2f}  {'BEATS' if v < best_deep else 'behind'} "
              f"best deep ({best_deep:.2f}), {r['leaves']:,} leaves")

    (DATA / "gbm_faithful.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/gbm_faithful.json")


if __name__ == "__main__":
    main()
