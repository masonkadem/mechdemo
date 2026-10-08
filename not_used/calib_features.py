"""calib_features.py -- does richer LightGBM feature sets reduce per-subject calibration need?

For each feature set, train a GBM, then sweep K = 0,1,2,3,5,10 calibration anchors per test
subject and record DBP MAE. If demographics / fractal / APG features flatten the curve or lower
the K=0 (uncalibrated) point, that means the model needs LESS per-subject calibration -- a
practical advantage of the interpretable model.

Sets:  base (core cues)  |  +demographics  |  +fractal  |  +APG waves  |  all
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
CACHE = "data/_fam_cues_vitaldb_full_calfree_8000.pkl"

BASE = ["pat_peak", "rise", "aix", "notch", "decay", "peak", "period"]
DEMO = ["age", "sex", "bmi"]
FRACTAL = ["hfd", "katz_fd", "spec_ent"]
APG = ["apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "aging_idx"]

SETS = {
    "base":          (BASE, ()),
    "base+demo":     (BASE, ("age", "sex", "bmi")),
    "base+fractal":  (BASE + FRACTAL, ()),
    "base+APG":      (BASE + APG, ()),
    "all":           (BASE + FRACTAL + APG, ("age", "sex", "bmi")),
}
KS = [0, 1, 2, 3, 5, 10]


def calib_curve(pred, y, grp, Ks=KS):
    out = {}
    for K in Ks:
        if K == 0:
            out[0] = float(np.abs(pred[:, 1] - y[:, 1]).mean()); continue
        errs = []
        for g in np.unique(grp):
            idx = np.where(grp == g)[0]
            if len(idx) <= K:
                continue
            off = (y[idx[:K], 1] - pred[idx[:K], 1]).mean()
            errs.append(np.abs((pred[idx[K:], 1] + off) - y[idx[K:], 1]))
        out[K] = float(np.concatenate(errs).mean()) if errs else np.nan
    return out


def main():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    sctr, ytr, gtr, dmtr = blob["tr"]
    scva, yva, gva, dmva = blob["va"]
    scte, yte, gte, dmte = blob["te"]

    results = {}
    print("%-14s %s" % ("feature set", "  ".join(f"K={k}" for k in KS)))
    for tag, (cues, dkeys) in SETS.items():
        Xtr, _ = gbm.build_feature_table(sctr, dmtr, cues, n=len(ytr), demo_keys=dkeys)
        Xva, _ = gbm.build_feature_table(scva, dmva, cues, n=len(yva), demo_keys=dkeys)
        Xte, _ = gbm.build_feature_table(scte, dmte, cues, n=len(yte), demo_keys=dkeys)
        Xtr, Xva, Xte = map(gbm._impute, (Xtr, Xva, Xte))
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        m.fit(Xtr, ytr[:, 1], eval_set=[(Xva, yva[:, 1])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = np.stack([m.predict(Xte), m.predict(Xte)], 1)
        cur = calib_curve(pred, yte, gte)
        results[tag] = cur
        print("%-14s %s" % (tag, "  ".join(f"{cur[k]:.2f}" for k in KS)))

    (ROOT / "data" / "calib_features.json").write_text(json.dumps(results, indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    shades = {"base": "#bbbbbb", "base+demo": "#2f4b7c", "base+fractal": "#888888",
              "base+APG": "#555555", "all": "#000000"}
    for tag, cur in results.items():
        ax.plot(KS, [cur[k] for k in KS], "-o", ms=5, lw=1.6, color=shades.get(tag, "gray"),
                label=tag)
    ax.axvline(3, color="#c1543b", ls=":", lw=1, zorder=0)
    ax.text(3.1, ax.get_ylim()[1], "K=3 knee", fontsize=8, color="#c1543b", va="top")
    ax.set_xlabel("calibration anchors per subject (K)")
    ax.set_ylabel("DBP MAE (mmHg)")
    ax.set_title("Does richer physiology reduce calibration need?", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_calib_features.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_calib_features.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_calib_features.png")


if __name__ == "__main__":
    main()
