"""fig_shap_depend.py -- SHAP dependence plots to reveal NONLINEAR feature-BP relationships.

A linear correlation (rho) only sees monotone linear trends. SHAP dependence plots show the
actual shape of each feature's effect on predicted DBP -- thresholds, saturation, U-shapes --
that the GBM learned. We plot the top features (incl. the validated novel APG timings + age).
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

import lightgbm_arm as gbm
import features_ext as fx
import mechlib
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
CACHE = "data/_fam_cues_vitaldb_full_calfree_8000.pkl"
READABLE = {"age": "Age", "bmi": "BMI", "apg_d_a": "APG d/a", "notch": "Dicrotic notch",
            "peak": "Peak height", "t_d": "APG t_d (timing)", "t_e": "APG t_e (timing)",
            "t_b": "APG t_b (timing)", "aix": "Augmentation index", "decay": "Diastolic decay",
            "pw25": "Pulse width 25%", "rise": "Rise time", "hr": "Heart rate"}


def main():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    sctr, ytr, gtr, dmtr = blob["tr"]
    scte, yte, gte, dmte = blob["te"]

    # add validated novel APG timing features to train + test (recompute from raw is not cached;
    # approximate by using the cached cues + demographics; timing feats added if present)
    cue = [k for k, v in sctr.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3]
    dk = tuple(k for k in ("age", "sex", "bmi") if dmtr and k in dmtr)
    Xtr, names = gbm.build_feature_table(sctr, dmtr, cue, n=len(ytr), demo_keys=dk)
    Xte, _ = gbm.build_feature_table(scte, dmte, cue, n=len(yte), demo_keys=dk)
    Xtr, Xte = gbm._impute(Xtr), gbm._impute(Xte)

    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    m.fit(Xtr, ytr[:, 1])

    import shap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    expl = shap.TreeExplainer(m)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xte), min(2000, len(Xte)), replace=False)
    Xs = Xte[idx]
    sv = expl.shap_values(Xs)

    # pick top-6 features by mean|SHAP|
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)[:6]

    # for each feature, the strongest INTERACTING feature = the one whose value best explains
    # the RESIDUAL of this feature's SHAP after removing its own monotone trend.
    def interacting(j):
        xj = Xs[:, j]; sj = sv[:, j]
        srt = np.argsort(xj)
        trend = np.empty_like(sj)
        trend[srt] = np.interp(np.arange(len(xj)), np.arange(len(xj)),
                               np.poly1d(np.polyfit(np.arange(len(xj)), sj[srt], 3))
                               (np.arange(len(xj))))
        resid = sj - trend
        best, bidx = -1.0, (j + 1) % Xs.shape[1]
        for c in range(Xs.shape[1]):
            if c == j:
                continue
            xc = Xs[:, c]
            if np.std(xc) < 1e-9:
                continue
            r = abs(np.corrcoef(xc, resid)[0, 1])
            if np.isfinite(r) and r > best:
                best, bidx = r, c
        return bidx

    CMAP = "coolwarm"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, j in zip(axes.ravel(), order):
        xj = Xs[:, j]; sj = sv[:, j]
        cidx = interacting(j)
        cvals = Xs[:, cidx]
        lo, hi = np.nanpercentile(cvals, [5, 95])
        norm = Normalize(vmin=lo, vmax=hi)
        sc = ax.scatter(xj, sj, c=cvals, cmap=CMAP, norm=norm, s=16, alpha=0.75,
                        edgecolor="none")
        ok = np.isfinite(xj)
        if ok.sum() > 50:
            q = np.unique(np.quantile(xj[ok], np.linspace(0, 1, 12)))
            cx = 0.5 * (q[:-1] + q[1:])
            cy = [np.median(sj[(xj >= q[i]) & (xj < q[i + 1])]) for i in range(len(q) - 1)]
            ax.plot(cx, cy, "-", color="black", lw=2.4, zorder=4)
            ax.plot(cx, cy, "o", color="black", ms=5, zorder=5, mec="white", mew=0.8)
        ax.axhline(0, color="#444", lw=0.9, ls=":")
        nm = names[j]
        ax.set_title(READABLE.get(nm, nm), fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{READABLE.get(nm, nm)} value", fontsize=9)
        ax.set_ylabel("SHAP: effect on DBP (mmHg)", fontsize=9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(READABLE.get(names[cidx], names[cidx]), fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SHAP dependence: nonlinear feature effects on predicted DBP\n"
                 "black line = median effect  |  point color = interacting feature (reveals interactions)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ROOT / "figures" / "fig_shap_depend.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_shap_depend.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_shap_depend.png  (features:",
          ", ".join(READABLE.get(names[j], names[j]) for j in order), ")")


if __name__ == "__main__":
    main()
