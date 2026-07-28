"""fig_shap12.py -- two SHAP dependence multipanels (>=12 features each) for SBP and DBP,
colored by the strongest interacting feature only when the exact SHAP interaction exceeds a
threshold. From the 8k cached feature study.
"""
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
CACHE = "data/_feat_full_vitaldb_full_calfree_8000.pkl"

# readable-but-short names + concise formula for the footnote
NAME = {
    "rr_pnn50": ("pNN50", "% of RR intervals differing >50 ms"),
    "rr_mean": ("RR mean", "mean R-R interval"),
    "rr_sdnn": ("RR SDNN", "SD of R-R intervals"),
    "rr_cv": ("RR CV", "R-R SD / mean"),
    "dw25": ("dias. width 25%", "diastolic width at 25% pulse height"),
    "dw10": ("dias. width 10%", "diastolic width at 10% pulse height"),
    "ppg_p10": ("PPG p10", "10th percentile of PPG amplitude"),
    "ppg_p25": ("PPG p25", "25th percentile of PPG amplitude"),
    "ppg_skew_g": ("PPG skew", "skewness of the PPG segment"),
    "t_d": ("APG d-time", "time from foot to APG d landmark"),
    "t_e": ("APG e-time", "time from foot to APG e landmark"),
    "t_c": ("APG c-time", "time from foot to APG c landmark"),
    "apg_e_a": ("APG e/a", "APG e-wave / a-wave amplitude"),
    "apg_d_a": ("APG d/a", "APG d-wave / a-wave amplitude"),
    "pow_hf": ("HF power", "PPG power in 5-10 Hz band"),
    "qrs_amp_mean": ("QRS amp", "median R-wave amplitude"),
    "qrs_amp_std": ("QRS amp SD", "SD of R-wave amplitude"),
    "vpg_min": ("VPG min", "min PPG velocity (steepest downslope)"),
    "vpg_ratio": ("VPG ratio", "max/|min| PPG velocity"),
    "amp_cv": ("amp CV", "beat-to-beat amplitude CV"),
    "pat_peak": ("PAT (peak)", "R to PPG systolic-peak delay"),
}


def nice(k):
    return NAME.get(k, (k, ""))[0]


THRESH = 0.05


def build_panel(target, tname, path, n_panels=12):
    b = pickle.load(open(CACHE, "rb"))
    F, y = b["F"], b["y"]
    keys = [k for k, v in F.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3
            and np.nanstd(np.asarray(v, float)) > 1e-9]
    M = np.column_stack([np.nan_to_num(np.asarray(F[k], float),
                                       nan=np.nanmedian(np.asarray(F[k], float))) for k in keys])
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    m.fit(M, y[:, target])
    expl = shap.TreeExplainer(m)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(M), min(2000, len(M)), replace=False)
    Xs = M[idx]; sv = expl.shap_values(Xs)
    order = np.argsort(-np.abs(sv).mean(0))[:n_panels]

    iv = expl.shap_interaction_values(M[idx[:400]])
    inter = np.abs(iv).mean(0); np.fill_diagonal(inter, 0.0)
    main = np.abs(np.array([iv[:, i, i] for i in range(len(keys))]).T).mean(0)

    nc = 4; nr = int(np.ceil(n_panels / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.7 * nr))
    for ax, j in zip(axes.ravel(), order):
        xj = Xs[:, j]; sj = sv[:, j]
        strength = inter[j] / (main[j] + main + 1e-9)
        c = int(np.argmax(strength)); sval = strength[c]
        # ALWAYS color by the top interactor; the printed strength shows if it is strong or weak
        cv = Xs[:, c]; lo, hi = np.nanpercentile(cv, [5, 95])
        sc = ax.scatter(xj, sj, c=cv, cmap="coolwarm", norm=Normalize(lo, hi), s=10, alpha=0.75,
                        edgecolor="none")
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        strong = "" if sval >= THRESH else " (weak)"
        cb.set_label(f"{nice(keys[c])}  {sval:.2f}{strong}", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        q = np.unique(np.quantile(xj, np.linspace(0, 1, 11)))
        cx = 0.5 * (q[:-1] + q[1:])
        cy = [np.median(sj[(xj >= q[i]) & (xj < q[i + 1])]) for i in range(len(q) - 1)]
        ax.plot(cx, cy, "-", color="black", lw=2, zorder=4)
        ax.axhline(0, color="#666", lw=0.7, ls=":")
        ax.set_title(nice(keys[j]), fontsize=9.5, fontweight="bold")
        ax.set_xlabel("value", fontsize=7.5); ax.set_ylabel(f"SHAP {tname}", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")
    # footnote: concise formula for each plotted feature
    foot = "; ".join(f"{nice(keys[j])} = {NAME.get(keys[j], ('', ''))[1]}"
                     for j in order if keys[j] in NAME)
    fig.text(0.5, 0.005, foot, ha="center", fontsize=6.8, wrap=True, color="#333")
    fig.suptitle(f"SHAP dependence for {tname}: nonlinear effect (black line) + top interacting "
                 f"feature (color, with interaction strength)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(path, dpi=160, bbox_inches="tight")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {Path(path).name}  ({tname}: {', '.join(keys[j] for j in order)})")


def main():
    build_panel(0, "SBP", FIG / "fig_shap12_sbp.png")
    build_panel(1, "DBP", FIG / "fig_shap12_dbp.png")


if __name__ == "__main__":
    main()
