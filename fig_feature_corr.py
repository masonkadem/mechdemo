"""fig_feature_corr.py -- publication feature/BP correlation matrix.

Correlation (Spearman) of every physiological feature with SBP and DBP, grouped by signal
source: ECG-derived, PPG morphology, APG (2nd-derivative) waves, PTT/timing, demographics.
Diverging grey colormap, values annotated. Uses the cached subject-diverse cue features.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parent
CACHE = "data/_fam_cues_vitaldb_full_calfree_8000.pkl"

# feature -> (readable name, group). Order defines row order (grouped).
FEATURES = [
    # PTT / timing (ECG+PPG)
    ("pat", "PAT (R->foot)", "PTT / timing"),
    ("pat_peak", "PAT (R->peak)", "PTT / timing"),
    ("ptt_var", "PTT variability", "PTT / timing"),
    ("xcorr_peak", "ECG-PPG xcorr", "PTT / timing"),
    # ECG-derived
    ("hr", "Heart rate", "ECG"),
    ("period", "Cardiac period", "ECG"),
    ("rr_mean", "RR mean", "ECG"),
    ("rr_sdnn", "RR SDNN", "ECG"),
    ("rr_rmssd", "RR RMSSD", "ECG"),
    ("qrs_amp", "QRS amplitude", "ECG"),
    # PPG morphology
    ("rise", "Rise time", "PPG morphology"),
    ("crest", "Crest time", "PPG morphology"),
    ("aix", "Augmentation idx", "PPG morphology"),
    ("notch", "Dicrotic notch", "PPG morphology"),
    ("decay", "Diastolic decay", "PPG morphology"),
    ("pw25", "Pulse width 25%", "PPG morphology"),
    ("pw50", "Pulse width 50%", "PPG morphology"),
    ("sys_area", "Systolic area", "PPG morphology"),
    ("kurt", "Kurtosis", "PPG morphology"),
    ("peak", "Peak height", "PPG morphology"),
    ("amp", "Amplitude", "PPG morphology"),
    ("vpg_max", "VPG max (1st deriv)", "PPG morphology"),
    # APG (2nd derivative) waves
    ("apg_b_a", "APG b/a", "APG (2nd deriv)"),
    ("apg_c_a", "APG c/a", "APG (2nd deriv)"),
    ("apg_d_a", "APG d/a", "APG (2nd deriv)"),
    ("apg_e_a", "APG e/a", "APG (2nd deriv)"),
    ("aging_idx", "Aging index", "APG (2nd deriv)"),
    # complexity
    ("hfd", "Fractal (Higuchi)", "Complexity"),
    ("katz_fd", "Fractal (Katz)", "Complexity"),
    ("spec_ent", "Spectral entropy", "Complexity"),
    # demographics
    ("age", "Age", "Demographics"),
    ("sex", "Sex", "Demographics"),
    ("bmi", "BMI", "Demographics"),
]


def main():
    with open(CACHE, "rb") as f:
        blob = pickle.load(f)
    sc, y, g, dm = blob["tr"]
    sbp, dbp = y[:, 0], y[:, 1]

    def getcol(k):
        if k in sc:
            return np.asarray(sc[k], float)
        if dm and k in dm:
            return np.asarray(dm[k], float)
        return None

    rows, labels, groups = [], [], []
    for key, name, grp in FEATURES:
        v = getcol(key)
        if v is None or np.isfinite(v).mean() < 0.3:
            continue
        m = np.isfinite(v)
        rs = spearmanr(v[m], sbp[m]).correlation
        rd = spearmanr(v[m], dbp[m]).correlation
        rows.append([rs, rd]); labels.append(name); groups.append(grp)
    M = np.array(rows)

    fig, ax = plt.subplots(figsize=(4.6, 0.30 * len(labels) + 1.5))
    norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
    im = ax.imshow(M, cmap="RdGy_r", norm=norm, aspect="auto")
    ax.set_xticks([0, 1], ["SBP", "DBP"], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(2):
            val = M[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                    color=("white" if abs(val) > 0.38 else "black"))
    # group separators + labels on the right
    prev = None
    for i, grp in enumerate(groups):
        if grp != prev:
            if i > 0:
                ax.axhline(i - 0.5, color="black", lw=1.0)
            prev = grp
    # bracket labels
    seen = {}
    for i, grp in enumerate(groups):
        seen.setdefault(grp, []).append(i)
    for grp, idxs in seen.items():
        ax.text(1.75, np.mean(idxs), grp, rotation=0, va="center", ha="left",
                fontsize=7.5, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("Feature correlation with blood pressure\n(Spearman $\\rho$; VitalDB)",
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.28, shrink=0.5)
    cbar.set_label("Spearman $\\rho$", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_feature_corr.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_feature_corr.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] fig_feature_corr.png / .pdf  ({len(labels)} features)")


if __name__ == "__main__":
    main()
