"""fig_summary.py -- four panels covering what the project established.

a  the audit is validated end-to-end on raw waveforms (synthetic alpha dial)
b  no trained model is faithful (per-subject slopes, CIs span zero)
c  PAT buys no calibration; demographics halve the anchor requirement
d  cross-dataset transfer sits at the mean-predictor floor
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9a9a9a", "#4a7c59"


def main():
    sw = json.load(open(ROOT / "data" / "synth_waveform_audit.json"))
    w2 = json.load(open(ROOT / "data" / "weekend2_results.json"))
    cf = json.load(open(ROOT / "data" / "calib_families.json"))

    fig, ax = plt.subplots(2, 2, figsize=(10.2, 7.4))

    # ---- a: synthetic alpha dial -------------------------------------------
    A = ax[0, 0]
    al = [0.0, 0.25, 0.5, 0.75, 1.0]
    sl = [sw["alphas"][str(a)]["slope"] for a in al]
    A.plot(al, sl, "o-", color=NAVY, lw=1.8, ms=6)
    A.axhline(0, color="k", lw=0.6, alpha=0.4)
    A.set_xlabel("α  (fraction of BP routed through PTT)", fontsize=9)
    A.set_ylabel("audit slope (mmHg / ms shift)", fontsize=9)
    A.text(0.03, 0.06, f"r = {sw['slope_vs_alpha_r']:+.3f}", transform=A.transAxes,
           fontsize=9.5, fontweight="bold")
    A.text(0.03, 0.90, "faithful ↓", transform=A.transAxes, fontsize=8, color=GREY)
    A.set_title("a   audit validated on raw waveforms", loc="left", fontsize=10.5,
                fontweight="bold")
    A.spines[["top", "right"]].set_visible(False)

    # ---- b: real models, per-subject slopes with CI --------------------------
    B = ax[0, 1]
    st = w2["1_corrected_audit"]
    names = list(st.keys())
    y = np.arange(len(names))
    med = [st[k]["slope_median"] for k in names]
    lo = [st[k]["slope_median"] - st[k]["slope_lo"] for k in names]
    hi = [st[k]["slope_hi"] - st[k]["slope_median"] for k in names]
    B.errorbar(med, y, xerr=[lo, hi], fmt="o", color=NAVY, ms=5, lw=1.2, capsize=3)
    B.axvline(0, color="k", lw=0.9, ls="--")
    B.axvline(sl[-1], color=RED, lw=1.4, label=f"synthetic α=1.0 ({sl[-1]:.2f})")
    B.set_yticks(y, [n.replace("1d", "") for n in names], fontsize=8.5)
    B.set_xlabel("audit slope (mmHg / ms shift)", fontsize=9)
    B.legend(fontsize=7.5, frameon=False, loc="lower left")
    B.set_title("b   no trained model is faithful", loc="left", fontsize=10.5,
                fontweight="bold")
    B.spines[["top", "right"]].set_visible(False)

    # ---- c: anchor curves ----------------------------------------------------
    C = ax[1, 0]
    KS = [0, 1, 2, 3, 5, 10, 20]
    show = [("all 83 (waveform)", NAVY, "-"), ("all 83 + demo", GREEN, "-"),
            ("pat / arrival time", RED, "-"), ("demographics", GREY, "--")]
    for tag, col, ls in show:
        if tag not in cf:
            continue
        C.plot(KS, [cf[tag]["curve"][str(k)] for k in KS], ls, color=col, lw=1.7,
               marker="o", ms=3.6, label=tag)
    C.axhline(5.59, color="k", lw=0.9, ls=":", label="subject-mean floor")
    C.set_xlabel("calibration anchors per subject", fontsize=9)
    C.set_ylabel("DBP MAE (mmHg)", fontsize=9)
    C.legend(fontsize=7.2, frameon=False)
    C.set_title("c   PAT buys no calibration; demographics halve it", loc="left",
                fontsize=10.5, fontweight="bold")
    C.spines[["top", "right"]].set_visible(False)

    # ---- d: OOD vs mean-predictor floor --------------------------------------
    D = ax[1, 1]
    sets = ["ID", "MIMIC-BP", "BCG", "Sensors", "UCI2", "PPG-BP"]
    const = [9.43, 11.39, 7.92, 8.02, 8.49, 10.89]
    oracle = [9.43, 10.22, 7.41, 8.23, 8.59, 8.84]
    best = [8.07, 10.30, 7.60, 8.10, 8.40, 8.90]
    x = np.arange(len(sets)); w = 0.27
    D.bar(x - w, const, w, label="constant predictor", color=GREY)
    D.bar(x, oracle, w, label="oracle set mean", color=NAVY, alpha=0.75)
    D.bar(x + w, best, w, label="best model", color=RED)
    D.set_xticks(x, sets, fontsize=8, rotation=20)
    D.set_ylabel("DBP MAE (mmHg)", fontsize=9)
    D.legend(fontsize=7.2, frameon=False)
    D.set_title("d   transfer sits at the mean-predictor floor", loc="left",
                fontsize=10.5, fontweight="bold")
    D.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_summary.{ext}", dpi=210, bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_summary.png / .pdf")


if __name__ == "__main__":
    main()
