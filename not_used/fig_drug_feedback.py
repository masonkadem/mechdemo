"""fig_drug_feedback.py -- why VitalDB cannot answer a naive pharmacodynamic question.

Phenylephrine raises BP by 15-30 mmHg in a controlled setting. Measured naively across VitalDB,
the response to a rate step-up is +0.85 mmHg (p=0.092, ns). The reason is not a weak drug: it is
that the clinician is a controller in the loop. Vasopressors are given BECAUSE MAP has fallen,
and are titrated to a target, so treatment is a function of the outcome and a before/after
contrast measures the residual error of the control loop rather than the drug effect.

Panel a: MAP aligned to phenylephrine rate step-ups (n=1066 events, 35 cases), stratified by the
pre-dose MAP trend. The treated group is BELOW baseline when dosed and RETURNS to baseline -- the
signature of restoration to target, not elevation above it.

Panel b: the same events as a before/after scatter. Points sit along the identity line, which is
what closed-loop control produces and what makes the naive estimate ~0.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREY = "#2f4b7c", "#c1543b", "#8a8a8a"


def main():
    d = np.load(ROOT / "data" / "_drug_feedback.npz")
    C, PRE = d["C"], int(d["pre"])
    fall, flat, rise = d["fall"], d["flat"], d["rise"]
    t = np.arange(C.shape[1]) - PRE

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- panel a: event-aligned MAP by pre-dose trend -----------------------
    ax = axes[0]
    for mask, col, lab in [(fall, RED, "MAP falling before dose"),
                           (flat, GREY, "MAP flat"),
                           (rise, NAVY, "MAP rising")]:
        if mask.sum() < 5:
            continue
        m = np.median(C[mask], 0)
        lo = np.percentile(C[mask], 40, 0)
        hi = np.percentile(C[mask], 60, 0)
        ax.plot(t, m, color=col, lw=1.8, label=f"{lab} (n={int(mask.sum())})")
        ax.fill_between(t, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axvline(0, color="k", lw=1.0, ls="--")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.text(4, ax.get_ylim()[1] * 0.92, "phenylephrine\nrate increase", fontsize=7.5, va="top")
    ax.set_xlabel("time relative to dose increase (s)", fontsize=9)
    ax.set_ylabel("MAP relative to case baseline (mmHg)", fontsize=9)
    ax.legend(fontsize=7.2, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("a", loc="left", fontsize=11, fontweight="bold")

    # ---- panel b: before vs after ------------------------------------------
    ax = axes[1]
    pre = C[:, PRE - 30:PRE].mean(1)
    post = C[:, PRE + 120:PRE + 180].mean(1)
    ax.scatter(pre, post, s=5, alpha=0.28, color=NAVY, edgecolors="none")
    lim = [np.percentile(pre, 1) - 2, np.percentile(pre, 99) + 2]
    ax.plot(lim, lim, color="k", lw=1.0, ls="--", label="no change")
    # a real 15-30 mmHg pressor effect would sit on this line
    ax.plot(lim, [x + 20 for x in lim], color=RED, lw=1.2,
            label="expected pressor effect (+20)")
    ax.set_xlim(lim); ax.set_ylim(lim[0] - 2, lim[1] + 22)
    ax.set_xlabel("MAP before dose (rel. baseline, mmHg)", fontsize=9)
    ax.set_ylabel("MAP after dose (mmHg)", fontsize=9)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b", loc="left", fontsize=11, fontweight="bold")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_drug_feedback.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_drug_feedback.png / .pdf")

    med_fall = np.median(post[fall] - pre[fall])
    print(f"[fig] treated group: pre {np.median(pre[fall]):+.1f} -> post "
          f"{np.median(post[fall]):+.1f} mmHg (net {med_fall:+.1f})")


if __name__ == "__main__":
    main()
