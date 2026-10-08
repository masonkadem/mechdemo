"""fig_what_models_use.py -- not "the model ignores the physics", but what it uses instead.

"Anti-faithful to arrival time" is a negative claim, and it invites the obvious rebuttal that
arrival time was measured badly. The ablation answers that directly: hand the model the
governing law's variable alongside pulse morphology and it declines to use it. Accuracy is
identical with and without it, and no timing feature enters the top of the ranking even when
one is available.

  panel a   accuracy by feature set -- arrival time is the weakest signal on offer
  panel b   the controlled comparison: adding PAT to morphology changes nothing

    python fig_what_models_use.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
K = "20"                       # readings; the ordering is the same at every k

ARMS = [
    ("PAT only\n(the governing law's variable)", "PAT only", "#C0392B"),
    ("morphology only\n(no demographics, no rate)", "no shortcut, no demo", "#0072B2"),
    ("morphology + PAT", "PAT forced (PAT + morphology)", "#0072B2"),
    ("morphology + demographics", "full (83 + demo)", "#3C3C3C"),
]


def main():
    d = json.loads((DATA / "gbm_mechanism.json").read_text())
    lbl = [a[0] for a in ARMS]
    val = [d[a[1]]["curve"][K] for a in ARMS]
    col = [a[2] for a in ARMS]
    nf = [d[a[1]]["n_feat"] for a in ARMS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
                         "legend.fontsize": 9.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.5, 1]})

    y = np.arange(len(lbl))[::-1]
    ax[0].barh(y, val, 0.62, color=col)
    for yy, v, n in zip(y, val, nf):
        ax[0].text(v + 0.04, yy, f"{v:.2f}  ({n} features)", va="center", fontsize=9.5,
                   color="#333")
    ax[0].set_yticks(y); ax[0].set_yticklabels(lbl, fontsize=10)
    ax[0].set_xlabel(f"diastolic BP error at $k$ = {K} cuff readings (mmHg)")
    ax[0].set_xlim(0, max(val) * 1.28)
    ax[0].set_title("a   arrival time is the weakest signal available", fontsize=11.5, loc="left")
    ax[0].spines[["top", "right"]].set_visible(False)

    # ---- panel b: the controlled comparison
    m_only = d["no shortcut, no demo"]["curve"][K]
    m_pat = d["PAT forced (PAT + morphology)"]["curve"][K]
    pat_only = d["PAT only"]["curve"][K]
    ax[1].bar([0, 1], [m_only, m_pat], 0.5, color=["#0072B2", "#0072B2"])
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["morphology", "morphology\n+ PAT"], fontsize=10)
    ax[1].set_ylabel(f"DBP error at $k$ = {K} (mmHg)")
    ax[1].set_ylim(0, max(m_only, m_pat) * 1.5)
    for x, v in zip([0, 1], [m_only, m_pat]):
        ax[1].text(x, v + 0.08, f"{v:.2f}", ha="center", fontsize=11, color="#333")
    ax[1].annotate("", xy=(1, m_pat * 1.12), xytext=(0, m_only * 1.12),
                   arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=1.6))
    ax[1].text(0.5, m_pat * 1.18, f"Δ = {abs(m_pat - m_only):.2f} mmHg", ha="center",
               fontsize=11, color="#C0392B", fontweight="bold")
    ax[1].text(0.5, m_pat * 1.30,
               "the governing law's variable,\nadded to a model that has\nmorphology, contributes nothing",
               ha="center", fontsize=9.5, color="#555")
    ax[1].set_title("b   the controlled test", fontsize=11.5, loc="left")
    ax[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("What the model actually uses: pulse SHAPE, not pulse TIMING\n"
                 "top features are trough depth, shape asymmetry and downstroke width "
                 "\u2014 no timing feature appears", fontsize=11)
    out = ROOT / "figures" / "fig_what_models_use.png"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}")
    print(f"\nPAT only {pat_only:.2f} | morphology {m_only:.2f} | morphology+PAT {m_pat:.2f}")
    print(f"adding PAT to morphology: {m_pat - m_only:+.3f} mmHg")


if __name__ == "__main__":
    main()
