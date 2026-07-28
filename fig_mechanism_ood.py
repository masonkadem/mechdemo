"""fig_mechanism_ood.py -- does the roll-audit predict OOD failure?

Scatter: mechanism strength (|PAT slope|, how strongly the model uses arrival time) vs the
OOD penalty (MIMIC-BP DBP MAE minus in-distribution DBP MAE), one point per architecture,
with the least-squares fit and Pearson r. Black-and-white, publication style.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def main():
    d = json.loads((ROOT / "data" / "ood_benchmark_ecgppg_full.json").read_text())["models"]
    names = [n for n in d if not n.startswith("_")]
    absslope, gap, frac, labels = [], [], [], []
    for n in names:
        a = d[n]["audit"]["dbp"]; o = d[n]["ood"]
        absslope.append(abs(a["dBP_dPTT"]))
        gap.append(o["mimic_bp"]["mae_dbp"] - o["id"]["mae_dbp"])
        frac.append(a["frac_correct_sign"])
        labels.append(n)
    x, y = np.array(absslope), np.array(gap)
    r = np.corrcoef(x, y)[0, 1]
    b, a0 = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    xs = np.linspace(x.min() - 2, x.max() + 2, 50)
    ax.plot(xs, a0 + b * xs, color="black", lw=1.3, ls="--", zorder=1,
            label=f"fit (r = {r:.2f})")
    # marker shade encodes correct-sign fraction (darker = more physiological)
    for xi, yi, fi, lab in zip(x, y, frac, labels):
        ax.scatter(xi, yi, s=130, c=[[1 - fi, 1 - fi, 1 - fi]], edgecolor="black",
                   linewidth=1.1, zorder=3)
        dx = 1.2 if lab != "xresnet1d50" else -1.2
        ha = "left" if dx > 0 else "right"
        ax.annotate(lab, (xi, yi), xytext=(dx, 6 if lab != "transformer" else -12),
                    textcoords="offset points", fontsize=8.5, ha=ha)
    ax.set_xlabel("roll-audit sensitivity  |dDBP/dPTT|  (mmHg/s, relative)\n"
                  "(how strongly the model responds to the arrival-time perturbation)")
    ax.set_ylabel("OOD penalty on MIMIC-BP\n(DBP MAE increase vs in-distribution, mmHg)")
    ax.set_title("Roll-audit sensitivity is associated with OOD robustness "
                 f"(r = {r:.2f}, n = 5)", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    # colorbar-like note for the shade
    ax.text(0.02, 0.03, "darker marker = larger fraction of negative-slope segments",
            transform=ax.transAxes, fontsize=7.5, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "fig_mechanism_ood.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_mechanism_ood.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] fig_mechanism_ood.png / .pdf   (r = {r:.2f}, slope {b:.3f})")


if __name__ == "__main__":
    main()
