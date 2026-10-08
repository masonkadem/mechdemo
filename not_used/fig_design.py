"""fig_design.py -- Figure 1: study design + data landscape (publication quality).

Panels (lowercase labels):
  a  study design: VitalDB -> models -> two audit tracks -> OOD sets (clean, few boxes)
  b  example raw ECG+PPG traces: VitalDB (train) vs MIMIC-BP vs BCG (the morphology the model sees)
  c  SBP/DBP violin distributions across all datasets (the label shift)
  d  the roll-audit method: shift PPG later -> longer arrival time -> lower BP (faithful sign)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

# colorblind-safe categorical order (Okabe-Ito), fixed by dataset identity
CB = {"train": "#555555", "mimic": "#0072B2", "bcg": "#E69F00", "sensors": "#009E73",
      "uci2": "#CC79A7", "ppgbp": "#D55E00", "ecg": "#333333", "ppg": "#0072B2",
      "shift": "#D55E00", "good": "#009E73"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "font.family": "DejaVu Sans"})


def _plabel(ax, letter, x=-0.06, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")


def panel_schematic(ax):
    """Clean 4-stage flow, minimal boxes."""
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def box(x, y, w, h, text, fc, fs=8.5, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    fc=fc, ec="#333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.3, color="#666"))

    box(0.1, 1.3, 1.9, 1.4, "VitalDB\ntrain", "#dcdcdc", fs=9)
    box(2.9, 2.3, 2.5, 1.3, "5 deep models\n+ LightGBM", "#4C72B0", fs=8.5, tc="white")
    box(6.3, 2.9, 3.6, 0.9, "audit  (roll + probe battery)", "#bcd7ea", fs=8.5)
    box(6.3, 1.6, 3.6, 0.9, "OOD:  MIMIC-BP + 4 PPG sets", "#ededed", fs=8.5)

    arrow(2.0, 2.0, 2.9, 2.7)
    arrow(5.4, 3.0, 6.3, 3.35)
    arrow(5.4, 2.7, 6.3, 2.0)
    arrow(8.1, 2.9, 8.1, 2.5)
    _plabel(ax, "a", x=-0.02, y=1.02)
    ax.set_title("Study design", fontsize=10, loc="left", x=0.0)


def panel_traces(ax):
    """One clean beat-train per dataset, offset vertically. Shows the raw morphology shift."""
    z = np.load(ROOT / "data" / "_fig1_traces.npz")
    fs = int(z["fs"])
    t = np.arange(z["vit"].shape[0]) / fs
    # VitalDB ECG + PPG, then MIMIC PPG, then BCG PPG -- offset so all are visible
    ax.plot(t, z["vit"][:, 0] * 0.5 + 5.2, color=CB["ecg"], lw=1.0)
    ax.text(t[-1] + 0.05, 5.2, "ECG (VitalDB)", fontsize=7.5, va="center", color=CB["ecg"])
    ax.plot(t, z["vit"][:, 1] * 0.6 + 3.6, color=CB["train"], lw=1.2)
    ax.text(t[-1] + 0.05, 3.6, "PPG (VitalDB)", fontsize=7.5, va="center", color=CB["train"])
    ax.plot(t, z["mim"][:, 1] * 0.6 + 2.0, color=CB["mimic"], lw=1.2)
    ax.text(t[-1] + 0.05, 2.0, "PPG (MIMIC-BP)", fontsize=7.5, va="center", color=CB["mimic"])
    ax.plot(t, z["bcg"] * 0.6 + 0.4, color=CB["bcg"], lw=1.2)
    ax.text(t[-1] + 0.05, 0.4, "PPG (BCG)", fontsize=7.5, va="center", color=CB["bcg"])
    ax.set_xlim(0, t[-1] + 1.4); ax.set_ylim(-0.6, 6.2)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _plabel(ax, "b", x=-0.04)
    ax.set_title("Example waveforms  (per-segment normalized)", fontsize=10, loc="left")


def panel_violins(ax_s, ax_d):
    z = np.load(ROOT / "data" / "_fig1_bp.npz")
    order = ["VitalDB_(train)", "MIMIC-BP", "BCG", "Sensors", "UCI2", "PPG-BP"]
    labels = ["VitalDB", "MIMIC", "BCG", "Sensors", "UCI2", "PPG-BP"]
    cols = [CB["train"], CB["mimic"], CB["bcg"], CB["sensors"], CB["uci2"], CB["ppgbp"]]
    for ax, ti, ylab, pl in [(ax_s, 0, "SBP (mmHg)", "c"), (ax_d, 1, "DBP (mmHg)", None)]:
        data = [z[k][:, ti] for k in order]
        vp = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.85)
        for i, b in enumerate(vp["bodies"]):
            b.set_facecolor(cols[i]); b.set_alpha(0.6 if i else 0.4); b.set_edgecolor(cols[i])
            b.set_linewidth(1.0)
        for i, dta in enumerate(data):
            q1, med, q3 = np.percentile(dta, [25, 50, 75])
            ax.plot([i + 1, i + 1], [q1, q3], color="black", lw=1.2, zorder=3)
            ax.plot(i + 1, med, "o", color="white", mec="black", ms=4, zorder=4)
        vq1, vq3 = np.percentile(data[0], [25, 75])
        ax.axhspan(vq1, vq3, color=CB["train"], alpha=0.08, zorder=0)
        ax.set_xticks(range(1, len(order) + 1), labels, fontsize=7.5, rotation=20)
        ax.set_ylabel(ylab); ax.margins(x=0.02)
        if pl:
            _plabel(ax, pl, x=-0.16)
    ax_s.set_title("Blood-pressure shift  (shaded = VitalDB IQR)", fontsize=10, loc="left")


def panel_audit(ax):
    """Roll PPG later -> longer arrival -> lower BP. No textbox; one caption line."""
    fs = 125
    t = np.linspace(0, 6, 6 * fs)
    beat = np.exp(-((t % 1.2 - 0.35) ** 2) / 0.004) + 0.35 * np.exp(-((t % 1.2 - 0.62) ** 2) / 0.01)
    sh = int(0.14 * fs)
    ax.plot(t, beat, color=CB["ppg"], lw=1.6, label="PPG")
    ax.plot(t, np.roll(beat, sh), color=CB["shift"], lw=1.6, ls="--", label="PPG rolled +Δ (later)")
    # arrival-time arrow on the first beat
    ax.annotate("", xy=(0.35 + 0.14, 1.18), xytext=(0.35, 1.18),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
    ax.text(0.5, 1.28, "Δ", fontsize=10, ha="center")
    ax.set_xlim(0, 4.2); ax.set_ylim(-0.15, 1.5)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    _plabel(ax, "d", x=-0.04)
    ax.set_title("Causal roll-audit:  roll PPG later  →  longer arrival time (PTT)  →  "
                 "lower BP is faithful physiology", fontsize=9.5, loc="left")


def main():
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.62, 0.95, 1.15, 0.72],
                          hspace=0.62, wspace=0.26)
    panel_schematic(fig.add_subplot(gs[0, :]))
    panel_traces(fig.add_subplot(gs[1, :]))
    panel_violins(fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]))
    panel_audit(fig.add_subplot(gs[3, :]))
    fig.savefig(FIG / "fig0_design.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig0_design.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig0_design.png / .pdf")


if __name__ == "__main__":
    main()
