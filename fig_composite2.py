"""fig_composite2.py -- composite figure 2 (3 stacked panels):
  a  clean synthetic ECG+PPG+VPG+APG waveform (feature-extraction schematic)
  b  ID vs OOD DBP MAE table, deep models + LightGBM
  c  mechanism (roll-audit sensitivity) vs OOD scatter, minimal text
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fig_waveform_clean as wc

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
NAVY, RED, GREEN, ORANGE = "#2f4b7c", "#c1543b", "#3b8c5a", "#d98c3f"


def draw_waveform(gs, fig):
    """Reproduce the clean synthetic waveform into a 4-row subgrid of `gs`."""
    fs = 500
    t = np.linspace(-0.25, 0.9, int(1.15 * fs))
    ecg = wc.synth_ecg(t); foot = 0.22
    ppg = wc.synth_ppg(t, foot); ppg = ppg / ppg.max()
    sm = savgol_filter(ppg, 61, 3)
    vpg = savgol_filter(np.gradient(sm), 41, 3) * fs / 100
    apg = savgol_filter(np.gradient(np.gradient(sm)), 51, 3) * (fs / 100) ** 2
    sub = gs.subgridspec(4, 1, height_ratios=[1, 1.3, 0.9, 0.9], hspace=0.12)
    axs = [fig.add_subplot(sub[i]) for i in range(4)]
    for ax in axs:
        ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#e8e8e8", lw=0.8, zorder=0); ax.set_xlim(-0.25, 0.9)
    axs[0].plot(t, ecg, color=NAVY, lw=1.8); axs[0].plot(0, 1, "o", color=RED, ms=6)
    axs[0].annotate("R", (0, 1), xytext=(4, -2), textcoords="offset points", fontsize=10,
                    fontweight="bold", color=RED)
    axs[0].set_ylabel("ECG", fontsize=10, rotation=0, ha="right", va="center")
    pk = np.argmax(ppg); fi = np.argmin(np.abs(t - foot))
    axs[1].plot(t, ppg, color=GREEN, lw=2)
    axs[1].plot(t[fi], ppg[fi], "o", color="black", ms=6); axs[1].plot(t[pk], ppg[pk], "o", color=RED, ms=6)
    post = ppg[pk:]; ni = pk + np.argmin(post[:int(0.25 * fs)])
    axs[1].plot(t[ni], ppg[ni], "o", color=NAVY, ms=6)
    axs[1].annotate("", xy=(foot, -0.12), xytext=(0, -0.12),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
    axs[1].text(foot / 2, -0.22, "PAT", color=ORANGE, fontsize=9, ha="center", fontweight="bold")
    axs[1].set_ylabel("PPG", fontsize=10, rotation=0, ha="right", va="center"); axs[1].set_ylim(-0.3, 1.15)
    axs[2].plot(t, vpg, color=ORANGE, lw=1.8)
    axs[2].plot(t[np.argmax(vpg)], vpg.max(), "o", color=RED, ms=5)
    axs[2].plot(t[np.argmin(vpg)], vpg.min(), "o", color=NAVY, ms=5)
    axs[2].set_ylabel("VPG", fontsize=9, rotation=0, ha="right", va="center")
    axs[3].plot(t, apg, color=RED, lw=1.8)
    win = (t > foot - 0.01) & (t < foot + 0.45); idxw = np.where(win)[0]; aw = apg[idxw]
    pk_i = idxw[find_peaks(aw)[0]]; tr_i = idxw[find_peaks(-aw)[0]]
    ext = [pk_i[0]]
    for arr in [tr_i, pk_i, tr_i, pk_i]:
        cand = arr[arr > ext[-1]]
        if len(cand):
            ext.append(cand[0])
    for k, lbl in zip(ext[:5], "abcde"):
        axs[3].plot(t[k], apg[k], "o", color="black", ms=4)
        axs[3].annotate(lbl, (t[k], apg[k]), xytext=(0, 7 if apg[k] > 0 else -13),
                        textcoords="offset points", fontsize=9, fontweight="bold", ha="center")
    axs[3].set_ylabel("APG", fontsize=9, rotation=0, ha="right", va="center"); axs[3].margins(y=0.3)
    axs[3].set_xlabel("time relative to ECG R-peak (s)", fontsize=10)
    for gx in (t[np.argmax(vpg)], t[pk]):
        for ax in axs:
            ax.axvline(gx, color="#d5d5d5", lw=0.9, ls="--", zorder=0)
    axs[0].text(-0.02, 1.18, "a", transform=axs[0].transAxes, fontsize=14, fontweight="bold")
    axs[0].set_title("Feature extraction from ECG, PPG and its 1st/2nd derivatives",
                     fontsize=10, loc="left")


def main():
    fig = plt.figure(figsize=(9.5, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 0.9, 1.0], hspace=0.32)

    draw_waveform(gs[0], fig)

    # ---- b: ID vs OOD table (deep + LightGBM) ----
    e = json.loads((ROOT / "data" / "ood_benchmark_ecgppg_full.json").read_text())["models"]
    gfam = json.loads((ROOT / "data" / "gbm_families.json").read_text())
    rows = []
    for n, disp in [("lenet1d", "LeNet1d"), ("inception1d", "Inception1d"),
                    ("xresnet1d50", "XResNet50"), ("xresnet1d101", "XResNet101"),
                    ("transformer", "Transformer")]:
        o = e[n]["ood"]
        rows.append([disp, "deep", f"{o['id']['mae_dbp']:.1f}", f"{o['mimic_bp']['mae_dbp']:.1f}"])
    gw = gfam["waveform+demo"]
    rows.append(["LightGBM (features)", "interp.", f"{gw['id_dbp']:.1f}",
                 f"{gw['ood'].get('mimic_bp', {}).get('dbp', float('nan')):.1f}"])

    axb = fig.add_subplot(gs[1]); axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    hdr = ["Model", "Type", "ID DBP", "MIMIC-BP (OOD) DBP"]
    xcol = [0.0, 0.34, 0.52, 0.72]; align = ["left", "left", "right", "right"]
    n = len(rows); yhead = 0.88; dy = (yhead - 0.08) / (n + 1)

    def put(x, y, s, ha, bold=False, it=False):
        axb.text(x, y, s, ha=ha, va="center", fontsize=9,
                 fontweight=("bold" if bold else "normal"), fontstyle=("italic" if it else "normal"))
    for x, h, a in zip(xcol, hdr, align):
        put(x, yhead, h, a, bold=True)
    axb.plot([0, 0.86], [yhead + 0.5 * dy] * 2, color="black", lw=1.3)
    axb.plot([0, 0.86], [yhead - 0.5 * dy] * 2, color="black", lw=0.8)
    for i, r in enumerate(rows):
        yr = yhead - (i + 1) * dy
        for x, v, a, ci in zip(xcol, r, align, range(4)):
            put(x, yr, v, a, bold=(ci == 0), it=(ci == 1))
    axb.plot([0, 0.86], [yhead - (n + 0.5) * dy] * 2, color="black", lw=1.3)
    axb.text(-0.06, 1.02, "b", transform=axb.transAxes, fontsize=14, fontweight="bold")
    axb.set_title("Accuracy: identical in-distribution, divergent out-of-distribution "
                  "(deep models and the interpretable LightGBM)", fontsize=10, loc="left")

    # ---- c: mechanism (roll sensitivity) vs OOD, minimal text ----
    names = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
    absslope = [abs(e[n]["audit"]["dbp"]["dBP_dPTT"]) for n in names]
    gap = [e[n]["ood"]["mimic_bp"]["mae_dbp"] - e[n]["ood"]["id"]["mae_dbp"] for n in names]
    frac = [e[n]["audit"]["dbp"]["frac_correct_sign"] for n in names]
    r = np.corrcoef(absslope, gap)[0, 1]
    axc = fig.add_subplot(gs[2])
    b, a0 = np.polyfit(absslope, gap, 1)
    xs = np.linspace(min(absslope) - 2, max(absslope) + 2, 50)
    axc.plot(xs, a0 + b * xs, color="black", lw=1.3, ls="--", zorder=1, label=f"r = {r:.2f}")
    for xi, yi, fi, lab in zip(absslope, gap, frac, names):
        axc.scatter(xi, yi, s=120, c=[[1 - fi, 1 - fi, 1 - fi]], edgecolor="black", lw=1.1, zorder=3)
        axc.annotate(lab, (xi, yi), fontsize=8, xytext=(5, 5), textcoords="offset points")
    axc.set_xlabel("roll-audit sensitivity  |dDBP/dΔ|  (relative)")
    axc.set_ylabel("OOD penalty on MIMIC-BP (mmHg)")
    axc.legend(frameon=False, fontsize=9, loc="upper right")
    axc.spines[["top", "right"]].set_visible(False)
    axc.text(-0.08, 1.03, "c", transform=axc.transAxes, fontsize=14, fontweight="bold")
    axc.set_title("Mechanism sensitivity vs OOD robustness", fontsize=10, loc="left")

    fig.savefig(FIG / "fig_composite2.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / "fig_composite2.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_composite2.png / .pdf")


if __name__ == "__main__":
    main()
