"""fig_channel_compare.py -- ECG+PPG vs PPG-only comparison, SBP and DBP, per model + LightGBM.
Shows the ECG channel adds little accuracy -- consistent with models not using arrival-time
physics. PPG-only cannot support the PAT roll-audit (no timing reference); its mechanism is
tested by intra-beat perturbations instead (ppg_audit.py).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def main():
    ecg = json.loads((ROOT / "data" / "ood_benchmark_ecgppg_full.json").read_text())["models"]
    ppg = json.loads((ROOT / "data" / "ood_benchmark_ppg.json").read_text())["models"]
    disp = {"lenet1d": "LeNet1d", "inception1d": "Inception1d", "xresnet1d50": "XResNet50",
            "xresnet1d101": "XResNet101", "transformer": "Transformer"}
    names = list(disp)

    # LightGBM full-data numbers (from earlier runs): ECG+PPG 12.71/8.10, PPG-only 12.72/8.07
    lgb_ecg = (12.71, 8.10); lgb_ppg = (12.72, 8.07)

    rows = []                                              # (model, ecgSBP, ecgDBP, ppgSBP, ppgDBP)
    for n in names:
        eo = ecg[n]["ood"]["id"]; po = ppg[n]["ood"]["id"]
        rows.append([disp[n], eo["mae_sbp"], eo["mae_dbp"], po["mae_sbp"], po["mae_dbp"]])
    rows.append(["LightGBM", lgb_ecg[0], lgb_ecg[1], lgb_ppg[0], lgb_ppg[1]])

    fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # grouped header (group labels on top row, underline below them, sub-headers under that)
    ax.plot([0, 0.86], [0.99, 0.99], color="black", lw=1.3)          # toprule
    ax.text(0.0, 0.90, "Model", fontsize=9.5, fontweight="bold")
    ax.text(0.405, 0.93, "ECG + PPG", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(0.745, 0.93, "PPG only", fontsize=9.5, fontweight="bold", ha="center")
    ax.plot([0.30, 0.52], [0.905, 0.905], color="black", lw=0.7)     # group underlines
    ax.plot([0.635, 0.855], [0.905, 0.905], color="black", lw=0.7)
    sub = ["SBP", "DBP", "SBP", "DBP"]; xs = [0.35, 0.49, 0.685, 0.83]
    for x, s in zip(xs, sub):
        ax.text(x, 0.865, s, fontsize=8.5, ha="right")
    n = len(rows); yhead = 0.80; dy = (yhead - 0.06) / (n + 1)
    ax.plot([0, 0.86], [0.835, 0.835], color="black", lw=0.8)        # midrule under sub-headers
    for i, r in enumerate(rows):
        yr = yhead - i * dy
        ax.text(0.0, yr, r[0], fontsize=9, fontweight="bold" if i < n else "normal")
        for x, v in zip(xs, r[1:]):
            ax.text(x, yr, f"{v:.1f}", fontsize=9, ha="right")
    ax.plot([0, 0.86], [yhead - (n - 0.5) * dy] * 2, color="black", lw=1.3)
    ax.text(0.0, yhead - (n + 0.6) * dy,
            "MAE (mmHg), in-distribution. Removing ECG costs <0.5 mmHg -- the arrival-time "
            "channel\nadds little, consistent with the roll-audit showing models barely use PAT.",
            fontsize=7.8, style="italic", color="#444", va="top")
    ax.set_title("Does the ECG channel help?  ECG+PPG vs PPG-only", fontsize=11, loc="left")
    fig.savefig(FIG / "fig_channel_compare.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_channel_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_channel_compare.png / .pdf")


if __name__ == "__main__":
    main()
