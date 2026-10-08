"""fig_calibration.py -- how many cuff readings does each model need?

What a calibration anchor is
----------------------------
An anchor is one reference cuff measurement taken from the person the device is worn by. With k
anchors the device fits a single number for that person -- an offset added to every subsequent
prediction -- and is then scored on the segments it has not seen. k = 0 is the calibration-free
case, where the model must be right about a stranger from the waveform alone.

This is the quantity a product decision actually turns on. "How accurate is the model" is
answered by ID error; "how much does the user have to do before it works" is answered by the
anchor count, and the two order models differently.

Anchors are drawn at random within each subject, never the first k. First-k anchors sit adjacent
in time to the scored segments and absorb local drift as well as the per-subject offset, which
flatters every number by roughly 0.2-0.3 mmHg.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAVY, RED, GREEN, GREY, ORANGE = "#2f4b7c", "#c1543b", "#3b8c5a", "#9aa0a6", "#d98c3f"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
KS = [0, 1, 2, 3, 5, 10, 20]


def main():
    cam = json.loads((DATA / "calib_all_models.json").read_text())
    fai = json.loads((DATA / "gbm_faithful.json").read_text())

    curves = {}
    for k, v in cam.items():
        if isinstance(v, dict) and "curve" in v:
            curves[k] = {int(a): b for a, b in v["curve"].items()}
    for k, v in fai.items():
        if isinstance(v, dict) and "curve" in v:
            curves["faithful: " + k] = {int(a): b for a, b in v["curve"].items()}

    show = [
        ("gbm deep (83) + demo", GREEN, "-", "LightGBM + demographics"),
        ("gbm default (83)", NAVY, "-", "LightGBM, waveform only"),
        ("faithful: PAT + morphology (small)", ORANGE, "-",
         "Faithful: PAT + morphology (4.5k leaves)"),
        ("xresnet1d50", RED, "--", "XResNet50 (887k parameters)"),
        ("transformer", GREY, "--", "Transformer (107k parameters)"),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    target = None
    for key, col, ls, lab in show:
        if key not in curves:
            continue
        c = curves[key]
        ys = [c.get(k, np.nan) for k in KS]
        ax.plot(KS, ys, ls, color=col, lw=1.9, marker="o", ms=4.2, label=lab)
        if key == "xresnet1d50":
            target = c.get(20)
    if target:
        ax.axhline(target, color=RED, lw=0.9, ls=":", alpha=0.8)
        ax.text(20, target - 0.10, f"best deep net at k=20 ({target:.2f})", fontsize=7.5,
                color=RED, ha="right", va="top")
    ax.set_xlabel("k  =  cuff readings collected from this person", fontsize=9)
    ax.set_ylabel("DBP MAE on held-out segments (mmHg)", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="upper right")
    ax.text(0.02, 0.04, "k = 0 is calibration-free", transform=ax.transAxes,
            fontsize=7.5, color=GREY)

    rows = []
    for key, col, ls, lab in show:
        if key not in curves or not target:
            continue
        c = curves[key]
        hit = next((k for k in KS if c.get(k, 9e9) <= target), None)
        rows.append((lab, hit, col))

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_calibration.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_calibration.png")
    for lab, v, _ in rows:
        print(f"  {lab:44s} {v if v is not None else '>20'}")


if __name__ == "__main__":
    main()
