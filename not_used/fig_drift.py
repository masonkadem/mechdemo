"""fig_drift.py -- the figure the calibration-burden curve could not draw.

fig_calibration_strategy.py answers "how many cuff readings?". This answers "and how long do
they last?", which PulseDB cannot be asked because it shuffles segment order. ISO 81060-3 and
IEEE 1708a both require a post-calibration drift characterisation and a recalibration
interval; no published benchmark on this data reports either.

  panel a   drift of the offset-calibrated model, one line per k -- does more calibration
            data buy a LONGER-lasting calibration, or only a better initial one?
  panel b   every arm at a fixed k, against the uncalibrated population model, which is the
            line a calibration must stay below to be worth anything at all

    python fig_drift.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
C_K = {1: "#C8C8C8", 2: "#9FB6D0", 5: "#0072B2", 20: "#003F5C"}
C_ARM = {"offset": "#0072B2", "cuff_avg": "#3C3C3C", "equation": "#009E73",
         "uncalibrated": "#C0392B"}
LABEL = {"offset": "frozen model + per-patient offset", "cuff_avg": "cuff average (no waveform)",
         "equation": "equation, 2 params ($a + b/\\mathrm{PAT}^2$)",
         "uncalibrated": "population model, no calibration"}


def mid(r):
    """Bin centre in minutes; the open-ended last bin is plotted at a finite stand-in."""
    return (r["lo"] + min(r["hi"], r["lo"] + 120)) / 2.0   # open last bin: plot just past its start


def main():
    blob = json.loads((DATA / "calibration_drift.json").read_text())
    rows = blob["rows"]
    ks = sorted({r["k"] for r in rows})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
                         "legend.fontsize": 9, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ---- panel a: what calibration BUYS, plotted directly.
    # Two lines that cross cannot be subtracted by eye, and the quantity a reader wants is the
    # difference: how much better is a calibrated model than an uncalibrated one, right now.
    for k in ks:
        rs = sorted([r for r in rows if r["k"] == k], key=mid)
        if len(rs) < 2:
            continue
        ax[0].plot([mid(r) for r in rs], [r["uncalibrated"] - r["offset"] for r in rs], "o-",
                   color=C_K.get(k, "#666"), lw=2.4, ms=5, mec="white", mew=0.9,
                   label=f"k = {k} reading{'s' if k > 1 else ''}")
    ax[0].axhline(0, color="#333", lw=1.1)
    ax[0].text(ax[0].get_xlim()[1] * 0.98, 0.06, "calibration helps above this line",
               ha="right", va="bottom", fontsize=8.5, color="#333")
    ax[0].text(ax[0].get_xlim()[1] * 0.98, -0.06, "calibration HURTS below it",
               ha="right", va="top", fontsize=8.5, color="#C0392B")
    ax[0].set_title("a   what per-patient calibration actually buys", fontsize=11, loc="left")
    ax[0].set_xlabel("minutes since the calibration window closed")
    ax[0].set_ylabel("benefit vs no calibration (mmHg)")
    ax[0].legend(frameon=False, loc="upper right")

    # ---- panel b: every arm at the largest k that has an equation arm
    kb = max([k for k in ks if any(r["k"] == k and r.get("equation") for r in rows)] or [ks[-1]])
    rs = sorted([r for r in rows if r["k"] == kb], key=mid)
    for arm in ("cuff_avg", "equation", "offset", "uncalibrated"):
        xs = [mid(r) for r in rs if r.get(arm) is not None]
        ys = [r[arm] for r in rs if r.get(arm) is not None]
        if len(xs) < 2:
            continue
        ax[1].plot(xs, ys, "o-" if arm != "uncalibrated" else "--", color=C_ARM[arm],
                   lw=2.4 if arm == "offset" else 1.8, ms=5, mec="white", mew=0.9,
                   label=LABEL[arm])
    ax[1].set_title(f"b   all arms at k = {kb}", fontsize=11, loc="left")
    ax[1].set_xlabel("minutes since the calibration window closed")
    ax[1].set_ylabel("diastolic BP error (mmHg)")
    ax[1].legend(frameon=False)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    # AAMI 5 mmHg is an ERROR threshold, so it belongs only on the error axis (panel b).
    # Drawing it on panel a, whose y is a difference of two errors, would be meaningless.
    ax[1].axhline(5.0, color="#BBB", lw=0.8, ls=":", zorder=0)
    ax[1].text(ax[1].get_xlim()[1], 5.05, "AAMI 5 mmHg", ha="right", va="bottom",
               fontsize=8, color="#999")

    n = blob["n_segments"]
    fig.suptitle(f"Per-patient calibration on continuous recordings: "
                 f"{blob['n_test_cases']} held-out patients, {n:,} segments, "
                 f"one surgery each", fontsize=11.5)
    out = ROOT / "figures" / "fig_drift.png"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
