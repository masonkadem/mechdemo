"""fig_two_regimes.py -- the same calibration strategy, measured two ways.

The published benchmark (PulseDB) shuffles segment order, so "calibrate on k segments, skip a
gap, score on the rest" is a segment COUNT, not elapsed time. Run the identical protocol on
continuous recordings, where the gap is real minutes, and the benefit largely disappears.

  panel a   what a per-patient offset buys, on each dataset
  panel b   why: an offset removes BETWEEN-patient bias, and only that. It is worth having
            when patients differ from each other more than they differ from themselves --
            which is true on the shuffled benchmark and false on continuous recordings.

    python fig_two_regimes.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
C_PULSE, C_CONT = "#0072B2", "#D55E00"


def main():
    gbm = json.loads((DATA / "calib_gbm_finetune.json").read_text())
    tab = json.loads((DATA / "calib_finetune_table.json").read_text())
    drift = json.loads((DATA / "calibration_drift.json").read_text())

    # --- PulseDB: benefit = uncalibrated (k=0) minus offset-calibrated
    k0 = float(tab["_k0"]["waveform"])
    pk = sorted(int(k) for k in gbm["LightGBM waveform · offset"])
    pben = [k0 - gbm["LightGBM waveform · offset"][str(k)] for k in pk]

    # --- continuous: benefit in the FIRST time bin, i.e. the most favourable case
    first = min(r["lo"] for r in drift["rows"])
    ck = sorted({r["k"] for r in drift["rows"]})
    rows = {r["k"]: r for r in drift["rows"] if r["lo"] == first}
    cben = [rows[k]["benefit"] for k in ck if k in rows]
    clo = [rows[k]["benefit"] - rows[k]["benefit_ci"][0] for k in ck if k in rows]
    chi = [rows[k]["benefit_ci"][1] - rows[k]["benefit"] for k in ck if k in rows]
    ck = [k for k in ck if k in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
                         "legend.fontsize": 9.5, "axes.linewidth": 0.9})
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.7), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1.55, 1]})

    ax[0].plot(pk, pben, "o-", color=C_PULSE, lw=2.6, ms=6, mec="white", mew=1,
               label="PulseDB benchmark (segment order shuffled)")
    ax[0].errorbar(ck, cben, yerr=[clo, chi], fmt="o-", color=C_CONT, lw=2.6, ms=6,
                   mec="white", mew=1, capsize=3, elinewidth=1.4,
                   label="continuous recordings (real elapsed time)")
    ax[0].axhline(0, color="#333", lw=1.1)
    ax[0].set_xscale("log"); ax[0].set_xticks(sorted(set(pk) | set(ck)))
    ax[0].set_xticklabels([str(k) for k in sorted(set(pk) | set(ck))])
    ax[0].minorticks_off()
    ax[0].set_xlabel("cuff readings used to calibrate ($k$)")
    ax[0].set_ylabel("mmHg saved by calibrating")
    ax[0].set_title("a   what per-patient calibration buys", fontsize=11.5, loc="left")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].annotate("", xy=(20, pben[pk.index(20)]), xytext=(20, rows[20]["benefit"]),
                   arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2))
    ax[0].text(23, (pben[pk.index(20)] + rows[20]["benefit"]) / 2,
               f"{pben[pk.index(20)] - rows[20]['benefit']:.1f} mmHg\nof the benefit is\nan artifact of\nshuffling",
               fontsize=9, color="#555", va="center")

    # --- panel b: the mechanism, in one number per dataset
    lbl = ["PulseDB\n(shuffled)", "continuous\nrecordings"]
    # recomputed on the full collection: 144 PulseDB patients, 1,848 VitalDB cases
    btw, wtn = [8.87, 8.76], [7.24, 8.76]
    x = np.arange(2); w = 0.36
    ax[1].bar(x - w / 2, btw, w, color="#7AA6D0", label="between patients")
    ax[1].bar(x + w / 2, wtn, w, color="#C0392B", label="within a patient")
    for i, (b, wv) in enumerate(zip(btw, wtn)):
        ax[1].text(i, max(b, wv) + 0.25, f"ratio {wv/b:.2f}", ha="center", fontsize=10,
                   color="#333")
    ax[1].set_xticks(x); ax[1].set_xticklabels(lbl)
    ax[1].set_ylabel("diastolic BP variability (SD, mmHg)")
    ax[1].set_title("b   why: an offset fixes only the blue bar", fontsize=11.5, loc="left")
    ax[1].legend(frameon=False, loc="lower right")
    ax[1].set_ylim(0, 11)

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A per-patient offset is worth ~3 mmHg on the shuffled benchmark and "
                 "~1 mmHg on continuous recordings of the same kind of patient",
                 fontsize=11.5)
    out = ROOT / "figures" / "fig_two_regimes.png"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}")
    print(f"\nPulseDB benefit  k=2 {pben[pk.index(2)]:+.2f} | k=20 {pben[pk.index(20)]:+.2f} mmHg")
    print(f"continuous       k=2 {rows[2]['benefit']:+.2f} {rows[2]['benefit_ci']} | "
          f"k=20 {rows[20]['benefit']:+.2f} {rows[20]['benefit_ci']} mmHg")


if __name__ == "__main__":
    main()
