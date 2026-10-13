"""fig_supp.py -- supplementary checks for fig_story.

  a  the law in the data: share of 600 VitalDB patients whose within-patient BP falls as PAT
     lengthens, at four quality-filter levels; dotted: noise ceiling (pat_bp_quality.py)
  b  is it arrival time? Median response to an earlier PPG vs to a later ECG (the same PAT
     change), every run; on the dashed identity line the model responds to the ECG-PPG gap.
     Stars: controls
  c  unseen vs seen patients: share with the textbook sign on the 144 held-out patients
     (CalFree) and on 150 patients seen in training (PulseDB CalBased), mean of each
     configuration's runs; ticks: the data
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_story import COL, GREY, GREEN, data_levels, legend, load_json, mark, plabel, runs

ROOT = Path(__file__).resolve().parent


def main():
    q = load_json("pat_bp_quality_vitaldb.json")
    ctl = load_json("real_audit_controls.json")["models"]
    R = runs(); data_frac = data_levels()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.35})
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)

    # a: the law in the data
    a_ = ax[0]
    levels = [k for k in q if k.startswith("L")]
    xs = np.arange(len(levels))
    for t, lab in (("sbp", "SBP"), ("dbp", "DBP")):
        r = [q[l][f"PPG_{lab}"] for l in levels]
        a_.plot(xs, [100 * v["frac_negative"] for v in r], "-o", ms=5, color=COL[t], label=lab)
        a_.plot(xs, [100 * v["noise_ceiling"] for v in r], ":", color=COL[t], lw=1.2)
    a_.axhline(50, color="#bbb", lw=.8)
    a_.set_xticks(xs, ["All", "Clean\nbeats", "Arterial\nline agrees", "BP\nvaries"], fontsize=8)
    a_.set_ylim(40, 100); a_.set_ylabel("Patients with textbook sign (%)")
    a_.legend(fontsize=8, frameon=False, loc="lower right")
    plabel(a_, "a", "The law in the data")

    # b: does the model respond to the ECG-PPG gap?
    a_ = ax[1]
    pts = [(r["ppg"], r["ecg"]) for r in R] + [(ctl[c]["ppg"]["median"], ctl[c]["ecg"]["median"])
                                               for c in ("textbook", "inverted")]
    lim = max(abs(v) for pt in pts for v in pt) * 1.15
    a_.plot([-lim, lim], [-lim, lim], color="k", lw=.8, ls="--")
    a_.axhline(0, color="#bbb", lw=.6); a_.axvline(0, color="#bbb", lw=.6)
    for r in R:
        mark(r, a_, r["ppg"], r["ecg"], ms=6)
    a_.plot(ctl["textbook"]["ppg"]["median"], ctl["textbook"]["ecg"]["median"], "*", ms=11, color=GREEN)
    a_.plot(ctl["inverted"]["ppg"]["median"], ctl["inverted"]["ecg"]["median"], "*", ms=11, color=GREY)
    a_.text(0.16, 0.03, "Textbook sign", transform=a_.transAxes, fontsize=7.5, color=GREY, va="bottom")
    a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim)
    a_.set_xlabel("Response to earlier PPG (mmHg/ms)"); a_.set_ylabel("Response to later ECG (mmHg/ms)")
    plabel(a_, "b", "Is it the ECG–PPG gap?")

    # c: unseen vs seen patients, configuration means
    a_ = ax[2]
    a_.axhline(50, color="#bbb", lw=.8)
    for t in ("sbp", "dbp"):
        for dep in (1, 3):
            for big in (False, True):
                g = [r for r in R if r["target"] == t and r["depth"] == dep and r["big"] == big
                     and r["seen"] is not None]
                if not g:
                    continue
                u, s = 100 * np.mean([r["frac"] for r in g]), 100 * np.mean([r["seen"] for r in g])
                a_.plot([0, 1], [u, s], "-", color=COL[t], lw=1, alpha=.6)
                mark(g[0], a_, 0, u); mark(g[0], a_, 1, s)
        a_.plot([1.3], [100 * data_frac[t]], "_", ms=14, mew=2, color=COL[t])
    a_.text(1.3, 100 * data_frac["sbp"] + 4, "Data", fontsize=8, ha="center")
    a_.set_xticks([0, 1], ["Unseen\npatients", "Seen\npatients"]); a_.set_xlim(-.4, 1.55)
    a_.set_ylim(0, 100); a_.set_ylabel("Patients with textbook sign (%)")
    plabel(a_, "c", "Does knowing the patient help?")

    legend(fig, y=-0.08)
    fig.savefig(ROOT / "figures" / "fig_supp.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_supp.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_supp.png / .pdf")


if __name__ == "__main__":
    main()
