"""fig_targets.py -- SBP vs DBP: does the arrival-time law hold in the data, and do models follow it?

  a  share of patients whose own data show the expected sign (longer PAT, lower BP), at four
     quality-filter levels, with the noise ceiling (pat_bp_quality.py)
  b  every transformer training run against the data, by target (real_audit_controls.py)
  c  improvement over predicting the mean vs share of patients with the expected sign
  d  response to an earlier PPG vs a later ECG (equal PAT change): runs that read the ECG-PPG
     gap fall on the identity line

Reads data/pat_bp_quality_vitaldb.json, data/real_audit_controls.json (DBP) and
data/real_audit_controls_sbp.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COL = {"SBP": "#c1543b", "DBP": "#2f4b7c"}
GREY = "#9aa0a6"


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def plabel(ax, s, title):
    ax.text(-0.2, 1.08, s, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=9.5, loc="left", pad=6)


def runs(res):
    """(name, MAE gain over the mean, % expected sign, CI, PPG slope, ECG slope) per model run."""
    out = []
    for k, m in res["models"].items():
        if k in ("textbook", "inverted"):
            continue
        r = m["ppg"]; n = r["n_subjects"]; kk = round(r["frac_negative"] * n)
        lo, hi = wilson(kk, n)
        out.append((k, res["baseline_mae"] - m["test_mae"], 100 * r["frac_negative"],
                    (100 * lo, 100 * hi), m["ppg"]["median"], m["ecg"]["median"]))
    return out


def main():
    q = json.loads((DATA / "pat_bp_quality_vitaldb.json").read_text())
    res = {"DBP": json.loads((DATA / "real_audit_controls.json").read_text()),
           "SBP": json.loads((DATA / "real_audit_controls_sbp.json").read_text())}

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.7), gridspec_kw={"wspace": 0.45})
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)

    # ---------------- a: the law in the data, by quality level
    a_ = ax[0]
    levels = [k for k in q if k.startswith("L")]
    xs = np.arange(len(levels))
    for bp in ("SBP", "DBP"):
        r = [q[l][f"PPG_{bp}"] for l in levels]
        a_.plot(xs, [100 * v["frac_negative"] for v in r], "-o", ms=5, color=COL[bp], label=bp)
        a_.plot(xs, [100 * v["noise_ceiling"] for v in r], ":", color=COL[bp], lw=1.2)
    a_.axhline(50, color="#bbb", lw=.8)
    a_.set_xticks(xs, ["All", "Clean\nbeats", "Arterial\nline agrees", "BP\nvaries"], fontsize=8)
    a_.set_ylim(40, 100); a_.set_ylabel("Patients with expected sign (%)")
    a_.legend(fontsize=8, frameon=False, loc="lower right")
    a_.text(0.02, 0.97, "Dotted: noise ceiling", transform=a_.transAxes, fontsize=7.5,
            color=GREY, va="top")
    plabel(a_, "a", "The law in the data")

    # ---------------- b: every run vs the data, by target
    a_ = ax[1]
    a_.axhline(50, color="#bbb", lw=.8)
    rng = np.random.default_rng(0)
    for i, bp in enumerate(("DBP", "SBP")):
        d = res[bp]["data"]["ppg_pat"]
        a_.plot([i - .3, i + .3], [100 * d["frac_negative"]] * 2, color="k", lw=1.6)
        for _, _, f, (lo, hi), _, _ in runs(res[bp]):
            x = i + rng.uniform(-.15, .15)
            a_.plot([x, x], [lo, hi], color=COL[bp], lw=1, alpha=.5)
            a_.plot(x, f, "o", color=COL[bp], ms=5)
    a_.text(1.33, 100 * res["SBP"]["data"]["ppg_pat"]["frac_negative"], "Data", fontsize=8, va="center")
    a_.set_xticks([0, 1], ["DBP\nmodels", "SBP\nmodels"], fontsize=8.5)
    a_.set_xlim(-.6, 1.6); a_.set_ylim(0, 100)
    a_.set_ylabel("Patients with expected sign (%)")
    plabel(a_, "b", "Models vs. data")

    # ---------------- c: learning beyond the mean vs following the law
    a_ = ax[2]
    a_.axhline(50, color="#bbb", lw=.8)
    for bp in ("DBP", "SBP"):
        rr = runs(res[bp])
        g = np.array([r[1] for r in rr]); f = np.array([r[2] for r in rr])
        a_.plot(g, f, "o", color=COL[bp], ms=6, label=f"{bp} (r = {np.corrcoef(g, f)[0, 1]:.2f})")
    a_.set_xlabel("Improvement over mean predictor (mmHg)")
    a_.set_ylabel("Patients with expected sign (%)"); a_.set_ylim(0, 100)
    a_.legend(fontsize=8, frameon=False, loc="lower right")
    plabel(a_, "c", "Accuracy vs. following the law")

    # ---------------- d: does the model read the ECG-PPG gap?
    a_ = ax[3]
    allv = [v for bp in res for r in runs(res[bp]) for v in r[4:6]]
    lim = max(abs(v) for v in allv) * 1.15
    a_.plot([-lim, lim], [-lim, lim], color="k", lw=.8, ls="--")
    a_.axhline(0, color="#bbb", lw=.6); a_.axvline(0, color="#bbb", lw=.6)
    for bp in ("DBP", "SBP"):
        rr = runs(res[bp])
        a_.plot([r[4] for r in rr], [r[5] for r in rr], "o", color=COL[bp], ms=6, label=bp)
    a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim)
    a_.set_xlabel("Response to earlier PPG (mmHg/ms)")
    a_.set_ylabel("Response to later ECG (mmHg/ms)")
    a_.text(0.03, 0.03, "Expected\nsign", transform=a_.transAxes, fontsize=7.5, color=GREY, va="bottom")
    a_.legend(fontsize=8, frameon=False, loc="lower right")
    plabel(a_, "d", "Do runs read the ECG–PPG gap?")

    fig.savefig(ROOT / "figures" / "fig_targets.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_targets.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_targets.png / .pdf")


if __name__ == "__main__":
    main()
