"""fig_story.py -- the results in one figure (2 x 2): simulation on top, VitalDB below.

  a  simulated ECG and PPG (beat shapes fitted to VitalDB, 400 ms PPG channel offset), drawn at
     the heart rate of the VitalDB example in c; dashed lines join each R peak to its PPG foot
  b  simulated ECG/PPG with a known PTT -> BP law: audit slope vs the slope injected into the
     data (alpha x -0.22 mmHg/ms), mean +- SD over 3 seeds; dashed: perfect recovery
  c  VitalDB ECG and PPG, each R peak joined to the PPG foot it is paired with
  d  improvement over predicting the training mean against the audit result: the share of 144
     held-out patients with the transit-time sign (longer PAT, lower BP). Markers and error bars:
     mean +- SD of each configuration's 3 training runs (single runs in focus_models.json).
     Right margin: the data itself (within-patient PAT vs BP, quality-filtered; fig_supp a), and
     controls built from the measured PAT with the textbook (+, 100%) or inverted (-, 0%) law.
Models: 1-layer (triangle) and 3-layer (circle) transformers, trained on 12k (hollow) or 140k
(filled) segments of the official PulseDB VitalDB training set, 3 seeds each (focus_models.py).
Supplementary checks (law by quality level, ECG-PPG gap, seen patients) are in fig_supp.py.

Reads data/synth_waveform_audit.json, pat_bp_quality_vitaldb.json, focus_models.json and the
controls in real_audit_controls.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import synth_waveform_audit as SW
from fig_audit import stacked

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COL = {"sbp": "#c1543b", "dbp": "#2f4b7c"}
NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a"
STYLE = {"font.family": "Arial", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
         "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
         "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.3,
         "savefig.dpi": 300, "pdf.fonttype": 42}


def plabel(ax, s, title, x=-0.2, y=1.09):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.set_title(title, loc="left", pad=5)


def load_json(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def runs():
    """Every focus-model run (all seeds; superseded learning-rate runs excluded)."""
    out = []
    for k, m in load_json("focus_models.json").items():
        if "_lr" in k:
            continue
        c, cb = m["calfree"], m.get("calbased")
        out.append(dict(target=m["target"], depth=m["depth"], big=m["data"] == "140k",
                        seed=m.get("seed", 0), gain=c["mean_mae"] - c["mae"],
                        frac=c["ppg"]["frac_negative"], ppg=c["ppg"]["median"], ecg=c["ecg"]["median"],
                        both=c["both"]["median"], seen=cb["ppg"]["frac_negative"] if cb else None))
    return out


def mark(r, ax, x, y, ms=7, alpha=1.0, z=3):
    ax.plot(x, y, "^" if r["depth"] == 1 else "o", ms=ms, color=COL[r["target"]], alpha=alpha,
            mfc=COL[r["target"]] if r["big"] else "white", mew=1.3, zorder=z)


def data_levels():
    q = load_json("pat_bp_quality_vitaldb.json")
    return {"sbp": q["L2 ABP agrees"]["PPG_SBP"]["frac_negative"],
            "dbp": q["L2 ABP agrees"]["PPG_DBP"]["frac_negative"]}


def legend(fig, y=-0.02):
    k = dict(ls="", color="#555", ms=6, mew=1.1)
    h = [plt.Line2D([], [], marker="^", label="1 layer", **k),
         plt.Line2D([], [], marker="o", label="3 layers", **k),
         plt.Line2D([], [], marker="o", mfc="white", label="12k segments", **k),
         plt.Line2D([], [], marker="o", label="140k segments", **k),
         plt.Line2D([], [], color=COL["sbp"], lw=2, label="SBP"),
         plt.Line2D([], [], color=COL["dbp"], lw=2, label="DBP")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, y))


def main():
    plt.rcParams.update(STYLE)
    sw = load_json("synth_waveform_audit.json")
    ctl = load_json("real_audit_controls.json")["models"]
    R = runs(); data_frac = data_levels()

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38, left=0.09, right=0.98, top=0.93, bottom=0.12)
    ax_b, ax_d = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    pairs = []
    for row in (0, 1):
        sub = gs[row, 0].subgridspec(2, 1, hspace=0.12)
        e = fig.add_subplot(sub[0]); pairs.append((e, fig.add_subplot(sub[1], sharex=e)))
    for a_ in (ax_b, ax_d) + pairs[0] + pairs[1]:
        a_.spines[["top", "right"]].set_visible(False)
    for a_ in (ax_b, ax_d):
        a_.set_box_aspect(1)

    def signals(axes, ecg, ppg, fs, letter, title):
        ax_e, ax_p = axes
        stacked(ax_e, ax_p, ecg, ppg, fs, 0.0, 3.0)
        ax_p.set_xlabel(""); ax_p.spines["bottom"].set_visible(False)
        ax_p.tick_params(axis="x", bottom=False, labelbottom=False)
        for a_ in axes:
            a_.spines["left"].set_visible(False)
            a_.set_ylabel(a_.get_ylabel(), rotation=0, ha="right", va="center", fontsize=8.5)
        plabel(ax_e, letter, title, y=1.2)      # half-height axis: lift the letter to match b/d

    # VitalDB example first, so the simulated example can be drawn at the same heart rate
    d = mechlib.load_mini(str(DATA / "vitaldb_mini.npz")); fs = int(d["fs"])
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    i_ex = next(i for i in range(300)
                if np.sum(mechlib.segment_fiducials(Xte[i, :, 0], Xte[i, :, 1], fs)["feet"] < 3 * fs) >= 3)
    hr = 60.0 / (np.median(np.diff(mechlib.r_peaks(Xte[i_ex, :, 0], fs))) / fs)

    # a: simulated ECG and PPG (beat shapes fitted to VitalDB, 400 ms PPG offset)
    seg = SW.make_segment(np.random.default_rng(3), hr=hr, ptt_ms=210)
    signals(pairs[0], seg[:, 0], seg[:, 1], SW.FS, "a", "Simulated ECG and PPG")

    # b: audit recovers the law (simulated waveforms)
    # measured vs ground truth: the injected slope is alpha x the law written into the simulator
    alw = [0.0, 0.25, 0.5, 0.75, 1.0]
    LAW = -0.22                                              # mmHg/ms, synth_waveform_audit.make_dataset
    inj = np.array([a0 * LAW for a0 in alw])
    mea = np.array([sw["alphas"][str(a0)]["slope"] for a0 in alw])
    lim = [LAW * 1.1, 0.01]
    ax_b.plot(lim, lim, color="k", lw=.8, ls="--")
    for a0, x in zip(alw, inj):
        sv = sw["alphas"][str(a0)]["slopes"]
        ax_b.errorbar(x, np.mean(sv), yerr=np.std(sv), fmt="o", ms=4.5, color=NAVY, elinewidth=0.8, capsize=2)
    gain = float(np.polyfit(inj, mea, 1)[0])
    ax_b.text(0.97, 0.05, f"Recovered: {gain:.0%} of injected", transform=ax_b.transAxes, ha="right",
              fontsize=7.5, color="#333")
    ax_b.set_xlim(lim); ax_b.set_ylim(lim)
    ax_b.set_xlabel("Injected slope (mmHg/ms)"); ax_b.set_ylabel("Audit slope (mmHg/ms)")
    plabel(ax_b, "b", "Audit recovers the law (simulation)")

    # c: VitalDB example
    signals(pairs[1], Xte[i_ex, :, 0], Xte[i_ex, :, 1], fs, "c", "VitalDB ECG and PPG")

    # d: accuracy vs audit, every run, with the data and the controls
    a_ = ax_d
    gmax = max(r["gain"] for r in R)
    # reference marks sit in a margin right of the models, as short segments with labels above
    xa = gmax + 0.3; xb = xa + 0.28; xr = xb + 0.05

    def ref(y, col, ls, label):
        a_.plot([xa, xb], [y, y], color=col, lw=1.4, ls=ls, solid_capstyle="butt")
        a_.text((xa + xb) / 2, y + 2, label, fontsize=7, color=col, ha="center", va="bottom")

    ref(100 * ctl["textbook"]["ppg"]["frac_negative"], GREEN, "-", "+ control")
    ref(100 * ctl["inverted"]["ppg"]["frac_negative"], GREY, "-", "− control")
    a_.axhline(50, color="#ddd", lw=.8)
    for t, lab in (("sbp", "SBP"), ("dbp", "DBP")):
        ref(100 * data_frac[t], COL[t], "-", f"{lab} data")
        rt = [r for r in R if r["target"] == t]
        for dep in (1, 3):
            for big in (False, True):
                g = [r for r in rt if r["depth"] == dep and r["big"] == big]
                if g:
                    gx = [r["gain"] for r in g]; gy = [100 * r["frac"] for r in g]
                    a_.errorbar(np.mean(gx), np.mean(gy), xerr=np.std(gx), yerr=np.std(gy), fmt="none",
                                ecolor=COL[t], elinewidth=0.8, capsize=2, zorder=3)
                    mark(g[0], a_, np.mean(gx), np.mean(gy), ms=7, z=4)
    a_.set_xlim(-0.05, xr); a_.set_ylim(-3, 106)
    a_.set_xticks([0, 0.5, 1.0, 1.5, 2.0] if gmax > 1.6 else [0, 0.5, 1.0, 1.5])
    a_.set_xlabel("Improvement over mean (mmHg)"); a_.set_ylabel("Transit-time sign (% of patients)")
    plabel(a_, "d", "Accuracy vs. audit")

    legend(fig, y=0.0)
    fig.savefig(ROOT / "figures" / "fig_story.png", dpi=600, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_story.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_story.png / .pdf")


if __name__ == "__main__":
    main()
