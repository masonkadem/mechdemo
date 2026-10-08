"""fig_composite1.py -- composite figure 1 (3 stacked panels):
  a  SBP/DBP violin distributions across datasets (black outline, publication)
  b  dataset information table
  c  causal roll-audit schematic (reuse the synthetic-beat roll illustration)
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import physics_audit as pa

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
B = "C:/Users/mason/OneDrive - McMaster University/2026/BP"


def _age(v):
    v = v[np.isfinite(v)]; v = v[v >= 10]
    return v.mean() if len(v) else np.nan


def collect():
    recs = []
    d = np.load(ROOT / "data" / "vitaldb_full_calfree.npz")
    ytr = np.concatenate([d["ytr"], d["yva"]]); gtr = np.concatenate([d["gtr"], d["gva"]])
    recs.append(("VitalDB (train)", "train", len(ytr), len(np.unique(gtr)), True,
                 "age, sex, BMI", _age(d["age_tr"]), ytr))
    recs.append(("VitalDB (test)", "id", len(d["yte"]), len(np.unique(d["gte"])), True,
                 "age, sex, BMI", _age(d["age_te"]), d["yte"]))
    m = pa.load_mimic_bp(B, channels=("ppg",), max_patients=1524)
    _, k = pa.window_segments(m["X"][:1], 1250)
    recs.append(("MIMIC-BP", "ood", len(m["y"]) * k, len(np.unique(m["g"])), True, "--", np.nan, m["y"]))
    for nm, path in [("BCG", "data/bcg_dataset"),
                     ("Sensors", "C:/Users/mason/Downloads/sensors_dataset/sensors_dataset"),
                     ("UCI2", "data/uci2_dataset/uci2_dataset"),
                     ("PPG-BP", "C:/Users/mason/Downloads/ppgbp_dataset/ppgbp_dataset")]:
        e = pa.load_bpbenchmark(path, nm)
        demo = ("age, sex, BMI" if nm == "PPG-BP" else "age, sex") if e["demo"] else "--"
        ag = _age(e["demo"]["age"]) if e["demo"] else np.nan
        recs.append((nm, "ood", len(e["y"]), len(np.unique(e["g"])), False, demo, ag, e["y"]))
    return recs


def main():
    recs = collect()
    shade = {"train": "#3a3a3a", "id": "#7a7a7a", "ood": "#c8c8c8"}
    labels = [r[0] for r in recs]; cols = [shade[r[1]] for r in recs]; ys = [r[7] for r in recs]

    fig = plt.figure(figsize=(9.5, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 0.7], hspace=0.42)

    # ---- a: violins (black outline) ----
    gsa = gs[0].subgridspec(1, 2, wspace=0.22)
    for sub, ti, ylab in [(0, 0, "SBP (mmHg)"), (1, 1, "DBP (mmHg)")]:
        ax = fig.add_subplot(gsa[sub])
        data = [y[:, ti] for y in ys]
        vp = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.85)
        for j, b in enumerate(vp["bodies"]):
            b.set_facecolor(cols[j]); b.set_alpha(0.9); b.set_edgecolor("black"); b.set_linewidth(1.1)
        for j, dta in enumerate(data):
            q1, med, q3 = np.percentile(dta, [25, 50, 75])
            ax.plot([j + 1, j + 1], [q1, q3], color="black", lw=1.1, zorder=3)
            ax.plot(j + 1, med, "o", color="white", mec="black", ms=4, zorder=4)
        iqr = np.percentile(ys[0][:, ti], [25, 75])
        ax.axhspan(iqr[0], iqr[1], color="black", alpha=0.06, zorder=0)
        ax.set_xticks(range(1, len(recs) + 1), labels, fontsize=7.5, rotation=25, ha="right")
        ax.set_ylabel(ylab); ax.spines[["top", "right"]].set_visible(False)
        if sub == 0:
            ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.set_title("Blood-pressure distribution shift  (shaded = VitalDB IQR)",
                         fontsize=10, loc="left")

    # ---- b: dataset table (booktabs) ----
    axb = fig.add_subplot(gs[1]); axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
    cols_l = ["Dataset", "Role", "Segments", "Subjects", "SBP/DBP", "Age", "Channels", "Demographics"]
    xcol = [0.00, 0.17, 0.31, 0.44, 0.57, 0.70, 0.79, 0.90]
    align = ["left", "left", "right", "right", "center", "right", "left", "left"]
    n = len(recs); yhead = 0.92; dy = (yhead - 0.05) / (n + 1)
    role_t = {"train": "train", "id": "in-dist", "ood": "OOD"}

    def put(x, y, s, ha, bold=False, it=False):
        axb.text(x, y, s, ha=ha, va="center", fontsize=8.3,
                 fontweight=("bold" if bold else "normal"), fontstyle=("italic" if it else "normal"))
    for x, lab, a in zip(xcol, cols_l, align):
        put(x, yhead, lab, a, bold=True)
    axb.plot([0, 1], [yhead + 0.5 * dy] * 2, color="black", lw=1.3)
    axb.plot([0, 1], [yhead - 0.5 * dy] * 2, color="black", lw=0.8)
    for i, (lab, role, ns, nsub, ecg, demo, ag, y) in enumerate(recs):
        yr = yhead - (i + 1) * dy
        seg = f"{ns/1000:.0f}k" if ns >= 1000 else str(ns)
        vals = [lab, role_t[role], seg, f"{nsub:,}",
                f"{y[:,0].mean():.0f}/{y[:,1].mean():.0f}",
                f"{ag:.0f}" if np.isfinite(ag) else "--",
                "ECG+PPG" if ecg else "PPG", demo]
        for x, v, a, ci in zip(xcol, vals, align, range(len(vals))):
            put(x, yr, v, a, bold=(ci == 0), it=(ci == 1))
    axb.plot([0, 1], [yhead - (n + 0.5) * dy] * 2, color="black", lw=1.3)
    axb.text(-0.06, 1.04, "b", transform=axb.transAxes, fontsize=14, fontweight="bold")
    axb.set_title("Datasets: one training source, one in-distribution and five OOD tests",
                  fontsize=10, loc="left")

    # ---- c: causal roll-audit schematic ----
    axc = fig.add_subplot(gs[2])
    fs = 125; t = np.linspace(0, 5, 5 * fs)
    beat = np.exp(-((t % 1.1 - 0.32) ** 2) / 0.004) + 0.35 * np.exp(-((t % 1.1 - 0.6) ** 2) / 0.01)
    sh = int(0.14 * fs)
    axc.plot(t, beat, color="#2f4b7c", lw=1.8, label="PPG")
    axc.plot(t, np.roll(beat, sh), color="#c1543b", lw=1.8, ls="--", label="PPG rolled later (+\u0394)")
    axc.annotate("", xy=(0.32 + 0.14, 1.2), xytext=(0.32, 1.2),
                 arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    axc.text(0.48, 1.3, "\u0394", fontsize=12, ha="center")
    axc.set_xlim(0, 3.4); axc.set_ylim(-0.15, 1.55); axc.set_yticks([])
    axc.set_xlabel("time (s)"); axc.spines[["top", "right", "left"]].set_visible(False)
    axc.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    axc.text(-0.06, 1.05, "c", transform=axc.transAxes, fontsize=14, fontweight="bold")
    axc.set_title("Causal roll-audit: shift PPG in time, measure the BP response "
                  "(relative sensitivity across models)", fontsize=10, loc="left")

    fig.savefig(FIG / "fig_composite1.png", dpi=170, bbox_inches="tight")
    fig.savefig(FIG / "fig_composite1.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_composite1.png / .pdf")


if __name__ == "__main__":
    main()
