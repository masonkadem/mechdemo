"""fig_datasets.py -- publication datasets figure for the mechdemo tab.

Two panels, kept clean (no KS, no em dashes):
  a  a plain publication table: role, segments, subjects, channels, demographics per dataset
  b  SBP+DBP violin distributions across all datasets, VitalDB IQR shaded as the reference

Palette matches app_faithfulness.py (NAVY train, GREEN ID, ORANGE OOD). Writes
figures/fig_datasets.png (+ .pdf) so the Streamlit tab can st.image() it.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import mechlib
import physics_audit as pa

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
B = "C:/Users/mason/OneDrive - McMaster University/2026/BP"

NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a"
ORANGE = "#d98c3f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})


def _age(v):
    v = v[np.isfinite(v)]; v = v[v >= 10]
    return v.mean() if len(v) else np.nan


def collect():
    """Return ordered dataset records:
    (label, role, n_seg, n_subj, has_ecg, demo_str, age, y[SBP,DBP])."""
    recs = []
    full = ROOT / "data" / "vitaldb_full_calfree.npz"
    if full.exists():
        d = np.load(full)
        ytr = np.concatenate([d["ytr"], d["yva"]])
        gtr = np.concatenate([d["gtr"], d["gva"]])
        recs.append(("VitalDB (train)", "train", len(ytr), len(np.unique(gtr)), True,
                     "age, sex, BMI", _age(d["age_tr"]), ytr))
        recs.append(("VitalDB (test)", "id", len(d["yte"]), len(np.unique(d["gte"])), True,
                     "age, sex, BMI", _age(d["age_te"]), d["yte"]))
    else:
        d = mechlib.load_mini("data/vitaldb_mini_deep.npz")
        y = np.concatenate([d["ytr"], d["yva"], d["yte"]])
        recs.append(("VitalDB (train)", "train", len(y), 894, True, "n/a", np.nan, y))

    m = pa.load_mimic_bp(B, channels=("ppg",), max_patients=1524)
    _, k = pa.window_segments(m["X"][:1], 1250)
    recs.append(("MIMIC-BP", "ood", len(m["y"]) * k, len(np.unique(m["g"])), True, "n/a", np.nan, m["y"]))
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
    role_c = {"train": NAVY, "id": GREEN, "ood": ORANGE}
    role_t = {"train": "train", "id": "in-dist test", "ood": "OOD"}

    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.1], hspace=0.36)

    # ---- panel a: black-and-white publication table (booktabs style: horizontal rules only)
    axa = fig.add_subplot(gs[0]); axa.axis("off")
    axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    col_labels = ["Dataset", "Role", "Segments", "Subjects", "SBP/DBP", "Age",
                  "Channels", "Demographics"]
    xcol = [0.00, 0.17, 0.30, 0.43, 0.57, 0.70, 0.79, 0.91]   # left x of each column
    align = ["left", "left", "right", "right", "center", "right", "left", "left"]
    n = len(recs)
    y_top = 0.90
    dy = (y_top - 0.06) / (n + 1)
    yhead = y_top

    def put(x, y, s, ha, bold=False, it=False):
        axa.text(x, y, s, ha=ha, va="center", fontsize=8.6,
                 fontweight=("bold" if bold else "normal"),
                 fontstyle=("italic" if it else "normal"))

    # header
    for x, lab, a in zip(xcol, col_labels, align):
        put(x, yhead, lab, a, bold=True)
    # booktabs rules
    axa.plot([0, 1], [yhead + 0.5 * dy] * 2, color="black", lw=1.3)     # toprule
    axa.plot([0, 1], [yhead - 0.5 * dy] * 2, color="black", lw=0.8)     # midrule
    # rows
    for i, (lab, role, ns, nsub, ecg, demo, ag, y) in enumerate(recs):
        yr = yhead - (i + 1) * dy
        seg = f"{ns/1000:.0f}k" if ns >= 1000 else str(ns)
        vals = [lab, role_t[role], seg, f"{nsub:,}",
                f"{y[:,0].mean():.0f} / {y[:,1].mean():.0f}",
                f"{ag:.0f}" if np.isfinite(ag) else "--",
                "ECG+PPG" if ecg else "PPG", demo]
        vals = [v if v != "n/a" else "--" for v in vals]
        for x, v, a, ci in zip(xcol, vals, align, range(len(vals))):
            put(x, yr, v, a, bold=(ci == 0), it=(ci == 1))
    axa.plot([0, 1], [yhead - (n + 0.5) * dy] * 2, color="black", lw=1.3)   # bottomrule
    axa.text(-0.06, 1.16, "a", transform=axa.transAxes, fontsize=14, fontweight="bold")
    axa.set_title("Datasets: one training source, one in-distribution and five out-of-distribution "
                  "tests.\nECG+PPG sets support the roll-audit; PPG-only sets do not.",
                  fontsize=9.5, loc="left", pad=16)

    # ---- panel b: BP violins, greyscale (train/test/OOD by fill shade, not color)
    labels = [r[0] for r in recs]
    shade = {"train": "#3a3a3a", "id": "#7a7a7a", "ood": "#b8b8b8"}
    cols = [shade[r[1]] for r in recs]
    ys = [r[7] for r in recs]
    vit_iqr = np.percentile(ys[0][:, 0], [25, 75]), np.percentile(ys[0][:, 1], [25, 75])

    gsb = gs[1].subgridspec(1, 2, wspace=0.22)
    axl = fig.add_subplot(gsb[0]); axr = fig.add_subplot(gsb[1])
    for sub, ti, ylab, iqr in [(0, 0, "SBP (mmHg)", vit_iqr[0]), (1, 1, "DBP (mmHg)", vit_iqr[1])]:
        axx = axl if sub == 0 else axr
        data = [y[:, ti] for y in ys]
        vp = axx.violinplot(data, showmeans=False, showextrema=False, widths=0.85)
        for j, b in enumerate(vp["bodies"]):
            b.set_facecolor(cols[j]); b.set_alpha(0.85); b.set_edgecolor("black"); b.set_linewidth(0.8)
        for j, dta in enumerate(data):
            q1, med, q3 = np.percentile(dta, [25, 50, 75])
            axx.plot([j + 1, j + 1], [q1, q3], color="black", lw=1.1, zorder=3)
            axx.plot(j + 1, med, "o", color="white", mec="black", ms=4, zorder=4)
        axx.axhspan(iqr[0], iqr[1], color="black", alpha=0.06, zorder=0)
        axx.set_xticks(range(1, len(recs) + 1), labels, fontsize=7.5, rotation=25, ha="right")
        axx.set_ylabel(ylab); axx.margins(x=0.02)
        if sub == 0:
            axx.text(-0.15, 1.03, "b", transform=axx.transAxes, fontsize=14, fontweight="bold")
            axx.set_title("Blood-pressure shift  (shaded band = VitalDB IQR)", fontsize=9.5, loc="left")

    fig.savefig(FIG / "fig_datasets.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_datasets.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_datasets.png / .pdf")


if __name__ == "__main__":
    main()
