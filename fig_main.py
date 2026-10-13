"""fig_main.py -- datasets, accuracy in and out of distribution, and calibration burden:
  a  SBP/DBP violins per dataset (distribution shift)
  b  dataset table (age = median[5-95])
  c  DBP MAE in distribution (VitalDB) and out of distribution (MIMIC-BP), with the
     predict-the-training-mean row: on MIMIC no model beats it, so that column ranks how flat a
     model's predictions are under distribution shift, not mechanism
  d  calibration burden

The mechanism panels of the earlier version (audit sensitivity vs OOD robustness; probe bars)
were built on the old PAT detector and are superseded by fig_audit.py and fig_targets.py.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics_audit as pa

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
B = "C:/Users/mason/OneDrive - McMaster University/2026/BP"
NAVY, RED, GREEN, ORANGE = "#2f4b7c", "#c1543b", "#3b8c5a", "#d98c3f"
# muted categorical fills for the violins (color, publication)
DCOL = {"train": "#4c72b0", "id": "#55a868", "ood": "#c44e52"}


def _agemr(v):
    v = v[np.isfinite(v)]; v = v[v >= 10]
    if not len(v):
        return None
    return int(np.median(v)), int(np.percentile(v, 5)), int(np.percentile(v, 95))


def datasets():
    recs = []
    d = np.load(ROOT / "data" / "vitaldb_full_calfree.npz")
    ytr = np.concatenate([d["ytr"], d["yva"]]); gtr = np.concatenate([d["gtr"], d["gva"]])
    recs.append(("VitalDB", "train", len(ytr), len(np.unique(gtr)), True, "age,sex,BMI",
                 _agemr(d["age_tr"]), ytr))
    recs.append(("VitalDB test", "id", len(d["yte"]), len(np.unique(d["gte"])), True, "age,sex,BMI",
                 _agemr(d["age_te"]), d["yte"]))
    m = pa.load_mimic_bp(B, channels=("ppg",), max_patients=1524)
    _, k = pa.window_segments(m["X"][:1], 1250)
    recs.append(("MIMIC-BP", "ood", len(m["y"]) * k, len(np.unique(m["g"])), True, "--", None, m["y"]))
    for nm, path in [("BCG", "data/bcg_dataset"),
                     ("Sensors", "C:/Users/mason/Downloads/sensors_dataset/sensors_dataset"),
                     ("UCI2", "data/uci2_dataset/uci2_dataset"),
                     ("PPG-BP", "C:/Users/mason/Downloads/ppgbp_dataset/ppgbp_dataset")]:
        e = pa.load_bpbenchmark(path, nm)
        demo = ("age,sex,BMI" if nm == "PPG-BP" else "age,sex") if e["demo"] else "--"
        recs.append((nm, "ood", len(e["y"]), len(np.unique(e["g"])), False, demo,
                     _agemr(e["demo"]["age"]) if e["demo"] else None, e["y"]))
    return recs


def table(ax, hdr, rows, xcol, align, bolds, title, plabel, fs=7.6):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(rows); yhead = 0.92; dy = (yhead - 0.05) / (n + 1)

    def put(x, y, s, ha, bold=False, it=False):
        ax.text(x, y, s, ha=ha, va="center", fontsize=fs,
                fontweight=("bold" if bold else "normal"), fontstyle=("italic" if it else "normal"))
    for x, h, a in zip(xcol, hdr, align):
        put(x, yhead, h, a, bold=True)
    xr = max(xcol) + 0.08
    ax.plot([0, xr], [yhead + 0.5 * dy] * 2, color="black", lw=1.2)
    ax.plot([0, xr], [yhead - 0.5 * dy] * 2, color="black", lw=0.7)
    for i, r in enumerate(rows):
        yr = yhead - (i + 1) * dy
        for x, v, a, ci in zip(xcol, r, align, range(len(r))):
            put(x, yr, v, a, bold=(ci in bolds and ci == 0), it=(ci == 1))
    ax.plot([0, xr], [yhead - (n + 0.5) * dy] * 2, color="black", lw=1.2)
    ax.text(-0.05, 1.06, plabel, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=9, loc="left", x=0.04)


def main():
    recs = datasets()
    e = json.loads((ROOT / "data" / "ood_benchmark_ecgppg_full.json").read_text())["models"]
    disp0 = {"lenet1d": "LeNet", "inception1d": "Incep", "xresnet1d50": "XR50",
             "xresnet1d101": "XR101", "transformer": "Trans"}
    fig = plt.figure(figsize=(11.5, 8.2))
    # rows: [a violins + b table] / [c DBP table + d calibration]
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.2],
                          width_ratios=[1.1, 1.1], hspace=0.45, wspace=0.3)

    # ---- a: violins (top-left) ----
    gsa = gs[0, 0].subgridspec(1, 2, wspace=0.35)
    labels = [r[0] for r in recs]; cols = [DCOL[r[1]] for r in recs]; ys = [r[7] for r in recs]
    for sub, ti, ylab in [(0, 0, "SBP"), (1, 1, "DBP")]:
        ax = fig.add_subplot(gsa[sub])
        data = [y[:, ti] for y in ys]
        vp = ax.violinplot(data, showextrema=False, widths=0.8)
        for j, b in enumerate(vp["bodies"]):
            b.set_facecolor(cols[j]); b.set_alpha(0.75); b.set_edgecolor("black"); b.set_linewidth(0.8)
        for j, dta in enumerate(data):
            ax.plot(j + 1, np.median(dta), "o", color="white", mec="black", ms=3, zorder=4)
        ax.set_xticks(range(1, len(recs) + 1), labels, fontsize=6.3, rotation=35, ha="right")
        ax.set_ylabel(f"{ylab} (mmHg)", fontsize=8.5); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if sub == 0:
            ax.text(-0.32, 1.05, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

    # ---- b: dataset table (top-right) ----
    def seg(ns):
        return f"{ns/1000:.0f}k" if ns >= 1000 else str(ns)
    role_t = {"train": "train", "id": "ID", "ood": "OOD"}
    drows = []
    for lab, role, ns, nsub, ecg, demo, ag, y in recs:
        agestr = f"{ag[0]} [{ag[1]}-{ag[2]}]" if ag else "--"
        drows.append([lab, role_t[role], seg(ns), f"{nsub:,}",
                      f"{y[:,0].mean():.0f}/{y[:,1].mean():.0f}", agestr,
                      "ECG+PPG" if ecg else "PPG"])
    table(fig.add_subplot(gs[0, 1]),
          ["Dataset", "Role", "Seg", "Subj", "SBP/DBP", "Age [5-95]", "Chan"],
          drows, [0.0, 0.22, 0.35, 0.46, 0.59, 0.73, 0.93],
          ["left", "left", "right", "right", "center", "left", "left"], {0}, "Datasets", "b", fs=7.0)

    # ---- d: DBP MAE + mechanism table (middle-left) ----
    gfam = json.loads((ROOT / "data" / "gbm_families.json").read_text())
    names = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
    slopes = {n: e[n]["audit"]["dbp"]["dBP_dPTT"] for n in names}
    rank = {n: i + 1 for i, n in enumerate(sorted(names, key=lambda n: -abs(slopes[n])))}
    dispn = {"lenet1d": "LeNet1d", "inception1d": "Inception1d", "xresnet1d50": "XResNet50",
             "xresnet1d101": "XResNet101", "transformer": "Transformer"}
    def pfmt(p):
        return f"{p/1e6:.1f}M" if p >= 1e6 else f"{p/1e3:.0f}k"
    erows = []
    for n in names:
        o = e[n]["ood"]; s = slopes[n]
        # slope comes from CORR_SLOPE below, keyed on the architecture name, not from `s`:
        # `s` is the pre-correction value produced by the NaN-imputing audit and is ~40x too large
        erows.append([dispn[n], pfmt(e[n].get("params", 0)), f"{o['id']['mae_dbp']:.1f}",
                      f"{o['mimic_bp']['mae_dbp']:.1f}", n, str(rank[n])])
    # honest interpretable model: full 83-feature LightGBM (ID 8.1), scored on MIMIC with the
    # SAME features (data/_gbm_full_ood.json). Demographics absent on MIMIC so not used there.
    full = json.loads((ROOT / "data" / "feature_study_full.json").read_text())
    fullood = json.loads((ROOT / "data" / "_gbm_full_ood.json").read_text())
    # the LightGBM rows are listed explicitly in erows2 below, from the sparsity sweep, so the
    # older cached numbers in feature_study_full.json are not appended here as well
    # Audit-slope column dropped: it belongs to panels e/f, and every CI spans zero so it
    # cannot order the rows. What this table shows instead is size against accuracy on the two
    # PulseDB protocols.
    #
    # NOTE for anyone reading the OOD column: MIMIC's DBP mean sits 4.5 mmHg below training, so
    # a constant predictor scores 10.33 there -- better than every model in this table. PAT-only
    # reaches 11.09 by having the flattest predictions (sd 4.45 against 5.88) and the worst
    # within-subject correlation with true BP (0.032 against 0.15-0.17). The column ranks
    # flatness under distribution shift, not mechanism. Kept out of the figure at the author's
    # request; it belongs in the caption.
    erows2 = [[r[0], r[1], r[2], r[3]] for r in erows]
    erows2 += [
        ["LightGBM, all features", "~1.8k par", "8.47", "15.69"],
        ["LightGBM single tree, 4 feat", "~190 par", "8.92", "12.24"],
        ["LightGBM single tree, PAT only", "~190 par", "9.34", "10.96"],
    ]
    # the reference every row must beat: a constant at the VitalDB training mean
    tr_mean = float(recs[0][7][:, 1].mean())
    id_const = float(np.abs(recs[1][7][:, 1] - tr_mean).mean())
    ood_const = float(np.abs(recs[2][7][:, 1] - tr_mean).mean())
    erows2.append(["Predict the training mean", "--", f"{id_const:.2f}", f"{ood_const:.2f}"])
    table(fig.add_subplot(gs[1, 0]),
          ["Model", "size", "ID", "OOD"],
          erows2, [0.0, 0.60, 0.78, 0.95],
          ["left", "right", "right", "right"], {0},
          "DBP MAE (mmHg): ID VitalDB, OOD MIMIC-BP", "c", fs=6.9)

    # ---- d: calibration burden (bottom-right) ----
    # A calibration anchor is one reference cuff reading taken from the wearer. With k anchors the
    # device fits a single per-subject offset and is scored on held-out segments; k = 0 is the
    # calibration-free case. This is a different question from panel c -- "how accurate" versus
    # "how much must the user do before it works" -- and the two order the models differently.
    axg = fig.add_subplot(gs[1, 1])
    cam = json.loads((ROOT / "data" / "calib_all_models.json").read_text())
    curves = {k: {int(a): b for a, b in v["curve"].items()}
              for k, v in cam.items() if isinstance(v, dict) and "curve" in v}
    KS = [0, 1, 2, 3, 5, 10, 20]
    showg = [("gbm deep (83) + demo", GREEN, "-", "GBM + demographics"),
             ("gbm default (83)", NAVY, "-", "GBM, waveform"),
             ("xresnet1d50", RED, "--", "XResNet50"),
             ("transformer", "#9aa0a6", "--", "Transformer")]
    tgt = curves.get("xresnet1d50", {}).get(20)
    for key, col, ls, lab in showg:
        if key not in curves:
            continue
        axg.plot(KS, [curves[key].get(k, np.nan) for k in KS], ls, color=col, lw=1.8,
                 marker="o", ms=4, label=lab)
    if tgt:
        axg.axhline(tgt, color=RED, lw=0.8, ls=":", alpha=0.8)
        axg.text(20, tgt + 0.06, f"best deep net", fontsize=7,
                 color=RED, ha="right", va="bottom")
    axg.set_xlabel("cuff readings per person  (k = 0: none)", fontsize=8.5)
    axg.set_ylabel("DBP MAE (mmHg)", fontsize=8.5)
    axg.tick_params(labelsize=7.5)
    axg.legend(fontsize=7.5, frameon=False, loc="upper right")
    axg.spines[["top", "right"]].set_visible(False)
    axg.set_box_aspect(0.85)
    axg.text(-0.24, 1.04, "d", transform=axg.transAxes, fontsize=13, fontweight="bold")
    axg.set_title("Calibration burden", fontsize=9, loc="left")

    fig.savefig(FIG / "fig_main.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIG / "fig_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_main.png / .pdf")


if __name__ == "__main__":
    main()
