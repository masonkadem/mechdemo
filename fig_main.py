"""fig_main.py -- combined Nature-style main figure merging composites 1 and 2, compact:
  a  small color SBP/DBP violins (distribution shift)
  b  short clean waveform: ECG + one cycle of PPG/VPG/APG (feature extraction)
  c  small square mechanism-sensitivity vs OOD scatter
  d  clean dataset table (age = median[5-95])
  e  ID vs OOD DBP table (deep + LightGBM)
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics_audit as pa
import fig_waveform_clean as wc

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


def waveform(ax_list):
    fs = 500
    t = np.linspace(-0.15, 0.75, int(0.9 * fs))            # short: ~1 cycle
    ecg = wc.synth_ecg(t); foot = 0.22
    ppg = wc.synth_ppg(t, foot); ppg = ppg / ppg.max()
    sm = savgol_filter(ppg, 61, 3)
    vpg = savgol_filter(np.gradient(sm), 41, 3) * fs / 100
    apg = savgol_filter(np.gradient(np.gradient(sm)), 51, 3) * (fs / 100) ** 2
    sigs = [(ecg, NAVY, "ECG"), (ppg, GREEN, "PPG"), (vpg, ORANGE, "VPG"), (apg, RED, "APG")]
    for ax, (s, c, lab) in zip(ax_list, sigs):
        ax.plot(t, s, color=c, lw=1.7)
        ax.set_yticks([]); ax.set_xlim(-0.15, 0.75)
        ax.set_ylabel(lab, fontsize=8.5, rotation=0, ha="right", va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#eee", lw=0.7, zorder=0)
    ax_list[0].plot(0, 1, "o", color=RED, ms=5)
    pk = np.argmax(ppg); fi = np.argmin(np.abs(t - foot))
    ax_list[1].plot(t[fi], ppg[fi], "o", color="black", ms=5)
    ax_list[1].plot(t[pk], ppg[pk], "o", color=RED, ms=5)
    ax_list[1].annotate("", xy=(foot, -0.12), xytext=(0, -0.12),
                        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax_list[1].text(foot / 2, -0.28, "PAT", fontsize=7.5, ha="center", color="#333")
    ax_list[1].set_ylim(-0.35, 1.15)
    win = (t > foot - 0.01) & (t < foot + 0.4); idxw = np.where(win)[0]; aw = apg[idxw]
    pk_i = idxw[find_peaks(aw)[0]]; tr_i = idxw[find_peaks(-aw)[0]]
    ext = [pk_i[0]]
    for arr in [tr_i, pk_i, tr_i, pk_i]:
        cand = arr[arr > ext[-1]]
        if len(cand):
            ext.append(cand[0])
    for kk, lbl in zip(ext[:5], "abcde"):
        ax_list[3].annotate(lbl, (t[kk], apg[kk]), xytext=(0, 5 if apg[kk] > 0 else -10),
                            textcoords="offset points", fontsize=8, fontweight="bold", ha="center")
    ax_list[3].margins(y=0.3)
    ax_list[3].set_xlabel("time from R-peak (s)", fontsize=8.5)


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
    fig = plt.figure(figsize=(11.5, 15.0))
    # rows: [a violins + b table] / [c waveform + d DBP table] / [e OOD scatter + f probe bars]
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 1.45, 1.0, 1.05],
                          width_ratios=[1.05, 1.15], hspace=0.55, wspace=0.3)

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

    # ---- c: short small waveform (middle-left) ----
    # equal-height traces with real breathing room: the squeezed version made the APG
    # landmarks unreadable, which is the one thing this panel exists to show
    gsc = gs[1, 0].subgridspec(4, 1, height_ratios=[1, 1.25, 1, 1], hspace=0.30)
    axc = [fig.add_subplot(gsc[i]) for i in range(4)]
    waveform(axc)
    # only the bottom trace carries the time axis; the extra spacing that un-squeezed the panel
    # also un-shared the x axes, which put a duplicate tick row under every trace
    for a in axc[:-1]:
        a.set_xticklabels([])
        a.set_xlabel("")
        a.tick_params(axis="x", length=0)
    axc[0].text(-0.09, 1.2, "c", transform=axc[0].transAxes, fontsize=13, fontweight="bold")
    axc[0].set_title("Feature extraction (ECG, PPG, 1st/2nd deriv.)", fontsize=8.5, loc="left")

    # ---- d: DBP MAE + mechanism table (middle-right) ----
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
    table(fig.add_subplot(gs[1, 1]),
          ["Model", "size", "ID", "OOD"],
          erows2, [0.0, 0.60, 0.78, 0.95],
          ["left", "right", "right", "right"], {0},
          "DBP MAE (mmHg): ID VitalDB, OOD MIMIC-BP", "d", fs=6.9)

    disp = disp0
    # ---- e: OOD penalty vs mechanism scatter (bottom-left, square) ----
    axe = fig.add_subplot(gs[2, 0])
    # Slopes come from fig_panel_e.py, which recomputes the audit with the `second_deriv`
    # arrival-time estimator. That estimator is used because it ties for the best agreement with
    # invasive arterial arrival time (r = +0.21 within subject, on 100% of segments), a criterion
    # fixed before any of these correlations were computed. The previously plotted numbers came
    # from the NaN-imputing audit via the tangent-foot estimator, which scores r = 0.03 against
    # the same ground truth and gives a positive (wrong-signed) correlation here.
    pe = json.loads((ROOT / "data" / "panel_e.json").read_text())
    absl = [abs(v) * 1000.0 for v in pe["slopes"][pe["primary"]]]      # mmHg per second
    gap = pe["ood_penalty"]
    r = pe["r_by_estimator"][pe["primary"]]
    b0, a0 = np.polyfit(absl, gap, 1)
    xs = np.linspace(min(absl) * 0.9, max(absl) * 1.06, 30)
    axe.plot(xs, a0 + b0 * xs, "k--", lw=1.1, label=f"r = {r:.2f}")
    for xi, yi, n in zip(absl, gap, pe["models"]):
        axe.scatter(xi, yi, s=80, color=NAVY, edgecolor="black", lw=1, zorder=3)
        axe.annotate(disp[n], (xi, yi), fontsize=6.8, xytext=(4, 4), textcoords="offset points")
    axe.set_xlabel("roll-audit sensitivity  |dDBP/dΔ|  (mmHg/s)", fontsize=8.5)
    axe.set_ylabel("OOD penalty (mmHg)", fontsize=8.5); axe.tick_params(labelsize=7.5)
    axe.legend(frameon=False, fontsize=8.5, loc="upper right")
    axe.spines[["top", "right"]].set_visible(False); axe.set_box_aspect(0.85)
    axe.text(-0.24, 1.04, "e", transform=axe.transAxes, fontsize=13, fontweight="bold")
    axe.set_title("Mechanism vs OOD robustness", fontsize=9, loc="left")

    # ---- f: probe summary bars (bottom-right): PAT vs period decodable per model ----
    axf = fig.add_subplot(gs[2, 1])
    ps = json.loads((ROOT / "data" / "_probe_summary.json").read_text())
    xn = np.arange(len(names)); w = 0.38
    pat = [ps[n]["pat"] for n in names]; per = [ps[n]["period"] for n in names]
    axf.bar(xn - w / 2, per, w, color="#c1543b", label="cardiac period (shortcut)")
    axf.bar(xn + w / 2, pat, w, color="#2f4b7c", label="PAT (arrival time)")
    # roll slope above the PAT bar, using the same corrected values as panel e
    pe_sl = {n: abs(v) * 1000.0 for n, v in zip(pe["models"], pe["slopes"][pe["primary"]])}
    for i, n in enumerate(names):
        axf.text(i + w / 2, pat[i] + 0.02, f"{pe_sl[n]:.0f}", ha="center", fontsize=6.8,
                 color="#2f4b7c", fontweight="bold")
    axf.set_xticks(xn, [disp[n] for n in names], fontsize=7.5)
    axf.set_ylabel("max linear-probe $R^2$", fontsize=8.5); axf.set_ylim(0, 0.98)
    axf.tick_params(labelsize=7.5); axf.legend(frameon=False, fontsize=7.5, loc="upper right")
    axf.spines[["top", "right"]].set_visible(False); axf.set_box_aspect(0.85)
    axf.text(-0.2, 1.04, "f", transform=axf.transAxes, fontsize=13, fontweight="bold")
    axf.set_title("Decodability is flat across models (number = |roll slope|, mmHg/s)", fontsize=8.7, loc="left")


    # ---- g: calibration burden (bottom, spanning) ----
    # A calibration anchor is one reference cuff reading taken from the wearer. With k anchors the
    # device fits a single per-subject offset and is scored on held-out segments; k = 0 is the
    # calibration-free case. This is a different question from panel d -- "how accurate" versus
    # "how much must the user do before it works" -- and the two order the models differently.
    axg = fig.add_subplot(gs[3, 0])
    cam = json.loads((ROOT / "data" / "calib_all_models.json").read_text())
    curves = {k: {int(a): b for a, b in v["curve"].items()}
              for k, v in cam.items() if isinstance(v, dict) and "curve" in v}
    KS = [0, 1, 2, 3, 5, 10, 20]
    showg = [("gbm deep (83) + demo", GREEN, "-", "LightGBM + demographics"),
             ("gbm default (83)", NAVY, "-", "LightGBM, waveform only"),
             ("xresnet1d50", RED, "--", "XResNet50 (887k par)"),
             ("transformer", "#9aa0a6", "--", "Transformer (107k par)")]
    tgt = curves.get("xresnet1d50", {}).get(20)
    for key, col, ls, lab in showg:
        if key not in curves:
            continue
        axg.plot(KS, [curves[key].get(k, np.nan) for k in KS], ls, color=col, lw=1.8,
                 marker="o", ms=4, label=lab)
    if tgt:
        axg.axhline(tgt, color=RED, lw=0.8, ls=":", alpha=0.8)
        axg.text(20, tgt + 0.06, f"best deep net ({tgt:.2f})", fontsize=7,
                 color=RED, ha="right", va="bottom")
    axg.set_xlabel("k  =  cuff readings collected from this person", fontsize=8.5)
    axg.set_ylabel("DBP MAE (mmHg)", fontsize=8.5)
    axg.tick_params(labelsize=7.5)
    axg.legend(fontsize=7.5, frameon=False, loc="upper right")
    axg.text(0.99, 0.95, "k = 0 is calibration-free", transform=axg.transAxes,
             fontsize=7.5, color="#9aa0a6", ha="right", va="top")
    axg.spines[["top", "right"]].set_visible(False)
    axg.set_box_aspect(0.85)          # same square as e and f, so the panel letters line up
    axg.text(-0.24, 1.04, "g", transform=axg.transAxes, fontsize=13, fontweight="bold")
    axg.set_title("Calibration burden", fontsize=9, loc="left")

    fig.savefig(FIG / "fig_main.png", dpi=185, bbox_inches="tight")
    fig.savefig(FIG / "fig_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_main.png / .pdf")


if __name__ == "__main__":
    main()
