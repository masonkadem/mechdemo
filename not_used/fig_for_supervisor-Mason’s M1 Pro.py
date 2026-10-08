"""fig_for_supervisor.py -- the single slide: what does calibrating a new patient cost, and does
a fine-tuned model beat a physiological equation given the same cuff readings?

All arms are fitted on the SAME k segments of the SAME held-out patients:

  uncalibrated      the population model, no per-patient fit at all
  cuff average      predict the mean of the k readings (no waveform used)
  equation          the classical Bramwell-Hill form, DBP = a + b/PTT^2, shown with two
                    different transit times (see below)
  fine-tuned head   per-patient linear head on a frozen transformer

Both equation arms use the SAME textbook formula; only the transit time differs.

  classical PTT        ECG R-peak -> PPG upstroke. Needs two sensors, and the PLETH channel in
                       this dataset lags the arterial line by a stable ~500 ms
                       (see fig_alignment.py), so this arm is handicapped by instrumentation.
  reflected-wave PTT   systolic peak -> dicrotic notch, i.e. the round trip of the wave
                       reflected from the periphery. Still a transit time, still Bramwell-Hill,
                       but measured INSIDE one PPG pulse, so no channel lag can touch it.

The reflected-wave version wins, which is the point: given this instrumentation, the best
available PTT is the one that never crosses channels.

    python fig_for_supervisor.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GAP = 50
KS = [2, 3, 5, 10, 20, 30, 50, 100]          # start at 2: k=1 only fits an offset for every arm
# published PulseDB CalFree DBP MAE, uncalibrated (Wang et al., benchmark paper)
LIT = (7.85, 8.05)


def col_invsq(v, rows):
    """1/PTT^2, the Bramwell-Hill regressor; gaps filled with the patient's own median."""
    z = 1.0 / np.clip(v, 0.02, None) ** 2
    o = z.copy()
    for i in rows.values():
        ok = np.isfinite(z[i])
        o[i] = np.where(ok, z[i], np.median(z[i][ok]) if ok.any() else 0.0)
    return o


def fit_predict(A, yc, B, alpha=10.0):
    """Ridge with alpha scaled by features-per-sample, identical for every arm."""
    if len(A) < 2:
        return np.full(len(B), yc.mean())
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-6
    if not keep.any():
        return np.full(len(B), yc.mean())
    A, B, mu, sd = A[:, keep], B[:, keep], mu[keep], sd[keep]
    return Ridge(alpha=alpha * max(1.0, A.shape[1] / len(A))
                 ).fit((A - mu) / sd, yc).predict((B - mu) / sd)


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, y = np.array(d["gte"]), np.array(d["yte"])[:, 1]
    rows = {p: np.where(g == p)[0] for p in np.unique(g)}

    def col(v):
        o = v.astype(float).copy()
        for i in rows.values():
            ok = np.isfinite(v[i])
            o[i] = np.where(ok, v[i], np.median(v[i][ok]) if ok.any() else 0.0)
        return o

    def curve(M):
        if M.ndim == 1:
            M = M[:, None]
        return {k: float(np.median(
            [np.abs(fit_predict(M[i[:k]], y[i[:k]], M[i[k + GAP:]]) - y[i[k + GAP:]]).mean()
             for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}

    # The equation arm is a genuine PTT in Bramwell-Hill form, DBP = a + b/PTT^2. The transit
    # time used is the DICROTIC NOTCH DELAY -- the round trip of the wave reflected from the
    # periphery, measured inside a single PPG pulse. That keeps the classical form while avoiding
    # the ECG/PPG channel lag; it also beats both cross-channel PTTs (see below).
    notch = np.load(DATA / "_calib_ppg_notch.npy")
    eq = curve(col_invsq(notch, rows))
    # classical two-sensor PTT, shown for contrast
    ptt = curve(col_invsq(np.load(DATA / "_calib_ptt_maxslope.npy"), rows))
    # invasive reference: PAT from the arterial line, the best a clinical setup could measure
    inv = curve(col_invsq(np.load(DATA / "_calib_pat_abp.npy"), rows))
    floor = {k: float(np.median([np.abs(y[i[:k]].mean() - y[i[k + GAP:]]).mean()
                                 for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    g50 = {r["k"]: r for r in json.load(open(DATA / "calibration_cost.json")) if r["gap"] == 50}
    model = {k: g50[k]["head"] for k in KS if k in g50}
    uncal = float(np.median([g50[k]["uncalibrated"] for k in g50]))

    for nm, c in (("uncalibrated", {k: uncal for k in KS}), ("cuff average", floor),
                  ("equation, notch PTT", eq), ("equation, classical PTT", ptt),
                  ("equation, invasive PAT", inv), ("fine-tuned head", model)):
        print(f"{nm:22s} " + "  ".join(f"k={k}:{c[k]:5.2f}" for k in (2, 5, 20, 100)))

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 10.5,
    })
    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)

    # reference: no per-patient calibration at all
    ax.axhline(uncal, color="#9A9A9A", ls=(0, (4, 3)), lw=1.3, zorder=1)
    ax.text(0.985, uncal - 0.10, "no calibration", transform=ax.get_yaxis_transform(),
            ha="right", va="top", fontsize=10, color="#7A7A7A")

    series = [
        (floor, "#3C3C3C", "-",  "s", 1.9, 5.5, "cuff average"),
        (ptt,   "#BFBFBF", "--", "o", 1.7, 5.5, "PAT equation (ECG→PPG)"),
        (inv,   "#7A9BC4", "--", "^", 1.7, 5.5, "PAT equation (ECG→artery, invasive)"),
        (eq,    "#009E73", "-",  "o", 2.7, 6.5, "PTT equation (PPG only)"),
        (model, "#D55E00", "-",  "o", 3.1, 7.0, "fine-tuned model"),
    ]
    for c, colr, ls, mk_, lw, ms, lab in series:
        xs = [k for k in KS if k in c]
        ax.plot(xs, [c[k] for k in xs], marker=mk_, ls=ls, color=colr, lw=lw, ms=ms,
                mec="white", mew=0.9, label=lab, zorder=3,
                clip_on=False, solid_capstyle="round")

    # a device asks for a handful of cuff readings, not twenty: shade where it actually operates
    ax.axvspan(1.85, 5, color="#F0EAE0", alpha=0.85, zorder=0)
    ax.text(3.05, 3.10, "realistic for a device", ha="center", va="bottom",
            fontsize=9.5, color="#8A7A5A")

    ax.set_xscale("log")
    ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off()
    ax.set_xlim(1.85, 118)
    ax.set_ylim(3.0, 8.45)
    ax.set_xlabel("cuff readings used for calibration ($k$)", labelpad=8)
    ax.set_ylabel("diastolic BP error (mmHg)", labelpad=8)
    ax.tick_params(length=4, pad=5)
    # legend below the axes: the model line sweeps the whole lower-right, so no in-axes
    # position stays clear of it
    leg = ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.12),
                    ncol=1, handlelength=2.2, labelspacing=0.55)
    for t in leg.get_texts():
        t.set_color("#1A1A1A")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#5A5A5A")
    out = ROOT / "figures" / "fig_for_supervisor.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
