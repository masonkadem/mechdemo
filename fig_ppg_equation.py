"""fig_ppg_equation.py -- the clean slide: within-PPG timing is the only viable equation input,
so how far does it get against no calibration model and against a fine-tuned network?

Cross-channel timing (ECG->PPG, ABP->PPG) is unusable in this dataset because the PLETH channel
carries an undocumented, per-case device latency (see fig_alignment.py). Everything measured
INSIDE the PPG pulse is immune to that. This asks what the best single-sensor equation achieves.

    python fig_ppg_equation.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS, GAP = 125, 50
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100]


def fit_predict(A, yc, B, alpha=10.0):
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
        return o[:, None]

    def curve(M):
        return {k: float(np.median(
            [np.abs(fit_predict(M[i[:k]], y[i[:k]], M[i[k + GAP:]]) - y[i[k + GAP:]]).mean()
             for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}

    ppg = curve(col(np.load(DATA / "_calib_ppg_syswidth.npy")))     # best within-PPG feature
    ptt = curve(col(1.0 / np.clip(np.load(DATA / "_calib_ptt_maxslope.npy"), .02, None) ** 2))
    floor = {k: float(np.median([np.abs(y[i[:k]].mean() - y[i[k + GAP:]]).mean()
                                 for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    g50 = {r["k"]: r for r in json.load(open(DATA / "calibration_cost.json")) if r["gap"] == 50}
    model = {k: g50[k]["head"] for k in KS if k in g50}

    for nm, c in (("PPG systolic width", ppg), ("classical PTT", ptt),
                  ("cuff average", floor), ("model", model)):
        print(f"{nm:22s} k=5 {c[5]:5.2f}  k=20 {c[20]:5.2f}  k=100 {c[100]:5.2f}")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True,
                           gridspec_kw={"width_ratios": [1, 1.35]})

    # (a) what the feature IS, drawn on a real pulse
    i0 = np.where(g == 1309)[0][0]
    p = np.array(d["Xte"][i0])[:, 1]
    sm = savgol_filter(p, 9, 3)
    rng = sm.max() - sm.min()
    pk = find_peaks(sm, distance=int(0.4 * FS), prominence=rng * 0.3)[0]
    a_, b_ = pk[1], pk[2]
    lo = max(0, a_ - int(0.30 * FS))
    ft = lo + int(np.argmin(sm[lo:a_]))
    half = (sm[a_] + sm[ft]) / 2
    m = (sm[ft:b_] > half).astype(int)
    idx = np.where(m)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    first = idx[:brk[0] + 1] if len(brk) else idx
    s0, s1 = ft + first[0], ft + first[-1]

    t = np.arange(len(p)) / FS
    w = slice(ft - int(0.12 * FS), b_ + int(0.06 * FS))
    ax[0].plot(t[w], sm[w], color="#0072B2", lw=2)
    ax[0].axhline(half, color="#888", ls=":", lw=1.2)
    ax[0].fill_between(t[s0:s1 + 1], half, sm[s0:s1 + 1], color="#009E73", alpha=0.30, lw=0)
    ax[0].annotate("", (s1 / FS, half), (s0 / FS, half),
                   arrowprops=dict(arrowstyle="<->", color="#009E73", lw=2.2))
    ax[0].annotate(f"systolic width\n{(s1-s0)/FS*1000:.0f} ms",
                   ((s0 + s1) / 2 / FS, half), (0, 10), textcoords="offset points",
                   ha="center", fontsize=10, color="#1E7A5A", fontweight="bold")
    ax[0].plot([ft / FS, a_ / FS], [sm[ft], sm[a_]], "o", color="#333", ms=7)
    ax[0].annotate("foot", (ft / FS, sm[ft]), (-6, -14), textcoords="offset points", fontsize=9)
    ax[0].annotate("peak", (a_ / FS, sm[a_]), (-10, 6), textcoords="offset points", fontsize=9)
    ax[0].annotate("half height", (t[w].max(), half), (-4, 5), textcoords="offset points",
                   fontsize=8.5, color="#666", ha="right")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("PPG (normalised)")
    ax[0].set_title("a  the feature: how long ejection lasts\n"
                    "     measured inside ONE PPG pulse — no ECG, no arterial line",
                    loc="left", fontsize=10.5)

    # (b) the calibration curves
    ax[1].plot(KS, [floor[k] for k in KS], "s-", color="#333333", lw=1.8, ms=5,
               label="no model — average of the cuff readings")
    ax[1].plot(KS, [ptt[k] for k in KS], "o--", color="#999999", lw=1.6, ms=5,
               label="equation, classical PTT (ECG+PPG, misaligned)")
    ax[1].plot(KS, [ppg[k] for k in KS], "o-", color="#009E73", lw=3.0, ms=6,
               label="equation, PPG systolic width (1 sensor)")
    mk = [k for k in KS if k in model]
    ax[1].plot(mk, [model[k] for k in mk], "o-", color="#D55E00", lw=3.0, ms=6,
               label="transformer + fine-tuned linear head")

    ax[1].axvspan(0.85, 2.6, color="#BBB", alpha=0.16, lw=0)
    ax[1].annotate("too few\nreadings", (1.5, 7.7), ha="center", fontsize=8, color="#666")
    ax[1].annotate("", (20, ppg[20]), (20, model[20]),
                   arrowprops=dict(arrowstyle="<->", color="#333", lw=1.6))
    ax[1].annotate(f"  {ppg[20]-model[20]:.2f} mmHg\n  still to gain",
                   (20.6, (ppg[20] + model[20]) / 2), fontsize=9, va="center")
    ax[1].annotate("", (100, floor[100]), (100, ppg[100]),
                   arrowprops=dict(arrowstyle="<->", color="#1E7A5A", lw=1.6))
    ax[1].annotate(f"{floor[100]-ppg[100]:.2f}\nmmHg", (104, (floor[100] + ppg[100]) / 2),
                   fontsize=8.5, va="center", color="#1E7A5A")
    ax[1].set_xscale("log"); ax[1].set_xticks(KS); ax[1].set_xticklabels(KS, fontsize=9)
    ax[1].minorticks_off()
    ax[1].set_xlabel("cuff readings used to calibrate this patient (k)")
    ax[1].set_ylabel("diastolic BP error (mmHg)")
    ax[1].set_title("b  what a single-sensor equation buys", loc="left", fontsize=11)
    ax[1].legend(frameon=False, fontsize=9, loc="upper right")

    for a2 in ax:
        a2.spines[["top", "right"]].set_visible(False)
    out = ROOT / "figures" / "fig_ppg_equation.png"
    fig.savefig(out, dpi=200)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
