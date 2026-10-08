"""fig_waveform_anno.py -- annotated ECG + PPG + VPG + APG figure showing where every feature
family is computed from. Publication overview panel.
"""
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREEN, ORANGE = "#2f4b7c", "#c1543b", "#3b8c5a", "#d98c3f"


def pick_clean_beat(d, fs):
    """Find a segment with clean ECG R-peaks and a well-formed PPG beat; return one beat window."""
    X = mechlib.normalize(d["Xte"][:200, :, [ECG, PPG]])
    for i in range(len(X)):
        ez = (X[i, :, 0] - X[i, :, 0].mean()) / (X[i, :, 0].std() + 1e-8)
        pz = (X[i, :, 1] - X[i, :, 1].mean()) / (X[i, :, 1].std() + 1e-8)
        r, _ = find_peaks(ez, distance=int(0.3 * fs), prominence=1.0)
        feet, _ = find_peaks(-pz, distance=int(0.4 * fs), prominence=0.3)
        if len(r) >= 2 and len(feet) >= 2:
            # take one R and the following PPG beat
            rp = r[0]
            nf = feet[feet > rp]
            if len(nf) >= 2 and nf[1] - nf[0] > int(0.5 * fs):
                return ez, pz, rp, nf[0], nf[1], fs
    return None


def main():
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    got = pick_clean_beat(d, fs)
    ez, pz, rp, f0, f1, fs = got
    # window: a bit before R to just past next foot
    a = max(rp - int(0.1 * fs), 0); b = min(f1 + int(0.05 * fs), len(pz))
    t = (np.arange(a, b) - rp) / fs * 1000                 # ms, R at 0
    e = ez[a:b]; p = pz[a:b]
    sm = savgol_filter(pz, max(int(0.05 * fs) | 1, 5), 3)
    v = np.gradient(sm)[a:b]; ap = np.gradient(np.gradient(sm))[a:b]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.4, 1, 1], "hspace": 0.12})

    # ECG
    ax = axes[0]
    ax.plot(t, e, color=NAVY, lw=1.6)
    rr = (rp - rp) / fs * 1000
    ax.plot(0, ez[rp], "v", color=RED, ms=10)
    ax.annotate("R peak", (0, ez[rp]), xytext=(6, -2), textcoords="offset points", fontsize=9, color=RED)
    ax.set_ylabel("ECG", fontsize=10)
    ax.text(0.01, 0.85, "HRV: RR interval, SDNN, RMSSD, LF/HF | QRS amp/width",
            transform=ax.transAxes, fontsize=8, color=NAVY)

    # PPG with fiducials
    ax = axes[1]
    beat_i = slice(f0 - a, f1 - a)
    tb = t[beat_i]; pb = p[beat_i]
    ax.plot(t, p, color=GREEN, lw=1.8)
    foot_t = (f0 - rp) / fs * 1000
    pk_rel = int(np.argmax(pb)); pk_t = tb[pk_rel]
    ax.plot(foot_t, p[f0 - a], "o", color="black", ms=7); ax.annotate("foot", (foot_t, p[f0 - a]),
            xytext=(-4, -14), textcoords="offset points", fontsize=8.5)
    ax.plot(pk_t, pb[pk_rel], "^", color=RED, ms=9); ax.annotate("systolic peak", (pk_t, pb[pk_rel]),
            xytext=(4, 4), textcoords="offset points", fontsize=8.5)
    # PAT arrow R->foot
    ax.annotate("", xy=(foot_t, p.min()), xytext=(0, p.min()),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax.text((0 + foot_t) / 2, p.min() - 0.15, "PAT", color=ORANGE, fontsize=9, ha="center")
    # dicrotic notch
    down = pb[pk_rel:]
    ni, _ = find_peaks(-down, prominence=0.03)
    if len(ni):
        nt = tb[pk_rel + ni[0]]; nv = pb[pk_rel + ni[0]]
        ax.plot(nt, nv, "s", color=NAVY, ms=7); ax.annotate("dicrotic notch", (nt, nv),
                xytext=(4, -12), textcoords="offset points", fontsize=8.5)
    ax.set_ylabel("PPG", fontsize=10)
    ax.text(0.01, 0.05, "morphology: rise/crest time, sys/dia widths (10-90%), areas, "
            "augmentation & reflection index, notch depth/timing, skew/kurtosis",
            transform=ax.transAxes, fontsize=8, color=GREEN)

    # VPG
    ax = axes[2]
    ax.plot(t, v, color=ORANGE, lw=1.6)
    vb = v[beat_i]
    ax.plot(tb[np.argmax(vb)], vb.max(), "^", color=RED, ms=8)
    ax.annotate("max slope", (tb[np.argmax(vb)], vb.max()), xytext=(4, 2),
                textcoords="offset points", fontsize=8.5)
    ax.plot(tb[np.argmin(vb)], vb.min(), "v", color=NAVY, ms=8)
    ax.axhline(0, color="#ccc", lw=0.7)
    ax.set_ylabel("VPG\n(1st deriv)", fontsize=10)
    ax.text(0.01, 0.05, "velocity: max/min slope, up/down ratio, landmark timings",
            transform=ax.transAxes, fontsize=8, color=ORANGE)

    # APG with a-e
    ax = axes[3]
    ax.plot(t, ap, color=RED, lw=1.6)
    apb = ap[beat_i]
    pks, _ = find_peaks(apb); trs, _ = find_peaks(-apb)
    ext = np.sort(np.concatenate([pks, trs])); ext = ext[ext < int(0.5 * fs)]
    ax.margins(y=0.25)                                     # headroom so a-e labels aren't clipped
    for k, lbl in zip(ext[:5], ["a", "b", "c", "d", "e"]):
        ax.plot(tb[k], apb[k], "o", color="black", ms=5)
        ax.annotate(lbl, (tb[k], apb[k]), xytext=(0, 7 if apb[k] > 0 else -13),
                    textcoords="offset points", fontsize=9, fontweight="bold", ha="center")
    ax.axhline(0, color="#ccc", lw=0.7)
    ax.set_ylabel("APG\n(2nd deriv)", fontsize=10)
    ax.set_xlabel("time relative to ECG R-peak (ms)", fontsize=10)
    ax.text(0.01, 0.05, "stiffness: a-e amplitudes & ratios (b/a, c/a...), a-e TIMINGS, "
            "aging indices (Takazawa, Ushiroyama)", transform=ax.transAxes, fontsize=8, color=RED)

    fig.suptitle("How the ~80 physiological features are computed from ECG + PPG and its derivatives",
                 fontsize=12, y=0.995)
    fig.savefig(ROOT / "figures" / "fig_waveform_anno.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_waveform_anno.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_waveform_anno.png / .pdf")


if __name__ == "__main__":
    main()
