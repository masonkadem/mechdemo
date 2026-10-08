"""fig_waveform_clean.py -- clean, synthetic, textbook-quality ECG + PPG + VPG + APG figure.

Real VitalDB beats are noisy and the ECG barely looks like an ECG. For a schematic that shows
where features come from, a clean SYNTHETIC beat reads far better. We build a canonical ECG
(P-QRS-T) and a canonical PPG pulse (systolic + reflected/diastolic wave), derive VPG/APG
analytically, and mark the fiducials with NO clutter -- meaning goes in the caption.
"""
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREEN, ORANGE, GREY = "#2f4b7c", "#c1543b", "#3b8c5a", "#d98c3f", "#888888"


def gauss(t, mu, s, a):
    return a * np.exp(-0.5 * ((t - mu) / s) ** 2)


def synth_ecg(t):
    """Canonical P-QRS-T complex, R at t=0."""
    e = (gauss(t, -0.16, 0.025, 0.12)                      # P
         + gauss(t, -0.02, 0.008, -0.15)                   # Q
         + gauss(t, 0.0, 0.010, 1.0)                       # R
         + gauss(t, 0.02, 0.008, -0.22)                    # S
         + gauss(t, 0.18, 0.045, 0.28))                    # T
    return e


def synth_ppg(t, foot=0.22):
    """Canonical finger PPG: systolic (percussion) wave + delayed reflected/diastolic wave.
    Rises from `foot` (arrival time after R). Slightly steeper upstroke so the APG shows a
    clear a(+) peak at the onset (physiological a-b-c-d-e convention)."""
    tt = t - foot
    sys = gauss(tt, 0.10, 0.050, 1.0)                       # systolic peak (steeper foot)
    refl = gauss(tt, 0.30, 0.070, 0.42)                    # reflected/diastolic wave
    p = sys + refl
    p[t < foot] = p[t < foot] * np.exp(-((foot - t[t < foot]) / 0.025))
    return p


def main():
    fs = 500
    t = np.linspace(-0.25, 0.9, int(1.15 * fs))
    ecg = synth_ecg(t)
    foot = 0.22
    ppg = synth_ppg(t, foot)
    ppg = ppg / ppg.max()
    # heavier smoothing so the analytic derivatives are clean (no glitch after 'a')
    sm = savgol_filter(ppg, 61, 3)
    vpg = savgol_filter(np.gradient(sm), 41, 3) * fs / 100
    apg = savgol_filter(np.gradient(np.gradient(sm)), 51, 3) * (fs / 100) ** 2

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.3, 0.9, 0.9], "hspace": 0.1})
    for ax in axes:
        ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.axhline(0, color="#e5e5e5", lw=0.8, zorder=0)

    # ECG
    ax = axes[0]
    ax.plot(t, ecg, color=NAVY, lw=2)
    ax.plot(0, ecg[np.argmin(np.abs(t))], "o", color=RED, ms=7, zorder=5)
    ax.annotate("R", (0, 1.0), xytext=(4, -2), textcoords="offset points", fontsize=11,
                fontweight="bold", color=RED)
    ax.set_ylabel("ECG", fontsize=11, rotation=0, ha="right", va="center")

    # PPG with foot / systolic peak / notch / reflected wave
    ax = axes[1]
    ax.plot(t, ppg, color=GREEN, lw=2.2)
    fi = np.argmin(np.abs(t - foot))
    pk = np.argmax(ppg)
    ax.plot(t[fi], ppg[fi], "o", color="black", ms=7, zorder=5)
    ax.plot(t[pk], ppg[pk], "o", color=RED, ms=7, zorder=5)
    # dicrotic notch = local min after peak
    post = ppg[pk:]
    nrel = np.argmin(post[: int(0.25 * fs)])
    ni = pk + nrel
    ax.plot(t[ni], ppg[ni], "o", color=NAVY, ms=7, zorder=5)
    # PAT bracket
    ax.annotate("", xy=(foot, -0.13), xytext=(0, -0.13),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
    ax.text(foot / 2, -0.22, "PAT", color=ORANGE, fontsize=10, ha="center", fontweight="bold")
    ax.set_ylabel("PPG", fontsize=11, rotation=0, ha="right", va="center")
    ax.set_ylim(-0.3, 1.15)

    # VPG
    ax = axes[2]
    ax.plot(t, vpg, color=ORANGE, lw=2)
    ax.plot(t[np.argmax(vpg)], vpg.max(), "o", color=RED, ms=6, zorder=5)
    ax.plot(t[np.argmin(vpg)], vpg.min(), "o", color=NAVY, ms=6, zorder=5)
    ax.set_ylabel("VPG\n1st deriv", fontsize=10, rotation=0, ha="right", va="center")

    # APG with a-e
    ax = axes[3]
    ax.plot(t, apg, color=RED, lw=2)
    # a-e: physiological convention a(+) b(-) c(+) d(-) e(+). Start at the first positive APG
    # peak after the foot, then alternate.
    from scipy.signal import find_peaks
    win = (t > foot - 0.01) & (t < foot + 0.45)
    idxw = np.where(win)[0]
    aw = apg[idxw]
    pk_i = idxw[find_peaks(aw)[0]]                          # positive extrema (a, c, e)
    tr_i = idxw[find_peaks(-aw)[0]]                         # negative extrema (b, d)
    ext = []
    a = pk_i[0]; ext.append(a)                              # a = first positive peak
    for arr, after in [(tr_i, a), (pk_i, None), (tr_i, None), (pk_i, None)]:
        cand = arr[arr > ext[-1]]
        if len(cand):
            ext.append(cand[0])
    ext = ext[:5]
    for k, lbl in zip(ext, ["a", "b", "c", "d", "e"]):
        ax.plot(t[k], apg[k], "o", color="black", ms=5, zorder=5)
        ax.annotate(lbl, (t[k], apg[k]), xytext=(0, 8 if apg[k] > 0 else -14),
                    textcoords="offset points", fontsize=10, fontweight="bold", ha="center")
    ax.set_ylabel("APG\n2nd deriv", fontsize=10, rotation=0, ha="right", va="center")
    ax.set_xlabel("time relative to ECG R-peak (s)", fontsize=11)
    ax.margins(y=0.3)

    # faint vertical guides linking the derivative alignment: PPG systolic peak lines up with
    # the VPG zero-crossing (slope=0); the PPG max-upstroke lines up with the VPG peak.
    t_peak = t[pk]
    t_upstroke = t[np.argmax(vpg)]
    for gx in (t_upstroke, t_peak):
        for ax in axes:
            ax.axvline(gx, color="#cfcfcf", lw=0.9, ls="--", zorder=0)

    axes[0].set_xlim(-0.25, 0.9)
    fig.savefig(ROOT / "figures" / "fig_waveform_clean.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_waveform_clean.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_waveform_clean.png / .pdf")


if __name__ == "__main__":
    main()
