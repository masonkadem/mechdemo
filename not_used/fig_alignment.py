"""fig_alignment.py -- one figure: PulseDB's PPG channel is misaligned, and what that does (and
does not) invalidate.

    python fig_alignment.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125


def peaks_of(v):
    sm = savgol_filter(v, 9, 3)
    return find_peaks(sm, distance=int(0.4 * FS),
                      prominence=(sm.max() - sm.min()) * 0.3)[0]


def foot_of(v, pk):
    sm = savgol_filter(v, 9, 3)
    lo = max(0, pk - int(0.30 * FS))
    return lo + int(np.argmin(sm[lo:pk]))


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g = np.array(d["gte"])
    subs = np.unique(g)

    pk2pk, ft2ft, rise_a, rise_p, nearest = [], [], [], [], []
    for s in subs[:40]:
        for sg in np.array(d["Xte"][np.where(g == s)[0][:8]]):
            abp, ppg = sg[:, 2], sg[:, 1]
            Pa, Pp = peaks_of(abp), peaks_of(ppg)
            if len(Pa) < 3 or len(Pp) < 3:
                continue
            Fa = [foot_of(abp, p) for p in Pa if p > int(0.30 * FS)]
            Fp = [foot_of(ppg, p) for p in Pp if p > int(0.30 * FS)]
            rise_a += [(p - f) / FS * 1000 for p, f in zip(Pa[-len(Fa):], Fa)]
            rise_p += [(p - f) / FS * 1000 for p, f in zip(Pp[-len(Fp):], Fp)]
            for t in Pa:
                c = Pp[(Pp >= t - 0.10 * FS) & (Pp < t + 0.35 * FS)]
                if len(c):
                    pk2pk.append((c[0] - t) / FS * 1000)
            for t in Fa:
                c = [x for x in Fp if t - 0.10 * FS <= x < t + 0.35 * FS]
                if c:
                    ft2ft.append((c[0] - t) / FS * 1000)
            for t in Pa[1:-1]:
                dl = (Pp - t) / FS * 1000
                nearest.append(dl[np.argmin(np.abs(dl))])

    pk2pk, ft2ft, nearest = map(np.array, (pk2pk, ft2ft, nearest))
    ra, rp = float(np.median(rise_a)), float(np.median(rise_p))
    print(f"peak-peak {np.median(pk2pk):+.0f} ms (SD {pk2pk.std():.0f}) | "
          f"foot-foot {np.median(ft2ft):+.0f} ms (SD {ft2ft.std():.0f})")
    print(f"rise: ABP {ra:.0f} ms, PPG {rp:.0f} ms -> predicted gap {rp-ra:+.0f}, "
          f"observed {np.median(pk2pk)-np.median(ft2ft):+.0f}")

    # a clean example beat for panel a
    ex = np.array(d["Xte"][np.where(g == 1309)[0][0]])
    Pa, Pp = peaks_of(ex[:, 2]), peaks_of(ex[:, 1])

    fig = plt.figure(figsize=(13.5, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

    # ---- a: the raw evidence, one segment ----------------------------------
    ax = fig.add_subplot(gs[0])
    t = np.arange(len(ex)) / FS
    w = slice(int(0.3 * FS), int(2.6 * FS))
    abp_n = (ex[:, 2] - ex[:, 2].min()) / np.ptp(ex[:, 2])
    ax.plot(t[w], abp_n[w], color="#009E73", lw=1.6, label="ABP (arterial line)")
    ax.plot(t[w], ex[w, 1], color="#0072B2", lw=1.6, label="PPG (finger)")
    for p in Pa[(Pa > w.start) & (Pa < w.stop)]:
        ax.axvline(p / FS, color="#009E73", ls=":", lw=1.1)
    for p in Pp[(Pp > w.start) & (Pp < w.stop)]:
        ax.axvline(p / FS, color="#0072B2", ls=":", lw=1.1)
    pa = Pa[(Pa > w.start) & (Pa < w.stop)][1]
    pp_ = Pp[Pp > w.start][np.argmin(np.abs(Pp[Pp > w.start] - pa))]
    ax.annotate("", (pp_ / FS, 1.06), (pa / FS, 1.06),
                arrowprops=dict(arrowstyle="<->", color="#C0392B", lw=2))
    ax.annotate("PPG channel lags by ~500 ms,\nso the nearest PPG peak\nbelongs to another beat",
                ((pa + pp_) / 2 / FS, 1.10), ha="center", fontsize=9, color="#C0392B",
                fontweight="bold")
    ax.set_ylim(-0.05, 1.42)
    ax.set_xlabel("time (s)"); ax.set_ylabel("normalised amplitude")
    ax.set_title("a  the pulses do not line up", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    # ---- b: which timings survive, measured on RAW VitalDB -------------------
    # Between-case spread of each timing measure, from the vitaldb API (6 cases). Anything that
    # crosses into the PLETH channel is unusable; anything measured inside one channel is fine.
    meas = [("PLETH rise\n(within PPG)", 26, False), ("ART rise\n(within ART)", 28, False),
            ("ECG → ART", 36, False), ("PLETH notch\n(within PPG)", 39, False),
            ("ART → PLETH", 147, True), ("ECG → PLETH", 243, True)]
    ax = fig.add_subplot(gs[1])
    cols = ["#C0392B" if bad else "#009E73" for _, _, bad in meas]
    ax.barh([m[0] for m in meas], [m[1] for m in meas], color=cols, height=0.62)
    ax.invert_yaxis()
    ax.axvline(50, color="#333", ls="--", lw=1.2)
    for j, (_, v, _) in enumerate(meas):
        ax.annotate(f"{v} ms", (v, j), (5, 0), textcoords="offset points",
                    va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 300)
    ax.set_xlabel("spread across cases (ms) — lower is more reliable")
    ax.set_title("b  what survives: anything inside one channel", loc="left", fontsize=11)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.annotate("crosses into PLETH →\nunusable", (250, 4.6), fontsize=8.5, ha="center",
                color="#C0392B", fontweight="bold")
    ax.annotate("raw VitalDB API, 6 cases", (0.98, 0.02), xycoords="axes fraction",
                fontsize=8, ha="right", color="#555")

    # ---- c: what it does and does not break --------------------------------
    ax = fig.add_subplot(gs[2])
    rows = json.load(open(DATA / "calibration_cost.json")) \
        if (DATA / "calibration_cost.json").exists() else []
    g50 = {r["k"]: r for r in rows if r["gap"] == 50}
    ks = [k for k in (1, 5, 20, 100) if k in g50]
    if ks:
        ax.plot(ks, [g50[k]["head"] for k in ks], "o-", color="#D55E00", lw=2.4, ms=6,
                label="transformer + fine-tuned head")
        ax.plot(ks, [g50[k]["offset"] for k in ks], "o-", color="#0072B2", lw=1.6, ms=5,
                label="average of cuff readings")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("cuff readings used to calibrate (k)")
        ax.set_ylabel("diastolic BP error (mmHg)")
    ax.set_title("c  but calibration is unaffected", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.annotate("a FIXED offset is absorbed by the per-patient\n"
                "intercept, so every calibration result stands",
                (0.5, -0.30), xycoords="axes fraction", fontsize=9, ha="center",
                color="#1E7A5A", fontweight="bold")

    for a_ in fig.axes:
        a_.spines[["top", "right"]].set_visible(False)
    out = ROOT / "figures" / "fig_alignment.png"
    fig.savefig(out, dpi=200)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
