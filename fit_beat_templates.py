"""fit_beat_templates.py -- fit the simulator's beat shapes to VitalDB's median heartbeat.

The generator in synth_waveform_audit.py builds each beat from Gaussian waves. Rather than choose
their sizes and timings by hand, fit them to real data so the simulation matches VitalDB by
construction:

  ECG  : P, Q, R, S, T Gaussians (+ baseline), fitted to the median R-aligned beat
  PPG  : systolic, dicrotic and diastolic Gaussians (+ baseline), fitted to the median
         foot-aligned beat; times are relative to the foot, so the injected delay IS the foot delay

Beats come from held-out VitalDB test segments in one heart-rate band (RR 0.75-0.95 s), each
min-max scaled within its window, so this fits shape, not amplitude. Also reports the per-beat
spread of T-wave height, which the generator uses for beat-to-beat variety. Prints the fitted
constants; they are copied into synth_waveform_audit.py.
"""
import numpy as np
from scipy.optimize import least_squares

import mechlib

FS = 125
E_WIN = (-0.35, 0.55)
P_WIN = (-0.15, 0.75)


def g(t, c, w, a):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


def collect(X, rr_band=(0.75, 0.95)):
    E, P = [], []
    for x in X:
        e, p = mechlib._z(x[:, 0]), mechlib._z(x[:, 1])
        r = mechlib.r_peaks(e, FS)
        if len(r) < 4 or not (rr_band[0] <= np.median(np.diff(r)) / FS <= rr_band[1]):
            continue
        for q in r:
            a, b = q + int(E_WIN[0] * FS), q + int(E_WIN[1] * FS)
            if a >= 0 and b <= len(e):
                w = e[a:b]; E.append((w - w.min()) / (np.ptp(w) + 1e-9))
        for f in mechlib.ppg_feet(p, FS):
            f = int(round(f)); a, b = f + int(P_WIN[0] * FS), f + int(P_WIN[1] * FS)
            if a >= 0 and b <= len(p):
                w = p[a:b]; P.append((w - w.min()) / (np.ptp(w) + 1e-9))
    return np.array(E), np.array(P)


def fit(t, y, names, x0, lo, hi):
    """y ~ base + sum of Gaussians; parameter vector [base, (c, w, a) per wave]."""
    def model(v):
        out = np.full_like(t, v[0])
        for k in range(len(names)):
            out += g(t, *v[1 + 3 * k:4 + 3 * k])
        return out
    r = least_squares(lambda v: model(v) - y, x0, bounds=(lo, hi))
    rmse = float(np.sqrt(np.mean(r.fun ** 2)))
    return r.x, model(r.x), rmse


def main():
    d = mechlib.load_mini("data/vitaldb_mini.npz")
    E, P = collect(d["Xte"][:, :, :2].astype(float))
    te = np.arange(E.shape[1]) / FS + E_WIN[0]; tp = np.arange(P.shape[1]) / FS + P_WIN[0]
    ye, yp = np.median(E, 0), np.median(P, 0)
    print(f"{len(E)} ECG beats, {len(P)} PPG beats")

    #            c      w      a        per wave; base first
    ecg_names = ["P", "Q", "R", "S", "T"]
    x0 = [0.1, -0.16, .03, .1, -0.035, .012, -.1, 0, .01, 1, 0.025, .012, -.1, 0.28, .06, .2]
    # Q must precede R and S follow it by >= 12 ms, so each wave keeps its physiological meaning
    lo = [-1, -0.25, .01, 0, -0.06, .004, -1, -0.01, .004, 0, 0.012, .004, -1, 0.15, .02, 0]
    hi = [1, -0.08, .08, .5, -0.015, .03, 0, 0.01, .03, 2, 0.06, .03, 0, 0.42, .15, 1]
    ve, fe, re = fit(te, ye, ecg_names, x0, lo, hi)

    ppg_names = ["systolic", "dicrotic", "diastolic"]
    x0 = [0.0, 0.14, .07, 1, 0.33, .08, .3, 0.5, .12, .1]
    lo = [-.5, 0.05, .02, 0, 0.22, .03, 0, 0.35, .03, 0]
    hi = [.5, 0.25, .15, 2, 0.45, .2, 1, 0.75, .3, 1]
    vp, fp, rp = fit(tp, yp, ppg_names, x0, lo, hi)

    print(f"\nECG fit RMSE {re:.3f} (of 1)   PPG fit RMSE {rp:.3f}")
    R_amp = ve[1 + 3 * 2 + 2]
    print("\nECG waves (time from R, width, height relative to R):")
    for k, n in enumerate(ecg_names):
        c, w, a = ve[1 + 3 * k:4 + 3 * k]
        print(f"  {n}: c={c:+.3f} s  w={w:.3f} s  a={a / R_amp:+.3f}")
    P_amp = vp[1 + 2]
    print("\nPPG waves (time from foot, width, height relative to systolic):")
    for k, n in enumerate(ppg_names):
        c, w, a = vp[1 + 3 * k:4 + 3 * k]
        print(f"  {n}: c={c:+.3f} s  w={w:.3f} s  a={a / P_amp:+.3f}")

    # per-beat T-wave height relative to R (beat-to-beat variety for the generator)
    base = np.median(E[:, :int(0.1 * FS)], 1)
    i_t = (te > 0.15) & (te < 0.45)
    t_rel = (E[:, i_t].max(1) - base) / (1 - base)
    print(f"\nT/R per beat: median {np.median(t_rel):.2f}, IQR {np.percentile(t_rel, 25):.2f}-"
          f"{np.percentile(t_rel, 75):.2f}")


if __name__ == "__main__":
    main()
