"""pat_estimators.py -- every published way to measure pulse arrival time, put through the audit.

Why this matters
----------------
Every arrival-time result in this project rests on ONE estimator: R-peak to intersecting-tangent
foot. That estimator is measurable on only ~44% of segments and recovers injected PTT at r = 0.56
even on clean synthetic waveforms. So "models do not use arrival time" could equally be "our one
estimator is too lossy to register the dependence". The literature uses at least six definitions
and they disagree systematically -- some are foot-based and noise-sensitive, others peak- or
derivative-based and more robust but more contaminated by pulse shape.

The estimators:

  foot_tangent   R -> intersecting-tangent foot. The classical choice, and what the project has
                 used throughout. Most physiologically defensible, least robust.
  foot_min       R -> the diastolic minimum itself. Simpler, noisier, no tangent fit.
  peak           R -> PPG systolic peak. Robust, but includes the upstroke, so it mixes arrival
                 time with ejection dynamics.
  max_slope      R -> steepest point of the upstroke (max of the first derivative). The standard
                 robust alternative; less shape-contaminated than the peak.
  second_deriv   R -> the 'a' wave of the second derivative, i.e. maximum acceleration. Often the
                 most repeatable fiducial in noisy recordings.
  xcorr          full-segment cross-correlation lag between ECG and PPG. Uses every sample rather
                 than one fiducial per beat, so it degrades gracefully where beat detection fails.
  xcorr_deriv    cross-correlation between the ECG and the PPG first derivative, which sharpens
                 the PPG feature being aligned.

For each estimator we report: how often it is measurable, how well it tracks a KNOWN injected PTT
on synthetic waveforms, and how faithfully it responds to the roll intervention. An estimator that
cannot track ground truth cannot support a claim about models either way.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

import mechlib
from mechlib import ECG, PPG, _z, _shift_channel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
LO, HI = 0.05, 0.5          # physiological bounds on arrival time (s)


def _rpeaks(ecg, fs):
    ez = _z(ecg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    return r


def _beat_delay(ecg, ppg, fs, locator):
    """Median R-to-fiducial delay, where `locator` returns an index within the search window."""
    ez, pz = _z(ecg), _z(ppg)
    r = _rpeaks(ecg, fs)
    if len(r) < 3:
        return np.nan
    out = []
    for rp in r:
        lo, hi = rp + int(LO * fs), min(rp + int(HI * fs), len(pz) - 1)
        if hi - lo < 5:
            continue
        idx = locator(pz, lo, hi, fs)
        if idx is None:
            continue
        d = (idx - rp) / fs
        if LO < d < HI:
            out.append(d)
    return float(np.median(out)) if len(out) >= 2 else np.nan


def est_foot_tangent(ecg, ppg, fs):
    return mechlib.segment_ptt(ecg, ppg, fs)          # the project's existing estimator


def est_foot_min(ecg, ppg, fs):
    return _beat_delay(ecg, ppg, fs,
                       lambda pz, lo, hi, f: lo + int(np.argmin(pz[lo:hi])))


def est_peak(ecg, ppg, fs):
    return _beat_delay(ecg, ppg, fs,
                       lambda pz, lo, hi, f: lo + int(np.argmax(pz[lo:hi])))


def est_max_slope(ecg, ppg, fs):
    def loc(pz, lo, hi, f):
        dv = np.gradient(pz[lo:hi])
        return lo + int(np.argmax(dv))
    return _beat_delay(ecg, ppg, fs, loc)


def est_second_deriv(ecg, ppg, fs):
    def loc(pz, lo, hi, f):
        seg = pz[lo:hi]
        if len(seg) < 7:
            return None
        sm = savgol_filter(seg, max(int(0.05 * f) | 1, 5), 3)
        return lo + int(np.argmax(np.gradient(np.gradient(sm))))
    return _beat_delay(ecg, ppg, fs, loc)


def _xcorr_lag(a, b, fs, max_s=HI):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    n = int(max_s * fs)
    c = np.correlate(b, a, "full")
    mid = len(c) // 2
    seg = c[mid:mid + n]
    if len(seg) < 3:
        return np.nan
    k = int(np.argmax(seg))
    if 0 < k < len(seg) - 1:                      # parabolic sub-sample refinement
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        k = k + (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
    d = k / fs
    return d if LO < d < HI else np.nan


def est_xcorr(ecg, ppg, fs):
    return _xcorr_lag(_z(ecg), _z(ppg), fs)


def est_xcorr_deriv(ecg, ppg, fs):
    return _xcorr_lag(_z(ecg), np.gradient(_z(ppg)), fs)


ESTIMATORS = {
    "foot_tangent": est_foot_tangent,
    "foot_min": est_foot_min,
    "peak": est_peak,
    "max_slope": est_max_slope,
    "second_deriv": est_second_deriv,
    "xcorr": est_xcorr,
    "xcorr_deriv": est_xcorr_deriv,
}


def batch(fn, X, fs):
    return np.array([fn(X[i, :, ECG], X[i, :, PPG], fs) for i in range(len(X))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    # ---- 1. ground truth: synthetic waveforms with a KNOWN injected PTT -----
    import synth_waveform_audit as SW
    Xs, _, ptt_true, _ = SW.make_dataset(args.n, 1.0, np.random.default_rng(5))
    print(f"[synthetic] {len(Xs)} segments with known injected PTT\n", flush=True)

    # ---- 2. real data, and how each estimator responds to the roll ----------
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= 60][:40]
    sel = np.concatenate([np.where(g == s)[0][:30] for s in subs])
    Xr = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    shifts = [-6, -2, 0]
    Xshift = {}
    for sh in shifts:
        Xd = Xr.copy()
        Xd[:, :, PPG] = _shift_channel(Xr[:, :, PPG], sh)
        Xshift[sh] = Xd

    print(f"{'estimator':14s} {'valid%':>7s} {'r vs truth':>11s} {'bias ms':>8s} "
          f"{'tracks roll':>12s} {'real valid%':>12s}")
    print("-" * 70)
    rows = []
    for name, fn in ESTIMATORS.items():
        est = batch(fn, Xs, SW.FS) * 1000.0
        ok = np.isfinite(est)
        r = float(np.corrcoef(est[ok], ptt_true[ok])[0, 1]) if ok.sum() > 30 else np.nan
        bias = float(np.median(est[ok] - ptt_true[ok])) if ok.sum() > 30 else np.nan

        # does the estimator follow an IMPOSED shift on real data? -48 ms should read -48 ms
        base = batch(fn, Xshift[0], FS) * 1000.0
        m48 = batch(fn, Xshift[-6], FS) * 1000.0
        both = np.isfinite(base) & np.isfinite(m48)
        tracked = float(np.median((m48 - base)[both])) if both.sum() > 20 else np.nan
        nominal = -48.0
        frac_track = tracked / nominal if np.isfinite(tracked) else np.nan
        realvalid = float(np.isfinite(base).mean())

        rows.append({"estimator": name, "synth_valid": float(ok.mean()), "r_truth": r,
                     "bias_ms": bias, "tracked_ms": tracked, "frac_tracked": frac_track,
                     "real_valid": realvalid})
        print(f"{name:14s} {100*ok.mean():6.0f}% {r:+11.3f} {bias:+8.1f} "
              f"{frac_track:11.0%} {100*realvalid:11.0f}%", flush=True)

    (DATA / "pat_estimators.json").write_text(json.dumps(rows, indent=2, default=float))
    best = max((x for x in rows if np.isfinite(x["r_truth"])), key=lambda x: x["r_truth"])
    print(f"\n[best vs ground truth] {best['estimator']}  r = {best['r_truth']:+.3f}, "
          f"measurable on {100*best['real_valid']:.0f}% of real segments")
    print("'tracks roll' is the fraction of an imposed -48 ms shift that the estimator recovers.")
    print("An estimator well below 100% is attenuating the intervention, which would show up as")
    print("model unfaithfulness even if the model responded perfectly.")
    print(f"\n[done] data/pat_estimators.json")


if __name__ == "__main__":
    main()
