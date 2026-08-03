"""pat_groundtruth.py -- rank PAT estimators against REAL arterial pressure, not synthetic data.

Why the synthetic ranking was not enough
----------------------------------------
pat_estimators.py scored each detector against a PTT that I injected into waveforms I also wrote.
That is circular in a specific way: my generator gives every pulse the same shape, so a
peak-based estimator looks unbiased when in real pulses the upstroke duration varies with
stiffness and would contaminate it. The synthetic test can rank noise-robustness but cannot rank
physiological validity.

PulseDB carries a third channel: the invasive arterial pressure waveform (channel 2, in mmHg).
Its own foot is the true arrival of the pressure pulse at the radial artery, measured
independently of any PPG detector. That gives a real reference:

    PAT_true = t(ABP foot) - t(ECG R peak)

Each PPG estimator is then scored on how well it recovers PAT_true, WITHIN subject as well as
pooled, since between-subject agreement can be driven by body size alone.

Two caveats kept in view rather than buried:
  * ABP is measured at the radial artery and PPG at the finger, so the two differ by a real
    peripheral transit of roughly 10-30 ms. A constant offset is expected and is not error.
  * PulseDB channels may carry independent device latencies (this was measured directly in
    VitalDB, where ART and PPG were not mutually synchronised). A fixed offset therefore cannot
    be interpreted; only the CORRELATION and the beat-to-beat tracking are trustworthy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy import stats

import mechlib
import pat_estimators as PE
from mechlib import _z

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
ABP = 2
LO, HI = 0.05, 0.5


def abp_pat(ecg, abp, fs):
    """Ground-truth arrival: ECG R peak to the ABP upstroke foot, via intersecting tangent.

    The ABP foot is a pressure measurement, not an optical one, so it does not share the failure
    modes of PPG foot detection (perfusion, motion, contact).
    """
    ez = _z(ecg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    if len(r) < 3:
        return np.nan
    az = _z(abp)
    sm = savgol_filter(az, max(int(0.04 * fs) | 1, 5), 3)
    dv = np.gradient(sm)
    out = []
    for rp in r:
        lo, hi = rp + int(LO * fs), min(rp + int(HI * fs), len(az) - 2)
        if hi - lo < 5:
            continue
        # steepest upstroke, then walk back to the local minimum before it = the foot
        k = lo + int(np.argmax(dv[lo:hi]))
        j = k
        while j > lo and sm[j - 1] <= sm[j]:
            j -= 1
        d = (j - rp) / fs
        if LO < d < HI:
            out.append(d)
    return float(np.median(out)) if len(out) >= 2 else np.nan


def within_r(x, y, g, min_seg=25):
    rs = []
    for s in np.unique(g):
        m = g == s
        if m.sum() < min_seg:
            continue
        a, b = x[m], y[m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_seg // 2 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            continue
        rs.append(stats.spearmanr(a[ok], b[ok]).statistic)
    return (float(np.nanmedian(rs)) if rs else np.nan), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=60)
    ap.add_argument("--subjects", type=int, default=60)
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= args.per_subject][:args.subjects]
    sel = np.concatenate([np.where(g == s)[0][:args.per_subject] for s in subs])
    Xraw = d["Xte"][sel]                       # keep raw: ABP must stay in mmHg
    Xn = mechlib.normalize(Xraw[:, :, [mechlib.ECG, mechlib.PPG]])
    gg = g[sel]
    print(f"[gt] {len(Xn)} segments, {len(subs)} subjects", flush=True)

    print("[gt] computing ground-truth PAT from the arterial waveform ...", flush=True)
    truth = np.array([abp_pat(Xraw[i, :, mechlib.ECG], Xraw[i, :, ABP], FS)
                      for i in range(len(Xraw))]) * 1000.0
    ok_t = np.isfinite(truth)
    print(f"[gt] ABP-based PAT measurable on {100*ok_t.mean():.0f}% of segments, "
          f"median {np.nanmedian(truth[ok_t]):.1f} ms", flush=True)

    rows = []
    print(f"\n{'estimator':14s} {'valid%':>7s} {'r pooled':>9s} {'r within':>9s} "
          f"{'bias ms':>8s} {'n subj':>7s}")
    print("-" * 60)
    for name, fn in PE.ESTIMATORS.items():
        est = PE.batch(fn, Xn, FS) * 1000.0
        m = ok_t & np.isfinite(est)
        if m.sum() < 200:
            print(f"{name:14s} too few valid"); continue
        rp = float(stats.spearmanr(est[m], truth[m]).statistic)
        rw, nsub = within_r(est, truth, gg)
        bias = float(np.median(est[m] - truth[m]))
        rows.append({"estimator": name, "valid": float(np.isfinite(est).mean()),
                     "r_pooled": rp, "r_within": rw, "bias_ms": bias, "n_subj": nsub})
        print(f"{name:14s} {100*np.isfinite(est).mean():6.0f}% {rp:+9.3f} {rw:+9.3f} "
              f"{bias:+8.1f} {nsub:7d}", flush=True)

    (DATA / "pat_groundtruth.json").write_text(json.dumps(rows, indent=2, default=float))
    good = [r for r in rows if np.isfinite(r["r_within"])]
    if good:
        best = max(good, key=lambda r: r["r_within"])
        print(f"\n[best within-subject] {best['estimator']}  r = {best['r_within']:+.3f}")
    print("\nWithin-subject is the column that matters: pooled agreement can be driven by body")
    print("size, which affects both the true transit and every estimator without either")
    print("tracking pressure. Bias is not interpretable here -- ABP is radial and PPG is")
    print("finger, so a real peripheral transit separates them, and PulseDB channels may")
    print("additionally carry independent device latencies.")
    print(f"\n[done] data/pat_groundtruth.json")


if __name__ == "__main__":
    main()
