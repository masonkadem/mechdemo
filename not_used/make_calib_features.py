"""make_calib_features.py -- produce the data/_calib_*.npy timing arrays.

These seven arrays drive every equation arm in fig_calibration_strategy.py,
fig_equation_variants.py, fig_for_supervisor.py, calib_finetune_table.py and
fig_ppg_equation.py -- five consumers, and until now zero producers. They were committed as
data with the code that made them absent, so the fiducial behind each equation arm could not
be audited. This closes that.

Every definition is stated explicitly below, in seconds, one value per CalFree test segment.
Naming follows the arrival-time convention in mechlib.compute_ptt: what an ECG->PPG interval
measures is PAT (= PEP + transit), not PTT. The file names are kept for compatibility with
the existing call sites, but the docstrings say what is actually measured.

  _calib_ptt_maxslope   R peak -> steepest point of the following PPG upstroke.
                        An arrival time to the finger. NOT peak-to-peak, and not a pure
                        transit time -- it contains the pre-ejection period.
  _calib_pat_abp        R peak -> arterial-line foot (intersecting tangent).
                        Arrival time to the radial artery: PEP + proximal arterial transit.
  _calib_ptt_true       arterial-line foot -> PPG foot.
                        The only genuinely site-to-site arterial transit available here,
                        and the only arm free of the pre-ejection period.
  _calib_ppg_rise       PPG foot -> systolic peak (crest time). Single sensor.
  _calib_ppg_notch      systolic peak -> dicrotic notch. A reflected-wave round trip
                        measured INSIDE one pulse, so no inter-channel lag can reach it.
  _calib_ppg_width      pulse width at 50% of systolic amplitude.
  _calib_ppg_ai         augmentation index: notch amplitude / systolic amplitude (unitless).

    python make_calib_features.py                 # writes, comparing to any existing arrays
    python make_calib_features.py --write         # overwrite the committed arrays
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
NAMES = ["ptt_maxslope", "pat_abp", "ptt_true", "ppg_rise", "ppg_notch", "ppg_width", "ppg_ai"]


def _seg(args):
    """All seven fiducial intervals for one segment. NaN where undetectable."""
    import numpy as np
    from scipy.signal import find_peaks
    import mechlib as ml

    ecg, ppg, abp = args
    out = {k: np.nan for k in NAMES}
    ez, pz, az = ml._z(ecg), ml._z(ppg), ml._z(abp)
    r, _ = find_peaks(ez, distance=max(int(0.3 * FS), 1), prominence=0.5)
    if len(r) < 3:
        return out

    dp = np.gradient(pz)
    ms, pat_a, ptt_t, rise, notch, width, ai = [], [], [], [], [], [], []
    for rp in r:
        # --- R -> steepest PPG upstroke (the "maxslope" arrival time)
        a0, b0 = rp + int(0.05 * FS), min(rp + int(ml.PAT_WIN * FS), len(pz))
        if a0 < b0:
            up = a0 + int(np.argmax(dp[a0:b0]))
            v = (up - rp) / FS
            if 0.05 < v < ml.PAT_WIN:
                ms.append(v)

        # --- R -> arterial foot, and arterial foot -> PPG foot
        fa = ml._foot_after_r(az, rp, FS, hi_s=0.4)
        fp = ml._foot_after_r(pz, rp, FS, hi_s=0.6)
        if fa is not None:
            v = (fa - rp) / FS
            if 0.04 < v < 0.4:
                pat_a.append(v)
            if fp is not None and fp > fa:
                v2 = (fp - fa) / FS
                if 0.01 < v2 < 0.4:
                    ptt_t.append(v2)

        pass

    # --- single-sensor PPG shape cues.
    # These are deliberately NOT anchored on the R peak. Anchoring them there makes them
    # inherit the ECG->PPG foot search, which fails whenever the PLETH channel lag pushes the
    # foot outside its window -- that dropped yield to 29% against the committed arrays' 85%+.
    # A crest time or a notch delay is a property of the PPG pulse alone, so detect the pulses
    # directly and never involve the ECG.
    feet = ml._pulse_feet(pz, FS)
    for j in range(len(feet) - 1):
        fi, nxt = int(feet[j]), int(feet[j + 1])
        hp = min(fi + int(0.4 * FS), nxt)
        if fi + 2 >= hp:
            continue
        pk = fi + int(np.argmax(pz[fi:hp]))
        rise.append((pk - fi) / FS)

        amp = pz[pk] - pz[fi]
        if amp > 1e-6:                                    # width at 50% of systolic amplitude
            half = pz[fi] + 0.5 * amp
            lo = fi + int(np.argmax(pz[fi:pk + 1] >= half)) if pk > fi else fi
            tail = pz[pk:min(pk + int(0.6 * FS), len(pz))]
            below = np.where(tail <= half)[0]
            if len(below):
                width.append((pk + below[0] - lo) / FS)

        down = pz[pk:min(pk + int(0.4 * FS), len(pz))]     # dicrotic notch on the downslope
        ni, _ = find_peaks(-down, prominence=0.05)
        if len(ni):
            n_i = pk + int(ni[0])
            notch.append((n_i - pk) / FS)
            if amp > 1e-6:
                ai.append((pz[n_i] - pz[fi]) / amp)

    def med(v, k=2):
        return float(np.median(v)) if len(v) >= k else np.nan

    out["ptt_maxslope"] = med(ms); out["pat_abp"] = med(pat_a); out["ptt_true"] = med(ptt_t)
    out["ppg_rise"] = med(rise); out["ppg_notch"] = med(notch)
    out["ppg_width"] = med(width); out["ppg_ai"] = med(ai)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite data/_calib_*.npy")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    X = np.asarray(d["Xte"], np.float64)
    if args.limit:
        X = X[:args.limit]
    print(f"[calib] {len(X)} CalFree test segments, {args.workers} workers", flush=True)

    jobs = [(X[i, :, 0], X[i, :, 1], X[i, :, 2]) for i in range(len(X))]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_seg, jobs, chunksize=256))

    arrs = {k: np.array([r[k] for r in rows], float) for k in NAMES}
    print(f"\n{'feature':16} {'finite':>7} {'median':>9}   vs committed array")
    for k, v in arrs.items():
        old_p = DATA / f"_calib_{k}.npy"
        line = f"{k:16} {np.isfinite(v).mean():7.1%} {np.nanmedian(v):9.4f}"
        if old_p.exists():
            o = np.load(old_p)
            m = np.isfinite(v) & np.isfinite(o[:len(v)])
            r = np.corrcoef(v[m], o[:len(v)][m])[0, 1] if m.sum() > 10 else np.nan
            line += (f"   old median {np.nanmedian(o):7.4f}  r={r:+.3f}  "
                     f"finite {np.isfinite(o).mean():.0%}")
        print(line)

    if args.write:
        for k, v in arrs.items():
            np.save(DATA / f"_calib_{k}.npy", v)
        print(f"\n[calib] wrote {len(arrs)} arrays to {DATA}")
    else:
        np.savez(DATA / "_calib_regenerated.npz", **arrs)
        print(f"\n[calib] dry run -> {DATA / '_calib_regenerated.npz'} (pass --write to replace)")


if __name__ == "__main__":
    main()
