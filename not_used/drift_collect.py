"""drift_collect.py -- raw VitalDB segments WITH elapsed time, for the calibration-drift study.

PulseDB shuffles segment order, so the question every cuffless-BP validation framework now
requires an answer to -- how long does one calibration last -- cannot be asked of it at all.
ISO 81060-3 / IEEE 1708a ask for post-calibration drift and a recalibration interval; the
published PulseDB benchmarks report neither, because the data cannot support it.

Raw VitalDB can: the recordings are continuous, so a segment's offset from the start of the
case is known. This collects segments in TEMPORAL ORDER with that offset attached, which is
the one thing vitaldb_raw.load_case does not keep.

    python drift_collect.py --n-cases 300 --max-seg 240
"""
import argparse
import json
from pathlib import Path

import numpy as np

import vitaldb_raw as vr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_case_timed(caseid, seg_s=vr.SEG_S, fs_out=125, max_segments=240, min_span_min=45):
    """Segments spread over the whole case, each tagged with elapsed seconds from case start.

    Returns None unless the surviving segments span at least min_span_min minutes -- a case
    that only yields a few minutes of usable signal cannot speak to drift over hours.
    """
    import vitaldb
    v = vitaldb.VitalFile(int(caseid), vr.TRACKS)
    d = v.to_numpy(vr.TRACKS, 1.0 / vr.FS)
    if d is None or len(d) < seg_s * vr.FS * 2:
        return None
    ecg, ppg, art = d[:, 0], d[:, 1], d[:, 2]

    L = seg_s * vr.FS
    n = len(d) // L
    keep = np.arange(n)
    if len(keep) > max_segments:                    # spread across the case, keep the order
        keep = np.unique(np.linspace(0, n - 1, max_segments).astype(int))

    segs, bps, ts, peps, ptts = [], [], [], [], []
    k = int(vr.FS // fs_out)
    for i in keep:
        s = i * L
        e, p, a = ecg[s:s + L], ppg[s:s + L], art[s:s + L]
        if len(e) < L:
            continue
        if not (np.isfinite(e).all() and np.isfinite(p).all() and np.isfinite(a).all()):
            continue
        if a.min() < vr.ART_LO or a.max() > vr.ART_HI:
            continue
        sbp, dbp = vr.bp_from_art(a)
        if not np.isfinite(sbp):
            continue
        pep, ptt, mfrac, spread = vr.beat_intervals(e, p, a, with_quality=True)
        segs.append(np.stack([e[::k], p[::k]], 1))
        bps.append([sbp, dbp])
        ts.append(s / vr.FS)                        # elapsed seconds from case start
        # arterial intervals are optional here: a segment with unusable feet is still a
        # perfectly good BP label for the drift question, so do not discard it
        peps.append(float(np.median(pep)) if len(pep) >= 3 and mfrac >= vr.MIN_MATCH_FRAC
                    else np.nan)
        ptts.append(float(np.median(ptt)) if len(ptt) >= 3 and mfrac >= vr.MIN_MATCH_FRAC
                    else np.nan)

    if len(segs) < 20:
        return None
    ts = np.asarray(ts)
    if (ts.max() - ts.min()) / 60.0 < min_span_min:
        return None
    return {"X": np.asarray(segs, np.float32), "y": np.asarray(bps, np.float32),
            "t": ts, "pep": np.asarray(peps), "ptt": np.asarray(ptts), "caseid": int(caseid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--max-seg", type=int, default=240)
    ap.add_argument("--min-span-min", type=float, default=45)
    ap.add_argument("--features", action="store_true", default=True,
                    help="store extracted features, not waveforms (needed at scale)")
    ap.add_argument("--out", default="data/vitaldb_drift.npz")
    args = ap.parse_args()

    ids = json.load(open(DATA / "vitaldb_scope.json"))["trio_caseids"]
    if args.n_cases > len(ids):                      # scope file caps at 2000; go wider
        import vitaldb
        extra = [int(c) for c in vitaldb.find_cases(vr.TRACKS)]
        ids = list(dict.fromkeys(list(ids) + extra))
    print(f"[scope] {len(ids)} candidate cases, target {args.n_cases}", flush=True)

    Fs, ys, ts, gs, peps, ptts = [], [], [], [], [], []
    done = 0
    for cid in ids:
        if done >= args.n_cases:
            break
        try:
            r = load_case_timed(cid, max_segments=args.max_seg,
                                min_span_min=args.min_span_min)
        except Exception as e:
            print(f"  case {cid}: {str(e)[:60]}", flush=True)
            continue
        if r is None:
            continue
        # Extract features HERE rather than storing waveforms. 2,000 cases of raw signal is
        # ~4 GB to hold in memory for compression, which does not fit; the features are ~80 MB
        # and are all the analysis needs. Extraction costs ~2 s/case against ~7 s of download,
        # so it is nearly free in wall-clock terms.
        if args.features:
            import features_full as ff
            import mechlib
            F = ff.compute_full(mechlib.normalize(np.asarray(r["X"], np.float64)), 125,
                                ppg_ch=1, ecg_ch=0)
            pat = mechlib.compute_ptt(np.asarray(r["X"], np.float64), 125, ecg_pos=0, ppg_pos=1)
            F["_pat"] = pat
            Fs.append(F)
        else:
            Fs.append({"_raw": r["X"]})
        ys.append(r["y"]); ts.append(r["t"])
        gs.append(np.full(len(r["X"]), cid)); peps.append(r["pep"]); ptts.append(r["ptt"])
        done += 1
        span = (r["t"].max() - r["t"].min()) / 60.0
        print(f"  case {cid:5d}: {len(r['X']):4d} segs over {span:6.1f} min "
              f"(total {done}/{args.n_cases})", flush=True)

    keys = sorted(set.intersection(*[set(f) for f in Fs]))
    blob = {k: np.concatenate([np.asarray(f[k], np.float32) for f in Fs]) for k in keys}
    blob.update(y=np.concatenate(ys), t=np.concatenate(ts), g=np.concatenate(gs),
                pep=np.concatenate(peps), ptt=np.concatenate(ptts), fs=np.int64(125),
                feat_keys=np.array(keys))
    out = ROOT / args.out
    np.savez_compressed(out, **blob)
    spans = [(tt.max() - tt.min()) / 60.0 for tt in ts]
    print(f"\n[drift] {len(blob['y'])} segments from {done} cases -> {out}")
    print(f"[drift] span per case: median {np.median(spans):.0f} min, "
          f"IQR {np.percentile(spans,25):.0f}-{np.percentile(spans,75):.0f}, max {max(spans):.0f}")


if __name__ == "__main__":
    main()
