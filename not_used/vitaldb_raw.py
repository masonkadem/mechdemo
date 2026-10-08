"""vitaldb_raw.py -- raw continuous VitalDB loader with PEP/PTT decomposition.

Two things this enables that PulseDB cannot.

1. PEP/PTT SEPARATION. PulseDB ships ECG+PPG only, so the best available arrival time is
   PAT = PEP + PTT. Only PTT carries arterial stiffness; PEP (pre-ejection period) is cardiac.
   VitalDB carries the ARTERIAL LINE, which puts a fiducial between the two:

       R peak  --PEP-->  ART foot  --PTT-->  PPG foot

   A first probe on one case put PEP at ~63% of PAT, i.e. most of the quantity this literature
   treats as a stiffness proxy is not vascular at all. That would explain directly why every
   audited model came out ANTI-faithful to PAT. This module exists to test that at scale rather
   than anecdotally.

2. UNALIGNED TIMING. PulseDB pre-segments into beat-aligned 10 s windows, so ECG and PPG arrive
   in fixed relative position and arrival time is partly baked into the windowing rather than
   learned. Windows cut at a FIXED grid (align=False, the default here) force a model to infer
   timing, which is the fairer test of whether it uses PAT.

Quality gating is not optional: raw ART includes flush artifacts and line disconnections (the
first case probed spans -24 to 292 mmHg). Physiological range checks and a beat-consistency
check are applied before any segment is emitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parent
TRACKS = ["SNUADC/ECG_II", "SNUADC/PLETH", "SNUADC/ART"]
FS = 500                      # native SNUADC rate
SEG_S = 10                    # segment length (seconds), matching PulseDB
# physiological gates
ART_LO, ART_HI = 20.0, 300.0
SBP_LO, SBP_HI = 70.0, 220.0
DBP_LO, DBP_HI = 30.0, 130.0
# R -> ART-foot is NOT pure pre-ejection period. VitalDB's arterial line is radial, so this
# interval is PEP (~100 ms) PLUS aorta-to-arm transit (~90 ms), and the measured median is
# ~190 ms. The old 200 ms ceiling therefore sat on top of the distribution and rejected about
# half of all beats -- which emptied 52% of windows entirely (0 valid beats) and silently
# selected for atypically short arrival times.
PEP_LO, PEP_HI = 40.0, 280.0      # ms; PEP + proximal arterial transit, not PEP alone
PTT_LO, PTT_HI = 30.0, 400.0      # ms; radial -> finger

# Quality gates on the beat matching itself (see beat_intervals).
MIN_MATCH_FRAC = 0.60   # matched (R, ART foot, PPG foot) triples, as a fraction of R peaks
MAX_PTT_SPREAD_RR = 0.5  # reject if PTT spread exceeds half an RR: that is a skipped foot


def _bp(x, lo, hi, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x)


def r_peaks(ecg, fs=FS):
    z = _bp(np.nan_to_num(ecg), 0.5, 20.0, fs)
    z = (z - z.mean()) / (z.std() + 1e-9)
    p, _ = find_peaks(z, distance=int(0.3 * fs), prominence=1.0)
    return p


def pulse_feet(w, fs=FS, refractory=0.4):
    """Upstroke feet via max of the first derivative, with a refractory period.

    The refractory matters: without it the dicrotic notch is detected as a second upstroke and
    the beat count roughly doubles (146 'upstrokes' for 74 R peaks on the first case probed),
    which corrupts every interval derived from it.
    """
    z = np.nan_to_num(w)
    z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-9)
    dv = np.gradient(_bp(z, 0.5, 10.0, fs))
    p, _ = find_peaks(dv, distance=int(refractory * fs), prominence=0.3 * np.std(dv))
    return p


def beat_intervals(ecg, ppg, art, fs=FS, with_quality=False):
    """Per-beat (PEP, PTT) in ms. PEP = R -> ART foot, PTT = ART foot -> PPG foot.

    with_quality also returns (match_frac, ptt_spread_rr):

      match_frac      matched triples / R peaks. A low value means the fiducials disagree
                      about how many beats occurred, which is the state in which the
                      surviving intervals are least trustworthy.
      ptt_spread_rr   (max PTT - min PTT) / median RR. The dominant failure is the PPG foot
                      detector missing one beat and locking onto the next, which adds a full
                      RR to that beat's PTT. Physiological beat-to-beat PTT variation is a few
                      ms; anything approaching half an RR is a skipped foot, not physiology.
    """
    r = r_peaks(ecg, fs)
    # Refractory scaled to THIS segment's heart rate. A fixed 0.4 s lets the arterial dicrotic
    # notch register as a second upstroke -- on case 13 that gave 1.45 ART "feet" per R peak,
    # so every interval derived from them paired the wrong beats. 0.7 x RR brings it to 1.04.
    rr_s = float(np.median(np.diff(r))) / fs if len(r) > 2 else 0.6
    ref = max(0.7 * rr_s, 0.25)
    fa, fp = pulse_feet(art, fs, refractory=ref), pulse_feet(ppg, fs, refractory=ref)
    pep, ptt = [], []
    for rp in r:
        na = fa[(fa > rp) & (fa < rp + int(0.4 * fs))]
        if not len(na):
            continue
        npp = fp[(fp > na[0]) & (fp < na[0] + int(0.5 * fs))]
        if not len(npp):
            continue
        a_ms = (na[0] - rp) / fs * 1000.0
        p_ms = (npp[0] - na[0]) / fs * 1000.0
        if PEP_LO <= a_ms <= PEP_HI and PTT_LO <= p_ms <= PTT_HI:
            pep.append(a_ms); ptt.append(p_ms)
    pep, ptt = np.array(pep), np.array(ptt)
    if not with_quality:
        return pep, ptt
    match_frac = len(pep) / max(len(r), 1)
    rr_ms = float(np.median(np.diff(r))) / fs * 1000.0 if len(r) > 1 else np.nan
    spread = (ptt.max() - ptt.min()) / rr_ms if (len(ptt) > 1 and rr_ms > 0) else np.nan
    return pep, ptt, match_frac, spread


def bp_from_art(art):
    """SBP/DBP from the arterial waveform, with physiological gating."""
    a = art[np.isfinite(art)]
    if len(a) < 100:
        return np.nan, np.nan
    sbp, dbp = float(np.percentile(a, 95)), float(np.percentile(a, 5))
    if not (SBP_LO <= sbp <= SBP_HI and DBP_LO <= dbp <= DBP_HI and sbp - dbp > 15):
        return np.nan, np.nan
    return sbp, dbp


def load_case(caseid, seg_s=SEG_S, align=False, fs_out=125, max_segments=400):
    """Return dict with segments (N, L, 2) [ECG, PPG], BP (N, 2), and per-segment PEP/PTT.

    align=False cuts on a fixed grid, so ECG/PPG timing is NOT normalized away -- the model
    must infer arrival time. align=True would cut at R peaks (PulseDB-style).
    """
    import vitaldb
    v = vitaldb.VitalFile(int(caseid), TRACKS)
    d = v.to_numpy(TRACKS, 1.0 / FS)
    if d is None or len(d) < seg_s * FS * 2:
        return None
    ecg, ppg, art = d[:, 0], d[:, 1], d[:, 2]

    L = seg_s * FS
    step = L                                    # non-overlapping
    n = len(d) // step
    keep = np.arange(n)
    if len(keep) > max_segments:                # spread across the case, not just the start
        keep = np.linspace(0, n - 1, max_segments).astype(int)

    segs, bps, peps, ptts = [], [], [], []
    for i in keep:
        s = i * step
        e, p, a = ecg[s:s + L], ppg[s:s + L], art[s:s + L]
        if len(e) < L:
            continue
        if not (np.isfinite(e).all() and np.isfinite(p).all() and np.isfinite(a).all()):
            continue
        if a.min() < ART_LO or a.max() > ART_HI:      # flush artifact / disconnection
            continue
        sbp, dbp = bp_from_art(a)
        if not np.isfinite(sbp):
            continue
        pep, ptt, mfrac, spread = beat_intervals(e, p, a, with_quality=True)
        if len(pep) < 3:                              # need a few consistent beats
            continue
        if mfrac < MIN_MATCH_FRAC:                    # fiducials disagree on the beat count
            continue
        if np.isfinite(spread) and spread > MAX_PTT_SPREAD_RR:   # a skipped PPG foot
            continue
        # downsample to the modelling rate
        k = int(FS // fs_out)
        segs.append(np.stack([e[::k], p[::k]], 1))
        bps.append([sbp, dbp])
        peps.append(float(np.median(pep))); ptts.append(float(np.median(ptt)))

    if not segs:
        return None
    return {"X": np.asarray(segs, np.float32), "y": np.asarray(bps, np.float32),
            "pep": np.asarray(peps), "ptt": np.asarray(ptts),
            "caseid": int(caseid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--max-seg", type=int, default=200)
    ap.add_argument("--out", default="data/vitaldb_raw_pep.npz")
    args = ap.parse_args()

    trio = json.load(open(ROOT / "data" / "vitaldb_scope.json"))["trio_caseids"]
    Xs, ys, gs, peps, ptts = [], [], [], [], []
    done = 0
    for cid in trio:
        if done >= args.n_cases:
            break
        try:
            r = load_case(cid, max_segments=args.max_seg)
        except Exception as e:
            print(f"  case {cid}: {str(e)[:60]}", flush=True); continue
        if r is None:
            continue
        Xs.append(r["X"]); ys.append(r["y"])
        gs.append(np.full(len(r["X"]), cid))
        peps.append(r["pep"]); ptts.append(r["ptt"])
        done += 1
        print(f"  case {cid:5d}: {len(r['X']):4d} segs  PEP {np.median(r['pep']):6.1f} ms  "
              f"PTT {np.median(r['ptt']):6.1f} ms  "
              f"PEP/PAT {np.median(r['pep'])/(np.median(r['pep'])+np.median(r['ptt'])):.2f}",
              flush=True)

    if not Xs:
        print("[raw] nothing passed quality gating"); return
    X = np.concatenate(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    pep = np.concatenate(peps); ptt = np.concatenate(ptts)
    np.savez_compressed(ROOT / args.out, X=X, y=y, g=g, pep=pep, ptt=ptt, fs=125)
    frac = pep / (pep + ptt)
    print(f"\n[raw] {len(X)} segments from {done} cases -> {args.out}")
    print(f"[raw] PEP {np.median(pep):.1f} ms | PTT {np.median(ptt):.1f} ms | "
          f"PEP/PAT median {np.median(frac):.2f} (IQR {np.percentile(frac,25):.2f}-"
          f"{np.percentile(frac,75):.2f})")


if __name__ == "__main__":
    main()
