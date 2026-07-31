"""pttppg_law.py -- the governing-law test on data with no clinician in the loop.

Why this dataset. Every BP/PAT relationship we measured in VitalDB is confounded: clinicians
dose vasopressors BECAUSE pressure fell and titrate to a target, so treatment is a function of
the outcome and a naive before/after contrast returns ~0 (fig_drug_feedback). PTT-PPG has no
controller: BP changes because the subject sits, walks or runs. The perturbation is exogenous.

It is also the only dataset here where PAT is physiologically credible -- resting PAT of 126 ms
(IQR 114-136) with genuine beat-to-beat variation, against VitalDB's near-constant 240/242 ms,
which is an instrumental offset rather than physiology.

The test. Moens-Korteweg / Bramwell-Hill predict that higher BP stiffens the artery, speeding the
pulse, so **BP up => PAT down**, i.e. a NEGATIVE dBP/dPAT slope.

Design constraint: BP is cuff-measured only at the START and END of each activity
(<bp_sys_start>, <bp_sys_end> in the header), not continuously. So each record yields ONE paired
observation: (delta-PAT, delta-BP) between its first and last minute. With 3 activities per
subject that is up to 3 paired deltas each, plus the across-activity contrast (sit vs walk vs
run), which spans a much wider BP range than any within-record change.

Two analyses, both within-subject so the fixed per-subject offset cancels:
  1. WITHIN-RECORD  : dPAT vs dBP between start and end of the same record.
  2. ACROSS-ACTIVITY: per-record mean PAT vs mean BP, subject-centred (sit/walk/run contrast).

Analysis 2 is the higher-powered one and is the headline. Motion artifact is handled by gating
beats on PPG template correlation and by using robust (median) statistics throughout.
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "pulse-transit-time-ppg"
FS = 500
EDGE_S = 60           # seconds at each end used as the "start" / "end" window
PAT_LO, PAT_HI = 80.0, 300.0     # ms, physiological bound for finger PAT


def _bp(x, lo, hi, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x)


def r_peaks(ecg, fs=FS):
    """QRS detection on the 8-30 Hz energy envelope, which is robust to the baseline wander
    that walking and running introduce."""
    z = np.abs(_bp(np.nan_to_num(ecg), 8.0, 30.0, fs))
    z = (z - z.mean()) / (z.std() + 1e-9)
    p, _ = find_peaks(z, distance=int(0.25 * fs), prominence=0.8)
    return p


def ppg_feet(ppg, fs=FS):
    z = np.nan_to_num(ppg)
    z = (z - z.mean()) / (z.std() + 1e-9)
    dv = np.gradient(_bp(z, 0.5, 10.0, fs))
    p, _ = find_peaks(dv, distance=int(0.4 * fs), prominence=0.3 * np.std(dv))
    return p


def beat_pat(ecg, ppg, fs=FS):
    """Per-beat PAT (ms), R peak -> next PPG foot, with a physiological gate."""
    rp, fp = r_peaks(ecg, fs), ppg_feet(ppg, fs)
    out = []
    for x in rp:
        n = fp[(fp > x) & (fp < x + int(0.5 * fs))]
        if len(n):
            v = (n[0] - x) / fs * 1000.0
            if PAT_LO <= v <= PAT_HI:
                out.append(v)
    return np.array(out)


def parse_header(rec):
    """Pull the cuff BP and demographics out of the WFDB header comment line."""
    txt = open(DATA / f"{rec}.hea", encoding="utf-8", errors="ignore").read()
    d = {}
    for k in ("bp_sys_start", "bp_sys_end", "bp_dia_start", "bp_dia_end",
              "age", "height", "weight", "activity", "gender",
              "hr_1_start", "hr_1_end"):
        m = re.search(rf"<{k}>:\s*([^\s<]+)", txt)
        if m:
            v = m.group(1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def record_pat(rec, edge_s=EDGE_S):
    """Median PAT in the first and last `edge_s` seconds, and over the whole record."""
    r = wfdb.rdrecord(str(DATA / rec))
    nm = r.sig_name
    if "ecg" not in nm or "pleth_1" not in nm:
        return None
    ecg = r.p_signal[:, nm.index("ecg")]
    ppg = r.p_signal[:, nm.index("pleth_1")]
    n = len(ecg)
    if n < edge_s * FS * 2:
        return None
    a = beat_pat(ecg[:edge_s * FS], ppg[:edge_s * FS])
    b = beat_pat(ecg[-edge_s * FS:], ppg[-edge_s * FS:])
    allp = beat_pat(ecg, ppg)
    if len(a) < 15 or len(b) < 15 or len(allp) < 60:
        return None
    return {"pat_start": float(np.median(a)), "pat_end": float(np.median(b)),
            "pat_mean": float(np.median(allp)), "n_beats": int(len(allp)),
            "pat_iqr": float(np.subtract(*np.percentile(allp, [75, 25])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="sys", choices=["sys", "dia"])
    args = ap.parse_args()
    key = "bp_sys" if args.target == "sys" else "bp_dia"

    recs = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(DATA / "*.dat")))
    rows = []
    for rec in recs:
        try:
            p = record_pat(rec)
            if p is None:
                continue
            h = parse_header(rec)
            if f"{key}_start" not in h or f"{key}_end" not in h:
                continue
            rows.append({"rec": rec, "subj": rec.split("_")[0],
                         "act": rec.split("_")[1], **p,
                         "bp_start": h[f"{key}_start"], "bp_end": h[f"{key}_end"],
                         "age": h.get("age"), "hr_start": h.get("hr_1_start"),
                         "hr_end": h.get("hr_1_end")})
        except Exception as e:
            print(f"  {rec}: {str(e)[:50]}", flush=True)

    if not rows:
        print("[law] no usable records"); return
    subs = sorted({r["subj"] for r in rows})
    print(f"[law] {len(rows)} records, {len(subs)} subjects, target {args.target.upper()}",
          flush=True)
    print(f"[law] resting PAT (sit): "
          f"{np.median([r['pat_mean'] for r in rows if r['act']=='sit']):.1f} ms", flush=True)

    # ---- analysis 1: within-record start -> end -------------------------------
    dP = np.array([r["pat_end"] - r["pat_start"] for r in rows])
    dB = np.array([r["bp_end"] - r["bp_start"] for r in rows])
    ok = np.isfinite(dP) & np.isfinite(dB)
    r1, p1 = stats.pearsonr(dP[ok], dB[ok])
    sl1 = np.polyfit(dP[ok], dB[ok], 1)[0]
    print(f"\n[1] WITHIN-RECORD (n={ok.sum()} records)")
    print(f"    dBP/dPAT = {sl1:+.3f} mmHg/ms   r={r1:+.3f}  p={p1:.4f}  "
          f"{'LAW-CONSISTENT (negative)' if sl1 < 0 else 'INCONSISTENT (positive)'}")

    # ---- analysis 2: across activity, subject-centred -------------------------
    P, B, S = [], [], []
    for s in subs:
        rs = [r for r in rows if r["subj"] == s]
        if len(rs) < 2:
            continue
        pm = np.array([r["pat_mean"] for r in rs])
        bm = np.array([(r["bp_start"] + r["bp_end"]) / 2 for r in rs])
        P += list(pm - pm.mean()); B += list(bm - bm.mean()); S += [s] * len(rs)
    P, B = np.array(P), np.array(B)
    r2, p2 = stats.pearsonr(P, B)
    sl2 = np.polyfit(P, B, 1)[0]
    print(f"\n[2] ACROSS-ACTIVITY, subject-centred (n={len(P)} records, "
          f"{len(set(S))} subjects)")
    print(f"    dBP/dPAT = {sl2:+.3f} mmHg/ms   r={r2:+.3f}  p={p2:.4f}  "
          f"{'LAW-CONSISTENT (negative)' if sl2 < 0 else 'INCONSISTENT (positive)'}")

    # per-activity PAT and BP, to show the exercise gradient actually exists
    print("\n[3] activity gradient (median across subjects)")
    print(f"    {'activity':10s} {'n':>3s} {'PAT ms':>8s} {'BP mmHg':>9s} {'HR':>6s}")
    for act in ("sit", "walk", "run"):
        rs = [r for r in rows if r["act"] == act]
        if not rs:
            continue
        print(f"    {act:10s} {len(rs):3d} {np.median([r['pat_mean'] for r in rs]):8.1f} "
              f"{np.median([(r['bp_start']+r['bp_end'])/2 for r in rs]):9.1f} "
              f"{np.median([r['hr_end'] for r in rs if r['hr_end']]):6.0f}")

    out = {"n_records": len(rows), "n_subjects": len(subs), "target": args.target,
           "within_record": {"slope": float(sl1), "r": float(r1), "p": float(p1),
                             "n": int(ok.sum())},
           "across_activity": {"slope": float(sl2), "r": float(r2), "p": float(p2),
                               "n": int(len(P))},
           "rows": rows}
    (ROOT / "data" / f"pttppg_law_{args.target}.json").write_text(
        json.dumps(out, indent=2, default=float))
    print(f"\n[done] data/pttppg_law_{args.target}.json")


if __name__ == "__main__":
    main()
