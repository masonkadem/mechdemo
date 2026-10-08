"""drug_event_ptt.py -- does PTT track blood pressure when a drug MOVES the pressure?

Every negative result in this project so far comes from resting anaesthetised patients, where
blood pressure barely moves and PTT explains 4-5% of within-subject variance. That is the regime
where the transit-time law is weakest, so it cannot decide whether the physics is wrong or simply
untested.

Phenylephrine is the test. It is a pure alpha-1 agonist: it raises pressure by constricting
arteries, i.e. through exactly the stiffness mechanism Bramwell-Hill describes. VitalDB records
the infusion pump rate alongside the waveforms, so each bolus is a natural experiment with a
known cause and a measurable effect.

Three questions, one pass over the data:

  1  event response   around each infusion onset, does PTT fall as BP rises?
  2  change model     predict dBP (not BP) from dPTT -- no per-patient offset to hide behind
  3  between-subject  does mean PTT predict stiffness-related patient attributes (age, PWV proxy)?

Sampling note: the drug track is sparse and must be read at 1 Hz. Resampling it to 125 Hz makes
NaN gaps that look like thousands of spurious on/off transitions.

    python drug_event_ptt.py --cases 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
TRACKS = ["SNUADC/ART", "SNUADC/PLETH", "SNUADC/ECG_II"]
DRUG = "Orchestra/PHEN_RATE"
PRE, POST = 90, 300          # seconds before onset / after onset used for the contrast
SETTLE = 120                 # the pressor response peaks ~2-4 min after a rate step, not 30 s:
                             # measured directly (case 20 rises +7 mmHg at ~140 s), so a window
                             # closing at 120 s scores the response before it has arrived


def pulse_feats(ppg, art, fs=FS):
    """Per-window timing features. All measured INSIDE one channel, so the ~500 ms PLETH
    channel lag cannot contaminate them (see fig_alignment.py)."""
    out = {}
    # raw VitalDB has NaN gaps wherever a channel dropped out; savgol cannot take them, and a
    # window that is mostly missing should not produce a feature at all
    if len(ppg) < fs * 10 or np.isfinite(ppg).mean() < 0.9 or np.isfinite(art).mean() < 0.9:
        return None
    ppg = np.nan_to_num(ppg, nan=float(np.nanmedian(ppg))) if np.isnan(ppg).any() else ppg
    art = np.nan_to_num(art, nan=float(np.nanmedian(art))) if np.isnan(art).any() else art
    if not (np.isfinite(ppg).all() and np.isfinite(art).all()):
        return None
    sm = savgol_filter(ppg, 11, 3)
    rng = sm.max() - sm.min()
    if rng < 1e-9:
        return None
    pk = find_peaks(sm, distance=int(0.4 * fs), prominence=rng * 0.3)[0]
    if len(pk) < 4:
        return None
    d2 = savgol_filter(np.gradient(np.gradient(sm)), 9, 3)
    ct, notch = [], []
    for j in range(len(pk) - 1):
        a, b = pk[j], pk[j + 1]
        lo = max(0, a - int(0.30 * fs))
        if a - lo < 4:
            continue
        ft = lo + int(np.argmin(sm[lo:a]))
        ct.append((a - ft) / fs * 1000)
        L = b - a
        w0, w1 = a + int(0.08 * L), a + int(0.60 * L)
        if w1 - w0 > 5:
            c = find_peaks(d2[w0:w1])[0]
            if len(c):
                notch.append((w0 + c[int(np.argmax(d2[w0:w1][c]))] - a) / fs * 1000)
    if not ct:
        return None
    out["crest_ms"] = float(np.median(ct))
    out["notch_ms"] = float(np.median(notch)) if notch else np.nan
    out["hr_bpm"] = float(60.0 / (np.median(np.diff(pk)) / fs))

    # true PTT: ABP foot -> PPG foot. Both landmarks mechanical, so PEP cancels; the fixed
    # channel offset inflates the absolute value but not its CHANGE, which is what we use.
    asm = savgol_filter(art, 11, 3)
    arng = asm.max() - asm.min()
    apk = find_peaks(asm, distance=int(0.4 * fs), prominence=arng * 0.3)[0] if arng > 1e-9 else []
    if len(apk) >= 3 and len(pk) >= 3:
        af = [max(0, p - int(0.30 * fs)) + int(np.argmin(asm[max(0, p - int(0.30 * fs)):p]))
              for p in apk if p > 5]
        pf = [max(0, p - int(0.30 * fs)) + int(np.argmin(sm[max(0, p - int(0.30 * fs)):p]))
              for p in pk if p > 5]
        dl = [(np.array(pf)[np.argmin(np.abs(np.array(pf) - t))] - t) / fs * 1000 for t in af]
        if dl:
            out["ptt_ms"] = float(np.median(dl))
    a_ok = art[np.isfinite(art) & (art > 30) & (art < 220)]
    if len(a_ok) < fs * 5:
        return None
    out["dbp"] = float(np.percentile(a_ok, 5))
    out["sbp"] = float(np.percentile(a_ok, 95))
    return out


def onsets(rate, min_run=60, min_gap=300, quiet=60):
    """Infusion events = a sustained RATE INCREASE after a quiet period.

    Phenylephrine here is usually a continuous background infusion (the pump is on ~85% of the
    time), so a zero-to-nonzero transition is normally a brief interruption ending, not a drug
    challenge -- the patient is already vasoconstricted on both sides of it and BP barely moves.
    A real challenge is a step UP in dose that then holds, preceded by a genuinely quiet window.
    """
    r = np.nan_to_num(rate)
    out = []
    for s in range(quiet, len(r) - min_run):
        if r[s] <= r[s - 1]:
            continue
        step = r[s] - r[s - 1]
        if step < 2.0:                                   # ignore tiny dose nudges
            continue
        if r[max(0, s - quiet):s].max() > 0.5:           # must follow a drug-free window
            continue
        if np.mean(r[s:s + min_run] > 0) < 0.8:          # and must actually hold
            continue
        if out and s - out[-1] < min_gap:
            continue
        out.append(int(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--out", default="data/drug_event_ptt.json")
    args = ap.parse_args()

    import vitaldb
    trks = pd.read_csv("https://api.vitaldb.net/trks")
    clin = pd.read_csv("https://api.vitaldb.net/cases").set_index("caseid")
    have = trks.groupby("caseid")["tname"].apply(set)
    trio = set(have[have.apply(lambda s: set(TRACKS) <= s)].index)
    phen = set(trks[trks.tname == DRUG].caseid)
    cand = sorted(trio & phen)[:args.cases]
    print(f"[scope] {len(cand)} cases with ART+PLETH+ECG and phenylephrine", flush=True)

    rows, subj = [], []
    for n, cid in enumerate(cand, 1):
        try:
            rate = vitaldb.VitalFile(cid, [DRUG]).to_numpy([DRUG], 1.0)[:, 0]
            ev = onsets(rate)
            if not ev:
                continue
            arr = vitaldb.VitalFile(cid, TRACKS).to_numpy(TRACKS, 1 / FS)
        except Exception as exc:
            print(f"  case {cid}: {str(exc)[:40]}", flush=True); continue
        art, ppg = arr[:, 0], arr[:, 1]
        got = 0
        for s in ev:
            a0, a1 = (s - PRE) * FS, s * FS
            b0, b1 = (s + SETTLE) * FS, (s + POST) * FS
            if a0 < 0 or b1 > len(art):
                continue
            f_pre = pulse_feats(ppg[a0:a1], art[a0:a1])
            f_post = pulse_feats(ppg[b0:b1], art[b0:b1])
            if not f_pre or not f_post:
                continue
            r = {"caseid": cid, "t": s}
            for k in ("dbp", "sbp", "ptt_ms", "notch_ms", "crest_ms", "hr_bpm"):
                r[f"pre_{k}"] = f_pre.get(k, np.nan)
                r[f"d_{k}"] = f_post.get(k, np.nan) - f_pre.get(k, np.nan)
            rows.append(r); got += 1
        if got:
            c = clin.loc[cid] if cid in clin.index else None
            # a resting reference window for this patient: 5 min before the first infusion
            s0 = ev[0]
            base = pulse_feats(ppg[max(0, (s0 - 360) * FS):(s0 - 60) * FS],
                               art[max(0, (s0 - 360) * FS):(s0 - 60) * FS])
            if base and c is not None:
                subj.append({"caseid": cid, "age": float(c.get("age", np.nan)),
                             "sex": 1.0 if str(c.get("sex")) == "M" else 0.0,
                             "bmi": float(c.get("bmi", np.nan)),
                             "height": float(c.get("height", np.nan)),
                             "htn": float(c.get("preop_htn", 0) or 0), **base})
        print(f"  [{n}/{len(cand)}] case {cid}: {got} usable events", flush=True)

    df = pd.DataFrame(rows)
    sf = pd.DataFrame(subj)
    df.to_csv(DATA / "drug_events.csv", index=False)
    sf.to_csv(DATA / "drug_subjects.csv", index=False)
    print(f"\n[data] {len(df)} events from {df.caseid.nunique()} cases; "
          f"{len(sf)} patients with baselines", flush=True)

    res = {}
    if len(df):
        d = df.dropna(subset=["d_dbp"])
        res["n_events"] = int(len(d))
        res["median_dDBP"] = float(d.d_dbp.median())
        res["frac_BP_rose"] = float((d.d_dbp > 0).mean())
        print(f"\n[1] EVENT RESPONSE  median dDBP {d.d_dbp.median():+.2f} mmHg, "
              f"BP rose in {(d.d_dbp>0).mean()*100:.0f}% of events")
        for f in ("ptt_ms", "notch_ms", "crest_ms", "hr_bpm"):
            m = d.dropna(subset=[f"d_{f}"])
            if len(m) > 10:
                r = float(np.corrcoef(m[f"d_{f}"], m.d_dbp)[0, 1])
                res[f"corr_d{f}_dDBP"] = r
                print(f"    corr(d {f:9s}, dDBP) = {r:+.3f}   (n={len(m)})"
                      f"   physics wants NEGATIVE for timing features")
    if len(sf) > 10:
        print(f"\n[3] BETWEEN-SUBJECT (n={len(sf)} patients)")
        for f in ("ptt_ms", "notch_ms", "crest_ms"):
            m = sf.dropna(subset=[f, "age"])
            if len(m) > 8:
                ra = float(np.corrcoef(m[f], m.age)[0, 1])
                rb = float(np.corrcoef(m[f], m.dbp)[0, 1])
                res[f"between_{f}_age"] = ra
                res[f"between_{f}_dbp"] = rb
                print(f"    {f:9s} vs age {ra:+.3f} | vs DBP {rb:+.3f}   (n={len(m)})")

    (DATA / Path(args.out).name).write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
