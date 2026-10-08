"""drug_response.py -- treatment-response tracking, with the indication confound handled.

drug_event_ptt.py contrasts a fixed pre-window against a fixed post-window around a
phenylephrine rate step, and finds a median rise of 0.49 mmHg with BP going up in 50% of
events -- a coin flip. That is almost certainly not pharmacology. A pressor is given BECAUSE
pressure is falling, so a fixed pre-window straddles the fall and a fixed post-window the
recovery: the net contrast is ~0 even when the drug works perfectly. Aligning on the infusion
step measures the clinician's reaction time, not the drug.

Three changes:

  nadir alignment   measure recovery from the BP TROUGH near the step, not from a fixed
                    pre-window. The trough is the state the clinician was reacting to.
  dose-response     relate the SIZE of the rate step to the size of the rise. Dose is not
                    purely determined by the trough, so this survives the confound in a way
                    a pre/post contrast does not.
  a validity gate   phenylephrine is a pure alpha-1 agonist, so a genuine pressor response
                    triggers baroreflex bradycardia: d(HR) vs d(BP) MUST come out negative.
                    If it does not, the extraction is broken and the feature correlations
                    below are not interpretable. The old run had +0.272 -- backwards.

Per-event rows are saved, so re-analysis does not mean re-downloading.

    python drug_response.py --cases 90
"""
import argparse
import json
from pathlib import Path

import numpy as np

import drug_event_ptt as de

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WIN_S = 10                # feature window
NADIR_BACK, NADIR_FWD = 150, 150      # s: search for the trough around the rate step
RESP_MAX = 360            # s: how long after the trough the response may take to peak


def case_series(caseid):
    """Feature time series across one case, plus the phenylephrine rate at 1 Hz."""
    import vitaldb
    v = vitaldb.VitalFile(int(caseid), de.TRACKS + [de.DRUG])
    w = v.to_numpy(de.TRACKS, 1.0 / de.FS)
    r = v.to_numpy([de.DRUG], 1.0)           # the drug track is sparse: read it at 1 Hz
    if w is None or r is None or len(w) < de.FS * 600:
        return None
    art, ppg, ecg = w[:, 0], w[:, 1], w[:, 2]
    rate = np.nan_to_num(r[:, 0], nan=0.0)

    L = WIN_S * de.FS
    rows = []
    for i in range(len(w) // L):
        s = i * L
        f = de.pulse_feats(ppg[s:s + L], art[s:s + L])
        if f is None:
            continue
        f["t"] = s / de.FS
        rows.append(f)
    if len(rows) < 30:
        return None
    return rows, rate


def analyse_case(caseid):
    got = case_series(caseid)
    if got is None:
        return []
    rows, rate = got
    t = np.array([r["t"] for r in rows])
    dbp = np.array([r["dbp"] for r in rows])
    hr = np.array([r.get("hr_bpm", np.nan) for r in rows])
    ptt = np.array([r.get("ptt_ms", np.nan) for r in rows])
    notch = np.array([r.get("notch_ms", np.nan) for r in rows])
    crest = np.array([r.get("crest_ms", np.nan) for r in rows])

    out = []
    for on in de.onsets(rate):
        pre = rate[max(0, on - 60):on]
        post = rate[on:on + 120]
        if not len(pre) or not len(post):
            continue
        step = float(np.median(post) - np.median(pre))
        if step <= 0:
            continue

        # --- nadir: the trough the clinician was reacting to
        m = (t >= on - NADIR_BACK) & (t <= on + NADIR_FWD)
        if m.sum() < 4:
            continue
        i_nad = np.where(m)[0][int(np.argmin(dbp[m]))]

        # --- response: the peak reached after the trough
        m2 = (t > t[i_nad]) & (t <= t[i_nad] + RESP_MAX)
        if m2.sum() < 4:
            continue
        i_pk = np.where(m2)[0][int(np.argmax(dbp[m2]))]

        def d(a):
            return float(a[i_pk] - a[i_nad]) if np.isfinite(a[i_pk]) and np.isfinite(a[i_nad]) else np.nan

        out.append({"caseid": int(caseid), "t_onset": float(t[i_nad]), "dose_step": step,
                    "dbp_nadir": float(dbp[i_nad]), "d_dbp": d(dbp), "d_hr": d(hr),
                    "d_ptt": d(ptt), "d_notch": d(notch), "d_crest": d(crest),
                    "rise_s": float(t[i_pk] - t[i_nad]), "kind": "drug"})

    # ---- MATCHED CONTROLS.
    # Taking a minimum and then a maximum after it cannot return a negative number, so the
    # drug "response" above is guaranteed positive by construction -- it measures regression
    # to the minimum, not pharmacology. Blood pressure troughs also recover on their own. So
    # score untreated troughs of similar depth in the SAME case by the identical procedure;
    # the drug effect is the difference, and the bias cancels because both arms carry it.
    drug_t = [e["t_onset"] for e in out]
    quiet = np.ones(len(t), bool)
    for on in de.onsets(rate):                       # exclude anything near a rate change
        quiet &= np.abs(t - on) > 600
    for e in list(out):
        quiet &= np.abs(t - e["t_onset"]) > 600
    for e in list(out):
        cand = np.where(quiet & (np.abs(dbp - e["dbp_nadir"]) < 3.0))[0]
        cand = cand[(t[cand] > t[0] + RESP_MAX) & (t[cand] < t[-1] - RESP_MAX)]
        if not len(cand):
            continue
        i_nad = int(cand[np.argmin(np.abs(dbp[cand] - e["dbp_nadir"]))])
        m2 = (t > t[i_nad]) & (t <= t[i_nad] + RESP_MAX)
        if m2.sum() < 4:
            continue
        i_pk = np.where(m2)[0][int(np.argmax(dbp[m2]))]

        def dc(a):
            return float(a[i_pk] - a[i_nad]) if np.isfinite(a[i_pk]) and np.isfinite(a[i_nad]) else np.nan

        out.append({"caseid": int(caseid), "t_onset": float(t[i_nad]), "dose_step": 0.0,
                    "dbp_nadir": float(dbp[i_nad]), "d_dbp": dc(dbp), "d_hr": dc(hr),
                    "d_ptt": dc(ptt), "d_notch": dc(notch), "d_crest": dc(crest),
                    "rise_s": float(t[i_pk] - t[i_nad]), "kind": "control"})
        quiet &= np.abs(t - t[i_nad]) > 600          # do not reuse the same control window
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=90)
    ap.add_argument("--out", default="data/drug_response.json")
    args = ap.parse_args()

    ids = de.scope_cases(args.cases) if hasattr(de, "scope_cases") else None
    if ids is None:
        import vitaldb
        cand = vitaldb.find_cases(de.TRACKS + [de.DRUG])
        ids = [int(c) for c in cand][:args.cases]
    print(f"[scope] {len(ids)} phenylephrine cases", flush=True)

    ev = []
    for i, cid in enumerate(ids, 1):
        try:
            e = analyse_case(cid)
        except Exception as exc:
            print(f"  [{i}/{len(ids)}] case {cid}: {str(exc)[:50]}", flush=True); continue
        ev += e
        print(f"  [{i}/{len(ids)}] case {cid}: {len(e)} events (total {len(ev)})", flush=True)

    if not ev:
        print("no events"); return
    drug = [e for e in ev if e["kind"] == "drug"]
    ctrl = [e for e in ev if e["kind"] == "control"]
    A = {k: np.array([e[k] for e in drug], float) for k in drug[0] if k not in ("caseid", "kind")}
    C = {k: np.array([e[k] for e in ctrl], float) for k in ctrl[0] if k not in ("caseid", "kind")} if ctrl else {}
    n = len(drug)
    print(f"\n[events] {n} drug + {len(ctrl)} matched controls, "
          f"{len({e['caseid'] for e in ev})} cases\n")

    print(f"[1] RESPONSE (nadir -> peak). min-then-max is positive BY CONSTRUCTION, so the")
    print(f"    control column is the only way to read the drug column.")
    md, mc = np.nanmedian(A["d_dbp"]), (np.nanmedian(C["d_dbp"]) if ctrl else np.nan)
    print(f"    drug    median dDBP {md:+.2f} mmHg | rose in {(A['d_dbp'] > 0).mean():.0%}"
          f" | time to peak {np.nanmedian(A['rise_s']):.0f} s")
    if ctrl:
        print(f"    control median dDBP {mc:+.2f} mmHg | rose in {(C['d_dbp'] > 0).mean():.0%}"
              f" | time to peak {np.nanmedian(C['rise_s']):.0f} s")
        # paired by case, since several events share a patient
        cases = sorted({e["caseid"] for e in drug} & {e["caseid"] for e in ctrl})
        pair = np.array([[np.nanmedian([e["d_dbp"] for e in drug if e["caseid"] == c]),
                          np.nanmedian([e["d_dbp"] for e in ctrl if e["caseid"] == c])]
                         for c in cases], float)
        pair = pair[np.isfinite(pair).all(1)]
        diff = pair[:, 0] - pair[:, 1]
        rng = np.random.default_rng(0)
        b = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(4000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"\n    DRUG EFFECT (paired by case, n={len(diff)}): "
              f"{np.median(diff):+.2f} mmHg  95%CI [{lo:+.2f}, {hi:+.2f}]  "
              f"{'excludes 0' if lo > 0 or hi < 0 else 'INCLUDES 0'}")

    print(f"\n[2] VALIDITY GATE -- baroreflex (needs a clearly negative correlation)")
    r_hr, n_hr = corr(A["d_hr"], A["d_dbp"])
    ok = np.isfinite(r_hr) and r_hr < -0.15      # -0.003 is not a baroreflex, it is nothing
    print(f"    corr(dHR, dDBP) = {r_hr:+.3f}  (n={n_hr})   "
          f"{'PASS' if ok else 'FAIL: no baroreflex signal -- the extraction is not capturing a pressor response'}")

    print(f"\n[3] DOSE-RESPONSE (survives confounding by indication)")
    r_d, n_d = corr(A["dose_step"], A["d_dbp"])
    print(f"    corr(dose step, dDBP) = {r_d:+.3f}  (n={n_d})")

    print(f"\n[4] DO TIMING FEATURES TRACK THE CHANGE?"
          f"{'' if ok else '   [uninterpretable: gate failed]'}")
    res = {}
    for k in ("d_ptt", "d_notch", "d_crest"):
        r, nn = corr(A[k], A["d_dbp"])
        res[k] = {"r": r, "n": nn}
        print(f"    corr({k:8}, dDBP) = {r:+.3f}  (n={nn})   physics wants NEGATIVE")

    json.dump({"n_events": n, "median_dDBP": float(np.nanmedian(A["d_dbp"])),
               "frac_rose": float((A["d_dbp"] > 0).mean()),
               "baroreflex_r": r_hr, "gate_passed": bool(ok),
               "dose_response_r": r_d, "features": res,
               "events": ev}, open(ROOT / args.out, "w"), indent=2, default=float)
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
