"""audit_subject.py -- SUBJECT-LEVEL causal roll-audit, with a validated positive control.

Why this file exists
--------------------
The segment-level audit (mechlib.causal_ptt_audit) fails its own positive control: a model
defined as BP = a/PAT + b, which uses arrival time and nothing else, returns slope ~0. Three
estimator pathologies, all diagnosed on VitalDB test segments, compound to cause this:

  1. NaN imputation. PAT is measurable on only ~44% of segments. audit_controls.py replaced
     unmeasurable PAT with a FIXED constant, so those segments predict the same value at every
     shift and contribute exactly zero slope. The audit reports the MEDIAN slope across
     segments, so the flat majority outvotes the responding minority -> ~0.
  2. Asymmetric saturation. Negative shifts track nearly 1:1 (-48 ms -> -45.7 ms measured) but
     positive shifts saturate (+48 ms -> +26.2 ms) as the foot leaves the detector's window.
  3. Beat-slip aliasing. At +48 ms the median dPAT is +26 ms but the MEAN is -3.4 ms: ~1% of
     segments jump a whole cardiac period (up to 225 ms). Those outliers flip the fitted sign.

The fix is to aggregate at the SUBJECT level, which is also the physiologically correct scale --
Moens-Korteweg relates BP to stiffness per subject, and BP varies far more between subjects than
between beats. Subject aggregation removes all three failures: unmeasurable segments are DROPPED
rather than imputed, beat-slips are outvoted within a subject, and saturation is confined to the
positive arm which we can simply not use.

Design
------
  * negative shifts only (the arm that tracks 1:1), plus 0
  * per-segment PAT measured; NON-FINITE SEGMENTS DROPPED, never imputed
  * per-segment dPAT screened for beat-slip (|dPAT| > 150 ms discarded)
  * regress subject-mean prediction on subject-mean MEASURED dPAT -- so the x-axis is the
    arrival-time change that actually occurred, not the nominal shift we asked for
  * slope in mmHg/ms; textbook physiology predicts NEGATIVE

Reports the same three controls as audit_controls.py so the fix is verifiable:
  positive  : BP = a/PAT + b, a<0 by construction  -> audit MUST return a clear negative slope
  negative  : BP = const                           -> audit MUST return ~0
  amplitude : BP = f(PPG amplitude)                -> audit should be ~0 (specificity)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

import mechlib
from mechlib import ECG, PPG, _shift_channel

ROOT = Path(__file__).resolve().parent
SLIP_MS = 150.0          # |dPAT| above this is a beat-slip, not a shift response
MIN_SEG = 3              # a subject needs this many valid segments to enter the regression


def subject_audit(predict_fn, X, groups, fs, deltas=(-6, -4, -2, 0), target=1,
                  slip_ms=SLIP_MS, min_seg=MIN_SEG):
    """Roll the PPG channel, measure the ACTUAL PAT change, and regress subject-mean
    prediction on subject-mean measured dPAT. Returns slope (mmHg/ms) and diagnostics."""
    base_pat = mechlib.compute_ptt(X, fs)
    ok0 = np.isfinite(base_pat)

    dpat = np.full((len(X), len(deltas)), np.nan)
    pred = np.full((len(X), len(deltas)), np.nan)
    for j, d in enumerate(deltas):
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], d)
        p = mechlib.compute_ptt(Xd, fs)
        good = ok0 & np.isfinite(p)
        dd = (p - base_pat) * 1000.0                      # ms
        good &= np.abs(dd) <= slip_ms                     # drop beat-slips
        dpat[good, j] = dd[good]
        pred[:, j] = predict_fn(Xd)[:, target]

    # a segment is usable only if EVERY delta gave a valid, non-slipped PAT
    keep = np.isfinite(dpat).all(1)
    x_sub, y_sub, n_sub = [], [], []
    for g in np.unique(groups[keep]):
        m = keep & (groups == g)
        if m.sum() < min_seg:
            continue
        x_sub.append(dpat[m].mean(0))
        y_sub.append(pred[m].mean(0))
        n_sub.append(int(m.sum()))
    if len(x_sub) < 3:
        return {"error": "too few subjects", "n_subjects": len(x_sub)}
    x = np.concatenate(x_sub)
    y = np.concatenate(y_sub)
    slope, intercept = np.polyfit(x, y, 1)

    # per-subject slopes -> a proper CI across subjects
    ps = [np.polyfit(xi, yi, 1)[0] for xi, yi in zip(x_sub, y_sub) if np.ptp(xi) > 1.0]
    ps = np.array(ps)
    return {
        "slope_mmHg_per_ms": float(slope),
        "slope_subj_median": float(np.median(ps)) if len(ps) else float("nan"),
        "slope_subj_lo": float(np.percentile(ps, 2.5)) if len(ps) else float("nan"),
        "slope_subj_hi": float(np.percentile(ps, 97.5)) if len(ps) else float("nan"),
        "frac_negative": float(np.mean(ps < 0)) if len(ps) else float("nan"),
        "n_subjects": len(x_sub),
        "n_segments_kept": int(keep.sum()),
        "frac_segments_kept": float(keep.mean()),
        "resp_range_mmHg": float(np.mean([np.ptp(yi) for yi in y_sub])),
        "dpat_range_ms": float(np.mean([np.ptp(xi) for xi in x_sub])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()

    d = mechlib.load_mini("data/vitaldb_full_calfree.npz")
    fs = d["fs"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["gte"]), min(args.n, len(d["gte"])), replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    y = d["yte"][sel]
    g = d["gte"][sel]
    print(f"[audit] {len(X)} segments, {len(np.unique(g))} subjects, fs={fs}", flush=True)

    pat = mechlib.compute_ptt(X, fs)
    ok = np.isfinite(pat)
    med = float(np.nanmedian(pat[ok]))
    print(f"[audit] PAT measurable on {100*ok.mean():.0f}% of segments, median {med:.3f} s",
          flush=True)

    # ---- controls -------------------------------------------------------------
    # positive: textbook-signed 1/PAT law. a<0 so longer PAT -> lower BP, by construction.
    a_pat = -3.0
    b_pat = float(y[ok, 1].mean() - a_pat / med)
    amp = np.array([np.ptp(x[:, PPG]) for x in X])
    amp_mod = Ridge(alpha=1.0).fit(amp[ok][:, None], y[ok, 1])
    dbp_mean = float(y[:, 1].mean())

    def make(kind):
        def fn(Xr):
            if kind == "positive":
                p = mechlib.compute_ptt(Xr, fs)
                # NOTE: still imputed here, because a predict_fn must return a value for every
                # row. The AUDIT is what now drops these rows -- that is precisely the fix.
                p = np.where(np.isfinite(p), p, med)
                o = a_pat / p + b_pat
            elif kind == "negative":
                o = np.full(len(Xr), dbp_mean)
            else:
                o = amp_mod.predict(np.array([np.ptp(x[:, PPG]) for x in Xr])[:, None])
            return np.stack([o, o], 1)
        return fn

    out = {}
    print(f"\n{'control':10s} {'slope':>8s} {'[95% CI subj]':>20s} {'frac<0':>7s} "
          f"{'range':>6s} {'dPAT':>6s} {'subj':>5s}", flush=True)
    for kind in ("positive", "negative", "amplitude"):
        r = subject_audit(make(kind), X, g, fs)
        out[kind] = r
        print(f"{kind:10s} {r['slope_mmHg_per_ms']:+8.4f} "
              f"[{r['slope_subj_lo']:+8.4f},{r['slope_subj_hi']:+8.4f}] "
              f"{r['frac_negative']:7.0%} {r['resp_range_mmHg']:6.2f} "
              f"{r['dpat_range_ms']:6.1f} {r['n_subjects']:5d}", flush=True)

    (ROOT / "data" / "audit_subject_controls.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\n[done] data/audit_subject_controls.json", flush=True)

    ok_pos = out["positive"]["slope_mmHg_per_ms"] < 0 and out["positive"]["frac_negative"] > 0.6
    ok_neg = abs(out["negative"]["slope_mmHg_per_ms"]) < 1e-3
    print(f"[gate] positive control negative-signed: {ok_pos}", flush=True)
    print(f"[gate] negative control ~zero          : {ok_neg}", flush=True)
    if not (ok_pos and ok_neg):
        print("[gate] AUDIT STILL NOT VALID -- do not interpret model slopes.", flush=True)


if __name__ == "__main__":
    main()
