"""within_subject.py -- does mechanistic faithfulness predict WITHIN-SUBJECT BP tracking?

Motivation
----------
Cross-dataset OOD evaluation turned out to be uninformative: on every external set the best
model merely matches a constant predictor (see the mean-predictor baseline), so there is no
transferable between-subject signal to rank models by. But in these surgical recordings the
within-subject BP variation (SBP sd 13.3 mmHg) is as large as the between-subject variation
(12.3 mmHg) -- every subject spans >30 mmHg. So the informative question is not "does the model
transfer across subjects" but "does it track BP CHANGE within a subject".

Deep nets do track it (within-subject r ~ 0.44-0.63) and, unlike ID accuracy (where all five
architectures sit at DBP 8.1-8.8 and are indistinguishable), the tracking scores SEPARATE the
models. That is the dissociation this project needs. Confirmed not to be a time/drift artifact:
partial r controlling for segment order is unchanged (0.631 -> 0.637).

Two things must be established before that is a finding:

  1. BASELINES. r=0.63 is only meaningful against the right null. If a heart-rate-only predictor
     tracks as well, the tracking is another shortcut and the mechanism story does not hold.
     We compare: within-subject mean (r=0 by construction), HR-only, PAT-only, HR+PAT.
  2. POWER. "Faithfulness predicts tracking" across 5 architectures is n=5 -- the same n that
     produced an r=-0.71 which later collapsed. Here we pair PER-SUBJECT audit slope with
     PER-SUBJECT tracking, giving n = number of test subjects, and bootstrap the correlation.

The audit used is the VALIDATED one: negative-arm sweep, non-finite PAT dropped (never imputed),
beat-slips discarded. On that sweep a faithful model has a POSITIVE dBP/d(shift): a shorter
arrival time means a higher predicted BP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
import ood_benchmark as ob
from mechlib import ECG, PPG, _shift_channel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "within_subject.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
DELTAS = (-6, -4, -2, 0)
SLIP_MS = 150.0
MIN_SD = 3.0          # a subject needs this much DBP variation to define a tracking correlation
MIN_SEG = 50


def per_subject_r(pred, y, groups, min_seg=MIN_SEG, min_sd=MIN_SD):
    """Within-subject correlation between prediction and truth, per subject."""
    out = {}
    for s in np.unique(groups):
        m = groups == s
        if m.sum() >= min_seg and np.std(y[m]) > min_sd and np.std(pred[m]) > 1e-6:
            out[int(s)] = float(np.corrcoef(pred[m], y[m])[0, 1])
    return out


def subject_audit_slopes(predict_fn, X, groups, fs, target=1):
    """Per-subject roll-audit slope (mmHg per ms of nominal shift), negative arm only.
    Non-finite / beat-slipped segments are DROPPED, which is the fix that makes the audit
    pass its positive control."""
    base = mechlib.compute_ptt(X, fs)
    ok0 = np.isfinite(base)
    nom = np.array([1000.0 * d / fs for d in DELTAS])
    P = np.full((len(X), len(DELTAS)), np.nan)
    keep = ok0.copy()
    for j, d in enumerate(DELTAS):
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], d)
        p = mechlib.compute_ptt(Xd, fs)
        dd = (p - base) * 1000.0
        keep &= np.isfinite(p) & (np.abs(dd) <= SLIP_MS)
        P[:, j] = predict_fn(Xd)[:, target]
    out = {}
    for s in np.unique(groups):
        m = (groups == s) & keep
        if m.sum() >= 10:
            sl = [np.polyfit(nom, P[i], 1)[0] for i in np.where(m)[0]]
            out[int(s)] = float(np.median(sl))
    return out, float(keep.mean())


def boot_ci(x, y, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    r = float(np.corrcoef(x, y)[0, 1])
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 1e-9 and np.std(y[i]) > 1e-9:
            bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=150)
    ap.add_argument("--audit-n", type=int, default=4000)
    args = ap.parse_args()

    d = mechlib.load_mini("data/vitaldb_full_calfree.npz")
    fs = d["fs"]
    g, y = d["gte"], d["yte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= args.per_subject]
    sel = np.concatenate([np.where(g == s)[0][:args.per_subject] for s in subs])
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    Y, G = y[sel], g[sel]
    print(f"[wsub] {len(X)} segments from {len(subs)} test subjects", flush=True)

    res = {"n_subjects": len(subs), "n_segments": int(len(X))}

    # ---- physiological baselines ----------------------------------------------
    print("[wsub] computing per-segment cues for baselines ...", flush=True)
    cues = mechlib.compute_scalars(X, fs)
    pat, hr = cues["pat"], cues["hr"]

    base_r = {}
    for tag, feat in [("HR only", hr), ("PAT only", pat), ("AIx only", cues["aix"]),
                      ("amp (null)", cues["amp"])]:
        f = np.asarray(feat, float)
        rs = []
        for s in np.unique(G):
            m = (G == s) & np.isfinite(f)
            if m.sum() >= MIN_SEG and np.std(Y[m, 1]) > MIN_SD and np.std(f[m]) > 1e-9:
                rs.append(abs(np.corrcoef(f[m], Y[m, 1])[0, 1]))
        base_r[tag] = float(np.nanmedian(rs)) if rs else float("nan")
        print(f"[wsub] baseline {tag:10s} |within-r| = {base_r[tag]:.3f}  (n={len(rs)})", flush=True)
    res["baselines"] = base_r

    # ---- deep models: tracking + per-subject faithfulness ----------------------
    rows = {}
    for mk in ARCHS:
        try:
            ck = torch.load(ROOT / "models" / f"{mk}_ecgppg_full.pt", map_location=DEVICE,
                            weights_only=False)
            m = ob.build_model(mk, n_ch=2, L=1250)
            m.load_state_dict(ck["state_dict"]); m.to(DEVICE).eval()
        except Exception as e:
            print(f"[wsub] {mk}: cannot load ({str(e)[:50]})", flush=True)
            continue
        fn = lambda Xr: ob.predict(m, Xr, DEVICE, ck["mu"], ck["sd"])
        p = fn(X)[:, 1]
        track = per_subject_r(p, Y[:, 1], G)
        ai = np.sort(np.random.default_rng(0).choice(len(X), min(args.audit_n, len(X)), False))
        slopes, frac_keep = subject_audit_slopes(fn, X[ai], G[ai], fs)
        common = sorted(set(track) & set(slopes))
        r, lo, hi = boot_ci([slopes[s] for s in common], [track[s] for s in common]) \
            if len(common) >= 10 else (float("nan"),) * 3
        rows[mk] = {"within_r_median": float(np.nanmedian(list(track.values()))),
                    "within_r_iqr": [float(np.nanpercentile(list(track.values()), 25)),
                                     float(np.nanpercentile(list(track.values()), 75))],
                    "mae_within": float(np.median([np.abs(p[G == s] - Y[G == s, 1]).mean()
                                                   for s in np.unique(G)])),
                    "audit_slope_median": float(np.median(list(slopes.values()))),
                    "frac_subj_faithful": float(np.mean([v > 0 for v in slopes.values()])),
                    "frac_segments_audited": frac_keep,
                    "r_slope_vs_tracking": r, "r_ci": [lo, hi], "n_paired": len(common)}
        q = rows[mk]
        print(f"[wsub] {mk:14s} track r {q['within_r_median']:.3f} | audit slope "
              f"{q['audit_slope_median']:+.4f} ({q['frac_subj_faithful']:.0%} subj faithful) | "
              f"r(slope,track) {r:+.3f} [{lo:+.2f},{hi:+.2f}] n={len(common)}", flush=True)
        res.setdefault("models", {})[mk] = q
        OUT.write_text(json.dumps(res, indent=2, default=float))

    # ---- across-architecture check (the weak n=5 version, reported for contrast)
    if res.get("models") and len(res["models"]) >= 3:
        a = [v["audit_slope_median"] for v in res["models"].values()]
        b = [v["within_r_median"] for v in res["models"].values()]
        res["across_arch_r"] = float(np.corrcoef(a, b)[0, 1])
        print(f"\n[wsub] across-architecture r(slope, tracking) = {res['across_arch_r']:+.3f} "
              f"(n={len(a)} -- underpowered, reported for contrast only)", flush=True)

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
