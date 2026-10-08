"""drift_analyze.py -- how many cuff readings, and how long do they last?

The calibration-burden curve (fig_calibration_strategy.py) answers the first half on PulseDB.
It cannot answer the second half, because PulseDB shuffles segment order: its "gap" axis is a
segment count, not elapsed time. ISO 81060-3 and IEEE 1708a both ask for post-calibration
drift and a recalibration interval, and the published PulseDB benchmarks report neither.

Raw VitalDB is continuous, so elapsed time is known. This measures error over BOTH axes:

    k  cuff readings used to calibrate      x      hours since that calibration

Design. Cases are split subject-disjointly. The population model is fitted on train cases and
frozen. Each held-out case is calibrated on its FIRST k segments in time order, then scored in
elapsed-time bins measured from the end of its own calibration window. No test case
contributes to the population model, and no calibration segment is ever scored.

    python drift_analyze.py
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
KS = [1, 2, 5, 20]
BINS_MIN = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 240), (240, 10 ** 4)]
FEAT_CACHE = DATA / "_drift_feats.pkl"


def _chunk(args):
    import numpy as np
    import features_full as ff
    import mechlib
    X = args
    F = ff.compute_full(mechlib.normalize(np.asarray(X, np.float64)), FS, ppg_ch=1, ecg_ch=0)
    pat = mechlib.compute_ptt(np.asarray(X, np.float64), FS, ecg_pos=0, ppg_pos=1)
    return F, pat


def extract(X, workers=8, block=2048):
    jobs = [X[i:i + block] for i in range(0, len(X), block)]
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_chunk, jobs), 1):
            parts.append(r)
            print(f"  features {i}/{len(jobs)}", flush=True)
    F = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    pat = np.concatenate([p[1] for p in parts])
    return F, pat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--data", default="vitaldb_drift.npz",
                    help="npz in data/; the at-scale collection stores features, not waveforms")
    args = ap.parse_args()

    d = np.load(DATA / args.data, allow_pickle=True)
    y, t, g = d["y"][:, 1], d["t"], d["g"].astype(int)              # y[:,1] = DBP

    if "feat_keys" in d.files:
        # collected at scale: features were extracted during download, waveforms discarded
        fk = [str(k) for k in d["feat_keys"]]
        F = {k: np.asarray(d[k], float) for k in fk if k != "_pat"}
        pat = np.asarray(d["_pat"], float)
        print(f"[data] {args.data}: {len(y)} segments, {len(np.unique(g))} patients, "
              f"{len(F)} stored features")
    elif FEAT_CACHE.exists():
        F, pat = pickle.load(open(FEAT_CACHE, "rb"))
        print(f"[cache] {FEAT_CACHE.name}")
    else:
        X = d["X"]
        print(f"[1/3] extracting features on {len(X)} segments ...", flush=True)
        F, pat = extract(X, args.workers)
        pickle.dump((F, pat), open(FEAT_CACHE, "wb"), protocol=4)

    keys = sorted(k for k in F
                  if np.isfinite(np.asarray(F[k], float)).mean() > 0.5
                  and np.nanstd(np.asarray(F[k], float)) > 1e-9)
    M = np.column_stack([np.asarray(F[k], float) for k in keys])
    med = np.nanmedian(M, 0)
    M = np.where(np.isfinite(M), M, med)

    cases = np.unique(g)
    rng = np.random.default_rng(0)
    perm = rng.permutation(cases)
    n_tr = int(len(cases) * args.train_frac)
    tr_cases, te_cases = set(perm[:n_tr]), perm[n_tr:]
    is_tr = np.array([c in tr_cases for c in g])
    print(f"\n[2/3] population model: {len(tr_cases)} train cases "
          f"({is_tr.sum()} segs) -> {len(te_cases)} test cases")

    import lightgbm as lgb
    mdl = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63,
                            min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                            random_state=0, n_jobs=1, verbosity=-1).fit(M[is_tr], y[is_tr])
    pred = mdl.predict(M)

    print("\n[3/3] error vs (k readings, time since calibration)\n")
    rows = []
    for k in KS:
        for lo, hi in BINS_MIN:
            e_off, e_avg, e_eqn, e_unc, n = [], [], [], [], 0
            for c in te_cases:
                m = np.where(g == c)[0]
                m = m[np.argsort(t[m])]
                if len(m) < k + 5:
                    continue
                cal, rest = m[:k], m[k:]
                t0 = t[cal[-1]]
                sel = rest[((t[rest] - t0) / 60.0 >= lo) & ((t[rest] - t0) / 60.0 < hi)]
                if len(sel) < 3:
                    continue
                n += 1
                off = float(np.mean(y[cal] - pred[cal]))
                e_off.append(np.abs(pred[sel] + off - y[sel]).mean())
                e_avg.append(np.abs(y[cal].mean() - y[sel]).mean())
                e_unc.append(np.abs(pred[sel] - y[sel]).mean())
                z = 1.0 / np.clip(pat, 0.02, None) ** 2
                zc, zs = z[cal], z[sel]
                if k >= 2 and np.isfinite(zc).sum() >= 2 and np.nanstd(zc) > 1e-9:
                    A = np.c_[np.ones(len(zc)), np.nan_to_num(zc, nan=np.nanmedian(zc))]
                    w = np.linalg.lstsq(A, y[cal], rcond=None)[0]
                    p = w[0] + w[1] * np.nan_to_num(zs, nan=np.nanmedian(zc))
                    e_eqn.append(np.abs(p - y[sel]).mean())
            if n < 10:
                continue
            # Bootstrap over PATIENTS. The benefit is paired within a patient (the same case
            # supplies both the calibrated and the uncalibrated error), so resample the pairs
            # -- a CI on the difference, not two independent CIs that would overstate the
            # uncertainty of their gap.
            eo, eu = np.array(e_off), np.array(e_unc)
            rg = np.random.default_rng(0)
            bs = [float(np.median(eu[i] - eo[i]))
                  for i in (rg.integers(0, n, n) for _ in range(3000))]
            lo_ci, hi_ci = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            rows.append({"k": k, "lo": lo, "hi": hi, "n_cases": n,
                         "offset": float(np.median(e_off)), "cuff_avg": float(np.median(e_avg)),
                         "uncalibrated": float(np.median(e_unc)),
                         "equation": float(np.median(e_eqn)) if e_eqn else None,
                         "benefit": float(np.median(eu - eo)),
                         "benefit_ci": [lo_ci, hi_ci]})
            r = rows[-1]
            sig = "*" if (lo_ci > 0 or hi_ci < 0) else " "
            print(f"  k={k:<3} {lo:>4}-{hi if hi<10**4 else 'inf':<4} min  n={n:3d}   "
                  f"offset {r['offset']:5.2f} | uncal {r['uncalibrated']:5.2f} | "
                  f"benefit {r['benefit']:+5.2f} [{lo_ci:+5.2f},{hi_ci:+5.2f}] {sig}")

    json.dump({"rows": rows, "n_train_cases": len(tr_cases), "n_test_cases": len(te_cases),
               "n_segments": int(len(y)), "bins_min": BINS_MIN, "ks": KS},
              open(DATA / "calibration_drift.json", "w"), indent=2)
    print(f"\n[done] {DATA / 'calibration_drift.json'}")


if __name__ == "__main__":
    main()
