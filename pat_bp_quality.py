"""pat_bp_quality.py -- does the within-patient PAT -> BP law get clearer on clean data?

In the raw data ~74% of patients show the textbook sign (longer PAT, lower DBP). This asks how
much of the remaining 26% is measurement error. Filters are fixed in advance, judge only signal
quality and how much BP varies (never the PAT-BP relation itself), and every level is reported.

Levels (cumulative):
  L0 all          every segment with a measurable PAT
  L1 clean beats  >= 80% of PPG feet agree with the segment delay, beat-to-beat delay SD
                  <= 10 ms, and the ECG R-peak count matches the arterial-line beat count (+-1)
  L2 ABP agrees   the segment's PPG-minus-ABP delay lies within 25 ms of the patient's median
                  offset: the PPG landmark moves with the arterial line, so the beat is not
                  mispaired (the arterial line is aligned with the ECG in VitalDB)
  L3 BP varies    patients with >= 8 segments and a DBP range >= 10 mmHg -- without BP change
                  there is no law to see

For each level: share of patients with a negative slope, its sign-test p-value, median slope,
and the noise ceiling -- the share expected if every patient had exactly the median slope and
only sampling noise moved the estimates. Slopes by least squares and by Theil-Sen (robust to
residual outliers). Repeated with arterial-line PAT (ECG -> ABP foot), which has no PPG offset.

Runs on VitalDB (surgical, 10 s segments) or MIMIC-BP (ICU, 30 s segments, --dataset mimic),
and also reports the ECG->PPG timing offset measured against the arterial line.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

import mechlib
from mechlib import ECG, PPG, ABP

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def segment_metrics(e, p, a, fs):
    """Per-segment PAT from PPG and from ABP, with the quality measures the filters use."""
    r = mechlib.r_peaks(e, fs)
    fp = mechlib.ppg_feet(p, fs); fa = mechlib.ppg_feet(a, fs)
    _, _, dp = mechlib.pair_beats(r, fp, fs)
    _, _, da = mechlib.pair_beats(r, fa, fs)
    nan = float("nan")
    return {
        "ppg_pat": float(np.median(dp)) * 1000 if len(dp) >= 3 else nan,
        "ppg_frac": len(dp) / max(len(fp), 1),
        "ppg_sd": float(np.std(dp)) * 1000 if len(dp) >= 3 else nan,
        "abp_pat": float(np.median(da)) * 1000 if len(da) >= 3 else nan,
        "abp_sd": float(np.std(da)) * 1000 if len(da) >= 3 else nan,
        "r_match": abs(len(r) - len(fa)) <= 1,
    }


def patient_slopes(x, y, g, min_seg=5):
    out = []
    for s in np.unique(g):
        m = (g == s) & np.isfinite(x)
        if m.sum() < min_seg or np.ptp(x[m]) < 5:
            continue
        lr = stats.linregress(x[m], y[m])
        ts = stats.theilslopes(y[m], x[m])[0]
        out.append((lr.slope, lr.stderr, ts))
    return np.array(out).reshape(-1, 3)


def summarise(S):
    sl, se, ts = S.T
    k = int((sl < 0).sum()); n = len(sl)
    ceiling = float(np.nanmean(stats.norm.cdf(-np.median(sl) / np.where(se > 0, se, np.nan))))
    return {"n_patients": n, "frac_negative": k / n, "p_sign": float(stats.binomtest(k, n, .5).pvalue),
            "median_slope": float(np.median(sl)), "noise_ceiling": ceiling,
            "frac_negative_theilsen": float(np.mean(ts < 0)), "median_theilsen": float(np.median(ts))}


def _metrics_chunk(args):
    X, fs = args
    return [segment_metrics(x[:, ECG], x[:, PPG], x[:, ABP], fs) for x in X]


def load(dataset, mimic_root, max_patients=None):
    """(X (N, L, 3) = [ECG, PPG, ABP], Y (N, 2) = [SBP, DBP], patient ids, fs)."""
    if dataset == "vitaldb":                    # 10 s segments, surgical (SNUH)
        d = mechlib.load_mini(str(DATA / "vitaldb_mini_deep.npz"))
        X = np.concatenate([d[k] for k in ("Xtr", "Xva", "Xte")]).astype(float)
        Y = np.concatenate([d[k] for k in ("ytr", "yva", "yte")])
        g = np.concatenate([d[k] for k in ("gtr", "gva", "gte")])
        return X, Y, g, int(d["fs"])
    import physics_audit as pa                  # 30 s segments, ICU (BIDMC), MIMIC-BP layout
    m = pa.load_mimic_bp(mimic_root, channels=("ecg", "ppg", "abp"), max_patients=max_patients)
    return m["X"].astype(float), m["y"], m["g"], 125


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["vitaldb", "mimic"], default="vitaldb")
    ap.add_argument("--mimic-root", default=str(ROOT.parent / "BP"))
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    ap.add_argument("--max-patients", type=int, default=0, help="MIMIC only: random subset; 0 = all")
    args = ap.parse_args()

    X, Y, g, fs = load(args.dataset, args.mimic_root, args.max_patients or None)
    print(f"[data] {args.dataset}: {len(X)} segments of {X.shape[1] / fs:.0f} s, "
          f"{len(np.unique(g))} patients", flush=True)
    chunks = [(X[i:i + 500], fs) for i in range(0, len(X), 500)]
    with ProcessPoolExecutor(args.workers) as ex:
        M = [m for part in ex.map(_metrics_chunk, chunks) for m in part]
    col = lambda k: np.array([m[k] for m in M], float)
    ppg, abp = col("ppg_pat"), col("abp_pat")
    frac, psd, rmatch = col("ppg_frac"), col("ppg_sd"), col("r_match").astype(bool)

    keep = {"L0 all": np.isfinite(ppg)}
    keep["L1 clean beats"] = keep["L0 all"] & (frac >= 0.8) & (psd <= 10) & rmatch
    off = ppg - abp
    agree = np.zeros(len(X), bool)
    for s in np.unique(g):
        m = (g == s) & keep["L1 clean beats"] & np.isfinite(off)
        if m.sum():
            agree[m] = np.abs(off[m] - np.median(off[m])) <= 25
    keep["L2 ABP agrees"] = keep["L1 clean beats"] & agree
    varied = np.zeros(len(X), bool)
    for s in np.unique(g):
        m = (g == s) & keep["L2 ABP agrees"]
        if m.sum() >= 8 and np.ptp(Y[m, 1]) >= 10:
            varied[m] = True
    keep["L3 BP varies"] = keep["L2 ABP agrees"] & varied

    # channel timing: ECG and ABP share a clock in both datasets, so PPG-minus-ABP delay well
    # beyond radial-to-finger transit (~10-50 ms) is an instrumental PPG offset
    ok = np.isfinite(ppg) & np.isfinite(abp)
    q = np.percentile(off[ok], [25, 50, 75])
    res = {"dataset": args.dataset, "timing_ms": {
        "ppg_pat_median": float(np.median(ppg[ok])), "abp_pat_median": float(np.median(abp[ok])),
        "ppg_minus_abp_median": float(q[1]), "ppg_minus_abp_iqr": [float(q[0]), float(q[2])]}}
    print(f"[timing] ECG->PPG foot {np.median(ppg[ok]):.0f} ms, ECG->ABP foot {np.median(abp[ok]):.0f} ms, "
          f"PPG minus ABP {q[1]:.0f} ms (IQR {q[0]:.0f}-{q[2]:.0f})", flush=True)
    print(f"{'level':16s} {'segs':>6s} | {'source':4s} {'BP':3s} {'pts':>4s} {'neg':>5s} {'ceil':>5s} "
          f"{'p':>8s} {'median':>8s} {'TS neg':>6s}")
    for lev, m in keep.items():
        res[lev] = {"segments": int(m.sum())}
        for src, x in (("PPG", ppg), ("ABP", abp)):
            for bi, bp in ((1, "DBP"), (0, "SBP")):
                xx = np.where(m, x, np.nan)
                r = summarise(patient_slopes(xx, Y[:, bi], g))
                res[lev][f"{src}_{bp}"] = r
                print(f"{lev:16s} {m.sum():6d} | {src:4s} {bp:3s} {r['n_patients']:4d} "
                      f"{r['frac_negative']:5.0%} {r['noise_ceiling']:5.0%} {r['p_sign']:8.1e} "
                      f"{r['median_slope']:+8.3f} {r['frac_negative_theilsen']:6.0%}", flush=True)
    out = DATA / f"pat_bp_quality_{args.dataset}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[done] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
