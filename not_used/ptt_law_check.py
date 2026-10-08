"""ptt_law_check.py -- does the real dBP/dPTT go NEGATIVE (textbook) across a wider BP range,
or is the +38 mmHg/s a within-segment artifact?

Four diagnostics:
  1. PTT reliability: split each subject's segments in half, correlate the two PTT estimates.
     Low reliability => the slope is noise-dominated.
  2. Range dependence: bin subjects by their BP RANGE (max-min DBP). If the negative law only
     appears in wide-range subjects, the +38 is a low-range / noise effect.
  3. ACROSS-subject law: regress every segment's DBP on PTT pooled across subjects (the
     population Moens-Korteweg regime), vs the within-subject slopes.
  4. Robust per-subject: use Theil-Sen (median-of-slopes) instead of least squares, which is
     immune to the few-beat outliers that inflate an OLS slope.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes

import mechlib
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()

    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["gte"]), min(args.n, len(d["gte"])), replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    dbp = d["yte"][sel][:, 1]; grp = d["gte"][sel]
    print(f"[law] {len(X)} seg / {len(np.unique(grp))} subjects; computing PTT...")
    ptt = mechlib.compute_ptt(X, fs)
    ok = np.isfinite(ptt) & np.isfinite(dbp)
    print(f"[law] PTT valid {ok.mean():.0%}, median {np.nanmedian(ptt)*1000:.0f} ms")

    # ---- 3. ACROSS-subject (population) law ----
    pop = np.polyfit(ptt[ok], dbp[ok], 1)[0]
    print(f"\n[law] POPULATION dDBP/dPTT (all segments pooled): {pop:+.1f} mmHg/s")
    # residualize subject means to get the pure WITHIN vs BETWEEN split
    subj_mean_ptt = {g: ptt[(grp == g) & ok].mean() for g in np.unique(grp)}
    subj_mean_dbp = {g: dbp[(grp == g) & ok].mean() for g in np.unique(grp)}
    bx = np.array([subj_mean_ptt[g] for g in np.unique(grp)])
    by = np.array([subj_mean_dbp[g] for g in np.unique(grp)])
    bm = np.isfinite(bx) & np.isfinite(by)
    between = np.polyfit(bx[bm], by[bm], 1)[0]
    print(f"[law] BETWEEN-subject dDBP/dPTT (subject means): {between:+.1f} mmHg/s  "
          f"(this is the Moens-Korteweg regime: stiffer subjects = shorter PTT + higher BP)")

    # ---- 1,2,4 per-subject ----
    ols, ts, ranges, rels, spreads = [], [], [], [], []
    for g in np.unique(grp):
        m = (grp == g) & ok
        if m.sum() < 10:
            continue
        p, b = ptt[m], dbp[m]
        if p.std() < 1e-4:
            continue
        ols.append(np.polyfit(p, b, 1)[0])
        ts.append(theilslopes(b, p)[0])                 # robust slope
        ranges.append(b.max() - b.min())
        spreads.append(p.std() * 1000)                  # PTT spread in ms
        # split-half PTT reliability
        h = len(p) // 2
        rels.append(np.corrcoef(np.sort(p)[:h].mean(), np.sort(p)[h:].mean())
                    if h > 2 else np.nan)
    ols, ts, ranges, spreads = map(np.array, (ols, ts, ranges, spreads))

    print(f"\n[law] WITHIN-subject dDBP/dPTT:")
    print(f"   OLS      median {np.median(ols):+.1f}  frac<0 {np.mean(ols<0):.0%}")
    print(f"   Theil-Sen median {np.median(ts):+.1f}  frac<0 {np.mean(ts<0):.0%}  (robust)")

    # ---- 2. range dependence ----
    print(f"\n[law] within-subject slope vs BP range (does negative law need wide range?):")
    for lo, hi, lab in [(0, 15, "narrow (<15)"), (15, 30, "mid (15-30)"), (30, 999, "wide (>30)")]:
        mask = (ranges >= lo) & (ranges < hi)
        if mask.sum() >= 3:
            print(f"   DBP range {lab:14s} n={mask.sum():3d}  median slope "
                  f"OLS {np.median(ols[mask]):+.0f}  Theil {np.median(ts[mask]):+.0f}")
    # correlation slope vs PTT spread (noise proxy)
    print(f"\n[law] corr(slope, PTT spread) = {np.corrcoef(ols, spreads)[0,1]:+.2f}  "
          f"(strong negative => small-spread subjects drive the positive slope = noise)")


if __name__ == "__main__":
    main()
