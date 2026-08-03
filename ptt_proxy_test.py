"""ptt_proxy_test.py -- which morphology features are actually proxies for PTT?

The question this answers
-------------------------
Every gradient-boosted arm leans on the same morphology features -- ppg_p10, ppg_skew_g, dw10,
t_c -- and none of them is an arrival time. That leaves two readings, and they have opposite
implications for faithfulness:

  (a) those features are PROXIES for pulse transit time. The model is then respecting the
      governing law through a different measurement channel, and is faithful in substance even
      though the audit's roll intervention does not register it.
  (b) they are unrelated to transit time and the model is exploiting some other correlate of BP,
      in which case the mechanism claim fails.

This distinguishes them by measuring, per feature, how much of the measured PTT it can recover:

  r_ptt        Spearman correlation with measured PTT, WITHIN subject. Within-subject is the
               right scale: between subjects, anything correlated with body size correlates with
               PTT for reasons that have nothing to do with pressure.
  R2_ptt       cross-validated R^2 of predicting PTT from that feature alone.
  partial      correlation with BP after PTT is regressed out. A feature that keeps its BP
               relationship once PTT is removed is carrying non-PTT information.

A genuine PTT proxy has high r_ptt and LOW partial correlation with BP -- it predicts BP because
it tracks transit time. A feature with low r_ptt and high partial is predicting BP some other way.

Also fits a model to predict PTT from morphology alone, since a wearable without ECG can only
respect the arrival-time law through such a reconstruction.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from gbm_mechanism import plain

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = 1
MIN_SEG = 40


def within_subject_r(x, y, groups, min_seg=MIN_SEG):
    """Median within-subject Spearman correlation, which removes between-subject confounding."""
    rs = []
    for s in np.unique(groups):
        m = groups == s
        if m.sum() < min_seg:
            continue
        a, b = x[m], y[m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_seg // 2 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            continue
        rs.append(stats.spearmanr(a[ok], b[ok]).statistic)
    return float(np.nanmedian(rs)) if rs else np.nan, len(rs)


def main():
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Fte, yte = full["Fte"], full["yte"]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    g = d["gte"][:len(yte)]
    dbp = yte[:, TARGET]

    # measured PTT is the reference. It is only valid on ~44% of segments, so every comparison
    # below is restricted to those; that is a limitation of the estimator, not of the features.
    ptt = np.asarray(Fte.get("pat_foot", Fte.get("pat")), float)
    ok = np.isfinite(ptt)
    print(f"[ptt] measured on {100*ok.mean():.0f}% of test segments "
          f"({ok.sum()} of {len(ptt)})", flush=True)

    keys = [k for k in Fte if k not in ("pat", "pat_foot", "pat_peak", "ptt_var")
            and np.isfinite(np.asarray(Fte[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Fte[k], float)) > 1e-9]

    rows = []
    for k in keys:
        x = np.asarray(Fte[k], float)
        m = ok & np.isfinite(x)
        if m.sum() < 2000:
            continue
        r_ptt, n_sub = within_subject_r(x[m], ptt[m], g[m])
        r_bp, _ = within_subject_r(x[m], dbp[m], g[m])
        # partial correlation of feature with BP, controlling for PTT (within subject, pooled)
        try:
            xz = x[m] - Ridge(alpha=1.0).fit(ptt[m, None], x[m]).predict(ptt[m, None])
            bz = dbp[m] - Ridge(alpha=1.0).fit(ptt[m, None], dbp[m]).predict(ptt[m, None])
            partial = float(stats.spearmanr(xz, bz).statistic)
        except Exception:
            partial = np.nan
        # how much of PTT does this one feature explain?
        sub = np.random.default_rng(0).choice(np.where(m)[0], min(20000, m.sum()), replace=False)
        try:
            r2 = float(np.mean(cross_val_score(Ridge(alpha=1.0), x[sub, None], ptt[sub],
                                               cv=3, scoring="r2")))
        except Exception:
            r2 = np.nan
        rows.append({"feature": k, "plain": plain(k), "r_ptt": r_ptt, "r_bp": r_bp,
                     "R2_ptt": r2, "partial_bp": partial, "n_subj": n_sub})

    rows.sort(key=lambda r: -abs(r["r_ptt"]) if np.isfinite(r["r_ptt"]) else 0)
    print(f"\n{'feature':16s} {'r(PTT)':>8s} {'R2(PTT)':>8s} {'r(BP)':>7s} "
          f"{'partial':>8s}  what it is")
    print("-" * 96)
    for r in rows[:18]:
        print(f"{r['feature']:16s} {r['r_ptt']:+8.3f} {r['R2_ptt']:8.3f} {r['r_bp']:+7.3f} "
              f"{r['partial_bp']:+8.3f}  {r['plain'][:44]}")

    # the features the models actually use
    print("\n[used] the features the GBM arms lean on:")
    for k in ("ppg_p10", "ppg_skew_g", "dw10", "t_c", "vpg_min", "t_e"):
        r = next((x for x in rows if x["feature"] == k), None)
        if r:
            verdict = ("PTT proxy" if abs(r["r_ptt"]) > 0.3 and abs(r["partial_bp"]) < 0.15
                       else "NOT a PTT proxy" if abs(r["r_ptt"]) < 0.15
                       else "partial proxy")
            print(f"  {k:14s} r(PTT) {r['r_ptt']:+.3f}  partial-BP {r['partial_bp']:+.3f}  "
                  f"-> {verdict}")

    # can morphology reconstruct PTT at all? (the wearable question)
    morph = [r["feature"] for r in rows if np.isfinite(r["r_ptt"])]
    M = np.column_stack([np.asarray(Fte[k], float) for k in morph])
    mm = ok & np.isfinite(M).all(1)
    if mm.sum() > 5000:
        sub = np.random.default_rng(0).choice(np.where(mm)[0], min(30000, mm.sum()),
                                              replace=False)
        import lightgbm as lgb
        sc = cross_val_score(lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                               num_leaves=31, verbosity=-1),
                             M[sub], ptt[sub], cv=3, scoring="r2")
        print(f"\n[recon] predicting measured PTT from all {len(morph)} morphology features: "
              f"R2 = {np.mean(sc):.3f} (3-fold)")
        print("        A wearable without ECG can only honour the arrival-time law through such")
        print("        a reconstruction, so this number bounds how faithful a PPG-only model")
        print("        can be to that law.")

    (DATA / "ptt_proxy_test.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n[done] data/ptt_proxy_test.json")


if __name__ == "__main__":
    main()
