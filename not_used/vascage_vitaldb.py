"""vascage_vitaldb.py -- can PPG morphology predict age, on VitalDB's own 144 CalFree patients?

The vascular-age question, asked with the data we actually have. fig_vascular_age.py needs
data/vascage_train.csv, which does not exist on this machine; but the PulseDB CalFree test
subset ships Age, Sex, BMI, Height and Weight for all 144 held-out patients, so panels a-c of
that figure can be answered directly from it.

  Age is a per-SUBJECT constant, so the unit of analysis is the subject, not the segment.
  Features are extracted per segment then median-aggregated to one row per patient, giving a
  144 x X design matrix. Cross-validation is therefore over subjects by construction, and no
  leakage is possible.

What this can and cannot say is in the printout: 144 subjects is small, and these are
anaesthetised operating-room recordings, so a null here is not a verdict on the idea.

    python vascage_vitaldb.py                 # 20 segments per patient
    python vascage_vitaldb.py --per-subj 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

import convert_pulsedb as cp
import features_full as ff

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "_vascage_cache.npz"
LIT = (6.0, 9.0)          # published PPG age-prediction MAE band, dedicated finger sensors


def build_cache(per_subj, seed=0):
    """Parse the CalFree .mat once, aggregate to one feature row per patient, cache it."""
    X, y, ids, demo = cp.parse_mat(cp.DB / "VitalDB_CalFree_Test_Subset.mat")
    ids = np.array([s.rsplit("_", 1)[0] for s in ids])
    age, sex = np.asarray(demo["age"], float), np.asarray(demo["sex"], float)
    bmi, hgt = np.asarray(demo["bmi"], float), np.asarray(demo["height"], float)

    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for p in np.unique(ids):
        idx = np.where(ids == p)[0]
        if len(idx) > per_subj:
            idx = rng.choice(idx, per_subj, replace=False)
        F = ff.compute_full(X[idx].astype(np.float64), 125, ppg_ch=1, ecg_ch=0)
        keys = sorted(F)
        with np.errstate(all="ignore"):
            rows.append([np.nanmedian(np.asarray(F[k], float)) for k in keys])
        meta.append((age[idx[0]], sex[idx[0]], bmi[idx[0]], hgt[idx[0]]))
        print(f"  {p}  n={len(idx):3d}  age={age[idx[0]]:.0f}", flush=True)

    M = np.array(rows, float)
    meta = np.array(meta, float)
    np.savez_compressed(CACHE, M=M, keys=np.array(keys), meta=meta)
    return M, keys, meta


def cv_mae(Xf, yv, kind, repeats=20):
    """Repeated 5-fold CV over SUBJECTS. Returns per-repeat MAE, so a spread is reportable."""
    out = []
    for r in range(repeats):
        pred = np.zeros(len(yv))
        for tr, te in KFold(5, shuffle=True, random_state=r).split(Xf):
            if kind == "mean":
                pred[te] = yv[tr].mean()
                continue
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
            A, B = (Xf[tr] - mu) / sd, (Xf[te] - mu) / sd
            if kind == "ridge":
                pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(A, yv[tr]).predict(B)
            else:
                import lightgbm as lgb
                # n_jobs=1: multithreaded LightGBM segfaults on this macOS/libomp build
                pred[te] = lgb.LGBMRegressor(
                    n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                    subsample=0.8, colsample_bytree=0.6, random_state=r,
                    n_jobs=1, verbosity=-1).fit(A, yv[tr]).predict(B)
        out.append(np.abs(pred - yv).mean())
    return np.array(out), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subj", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild:
        z = np.load(CACHE, allow_pickle=True)
        M, keys, meta = z["M"], list(z["keys"]), z["meta"]
        print(f"[cache] {CACHE.name}  {M.shape}")
    else:
        M, keys, meta = build_cache(args.per_subj)

    age, sex, bmi, hgt = meta.T

    # stiffness_index = height / delta_t: the one index with the dimensions of a velocity, so
    # a direct PWV proxy. compute_full returns it as NaN (height is per subject, not per
    # segment), so it is reconstructed here now that height is in hand.
    if "delta_t" in keys:
        dt = M[:, keys.index("delta_t")]
        with np.errstate(all="ignore"):
            M = np.c_[M, hgt / np.where(dt > 1e-6, dt, np.nan)]
        keys = keys + ["stiffness_index"]

    keep = [i for i, k in enumerate(keys)
            if np.isfinite(M[:, i]).mean() > 0.8 and np.nanstd(M[:, i]) > 1e-9]
    Xf = M[:, keep]
    col_med = np.nanmedian(Xf, 0)
    Xf = np.where(np.isfinite(Xf), Xf, col_med)
    ok = np.isfinite(age)
    Xf, age_v = Xf[ok], age[ok]

    print(f"\n{len(age_v)} patients | {Xf.shape[1]} usable features "
          f"| age {age_v.min():.0f}-{age_v.max():.0f}, median {np.median(age_v):.0f}, "
          f"SD {age_v.std():.1f}")

    res, oof = {}, {}
    for kind, label in (("mean", "predict the mean age (baseline)"),
                        ("ridge", "ridge on PPG features"),
                        ("gbm", "LightGBM on PPG features")):
        maes, pred = cv_mae(Xf, age_v, kind)
        oof[kind] = pred
        res[label] = {"mae_mean": float(maes.mean()), "mae_sd": float(maes.std()),
                      "r": float(np.corrcoef(pred, age_v)[0, 1]) if kind != "mean" else 0.0}
        print(f"  {label:34} MAE {maes.mean():5.2f} +/- {maes.std():.2f} y"
              f"   r = {res[label]['r']:+.3f}")

    base = res["predict the mean age (baseline)"]["mae_mean"]
    best = min(res[k]["mae_mean"] for k in res if "baseline" not in k)
    print(f"\nbest model beats the mean-age baseline by {base - best:.2f} years")

    # The +/- above is CV-SPLIT noise on one fixed set of 144 patients, so it says nothing
    # about generalising to new patients. Bootstrap the subjects themselves for that.
    oof_base = np.abs(oof["mean"] - age_v)
    oof_best = np.abs(oof["gbm" if res["LightGBM on PPG features"]["mae_mean"] <= res[
        "ridge on PPG features"]["mae_mean"] else "ridge"] - age_v)
    rng = np.random.default_rng(0)
    diffs = [np.mean(oof_base[b] - oof_best[b])
             for b in (rng.integers(0, len(age_v), len(age_v)) for _ in range(4000))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res["_gain_over_baseline"] = {"years": float(base - best), "ci95": [float(lo), float(hi)],
                                  "note": "bootstrap over the 144 subjects, not over CV splits"}
    print(f"  bootstrap over subjects: {base - best:.2f} y  95% CI [{lo:.2f}, {hi:.2f}]"
          f"   {'excludes 0' if lo > 0 else 'INCLUDES 0 -- not distinguishable from no signal'}")
    print(f"published band for dedicated finger PPG: {LIT[0]}-{LIT[1]} y MAE")
    print("caveats: 144 anaesthetised OR patients, one clinical PPG channel, no external "
          "validation. A null here is not a verdict on the idea.")

    res["_meta"] = {"n_patients": int(len(age_v)), "n_features": int(Xf.shape[1]),
                    "age_median": float(np.median(age_v)), "age_sd": float(age_v.std()),
                    "lit_band": LIT, "source": "PulseDB VitalDB CalFree test subset"}
    out = DATA / "vascular_age_vitaldb.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
