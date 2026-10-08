"""vascage_split.py -- PPG age prediction with a real train/test split.

vascage_vitaldb.py could only cross-validate within the 144 CalFree test patients, because
that was the only split whose features existed. The full feature table now covers ~1,150
train subjects as well, so the fair version is available: fit on train subjects, score on the
144 held-out ones, never having seen them.

Age is a per-SUBJECT constant, so the unit of analysis is the subject. Features are
median-aggregated per subject before anything is fitted, which makes leakage structurally
impossible rather than merely avoided.

    python vascage_split.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

import convert_pulsedb as cp
import run_feature_full_stream as rfs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LIT = (6.0, 9.0)          # published PPG age-prediction MAE band, dedicated finger sensors


def per_subject(F, keys, age, sub):
    """Median-aggregate segment features to one row per subject."""
    subs = np.unique(sub)
    M = np.empty((len(subs), len(keys)), np.float64)
    A = np.empty(len(subs))
    for i, s in enumerate(subs):
        m = sub == s
        with np.errstate(all="ignore"):
            M[i] = [np.nanmedian(np.asarray(F[k], float)[m]) for k in keys]
        A[i] = np.nanmedian(age[m])
    return M, A, subs


def main():
    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, Fte = blob["Ftr"], blob["Fte"]
    keys = sorted(k for k in Ftr
                  if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.5
                  and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9)

    # train subject ids: same metadata + same split the features were extracted under
    _, _, ids_all, _ = rfs.read_meta_cached(cp.DB / "VitalDB_Train_Subset.mat")
    tr_mask, base = rfs.train_mask_like_convert(ids_all)
    sub_tr = base[tr_mask]
    n = len(np.asarray(blob["ytr"]))
    assert len(sub_tr) == n, f"train ids {len(sub_tr)} != features {n}"

    Mtr, Atr, subs_tr = per_subject(Ftr, keys, np.asarray(blob["age_tr"], float), sub_tr)
    Mte, Ate, subs_te = per_subject(Fte, keys, np.asarray(blob["age_te"], float),
                                    np.asarray(blob["gte"]))
    assert not (set(map(str, subs_tr)) & set(map(str, subs_te))), "train/test subject overlap"

    med = np.nanmedian(Mtr, 0)
    Mtr = np.where(np.isfinite(Mtr), Mtr, med)
    Mte = np.where(np.isfinite(Mte), Mte, med)
    ok_tr, ok_te = np.isfinite(Atr), np.isfinite(Ate)
    Mtr, Atr, Mte, Ate = Mtr[ok_tr], Atr[ok_tr], Mte[ok_te], Ate[ok_te]

    print(f"train {len(Atr)} subjects | test {len(Ate)} subjects | {Mtr.shape[1]} features")
    print(f"train age {Atr.min():.0f}-{Atr.max():.0f} median {np.median(Atr):.0f} sd {Atr.std():.1f}")
    print(f"test  age {Ate.min():.0f}-{Ate.max():.0f} median {np.median(Ate):.0f} sd {Ate.std():.1f}\n")

    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-9
    A, B = (Mtr - mu) / sd, (Mte - mu) / sd
    res = {}

    base_pred = np.full(len(Ate), Atr.mean())
    res["predict train mean age"] = {"mae": float(np.abs(base_pred - Ate).mean()), "r": 0.0}

    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(A, Atr)
    pr = ridge.predict(B)
    res["ridge"] = {"mae": float(np.abs(pr - Ate).mean()),
                    "r": float(np.corrcoef(pr, Ate)[0, 1])}

    import lightgbm as lgb
    preds = []
    for seed in range(3):                                  # seeds, since single runs mislead
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.6,
                              random_state=seed, n_jobs=1, verbosity=-1).fit(A, Atr)
        preds.append(m.predict(B))
    pg = np.mean(preds, 0)
    maes = [float(np.abs(p - Ate).mean()) for p in preds]
    res["LightGBM"] = {"mae": float(np.mean(maes)), "mae_sd": float(np.std(maes)),
                       "r": float(np.corrcoef(pg, Ate)[0, 1])}

    for k, v in res.items():
        sdtxt = f" +/- {v['mae_sd']:.2f}" if "mae_sd" in v else ""
        print(f"  {k:22} MAE {v['mae']:5.2f}{sdtxt} y   r = {v['r']:+.3f}")

    best = min(res[k]["mae"] for k in res if "mean age" not in k)
    gain = res["predict train mean age"]["mae"] - best
    err_b = np.abs(base_pred - Ate)
    err_m = np.abs((pg if res["LightGBM"]["mae"] <= res["ridge"]["mae"] else pr) - Ate)
    rng = np.random.default_rng(0)
    diffs = [np.mean(err_b[i] - err_m[i])
             for i in (rng.integers(0, len(Ate), len(Ate)) for _ in range(4000))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\ngain over baseline {gain:.2f} y   95% CI [{lo:.2f}, {hi:.2f}]"
          f"   {'excludes 0' if lo > 0 else 'INCLUDES 0'}")
    print(f"published band (dedicated finger PPG): {LIT[0]}-{LIT[1]} y")

    res["_meta"] = {"n_train": int(len(Atr)), "n_test": int(len(Ate)),
                    "n_features": int(Mtr.shape[1]), "gain": float(gain),
                    "gain_ci": [float(lo), float(hi)], "lit_band": LIT,
                    "design": "train on train subjects, score on 144 held-out CalFree subjects"}
    out = DATA / "vascular_age_split.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
