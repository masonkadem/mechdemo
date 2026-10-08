"""calib_gbm_finetune.py -- LightGBM under the FINE-TUNING calibration protocol, with HR and
demographics added, against the transformer.

The population model is trained once on the training subjects; per-patient calibration then fits
a small head on the model's OUTPUT plus whatever extra per-segment signals we allow. Two ways to
calibrate a tree model, both reported, because they answer different questions:

  offset      add one scalar to the population prediction   (1 parameter)
  linear      fit [prediction, extra features] -> DBP       (2 + n parameters)

Demographics get their own arm even though age/sex/BMI are CONSTANT within a patient. That is
the point of including them: under CalBased (same patients in train and test) demographics
looked valuable, but a per-patient head cannot use a constant, so the honest question is whether
they help the POPULATION model instead.

    python calib_gbm_finetune.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GAP = 50
KS = [2, 3, 5, 10, 20, 50, 100]
TARGET = 1                                   # DBP
PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
              colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)


def fit_head(A, yc, B, alpha=10.0):
    """Per-patient ridge, alpha scaled by features-per-sample (same rule as every other arm)."""
    if len(A) < 2:
        return np.full(len(B), yc.mean())
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-6
    if not keep.any():
        return np.full(len(B), yc.mean())
    A, B, mu, sd = A[:, keep], B[:, keep], mu[keep], sd[keep]
    return Ridge(alpha=alpha * max(1.0, A.shape[1] / len(A))
                 ).fit((A - mu) / sd, yc).predict((B - mu) / sd)


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, y = np.array(d["gte"]), np.array(d["yte"])[:, TARGET]
    rows = {p: np.where(g == p)[0] for p in np.unique(g)}

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte = full["Ftr"], full["ytr"][:, TARGET], full["Fte"]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    ntr, nte = len(ytr), len(y)
    Mtr = np.column_stack([np.asarray(Ftr[k], float)[:ntr] for k in keys])
    Mte = np.column_stack([np.asarray(Fte.get(k, np.full(nte, np.nan)), float)[:nte]
                           for k in keys])
    med = gbm.column_medians(Mtr)
    Mtr, Mte = gbm._impute(Mtr, med), gbm._impute(Mte, med)

    Dtr = np.c_[np.array(d["age_tr"])[:ntr], np.array(d["sex_tr"])[:ntr],
                np.array(d["bmi_tr"])[:ntr]]
    Dte = np.c_[np.array(d["age_te"]), np.array(d["sex_te"]), np.array(d["bmi_te"])]

    HR = np.load(DATA / "_calib_hr.npy")
    hr = HR.copy()
    for i in rows.values():                       # fill gaps with the patient's own median
        ok = np.isfinite(HR[i])
        hr[i] = np.where(ok, HR[i], np.median(HR[i][ok]) if ok.any() else 0.0)

    print(f"[data] {len(keys)} waveform features, {ntr:,} train / {nte:,} test segments,"
          f" {len(rows)} held-out patients", flush=True)

    # ---- population models --------------------------------------------------
    variants = {}
    for name, Atr, Ate in (("waveform", Mtr, Mte),
                           ("waveform + demo", np.c_[Mtr, Dtr], np.c_[Mte, Dte])):
        m = lgb.LGBMRegressor(**PARAMS).fit(Atr, ytr)
        variants[name] = m.predict(Ate)
        k0 = float(np.median([np.abs(variants[name][i] - y[i]).mean() for i in rows.values()]))
        print(f"[k=0 population] LightGBM {name:16s} {k0:.2f} mmHg", flush=True)

    # ---- per-patient calibration -------------------------------------------
    res = {}

    def curve(build):
        out = {}
        for k in KS:
            e = []
            for i in rows.values():
                if k + GAP >= len(i) - 5:
                    continue
                c, t = i[:k], i[k + GAP:]
                A, B = build(c), build(t)
                e.append(float(np.abs(fit_head(A, y[c], B) - y[t]).mean()))
            out[k] = float(np.median(e))
        return out

    for name, pred in variants.items():
        # offset only: one scalar per patient
        off = {}
        for k in KS:
            e = [float(np.abs(pred[i[k + GAP:]] + (y[i[:k]] - pred[i[:k]]).mean()
                              - y[i[k + GAP:]]).mean())
                 for i in rows.values() if k + GAP < len(i) - 5]
            off[k] = float(np.median(e))
        res[f"LightGBM {name} · offset"] = off
        res[f"LightGBM {name} · linear"] = curve(lambda idx, p=pred: p[idx][:, None])
        res[f"LightGBM {name} · linear + HR"] = curve(
            lambda idx, p=pred: np.c_[p[idx], hr[idx]])

    # references
    res["cuff average"] = {k: float(np.median(
        [np.abs(y[i[:k]].mean() - y[i[k + GAP:]]).mean()
         for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    g50 = {r["k"]: r for r in json.load(open(DATA / "calibration_cost.json")) if r["gap"] == 50}
    res["transformer + fine-tuned head"] = {k: g50[k]["head"] for k in KS if k in g50}

    hdr = f"\n{'arm':38s} " + " ".join(f"{'k=' + str(k):>7s}" for k in KS)
    print(hdr); print("-" * len(hdr))
    for nm, c in res.items():
        print(f"{nm:38s} " + " ".join(f"{c.get(k, float('nan')):7.2f}" for k in KS))

    json.dump(res, open(DATA / "calib_gbm_finetune.json", "w"), indent=2)
    print(f"\n[done] data/calib_gbm_finetune.json")


if __name__ == "__main__":
    main()
