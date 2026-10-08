"""vascage_construct.py -- is "vascular age" from PPG actually vascular?

Every PPG age-prediction paper reports MAE against chronological age and then calls the
residual (predicted - chronological) a vascular age. That step is an assumption, not a
result: if the residual is just model error, the same numbers appear.

Testing it needs a measured stiffness reference on the same subjects, which is why it is
almost never done. VitalDB has one -- an arterial line -- and vitaldb_raw.py has already
decomposed arrival time on 200 cases into

    R --[PEP + proximal]--> ART foot --[PTT, distal arterial]--> PPG foot

PTT is the part that carries arterial stiffness. So the test is:

    does the age residual track MEASURED distal transit, after chronological age is
    partialled out?

PEP is the built-in negative control. It is cardiac timing, so a residual that tracks PEP as
strongly as PTT is not measuring vessels.

    python vascage_construct.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

import features_full as ff
import mechlib
import run_feature_full_stream as rfs
import convert_pulsedb as cp

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def partial_corr(x, y, z):
    """corr(x, y) with z regressed out of both."""
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[keep], y[keep], z[keep]
    Z = np.c_[np.ones(len(z)), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(x))


def boot_ci(fn, n, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    raw = np.load(DATA / "vitaldb_raw_pep.npz")
    X, g, pep, ptt = raw["X"], raw["g"].astype(int), raw["pep"], raw["ptt"]
    clin = pd.read_csv(DATA / "_vitaldb_clinical.csv").set_index("caseid")

    print(f"[1/4] extracting features on {len(X)} raw VitalDB segments ...", flush=True)
    F = ff.compute_full(mechlib.normalize(np.asarray(X, np.float64)), 125,
                        ppg_ch=1, ecg_ch=0)

    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr = blob["Ftr"]
    keys = sorted(k for k in Ftr
                  if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.5
                  and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9 and k in F)

    # ---- per case: median features, median measured PEP/PTT, chronological age
    cases = np.unique(g)
    M, AGE, PEPc, PTTc = [], [], [], []
    for c in cases:
        m = g == c
        with np.errstate(all="ignore"):
            M.append([np.nanmedian(np.asarray(F[k], float)[m]) for k in keys])
        AGE.append(float(clin.loc[c, "age"]))
        PEPc.append(float(np.median(pep[m]))); PTTc.append(float(np.median(ptt[m])))
    M, AGE = np.array(M, float), np.array(AGE)
    PEPc, PTTc = np.array(PEPc), np.array(PTTc)

    print(f"[2/4] training the age model on PulseDB train subjects ...", flush=True)
    _, _, ids_all, _ = rfs.read_meta_cached(cp.DB / "VitalDB_Train_Subset.mat")
    tr_mask, base = rfs.train_mask_like_convert(ids_all)
    sub_tr, age_tr = base[tr_mask], np.asarray(blob["age_tr"], float)
    subs = np.unique(sub_tr)
    Mtr = np.empty((len(subs), len(keys))); Atr = np.empty(len(subs))
    for i, s in enumerate(subs):
        mm = sub_tr == s
        with np.errstate(all="ignore"):
            Mtr[i] = [np.nanmedian(np.asarray(Ftr[k], float)[mm]) for k in keys]
        Atr[i] = np.nanmedian(age_tr[mm])

    med = np.nanmedian(Mtr, 0)
    Mtr = np.where(np.isfinite(Mtr), Mtr, med)
    M = np.where(np.isfinite(M), M, med)
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-9
    model = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit((Mtr - mu) / sd, Atr)

    print(f"[3/4] scoring {len(cases)} raw VitalDB cases ...", flush=True)
    pred = model.predict((M - mu) / sd)
    resid = pred - AGE
    mae = float(np.abs(resid).mean())
    r_age = float(np.corrcoef(pred, AGE)[0, 1])
    print(f"  transfer to raw VitalDB: MAE {mae:.2f} y   r(pred, chronological) = {r_age:+.3f}")

    print("\n[4/4] does the residual track MEASURED arterial transit?\n")
    out = {"_meta": {"n_cases": int(len(cases)), "transfer_mae": mae, "r_pred_age": r_age,
                     "n_features": len(keys)}}

    # physiological sanity: does measured transit itself track age?
    for nm, v in (("measured PTT (distal arterial)", PTTc), ("measured PEP (cardiac+proximal)", PEPc)):
        r = float(np.corrcoef(v, AGE)[0, 1])
        lo, hi = boot_ci(lambda i: np.corrcoef(v[i], AGE[i])[0, 1], len(AGE))
        print(f"  r({nm:32}, chronological age) = {r:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]")
        out[f"r_age_{nm.split()[1]}"] = {"r": r, "ci": [lo, hi]}

    print()
    for nm, v in (("PTT", PTTc), ("PEP", PEPc)):
        r, n = partial_corr(resid, v, AGE)
        lo, hi = boot_ci(lambda i: partial_corr(resid[i], v[i], AGE[i])[0], len(AGE))
        verdict = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
        print(f"  partial r(age residual, measured {nm} | chronological age) = {r:+.3f}"
              f"  95%CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        out[f"partial_resid_{nm}"] = {"r": r, "ci": [lo, hi], "n": n}

    json.dump(out, open(DATA / "vascular_age_construct.json", "w"), indent=2)
    print(f"\n[done] {DATA / 'vascular_age_construct.json'}")


if __name__ == "__main__":
    main()
