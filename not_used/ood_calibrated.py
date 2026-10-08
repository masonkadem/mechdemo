"""ood_calibrated.py -- does the physics feature's OOD advantage survive calibration?

gbm_constrained.json contains a result that the full-capacity audit hides. With a SINGLE
TREE, arrival time alone generalises far better than morphology:

    single tree, PAT only        ID 9.34   OOD 10.96
    single tree, no shortcuts    ID 8.88   OOD 13.49

Worse in-distribution, 2.5 mmHg better out of it. Give the model capacity and that reverses --
at 800 trees PAT-only is worst everywhere -- because capacity lets it fit dataset-specific
morphology that does not transfer.

The cell nobody has filled is OOD *with* per-patient calibration. It matters because a
deployed device is always out of distribution relative to its training set, and it is always
calibrated. If the physics feature keeps its advantage once calibrated, that is the argument
for building around it; if calibration erases the advantage, it is not.

                     ID      OOD      calibrated ID    calibrated OOD
    PAT only        9.34    10.96         6.33              ?
    morphology      8.88    13.49         5.92              ?

    python ood_calibrated.py
"""
import json
import pickle
from pathlib import Path

import numpy as np

import convert_pulsedb as cp
import features_full as ff
import mechlib
import run_feature_full_stream as rfs
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "_aami_feats.pkl"
GAP, K = 50, 20
PAT_KEYS = ["pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"]
RATE_KEYS = ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "r_count", "hrv_lfhf"]


def load_aami():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    X, y, ids, demo = cp.parse_mat(cp.DB / "VitalDB_AAMI_Test_Subset.mat")
    Xn = mechlib.normalize(np.asarray(X, np.float64)[:, :, [ECG, PPG]])
    print(f"[aami] extracting {len(Xn)} segments ...", flush=True)
    F = ff.compute_full(Xn, 125, ppg_ch=1, ecg_ch=0)
    g = np.array([hash(s) % (10 ** 9) for s in ids])
    blob = (F, y[:, 1], g)
    pickle.dump(blob, open(CACHE, "wb"), protocol=4)
    return blob


def offset_curve(pred, y, rows, k=K, gap=GAP):
    """Per-patient offset from the first k segments, scored after a gap."""
    out = []
    for idx in rows.values():
        if k + gap >= len(idx) - 5:
            continue
        cal, test = idx[:k], idx[k + gap:]
        off = float(np.mean(y[cal] - pred[cal]))
        out.append(float(np.abs(pred[test] + off - y[test]).mean()))
    return float(np.median(out)), len(out)


def uncal(pred, y, rows):
    return float(np.median([np.abs(pred[i] - y[i]).mean() for i in rows.values()]))


def main():
    import lightgbm as lgb

    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = blob["Ftr"], np.asarray(blob["ytr"], float)[:, 1]
    Fa, ya, ga = load_aami()

    shared = sorted(set(Ftr) & set(Fa))
    pat = [k for k in shared if k in PAT_KEYS]
    morph = [k for k in shared if k not in PAT_KEYS and k not in RATE_KEYS]
    print(f"[feat] shared {len(shared)} | PAT-only {len(pat)} | morphology {len(morph)}")

    rows = {p: np.where(ga == p)[0] for p in np.unique(ga)}
    print(f"[aami] {len(ya)} segments, {len(rows)} subjects\n")

    def build(keys, src, n):
        M = np.column_stack([np.asarray(src[k], float)[:n] for k in keys])
        return M

    res = {}
    for name, keys in (("PAT only", pat), ("morphology", morph),
                       ("morphology + PAT", morph + pat)):
        Mtr = build(keys, Ftr, len(ytr))
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        Ma = np.where(np.isfinite(build(keys, Fa, len(ya))), build(keys, Fa, len(ya)), med)

        for cap, cfg in (("single tree", dict(n_estimators=1, learning_rate=1.0, num_leaves=32)),
                         ("800 trees", dict(n_estimators=800, learning_rate=0.03,
                                            num_leaves=63, subsample=0.8,
                                            colsample_bytree=0.8, min_child_samples=50))):
            m = lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1, **cfg).fit(Mtr, ytr)
            pred = m.predict(Ma)
            u = uncal(pred, ya, rows)
            c, n = offset_curve(pred, ya, rows)
            res[f"{name} | {cap}"] = {"ood_uncal": u, "ood_cal20": c, "n_subj": n,
                                      "n_feat": len(keys)}
            print(f"  {name:18} {cap:12}  OOD uncal {u:6.2f} | OOD calibrated(k=20) {c:6.2f}"
                  f"  ({n} subjects)")

    print("\n--- does the physics feature keep its OOD advantage once calibrated? ---")
    for cap in ("single tree", "800 trees"):
        p = res[f"PAT only | {cap}"]; mm = res[f"morphology | {cap}"]
        print(f"  {cap:12}  uncalibrated: PAT {p['ood_uncal']:.2f} vs morph "
              f"{mm['ood_uncal']:.2f}  -> PAT {mm['ood_uncal']-p['ood_uncal']:+.2f}")
        print(f"  {'':12}  calibrated  : PAT {p['ood_cal20']:.2f} vs morph "
              f"{mm['ood_cal20']:.2f}  -> PAT {mm['ood_cal20']-p['ood_cal20']:+.2f}")

    json.dump(res, open(DATA / "ood_calibrated.json", "w"), indent=2)
    print(f"\n[done] {DATA / 'ood_calibrated.json'}")


if __name__ == "__main__":
    main()
