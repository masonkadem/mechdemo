"""mimic_ood_calibrated.py -- does the physics feature's OOD advantage survive calibration?

gbm_constrained.json holds a result the full-capacity audit hides. With a SINGLE TREE,
arrival time alone generalises far better than morphology:

    single tree, PAT only        ID 9.34   OOD 10.96
    single tree, no shortcuts    ID 8.88   OOD 13.49

Worse in-distribution, 2.5 mmHg better out of it. At 800 trees that reverses, because
capacity lets the model fit dataset-specific morphology that does not transfer.

The unfilled cell is OOD *with* per-patient calibration -- which is the only regime a real
device ever operates in: always out of distribution relative to its training set, and always
calibrated. If arrival time keeps its advantage once calibrated, that is the argument for
building an instrument around it.

OOD here is MIMIC (different hospital, different monitors), trained on PulseDB VitalDB.
MIMIC ships 30 windows of 30 s per patient; one 10 s slice is taken from the middle of each
window so that every scored segment carries its own label and no window is split across the
calibration/test boundary.

    python mimic_ood_calibrated.py --patients 600
"""
import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MIMIC = Path("/Users/masonkadem/Library/CloudStorage/OneDrive-McMasterUniversity/2026/BP")
def _cache(n):
    # "_full": the cache format changed from a single DBP column to both (SBP, DBP), so this
    # must not collide with the old _mimic_feats_{n}.pkl files -- those would load with the
    # wrong shape and either crash or, worse, silently misindex which column is which.
    return DATA / f"_mimic_feats_full_{n}.pkl"
L = 1250                      # 10 s at 125 Hz, matching PulseDB
PAT_KEYS = ["pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"]
RATE_KEYS = ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "r_count", "hrv_lfhf"]


def _one(pid):
    import numpy as np
    import features_full as ff
    import mechlib
    try:
        ecg = np.load(MIMIC / "ecg" / f"{pid}_ecg.npy")
        ppg = np.load(MIMIC / "ppg" / f"{pid}_ppg.npy")
        lab = np.load(MIMIC / "labels" / f"{pid}_labels.npy")
    except Exception:
        return None
    if ecg.ndim != 2 or ecg.shape[1] < L or len(lab) != len(ecg):
        return None
    a = (ecg.shape[1] - L) // 2                       # middle 10 s of each 30 s window
    X = np.stack([ecg[:, a:a + L], ppg[:, a:a + L]], axis=2).astype(np.float64)
    ok = np.isfinite(X).all(axis=(1, 2))
    ok &= np.isfinite(lab).all(1)
    if ok.sum() < 12:
        return None
    X, lab = X[ok], lab[ok]
    F = ff.compute_full(mechlib.normalize(X), 125, ppg_ch=1, ecg_ch=0)
    return pid, F, lab, int(ok.sum())          # lab[:,0]=SBP, lab[:,1]=DBP


def load_mimic(n_patients, workers=8):
    CACHE = _cache(n_patients)
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    pids = sorted({p.name.split("_")[0] for p in (MIMIC / "ecg").glob("*_ecg.npy")})[:n_patients]
    print(f"[mimic] {len(pids)} patients, extracting ...", flush=True)
    Fs, ys, gs = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_one, pids, chunksize=8), 1):
            if r is None:
                continue
            pid, F, y, n = r
            Fs.append(F); ys.append(y); gs.append(np.full(n, hash(pid) % (10 ** 9)))
            if i % 100 == 0:
                print(f"  {i}/{len(pids)}", flush=True)
    keys = sorted(set.intersection(*[set(f) for f in Fs]))
    F = {k: np.concatenate([np.asarray(f[k], float) for f in Fs]) for k in keys}
    blob = (F, np.concatenate(ys), np.concatenate(gs))
    pickle.dump(blob, open(CACHE, "wb"), protocol=4)
    return blob


def score(pred, y, rows, k, gap):
    unc = float(np.median([np.abs(pred[i] - y[i]).mean() for i in rows.values()]))
    cal = []
    for idx in rows.values():
        if k + gap >= len(idx) - 3:
            continue
        c, t = idx[:k], idx[k + gap:]
        cal.append(float(np.abs(pred[t] + np.mean(y[c] - pred[c]) - y[t]).mean()))
    return unc, (float(np.median(cal)) if cal else np.nan), len(cal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=1524)   # the full MIMIC cohort
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--target", choices=["sbp", "dbp"], default="dbp")
    args = ap.parse_args()
    ycol = 0 if args.target == "sbp" else 1
    import lightgbm as lgb

    blob = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = blob["Ftr"], np.asarray(blob["ytr"], float)[:, ycol]
    Fm, Ym, gm = load_mimic(args.patients)
    ym = np.asarray(Ym, float)[:, ycol]
    rows = {p: np.where(gm == p)[0] for p in np.unique(gm)}
    print(f"[mimic] target={args.target} | {len(ym)} segments, {len(rows)} patients, "
          f"median {np.median([len(v) for v in rows.values()]):.0f} segs/patient\n")

    shared = sorted(set(Ftr) & set(Fm))
    pat = [k for k in shared if k in PAT_KEYS]
    morph = [k for k in shared if k not in PAT_KEYS and k not in RATE_KEYS]
    print(f"[feat] shared {len(shared)} | PAT {len(pat)} | morphology {len(morph)}\n")

    res = {}
    for name, keys in (("PAT only", pat), ("morphology", morph)):
        Mtr = np.column_stack([np.asarray(Ftr[k], float)[:len(ytr)] for k in keys])
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        Mm = np.column_stack([np.asarray(Fm[k], float) for k in keys])
        Mm = np.where(np.isfinite(Mm), Mm, med)
        for cap, cfg in (("single tree", dict(n_estimators=1, learning_rate=1.0, num_leaves=32)),
                         ("800 trees", dict(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                            subsample=0.8, colsample_bytree=0.8,
                                            min_child_samples=50))):
            m = lgb.LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1, **cfg).fit(Mtr, ytr)
            u, c, n = score(m.predict(Mm), ym, rows, args.k, args.gap)
            res[f"{name} | {cap}"] = {"ood_uncal": u, "ood_cal": c, "n_pat": n,
                                      "k": args.k, "gap": args.gap}
            print(f"  {name:12} {cap:12}  OOD uncal {u:6.2f} | "
                  f"OOD calibrated(k={args.k}) {c:6.2f}   ({n} patients)")

    # A margin threshold, not just a sign check: dc=+0.05 is not a real "SURVIVES" against
    # arms sitting near 6.4-6.8 mmHg -- that was the earlier bug. Only call it surviving if
    # the gap is still a meaningful fraction of the calibrated error itself.
    MARGIN = 0.3
    print(f"\n--- does arrival time keep its OOD advantage once calibrated? (target={args.target}) ---")
    for cap in ("single tree", "800 trees"):
        p, mm = res[f"PAT only | {cap}"], res[f"morphology | {cap}"]
        du, dc = mm["ood_uncal"] - p["ood_uncal"], mm["ood_cal"] - p["ood_cal"]
        verdict = ("SURVIVES" if dc > MARGIN else
                  "REVERSED" if dc < -MARGIN else "erased by calibration (within noise)")
        print(f"  {cap:12}  uncalibrated  PAT advantage {du:+.2f} mmHg")
        print(f"  {'':12}  calibrated    PAT advantage {dc:+.2f} mmHg   {verdict}")

    out = DATA / f"mimic_ood_calibrated_{args.target}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
