"""gbm_external_ood.py -- our GBM on the paper's OWN four external OOD datasets.

Sensors, UCI, BCG and PPGBP are the four datasets the AI4HealthUOL paper calls "OOD" (Table
V) -- a stricter tier than the VitalDB<->MIMIC cross-source test in their Table IV, and the
only tier the paper itself uses that word for. All four ship locally in data/*_dataset/.

None of the four carry an ECG channel (verified: every "signal" array is 2D, segments x time,
single channel). So PAT cannot be computed on any of them -- this is necessarily a
morphology-only test, using features_full.compute_full with ecg_ch=None.

Design, matching the paper's own protocol as closely as possible:
  - trained ONCE on PulseDB VitalDB CalibFree (our train split), never touching these four
  - scored PER SEGMENT, uncalibrated -- the paper's Table V protocol has no per-patient offset
  - a calibrated (k=5, offset) variant is reported alongside for context, since every other
    comparison this session carries one, but it is NOT what Table V reports

    python gbm_external_ood.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio
import lightgbm as lgb

import features_full as ff
import mechlib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
TARGET = {"sbp": 0, "dbp": 1}  # SP/DP column order in the label arrays we build below

DATASETS = {
    "Sensors": DATA / "sensors_dataset",
    "BCG": DATA / "bcg_dataset",
    "PPGBP": DATA / "ppgbp_dataset",
    "UCI": DATA / "uci2_dataset" / "uci2_dataset",
}
CAPACITIES = [
    ("single tree (32 lv)", dict(n_estimators=1, num_leaves=32, learning_rate=1.0)),
    ("20 trees (32 lv)", dict(n_estimators=20, num_leaves=32, learning_rate=0.3)),
    ("150 trees (32 lv)", dict(n_estimators=150, num_leaves=32, learning_rate=0.1)),
    ("800 trees (63 lv)", dict(n_estimators=800, num_leaves=63, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, min_child_samples=50)),
]


def load_external(folder):
    """Concatenate every signal_fold_*.mat in the directory.

    Each dataset ships as a 5-way subject-disjoint split (signal_fold_0..4.mat), not five
    copies of the same data -- reading only fold_0 recovers roughly a fifth of the subjects
    (confirmed: BCG fold_0 alone gave 9 of the paper's reported 40 subjects, PPGBP fold_0 gave
    44 of 218). All five must be concatenated to match the dataset sizes Table I reports.
    """
    sigs, sbps, dbps, pats = [], [], [], []
    for i, path in enumerate(sorted(folder.glob("signal_fold_*.mat"))):
        m = sio.loadmat(path)
        sig = np.asarray(m["signal"], np.float64)
        sbp = np.asarray(m["SP"], np.float64).ravel()
        dbp = np.asarray(m["DP"], np.float64).ravel()
        # patient IDs are numeric in some datasets (Sensors, PPGBP) and STRING in others
        # (BCG "X1004", UCI "Part_4_0") -- keep as strings, np.unique works on either; prefix
        # with the fold index so the same numeric ID in different folds cannot collide
        pat = np.asarray([f"f{i}_{str(p).strip()}" for p in m["patient"].ravel()])
        sigs.append(sig); sbps.append(sbp); dbps.append(dbp); pats.append(pat)
    sig = np.concatenate(sigs); sbp = np.concatenate(sbps)
    dbp = np.concatenate(dbps); pat = np.concatenate(pats)
    ok = np.isfinite(sig).all(1) & np.isfinite(sbp) & np.isfinite(dbp)
    ok &= (sbp > 40) & (sbp < 250) & (dbp > 20) & (dbp < 150)   # physiological gate
    return sig[ok], sbp[ok], dbp[ok], pat[ok]


def score(pred, y, groups, k=5, gap=0):
    """Per-segment uncalibrated MAE (matches the paper's Table V protocol), plus an
    offset-calibrated variant for context -- NOT what Table V reports."""
    unc = float(np.mean(np.abs(pred - y)))
    cal = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < k + gap + 2:
            continue
        cal_idx, test_idx = idx[:k], idx[k + gap:]
        off = float(np.mean(y[cal_idx] - pred[cal_idx]))
        cal.append(float(np.abs(pred[test_idx] + off - y[test_idx]).mean()))
    return unc, (float(np.median(cal)) if cal else np.nan)


def main():
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr = full["Ftr"], np.asarray(full["ytr"], float)

    print("[1/2] extracting morphology features on the four external datasets ...\n")
    ext = {}
    for name, path in DATASETS.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} not found"); continue
        sig, sbp, dbp, pat = load_external(path)
        X = mechlib.normalize(sig)[:, :, None]                  # (N, L, 1), PPG only
        F = ff.compute_full(X, FS, ppg_ch=0, ecg_ch=None)        # no ECG -> no PAT features
        ext[name] = {"F": F, "sbp": sbp, "dbp": dbp, "g": pat, "n": len(sig)}
        print(f"  {name:8} {len(sig):6d} segments, {len(np.unique(pat)):4d} subjects")

    keys = sorted(set(Ftr) & set.intersection(*[set(e["F"]) for e in ext.values()]))
    print(f"\n[feat] {len(keys)} morphology features shared with every external set "
          f"(no PAT: no dataset here has ECG)\n")

    res = {}
    print(f"{'dataset':10} {'capacity':20} {'target':4} {'uncal (paper protocol)':>22} "
          f"{'calibrated (k=5, context)':>25}")
    for tname, ycol in TARGET.items():
        Mtr = np.column_stack([np.asarray(Ftr[k], float)[:len(ytr)] for k in keys])
        med = np.nanmedian(Mtr, 0)
        Mtr = np.where(np.isfinite(Mtr), Mtr, med)
        y_tr = ytr[:, ycol]

        for cap_name, params in CAPACITIES:
            m = lgb.LGBMRegressor(**params, random_state=0, n_jobs=1, verbosity=-1)
            m.fit(Mtr, y_tr)
            for dname, e in ext.items():
                Me = np.column_stack([np.asarray(e["F"][k], float) for k in keys])
                Me = np.where(np.isfinite(Me), Me, med)
                y_e = e["sbp"] if tname == "sbp" else e["dbp"]
                pred = m.predict(Me)
                unc, cal = score(pred, y_e, e["g"])
                res[f"{dname} | {tname} | {cap_name}"] = {
                    "uncal": unc, "cal_k5": cal, "n": e["n"],
                    "n_subj": int(len(np.unique(e["g"])))}
                print(f"{dname:10} {cap_name:20} {tname:4} {unc:22.2f} {cal:25.2f}")

    json.dump(res, open(DATA / "gbm_external_ood.json", "w"), indent=2)
    print(f"\n[done] {DATA / 'gbm_external_ood.json'}")


if __name__ == "__main__":
    main()
