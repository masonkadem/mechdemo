"""calbased_vs_anchor.py -- PulseDB's CalBased protocol against per-subject anchor calibration.

These are NOT the same thing, and conflating them is easy.

  CalBased (PulseDB)   The test subjects ARE the training subjects: 1,293 subjects, 100% overlap
                       with train (verified against the .mat subject IDs). The model has already
                       seen each person during training, so any per-subject offset is baked into
                       the weights. There is no calibration step at deployment -- and no way to
                       apply it to a new patient, because a new patient was not in training.

  CalFree (PulseDB)    144 subjects, 0% overlap. The honest generalisation setting.

  k-anchor (ours)      Train subject-disjoint, then fit ONE offset parameter per test subject
                       from k cuff readings. This is what a real device does: the patient is new,
                       and you buy accuracy with a few reference measurements.

So CalBased is an upper bound obtainable only for subjects already in the training set, whereas
k-anchor is the deployable quantity. Reporting them side by side shows how much of "calibrated"
performance in the literature is really subject memorisation.

Also reports the SFS-selected feature subset per variant, so the table can carry a top-features
column alongside the accuracy columns.

    python calbased_vs_anchor.py
    python calbased_vs_anchor.py --sfs      # also run sequential forward selection (slower)
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import h5py
import lightgbm as lgb

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CALBASED = Path("C:/Users/mason/OneDrive - McMaster University/2026/BP/dbdata/"
                "VitalDB_CalBased_Test_Subset.mat")
KS = [0, 1, 2, 3, 5, 10, 20]
TARGET = 1
MIN_SEG = 60


def load_calbased(max_seg=20000, seed=0):
    """Signals + labels + subject ids from the CalBased test subset."""
    f = h5py.File(CALBASED, "r")
    g = f["Subset"]
    n = g["SBP"].shape[1]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, min(max_seg, n), replace=False))
    sbp = np.array(g["SBP"][0, idx], float)
    dbp = np.array(g["DBP"][0, idx], float)
    sig = np.stack([g["Signals"][:, :, i] for i in idx])           # (N, 1250, 3)
    subj = []
    ds = g["Subject"]
    for i in idx:
        try:
            subj.append("".join(chr(c[0]) for c in f[ds[0, i]][:]))
        except Exception:
            subj.append("?")
    demo = {k: np.array(g[k][0, idx], float) for k in ("Age", "BMI", "Height", "Weight")
            if k in g}
    f.close()
    return sig, np.stack([sbp, dbp], 1), np.array(subj), demo


def anchor_curve(pred, y, groups, ks=KS, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        errs = []
        for s in np.unique(groups):
            idx = np.where(groups == s)[0]
            if len(idx) < MIN_SEG:
                continue
            if k == 0:
                hold, off = idx, 0.0
            else:
                a = rng.choice(idx, k, replace=False)
                hold = np.setdiff1d(idx, a)
                if len(hold) < 20:
                    continue
                off = float(np.mean(y[a] - pred[a]))
            errs.append(float(np.mean(np.abs(pred[hold] + off - y[hold]))))
        out[k] = float(np.median(errs)) if errs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfs", action="store_true")
    ap.add_argument("--max-seg", type=int, default=20000)
    args = ap.parse_args()

    if not CALBASED.exists():
        print(f"[err] CalBased subset not found at {CALBASED}"); return

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]

    def tbl(F, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in keys]
        if demo is not None:
            for v in demo:
                cols.append(np.asarray(v, float)[:n])
        return np.column_stack(cols)

    print("[cal] extracting CalBased features "
          f"({args.max_seg} segments) ...", flush=True)
    import mechlib
    import features_full as ff
    sig, ycb, subj, demo_cb = load_calbased(args.max_seg)
    fs = 125
    Xcb = mechlib.normalize(sig[:, :, [0, 1]])          # ECG, PPG channel order as in PulseDB
    Fcb = ff.compute_full(Xcb, fs)
    gcb = np.array([hash(s) % 100000 for s in subj])

    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)
    dtr = [d["age_tr"], d["bmi_tr"]]
    dte = [d["age_te"], d["bmi_te"]]
    dcb = [demo_cb.get("Age"), demo_cb.get("BMI")]

    res = {}
    print(f"\n{'variant':22s} {'CalFree k=0':>12s} {'CalFree k=5':>12s} "
          f"{'CalFree k=20':>13s} {'CalBased':>10s}")
    for name, use_demo in (("waveform (83)", False), ("waveform + demo", True)):
        Mtr = tbl(Ftr, len(ytr), dtr if use_demo else None)
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])

        p_cf = m.predict(gbm._impute(tbl(Fte, len(yte), dte if use_demo else None), med))
        curve = anchor_curve(p_cf, yte[:, TARGET], gte)

        p_cb = m.predict(gbm._impute(tbl(Fcb, len(ycb), dcb if use_demo else None), med))
        mae_cb = float(np.median([np.mean(np.abs(p_cb[gcb == s] - ycb[gcb == s, TARGET]))
                                  for s in np.unique(gcb)
                                  if (gcb == s).sum() >= 5]))
        res[name] = {"calfree_curve": curve, "calbased_mae": mae_cb}
        print(f"{name:22s} {curve[0]:12.2f} {curve[5]:12.2f} {curve[20]:13.2f} "
              f"{mae_cb:10.2f}")

    # ---- SFS: which features actually carry the calibrated signal ------------
    if args.sfs:
        print("\n[sfs] sequential forward selection on calibrated error ...", flush=True)
        sub = np.random.default_rng(0).choice(len(ytr), min(40000, len(ytr)), replace=False)
        chosen, remaining = [], list(keys)
        best_hist = []
        for step in range(12):
            scores = []
            for k in remaining:
                cols = chosen + [k]
                M = np.column_stack([np.asarray(Ftr[c], float)[sub] for c in cols])
                mm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=31,
                                       random_state=0, verbosity=-1)
                med2 = gbm.column_medians(M)
                mm.fit(gbm._impute(M, med2), ytr[sub, TARGET])
                Mt = np.column_stack([np.asarray(Fte.get(c, np.full(len(yte), np.nan)), float)
                                      for c in cols])
                pv = mm.predict(gbm._impute(Mt, med2))
                scores.append((anchor_curve(pv, yte[:, TARGET], gte, ks=[5])[5], k))
            scores.sort()
            best, feat = scores[0]
            chosen.append(feat); remaining.remove(feat); best_hist.append((feat, best))
            print(f"  {step+1:2d}. +{feat:16s} -> calibrated MAE {best:.3f}", flush=True)
        res["sfs_order"] = [{"feature": f, "mae_k5": v} for f, v in best_hist]

    (DATA / "calbased_vs_anchor.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nCalBased test subjects are 100% overlapping with train (1,293 of 1,293), so that")
    print("column is an upper bound available only for people already in the training set.")
    print("The k-anchor columns are the deployable quantity for a new patient.")
    print(f"\n[done] data/calbased_vs_anchor.json")


if __name__ == "__main__":
    main()
