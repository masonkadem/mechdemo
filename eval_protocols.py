"""eval_protocols.py -- every model on all three PulseDB protocols.

The three protocols, as defined in the PulseDB paper (Frontiers Digit Health 2023) and verified
against the .mat subject IDs here:

  CalFree   Training and test subjects are disjoint. 144 VitalDB subjects, 400 segments each,
            0% overlap with train. The honest generalisation setting.
  CalBased  The SAME subjects as training: 360 segments/subject go to train and 40 to test, so
            all 1,293 subjects appear in both. Verified: 100% overlap. This is not a calibration
            procedure -- the per-subject offset is memorised in the weights, and it cannot be
            applied to a patient who was not in training.
  AAMI      A second calibration-free setting built to the AAMI standard: >=85 subjects, >=255
            measurements, and >=5% of reference SBP below 100 and above 160 mmHg (likewise DBP
            below 60 and above 100). Our copy: 666 segments, 116 subjects, with 15.6% / 21.5% of
            SBP outside those bounds, so the spread requirement is met. This matters because
            CalFree and CalBased are dominated by near-normal pressures, where predicting the
            mean is nearly free; AAMI deliberately includes the tails where a model must be right
            about the physiology.

Reported alongside these: the k-anchor curve, which is what a device can actually do for a new
patient (fit one offset from k cuff readings), and a mean-predictor floor per protocol, without
which the AAMI numbers in particular are easy to over-read.

    python eval_protocols.py                 # LightGBM variants
    python eval_protocols.py --deep          # include the deep nets
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import h5py
import lightgbm as lgb

import mechlib
import lightgbm_arm as gbm
import features_full as ff

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BP = Path("C:/Users/mason/OneDrive - McMaster University/2026/BP/dbdata")
SUBSETS = {"CalBased": "VitalDB_CalBased_Test_Subset.mat",
           "AAMI": "VitalDB_AAMI_Test_Subset.mat"}
KS = [0, 1, 2, 3, 5, 10, 20]
TARGET = 1          # DBP
FS = 125


def load_subset(path, max_seg=0, seed=0):
    f = h5py.File(path, "r")
    g = f["Subset"]
    n = g["SBP"].shape[1]
    idx = np.arange(n)
    if max_seg and n > max_seg:
        idx = np.sort(np.random.default_rng(seed).choice(n, max_seg, replace=False))
    y = np.stack([np.array(g["SBP"][0, idx], float),
                  np.array(g["DBP"][0, idx], float)], 1)
    sig = np.stack([g["Signals"][:, :, i] for i in idx])
    ds = g["Subject"]
    subj = []
    for i in idx:
        try:
            subj.append("".join(chr(c[0]) for c in f[ds[0, i]][:]))
        except Exception:
            subj.append("?")
    demo = {k: np.array(g[k][0, idx], float) for k in ("Age", "BMI") if k in g}
    f.close()
    return sig, y, np.array(subj), demo


def anchor_curve(pred, y, groups, ks=KS, min_seg=20, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        errs = []
        for s in np.unique(groups):
            idx = np.where(groups == s)[0]
            if len(idx) < max(min_seg, k + 5):
                continue
            if k == 0:
                hold, off = idx, 0.0
            else:
                a = rng.choice(idx, k, replace=False)
                hold = np.setdiff1d(idx, a)
                off = float(np.mean(y[a] - pred[a]))
            errs.append(float(np.mean(np.abs(pred[hold] + off - y[hold]))))
        out[k] = float(np.median(errs)) if errs else float("nan")
    return out


def mean_floor(y, groups):
    """Error of predicting each subject's own mean -- the floor any model must beat."""
    v = [float(np.mean(np.abs(y[groups == s] - y[groups == s].mean())))
         for s in np.unique(groups) if (groups == s).sum() >= 3]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seg", type=int, default=15000)
    args = ap.parse_args()

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
                cols.append(np.asarray(v, float)[:n] if v is not None else np.full(n, np.nan))
        return np.column_stack(cols)

    # external protocols: extract features once, reuse for every variant
    ext = {}
    for name, fn in SUBSETS.items():
        p = BP / fn
        if not p.exists():
            print(f"[skip] {name}: not found"); continue
        print(f"[load] {name} ...", flush=True)
        sig, y, subj, demo = load_subset(p, args.max_seg)
        X = mechlib.normalize(sig[:, :, [0, 1]])
        F = ff.compute_full(X, FS)
        g = np.array([abs(hash(s)) % 1000000 for s in subj])
        ext[name] = {"F": F, "y": y, "g": g,
                     "demo": [demo.get("Age"), demo.get("BMI")],
                     "floor": mean_floor(y[:, TARGET], g),
                     "n_seg": len(y), "n_subj": len(set(subj))}
        print(f"       {len(y)} segments, {len(set(subj))} subjects, "
              f"subject-mean floor {ext[name]['floor']:.2f}", flush=True)

    variants = {
        "gbm default (83)": (dict(n_estimators=800, learning_rate=0.03, num_leaves=63), False),
        "gbm default + demo": (dict(n_estimators=800, learning_rate=0.03, num_leaves=63), True),
        "gbm deep + demo": (dict(n_estimators=1500, learning_rate=0.02, num_leaves=127), True),
        "gbm single tree": (dict(n_estimators=1, learning_rate=1.0, num_leaves=32), False),
    }
    dtr = [d["age_tr"], d["bmi_tr"]]
    dte = [d["age_te"], d["bmi_te"]]

    res = {}
    hdr = (f"{'variant':22s} {'CalFree k=0':>11s} {'k=5':>6s} {'k=20':>6s} " +
           " ".join(f"{n:>10s}" for n in ext))
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, (params, use_demo) in variants.items():
        Mtr = tbl(Ftr, len(ytr), dtr if use_demo else None)
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                              min_child_samples=50, random_state=0, verbosity=-1)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])

        p_cf = m.predict(gbm._impute(tbl(Fte, len(yte), dte if use_demo else None), med))
        curve = anchor_curve(p_cf, yte[:, TARGET], gte, min_seg=60)
        row = {"calfree_curve": curve}
        cells = []
        for pname, e in ext.items():
            pe = m.predict(gbm._impute(
                tbl(e["F"], len(e["y"]), e["demo"] if use_demo else None), med))
            mae = float(np.median([np.mean(np.abs(pe[e["g"] == s] - e["y"][e["g"] == s, TARGET]))
                                   for s in np.unique(e["g"]) if (e["g"] == s).sum() >= 3]))
            row[pname] = mae
            cells.append(f"{mae:10.2f}")
        res[name] = row
        print(f"{name:22s} {curve[0]:11.2f} {curve[5]:6.2f} {curve[20]:6.2f} "
              + " ".join(cells), flush=True)

    floor_cells = " ".join(f"{ext[n]['floor']:10.2f}" for n in ext)
    print(f"{'subject-mean floor':22s} {'--':>11s} {'--':>6s} {'--':>6s} {floor_cells}")

    res["_meta"] = {n: {k: e[k] for k in ("n_seg", "n_subj", "floor")}
                    for n, e in ext.items()}
    (DATA / "eval_protocols.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nCalBased shares 100% of its subjects with training, so it is an upper bound "
          "available only for\npeople already in the training set. AAMI is calibration-free "
          "like CalFree but deliberately\nincludes the BP tails, where predicting the mean "
          "stops being nearly free.")
    print(f"\n[done] data/eval_protocols.json")


if __name__ == "__main__":
    main()
