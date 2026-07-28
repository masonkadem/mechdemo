"""convert_pulsedb.py -- one-time: convert the FULL PulseDB VitalDB .mat subsets to a fast
.npz matching our pipeline format, WITH demographics (age, sex, BMI).

The demo cache (vitaldb_mini_deep.npz) was a 4% random subsample. This uses the official
patient-disjoint CalFree splits (matching the benchmark paper: 416,880 train segs / 1,158
subjects, 57,600 CalFree test segs) so the ID baseline, OOD gaps, and mechanism audit are on
the real training distribution -- and so VitalDB's own age/sex can drive the LightGBM demographics
analysis rather than only the external sets.

    python convert_pulsedb.py                    # full CalFree
    python convert_pulsedb.py --max-train 100000 # capped, faster iteration
"""
import argparse
import os
from pathlib import Path

import numpy as np

DB = Path("C:/Users/mason/OneDrive - McMaster University/2026/BP/dbdata")
OUT = Path(__file__).resolve().parent / "data" / "vitaldb_full_calfree.npz"


def parse_mat(path, want_demo=True):
    from mat73 import loadmat
    print(f"  loading {path.name} ...", flush=True)
    sub = loadmat(str(path))["Subset"]
    X = np.array(sub["Signals"], dtype=np.float32).transpose(0, 2, 1)     # (N,1250,3) ECG/PPG/ABP
    y = np.stack([np.array(sub["SBP"], np.float32), np.array(sub["DBP"], np.float32)], 1)
    ids = [s[0] if isinstance(s, list) else str(s) for s in sub["Subject"]]
    demo = None
    if want_demo:
        def col(k):
            return np.array(sub[k], np.float32).ravel() if k in sub else np.full(len(X), np.nan)
        g = sub.get("Gender")

        def _sex1(v):
            s = (v[0] if isinstance(v, list) and v else v)   # Gender comes as ['M'] per row
            s = str(s).strip().lower()
            return 1.0 if s.startswith(("m", "1")) else 0.0 if s.startswith(("f", "0")) else np.nan
        sex = np.array([_sex1(v) for v in (g if g is not None else [None] * len(X))])
        demo = {"age": col("Age"), "sex": sex, "bmi": col("BMI"),
                "height": col("Height"), "weight": col("Weight")}
    return X, y, ids, demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0, help="0 = all")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xtr_all, ytr_all, ids_all, demo_all = parse_mat(DB / "VitalDB_Train_Subset.mat")
    Xte, yte, ids_te, demo_te = parse_mat(DB / "VitalDB_CalFree_Test_Subset.mat")

    # subject-level train/val split (strip session suffix so sessions stay together)
    base = np.array([s.rsplit("_", 1)[0] for s in ids_all])
    pats = np.unique(base)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * args.val_frac))
    val_pats = set(pats[:n_val])
    val_mask = np.array([p in val_pats for p in base])

    # integer group index per full subject id
    allids = list(dict.fromkeys(ids_all + ids_te))
    to_int = {s: i for i, s in enumerate(allids)}
    g_all = np.array([to_int[s] for s in ids_all], np.int64)
    g_te = np.array([to_int[s] for s in ids_te], np.int64)

    tr = ~val_mask
    if args.max_train and tr.sum() > args.max_train:
        # subsample TRAIN segments only (keep all subjects represented)
        idx = np.where(tr)[0]
        keep = rng.choice(idx, args.max_train, replace=False)
        tr = np.zeros(len(tr), bool); tr[keep] = True

    def pack(prefix, mask, demo):
        d = {f"{prefix}": None}
        sel = {k: demo[k][mask] for k in demo} if demo else {}
        return sel

    out = dict(
        Xtr=Xtr_all[tr], ytr=ytr_all[tr], gtr=g_all[tr],
        Xva=Xtr_all[val_mask], yva=ytr_all[val_mask], gva=g_all[val_mask],
        Xte=Xte, yte=yte, gte=g_te, fs=np.int64(125),
        # demographics, aligned to each split
        age_tr=demo_all["age"][tr], sex_tr=demo_all["sex"][tr], bmi_tr=demo_all["bmi"][tr],
        age_va=demo_all["age"][val_mask], sex_va=demo_all["sex"][val_mask],
        age_te=demo_te["age"], sex_te=demo_te["sex"], bmi_te=demo_te["bmi"],
    )
    print(f"  train {out['Xtr'].shape} | val {out['Xva'].shape} | test {out['Xte'].shape}")
    print(f"  subjects: train+val {len(np.unique(g_all))}  test {len(np.unique(g_te))}")
    print(f"  saving {OUT} (this is large, ~{out['Xtr'].nbytes/1e9:.1f} GB train) ...", flush=True)
    np.savez(OUT, **out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
