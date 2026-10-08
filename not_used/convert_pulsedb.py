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

# PulseDB .mat subsets. This repo sits beside BP/ in the same OneDrive tree on both machines,
# so resolve rather than hardcode: PULSEDB_DIR wins, then the sibling path, then the old
# Windows location. Keeps the script runnable on the Mac without editing it each time.
_CANDIDATES = [os.environ.get("PULSEDB_DIR"),
               Path(__file__).resolve().parent.parent / "BP" / "dbdata",
               "C:/Users/mason/OneDrive - McMaster University/2026/BP/dbdata"]
DB = next((Path(p) for p in _CANDIDATES if p and Path(p).is_dir()), Path(_CANDIDATES[-1]))
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


def _sref(f, r):
    """MATLAB v7.3 stores char arrays as object refs; deref to a python str."""
    return "".join(chr(int(c)) for c in np.asarray(f[r][()]).ravel())


def read_meta_h5(path):
    """Everything EXCEPT the signals -- cheap, so the split can be decided before any bulk read."""
    import h5py
    with h5py.File(path, "r") as f:
        s = f["Subset"]
        y = np.stack([np.asarray(s["SBP"][0], np.float32),
                      np.asarray(s["DBP"][0], np.float32)], 1)
        ids = [_sref(f, r) for r in s["Subject"][0]]
        gen = [_sref(f, r) for r in s["Gender"][0]]
        sex = np.array([1.0 if g.strip().lower().startswith(("m", "1"))
                        else 0.0 if g.strip().lower().startswith(("f", "0")) else np.nan
                        for g in gen], np.float32)

        def col(k):
            return (np.asarray(s[k][0], np.float32) if k in s
                    else np.full(len(ids), np.nan, np.float32))
        demo = {"age": col("Age"), "sex": sex, "bmi": col("BMI"),
                "height": col("Height"), "weight": col("Weight")}
        n = s["Signals"].shape[2]
    return n, y, ids, demo


def read_signals_h5(path, want, block=8192):
    """Read only the wanted segments, in SEQUENTIAL blocks.

    mat73.loadmat materialises all 465,480 x 1250 x 3 as float64 (~14 GB) before anything can
    be subset, which does not fit on a 16 GB machine. Signals is stored (L, C, N), so walking
    N in blocks keeps peak memory at one block and the IO sequential.
    """
    import h5py
    want = np.sort(np.asarray(want))
    out = np.empty((len(want), 1250, 3), np.float32)
    with h5py.File(path, "r") as f:
        ds = f["Subset"]["Signals"]
        n = ds.shape[2]
        w = 0
        for a in range(0, n, block):
            b = min(a + block, n)
            sel = want[(want >= a) & (want < b)]
            if not len(sel):
                continue
            chunk = np.asarray(ds[:, :, a:b], np.float32)      # (1250, 3, b-a)
            out[w:w + len(sel)] = chunk[:, :, sel - a].transpose(2, 0, 1)
            w += len(sel)
            print(f"    {b}/{n} segments scanned, {w}/{len(want)} kept", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", action="store_true",
                    help="h5py block reader instead of mat73 (needed under ~32 GB RAM)")
    ap.add_argument("--max-train", type=int, default=0, help="0 = all")
    ap.add_argument("--max-val", type=int, default=0, help="0 = all")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    TRAIN_MAT, TEST_MAT = DB / "VitalDB_Train_Subset.mat", DB / "VitalDB_CalFree_Test_Subset.mat"
    if args.stream:
        # metadata first, so the split is decided before a single signal byte is read
        _, ytr_all, ids_all, demo_all = read_meta_h5(TRAIN_MAT)
        n_te, yte, ids_te, demo_te = read_meta_h5(TEST_MAT)
        Xtr_all = None
        print(f"  meta: {len(ids_all)} train segs, {n_te} test segs", flush=True)
    else:
        Xtr_all, ytr_all, ids_all, demo_all = parse_mat(TRAIN_MAT)
        Xte, yte, ids_te, demo_te = parse_mat(TEST_MAT)

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

    if args.max_val and val_mask.sum() > args.max_val:
        idx = np.where(val_mask)[0]
        keep = rng.choice(idx, args.max_val, replace=False)
        val_mask = np.zeros(len(val_mask), bool); val_mask[keep] = True

    if args.stream:
        print("  reading train signals ...", flush=True)
        Xtr = read_signals_h5(TRAIN_MAT, np.where(tr)[0])
        print("  reading val signals ...", flush=True)
        Xva = read_signals_h5(TRAIN_MAT, np.where(val_mask)[0])
        print("  reading test signals ...", flush=True)
        Xte = read_signals_h5(TEST_MAT, np.arange(n_te))
    else:
        Xtr, Xva = Xtr_all[tr], Xtr_all[val_mask]

    out = dict(
        Xtr=Xtr, ytr=ytr_all[tr], gtr=g_all[tr],
        Xva=Xva, yva=ytr_all[val_mask], gva=g_all[val_mask],
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
