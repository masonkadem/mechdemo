"""pulsedb_train_subset.py -- training data straight from the official PulseDB VitalDB training file.

Reads ../BP/dbdata/VitalDB_Train_Subset.mat (MATLAB v7.3 / HDF5; 465,480 segments, 1,293
patients, none of them in the CalFree test set) in contiguous slabs and keeps every `--stride`-th
segment, so every patient stays represented. ECG and PPG only, float32. 10% of patients are held
out for validation (patient-disjoint). Writes data/vitaldb_train_pulsedb.npz with the keys
real_audit_controls.py expects (Xtr, ytr, gtr, Xva, yva, gva, fs); labels are [SBP, DBP].

Replaces data/vitaldb_full_calfree.npz, whose Xtr array fails its CRC check on this machine.
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
MAT = ROOT.parent / "BP" / "dbdata" / "VitalDB_Train_Subset.mat"
SLAB = 8166                                  # the label arrays' HDF5 chunk length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="", help="output path (default data/vitaldb_train_pulsedb.npz)")
    args = ap.parse_args()
    t0 = time.time()
    with h5py.File(MAT, "r") as h:
        s = h["Subset"]
        n = s["SBP"].shape[1]
        keep = np.arange(0, n, args.stride)
        sbp = s["SBP"][0, keep]; dbp = s["DBP"][0, keep]
        subj = [''.join(chr(c) for c in h[r][:].flatten()) for r in s["Subject"][0, keep]]
        X = np.empty((len(keep), 1250, 2), np.float32)
        j = 0
        for a in range(0, n, SLAB):
            sig = s["Signals"][:, :2, a:min(a + SLAB, n)]          # (1250, 2, m), ECG and PPG
            loc = keep[(keep >= a) & (keep < a + SLAB)] - a
            X[j:j + len(loc)] = np.transpose(sig[:, :, loc], (2, 0, 1))
            j += len(loc)
            print(f"\r[read] {min(a + SLAB, n)}/{n} segments, {time.time() - t0:.0f}s", end="", flush=True)
    print()
    y = np.stack([sbp, dbp], 1).astype(np.float32)
    ids, g = np.unique(subj, return_inverse=True)
    ok = np.isfinite(X).all((1, 2)) & np.isfinite(y).all(1)
    rng = np.random.default_rng(0)
    val_p = rng.choice(len(ids), int(round(args.val_frac * len(ids))), replace=False)
    va = np.isin(g, val_p) & ok; tr = ~np.isin(g, val_p) & ok
    out = Path(args.out) if args.out else ROOT / "data" / "vitaldb_train_pulsedb.npz"
    np.savez(out, Xtr=X[tr], ytr=y[tr], gtr=g[tr], Xva=X[va], yva=y[va], gva=g[va], fs=np.int64(125))
    print(f"[done] {out.name}: train {tr.sum()} segments / {len(np.unique(g[tr]))} patients, "
          f"val {va.sum()} / {len(np.unique(g[va]))}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
