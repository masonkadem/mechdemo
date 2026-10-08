"""run_feature_full_stream.py -- the 83-feature library on the FULL train set, in parallel,
without ever holding the signals in memory.

run_feature_full.py reads d["Xtr"] out of vitaldb_full_calfree.npz, so the train set it can
reach is capped by whatever convert_pulsedb.py could materialise -- 150k segments on a 16 GB
machine, against the 396k the published numbers use. But LightGBM trains on FEATURES, not on
signals, and features are three orders of magnitude smaller: 396k x ~150 floats is ~475 MB,
while the signals are 7 GB. So this reads the .mat in blocks and keeps only the features.

Two costs that bit the first version, both fixed here:

  metadata   MATLAB v7.3 stores Subject/Gender as one HDF5 object reference PER ROW, so
             reading them means 931k individual dereferences -- ~20 min, and it repeated on
             every run. Now cached to a small npz next to the .mat.
  one core   compute_full is numpy/scipy, so no GPU helps, but segments are independent.
             Now spread over the performance cores (8 on an M1 Pro) instead of 1.

It reproduces convert_pulsedb.py's subject-level val split exactly (same seed, same shuffle),
so the train set is the one the npz would have described. Demographics aligned to Ftr are
written into the pickle too, since the npz's age_tr/sex_tr/bmi_tr are only as long as its
capped Xtr and would otherwise mis-align.

    python run_feature_full_stream.py                          # all ~396k, all cores
    python run_feature_full_stream.py --workers 4
    python run_feature_full_stream.py --train-n 5000 --out data/_probe.pkl
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "_feat_full_ALLtrain.pkl"
FS = 125


def meta_cache_path(mat_path):
    return ROOT / "data" / f"_meta_{Path(mat_path).stem}.npz"


def read_meta_cached(mat_path):
    """Metadata (ids, BP, demographics). Cached: the raw read is ~20 min of HDF5 derefs."""
    import convert_pulsedb as cp
    cp_path = meta_cache_path(mat_path)
    if cp_path.exists():
        z = np.load(cp_path, allow_pickle=True)
        demo = {k: z[f"demo_{k}"] for k in ("age", "sex", "bmi", "height", "weight")}
        print(f"[meta] cached {cp_path.name}  ({len(z['ids'])} segments)", flush=True)
        return len(z["ids"]), z["y"], list(z["ids"]), demo
    t0 = time.time()
    print(f"[meta] reading {Path(mat_path).name} (one-off, ~20 min) ...", flush=True)
    n, y, ids, demo = cp.read_meta_h5(mat_path)
    np.savez_compressed(cp_path, y=y, ids=np.array(ids),
                        **{f"demo_{k}": v for k, v in demo.items()})
    print(f"[meta] cached -> {cp_path.name} in {(time.time()-t0)/60:.1f} min", flush=True)
    return n, y, ids, demo


def _extract_chunk(job):
    """One worker: read only the blocks this chunk needs, extract, return features."""
    import h5py
    import mechlib
    import features_full as ff
    from mechlib import ECG, PPG

    path, want, block = job
    acc = None
    with h5py.File(path, "r") as f:
        ds = f["Subset"]["Signals"]                       # (1250, 3, N)
        lo, hi = int(want.min()), int(want.max()) + 1
        for a in range(lo - (lo % block), hi, block):
            b = min(a + block, ds.shape[2])
            sel = want[(want >= a) & (want < b)]
            if not len(sel):
                continue
            chunk = np.asarray(ds[:, :, a:b], np.float32)[:, :, sel - a].transpose(2, 0, 1)
            F = ff.compute_full(mechlib.normalize(chunk[:, :, [ECG, PPG]]),
                                FS, ppg_ch=1, ecg_ch=0)
            if acc is None:
                acc = {k: [] for k in F}
            for k in acc:
                acc[k].append(np.asarray(F[k], np.float32))
    return {k: np.concatenate(v) for k, v in acc.items()}


def extract_parallel(path, want, tag, workers, block=2048, per_task=8192):
    """Split the wanted segments across processes; results recombined in order."""
    want = np.sort(np.asarray(want))
    jobs = [(str(path), want[i:i + per_task], block) for i in range(0, len(want), per_task)]
    print(f"[{tag}] {len(want)} segments in {len(jobs)} tasks over {workers} workers",
          flush=True)
    parts, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, F in enumerate(ex.map(_extract_chunk, jobs), 1):
            parts.append(F)
            el = (time.time() - t0) / 60
            print(f"  [{tag}] task {i}/{len(jobs)}  {el:.1f} min elapsed, "
                  f"~{el / i * (len(jobs) - i):.1f} min left", flush=True)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def train_mask_like_convert(ids, val_frac=0.15, seed=42):
    """Reproduce convert_pulsedb.main()'s subject-level split, bit for bit."""
    base = np.array([s.rsplit("_", 1)[0] for s in ids])
    pats = np.unique(base)
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    val_pats = set(pats[:max(1, int(len(pats) * val_frac))])
    return ~np.array([p in val_pats for p in base]), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=0, help="0 = all ~396k")
    ap.add_argument("--workers", type=int, default=0, help="0 = performance cores")
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--out", default=str(OUT),
                    help="a capped --train-n run must NOT clobber the real cache")
    args = ap.parse_args()
    out_path = Path(args.out)
    if args.train_n and out_path == OUT:
        ap.error("--train-n is a partial run; pass --out to avoid overwriting the full cache")

    workers = args.workers or int(os.popen("sysctl -n hw.perflevel0.logicalcpu").read() or 8)
    import convert_pulsedb as cp
    TRAIN, TEST = cp.DB / "VitalDB_Train_Subset.mat", cp.DB / "VitalDB_CalFree_Test_Subset.mat"

    _, ytr_all, ids_all, demo_all = read_meta_cached(TRAIN)
    n_te, yte, ids_te, demo_te = read_meta_cached(TEST)
    tr, _ = train_mask_like_convert(ids_all)
    idx_tr = np.where(tr)[0]
    if args.train_n and len(idx_tr) > args.train_n:
        idx_tr = np.sort(np.random.default_rng(0).choice(idx_tr, args.train_n, replace=False))

    allids = list(dict.fromkeys(list(ids_all) + list(ids_te)))
    to_int = {s: i for i, s in enumerate(allids)}
    g_te = np.array([to_int[s] for s in ids_te], np.int64)
    n_sub = len({ids_all[i].rsplit("_", 1)[0] for i in idx_tr})
    print(f"train {len(idx_tr)} segments / {n_sub} subjects | test {n_te} / "
          f"{len(np.unique(g_te))} subjects\n", flush=True)

    Ftr = extract_parallel(TRAIN, idx_tr, "train", workers, args.block)
    Fte = extract_parallel(TEST, np.arange(n_te), "test", workers, args.block)

    blob = {"Ftr": Ftr, "ytr": ytr_all[idx_tr], "Fte": Fte, "yte": yte, "gte": g_te,
            # aligned demographics: the npz's train arrays are only as long as its capped Xtr
            "age_tr": demo_all["age"][idx_tr], "sex_tr": demo_all["sex"][idx_tr],
            "bmi_tr": demo_all["bmi"][idx_tr],
            "age_te": demo_te["age"], "sex_te": demo_te["sex"], "bmi_te": demo_te["bmi"]}
    pickle.dump(blob, open(out_path, "wb"), protocol=4)
    print(f"\n[done] {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  Ftr {len(Ftr)} features x {len(blob['ytr'])} segments")
    print(f"  Fte {len(Fte)} features x {len(yte)} segments")


if __name__ == "__main__":
    main()
