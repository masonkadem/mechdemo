"""run_feature_full.py -- extract the 83-feature library on the FULL PulseDB train set
(396k seg / 1100 subjects) and the full CalFree test set, then train the final LightGBM for
SBP and DBP with proper subject-disjoint evaluation. Caches features so reruns are instant.

Compares full-data results to the 8k study (feature_study.json).

    python run_feature_full.py --train-n 0 --test-n 0     # 0 = all
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb

import mechlib
import features_full as ff
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent


def extract(X, fs, tag):
    t0 = time.time()
    print(f"[full] extracting {tag}: {len(X)} segments ...", flush=True)
    F = ff.compute_full(X, fs, ppg_ch=1, ecg_ch=0)
    print(f"[full] {tag} done in {(time.time()-t0)/60:.1f} min", flush=True)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--train-n", type=int, default=0, help="0 = all 396k")
    ap.add_argument("--test-n", type=int, default=0, help="0 = all 57.6k")
    args = ap.parse_args()

    cache = ROOT / "data" / "_feat_full_ALLtrain.pkl"
    if cache.exists():
        print("[full] loading cached full features")
        blob = pickle.load(open(cache, "rb"))
        Ftr, ytr, Fte, yte, gte = (blob["Ftr"], blob["ytr"], blob["Fte"], blob["yte"], blob["gte"])
    else:
        d = mechlib.load_mini(args.data); fs = d["fs"]
        Xtr = mechlib.normalize(d["Xtr"][:, :, [ECG, PPG]])
        Xte = mechlib.normalize(d["Xte"][:, :, [ECG, PPG]])
        ytr, yte, gte = d["ytr"], d["yte"], d["gte"]
        if args.train_n:
            Xtr, ytr = Xtr[:args.train_n], ytr[:args.train_n]
        if args.test_n:
            Xte, yte, gte = Xte[:args.test_n], yte[:args.test_n], gte[:args.test_n]
        Ftr = extract(Xtr, fs, "train")
        Fte = extract(Xte, fs, "test")
        pickle.dump({"Ftr": Ftr, "ytr": ytr, "Fte": Fte, "yte": yte, "gte": gte},
                    open(cache, "wb"))
        print(f"[full] cached -> {cache.name}")

    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]

    def mat(F, n):
        M = np.column_stack([np.asarray(F[k], float) for k in keys])
        for j in range(M.shape[1]):
            c = M[:, j]; m = np.isfinite(c)
            c[~m] = np.median(c[m]) if m.any() else 0.0
        return M

    Mtr, Mte = mat(Ftr, len(ytr)), mat(Fte, len(yte))
    apg_novel = [k for k in ["t_b", "t_c", "t_d", "t_e", "apg_cd_a", "apg_bd_a", "apg_ce_a",
                             "takazawa", "ushiro", "reflect_be"] if k in keys]
    base_idx = [i for i, k in enumerate(keys) if k not in apg_novel]

    results = {"n_train": len(ytr), "n_test": len(yte), "n_features": len(keys)}
    print(f"\n[full] train {len(ytr):,} seg | test {len(yte):,} seg | {len(keys)} features")
    for t, tname in [(0, "SBP"), (1, "DBP")]:
        def fit(idx):
            m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=63,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
                                  random_state=0, verbosity=-1)
            m.fit(Mtr[:, idx], ytr[:, t])
            return m, float(np.abs(m.predict(Mte[:, idx]) - yte[:, t]).mean())
        m_all, mae_all = fit(list(range(len(keys))))
        _, mae_base = fit(base_idx)
        imp = sorted(zip(keys, m_all.feature_importances_.astype(float)), key=lambda x: -x[1])
        results[tname] = {"mae_all": mae_all, "mae_no_apgnovel": mae_base,
                          "apgnovel_gain": mae_base - mae_all,
                          "top15": imp[:15]}
        print(f"[full] {tname}: MAE {mae_all:.2f} | without APG-novel {mae_base:.2f} "
              f"(gain {mae_base - mae_all:+.3f})")
        print(f"       top: {', '.join(k for k, _ in imp[:8])}")

    (ROOT / "data" / "feature_study_full.json").write_text(json.dumps(results, indent=2, default=float))
    print("[done] data/feature_study_full.json")

    # compare to 8k
    small = ROOT / "data" / "feature_study.json"
    if small.exists():
        s = json.loads(small.read_text())
        print("\n[compare] 8k vs full DBP MAE: %.2f -> %.2f" %
              (s["DBP"]["all_mae"], results["DBP"]["mae_all"]))


if __name__ == "__main__":
    main()
