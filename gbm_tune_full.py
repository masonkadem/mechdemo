"""gbm_tune_full.py -- push the interpretable model on FULL data:
  1. Optuna-tune LightGBM hyperparameters harder (100 trials) on the full 396k-feature set.
  2. mlxtend SFS to find the best feature COMBINATION.
  3. Single-tree models with the top-k feature combinations (interpretable, visualizable).
  4. Report parameter counts (LightGBM leaves*trees) for size comparison with the deep nets.

Uses cached full-train features (data/_feat_full_ALLtrain.pkl).
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parent
import lightgbm_arm as gbm

TARGET = 1  # DBP


def gbm_nparams(model):
    """Approx LightGBM 'parameters' = total leaves across all trees (each leaf = 1 value)."""
    dump = model.booster_.dump_model()
    return sum(t["num_leaves"] for t in dump["tree_info"])


def main():
    full = pickle.load(open(ROOT / "data" / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]

    def mat(F, n):
        return np.column_stack([np.asarray(F.get(k, np.full(n, np.nan)), float) for k in keys])

    Mtr, Mte = mat(Ftr, len(ytr)), mat(Fte, len(yte))
    med = gbm.column_medians(Mtr)
    Mtr_i, Mte_i = gbm._impute(Mtr, med), gbm._impute(Mte, med)
    # inner val split from train (subject info not in this cache -> random; fine for tuning)
    rng = np.random.default_rng(0); perm = rng.permutation(len(ytr))
    va = perm[:len(perm) // 6]; tr = perm[len(perm) // 6:]
    print(f"[tune] {len(keys)} features, {len(tr):,} train / {len(va):,} val / {len(yte):,} test")
    results = {}

    # ---- 1. Optuna hard tune ----
    def obj(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 300, 2000),
                 learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 15, 255),
                 max_depth=trial.suggest_int("max_depth", 3, 14),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 random_state=0, verbosity=-1)
        m = lgb.LGBMRegressor(**p)
        m.fit(Mtr_i[tr], ytr[tr, TARGET], eval_set=[(Mtr_i[va], ytr[va, TARGET])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return float(np.abs(m.predict(Mtr_i[va]) - ytr[va, TARGET]).mean())

    print("[tune] Optuna 60 trials ...", flush=True)
    st = optuna.create_study(direction="minimize")
    st.optimize(obj, n_trials=60, show_progress_bar=False)
    best = lgb.LGBMRegressor(**st.best_params, random_state=0, verbosity=-1)
    best.fit(Mtr_i, ytr[:, TARGET])
    mae = float(np.abs(best.predict(Mte_i) - yte[:, TARGET]).mean())
    results["optuna_full"] = {"dbp": mae, "nparams": gbm_nparams(best), "params": st.best_params}
    print(f"[tune] tuned full: test DBP {mae:.3f}  ({gbm_nparams(best):,} leaves)")

    # ---- 2. SFS best combination (subsample rows: SFS is O(F^2 * n * cv) and saturates well
    # before 396k; use 20k for tractability) ----
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.base import clone
    base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, verbosity=-1)
    ss = rng.choice(len(Mtr_i), min(20000, len(Mtr_i)), replace=False)
    print(f"[tune] mlxtend SFS (up to 15 feats, 3-fold, {len(ss):,} rows) ...", flush=True)
    sfs = SFS(clone(base), k_features=(1, 15), forward=True, floating=False,
              scoring="neg_mean_absolute_error", cv=3, n_jobs=1)
    sfs.fit(Mtr_i[ss], ytr[ss, TARGET])
    sidx = list(sfs.k_feature_idx_)
    sm = clone(base).fit(Mtr_i[:, sidx], ytr[:, TARGET])
    smae = float(np.abs(sm.predict(Mte_i[:, sidx]) - yte[:, TARGET]).mean())
    results["sfs"] = {"dbp": smae, "n": len(sidx), "features": [keys[i] for i in sidx],
                      "nparams": gbm_nparams(sm)}
    print(f"[tune] SFS {len(sidx)} feats: DBP {smae:.3f}  {[keys[i] for i in sidx]}")

    # ---- 3. single-tree with top-k feature combinations ----
    imp = np.argsort(-best.feature_importances_)
    print("[tune] single-tree with top-k features:")
    tree_rows = {}
    for kk in [3, 5, 8, 12]:
        cols = list(imp[:kk])
        t = lgb.LGBMRegressor(n_estimators=1, num_leaves=32, max_depth=6, learning_rate=1.0,
                              min_child_samples=50, random_state=0, verbosity=-1)
        t.fit(Mtr_i[:, cols], ytr[:, TARGET])
        tmae = float(np.abs(t.predict(Mte_i[:, cols]) - yte[:, TARGET]).mean())
        tree_rows[kk] = {"dbp": tmae, "features": [keys[i] for i in cols],
                         "nparams": gbm_nparams(t)}
        print(f"   top-{kk:2d}: DBP {tmae:.2f}  ({gbm_nparams(t)} leaves)  {[keys[i] for i in cols[:4]]}...")
    results["single_tree"] = tree_rows

    (ROOT / "data" / "gbm_tune_full.json").write_text(json.dumps(results, indent=2, default=float))
    print("[done] data/gbm_tune_full.json")


if __name__ == "__main__":
    main()
