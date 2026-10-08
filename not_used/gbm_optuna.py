"""gbm_optuna.py -- upgraded interpretable-model study on the cached cue features:

  1. Optuna-tuned LightGBM (full) : tune hyperparameters on the validation split, then report
     ID + OOD DBP MAE. This is the strongest interpretable model.
  2. mlxtend SFS                  : proper Sequential Forward Selection with an inner CV, to
     find the feature COMBINATION that works best (not just greedy single-add).
  3. Optuna single-tree          : n_estimators=1 (one decision tree), tuned depth/leaves, so
     the whole tree can be exported and visualized.

Runs from the cached subject-diverse cue features (data/_fam_cues_*.pkl).

    python gbm_optuna.py --cache data/_fam_cues_vitaldb_full_calfree_8000.pkl --trials 60
"""
import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent

import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import lightgbm_arm as gbm
import physics_audit as pa
import mechlib

MIMIC = "C:/Users/mason/OneDrive - McMaster University/2026/BP"
TARGET = 1                                            # DBP

READABLE = {
    "apg_d_a": "APG d/a", "apg_b_a": "APG b/a", "apg_c_a": "APG c/a", "apg_e_a": "APG e/a",
    "pat": "PAT", "pat_peak": "PAT (peak)", "pat_foot": "PAT (foot)", "ptt_var": "PTT var",
    "xcorr_lag": "xcorr lag", "xcorr_peak": "xcorr peak", "xcorr_width": "xcorr width",
    "age": "Age", "sex": "Sex", "bmi": "BMI", "hr": "HR", "period": "Cardiac period",
    "rise": "Rise time", "aix": "AIx", "apg": "APG b/a", "notch": "Dicrotic notch",
    "decay": "Diastolic decay", "kurt": "Kurtosis", "peak": "Peak height", "amp": "Amplitude",
    "hfd": "Fractal (Higuchi)", "katz_fd": "Fractal (Katz)", "spec_ent": "Spectral entropy",
    "vpg_max": "VPG max", "aging_idx": "Aging index", "crest": "Crest time",
    "pw25": "Pulse width 25%", "pw50": "Pulse width 50%", "sys_area": "Systolic area",
    "rr_mean": "RR mean", "rr_sdnn": "RR SDNN", "rr_rmssd": "RR RMSSD", "qrs_amp": "QRS amp",
}


def build(sc, dm, cues, dkeys, n):
    X, nm = gbm.build_feature_table(sc, dm, cues, n=n, demo_keys=dkeys)
    return gbm._impute(X), nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/_fam_cues_vitaldb_full_calfree_8000.pkl")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--sfs-max", type=int, default=12, help="max features for SFS")
    ap.add_argument("--mimic-patients", type=int, default=200)
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        blob = pickle.load(f)
    sctr, ytr, gtr, dmtr = blob["tr"]
    scva, yva, gva, dmva = blob["va"]
    scte, yte, gte, dmte = blob["te"]
    ood = blob["ood"]

    cue_feats = [k for k, v in sctr.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3]
    demo_feats = tuple(k for k in ("age", "sex", "bmi") if dmtr and k in dmtr)
    Xtr, names = build(sctr, dmtr, cue_feats, demo_feats, len(ytr))
    Xva, _ = build(scva, dmva, cue_feats, demo_feats, len(yva))
    Xte, _ = build(scte, dmte, cue_feats, demo_feats, len(yte))
    print(f"[opt] {len(names)} features, {len(Xtr)} train / {len(Xva)} val / {len(Xte)} test seg")

    def ood_row(model_or_models, cols=None):
        """DBP MAE on each OOD set for a fitted model (single or [sbp,dbp] list)."""
        out = {}
        for nm, e in ood.items():
            edemo = e["demo"]
            Xf, _ = gbm.build_feature_table(e["sc"], edemo, cue_feats, n=len(e["y"]),
                                            demo_keys=demo_feats)
            Xf = gbm._impute(Xf)
            if cols is not None:
                Xf = Xf[:, cols]
            p = model_or_models.predict(Xf) if hasattr(model_or_models, "predict") \
                else model_or_models[TARGET].predict(Xf)
            bs = pa.bootstrap_mae(np.stack([p, p], 1), e["y"], e["g"])
            out[nm] = round(bs["mae"][TARGET], 2)
        return out

    results = {}

    # ---- 1. Optuna-tuned full LightGBM (DBP) --------------------------------
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=0, verbosity=-1,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr[:, TARGET], eval_set=[(Xva, yva[:, TARGET])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
        return float(np.abs(m.predict(Xva) - yva[:, TARGET]).mean())

    print(f"[opt] tuning full LightGBM ({args.trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = lgb.LGBMRegressor(**study.best_params, random_state=0, verbosity=-1)
    best.fit(Xtr, ytr[:, TARGET], eval_set=[(Xva, yva[:, TARGET])],
             callbacks=[lgb.early_stopping(40, verbose=False)])
    full_id = float(np.abs(best.predict(Xte) - yte[:, TARGET]).mean())
    imp = sorted(zip(names, best.feature_importances_.astype(float)), key=lambda t: -t[1])
    print(f"[opt] tuned full: ID DBP {full_id:.2f}  (val {study.best_value:.2f})")
    print("[opt] top features:", ", ".join(f"{READABLE.get(n,n)}={v:.0f}" for n, v in imp[:8]))
    results["optuna_full"] = {"id_dbp": full_id, "val_dbp": study.best_value,
                              "params": study.best_params,
                              "importance": [(READABLE.get(n, n), v) for n, v in imp],
                              "ood": ood_row(best)}

    # ---- 2. mlxtend SFS (best feature combination) --------------------------
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.base import clone
    base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, verbosity=-1)
    print(f"[opt] mlxtend SFS (forward, up to {args.sfs_max} features, 3-fold CV)...")
    sfs = SFS(clone(base), k_features=(1, args.sfs_max), forward=True, floating=False,
              scoring="neg_mean_absolute_error", cv=3, n_jobs=1)
    sfs.fit(Xtr, ytr[:, TARGET])
    sel_idx = list(sfs.k_feature_idx_)
    sel_names = [names[i] for i in sel_idx]
    sfs_model = clone(base).fit(Xtr[:, sel_idx], ytr[:, TARGET])
    sfs_id = float(np.abs(sfs_model.predict(Xte[:, sel_idx]) - yte[:, TARGET]).mean())
    print(f"[opt] SFS selected {len(sel_names)}: {[READABLE.get(n,n) for n in sel_names]}")
    print(f"[opt] SFS: ID DBP {sfs_id:.2f}")
    # per-k curve
    curve = {int(k): round(-v["avg_score"], 3) for k, v in sfs.get_metric_dict().items()}
    results["sfs"] = {"selected": sel_names, "selected_readable": [READABLE.get(n, n) for n in sel_names],
                      "id_dbp": sfs_id, "cv_curve": curve, "ood": ood_row(sfs_model, cols=sel_idx)}

    # ---- 3. Optuna single decision tree (visualizable) ----------------------
    def obj_tree(trial):
        p = dict(n_estimators=1, num_leaves=trial.suggest_int("num_leaves", 4, 64),
                 max_depth=trial.suggest_int("max_depth", 2, 8),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
                 learning_rate=1.0, random_state=0, verbosity=-1)
        m = lgb.LGBMRegressor(**p)
        m.fit(Xtr, ytr[:, TARGET])
        return float(np.abs(m.predict(Xva) - yva[:, TARGET]).mean())

    print("[opt] tuning single tree (n_estimators=1)...")
    st = optuna.create_study(direction="minimize")
    st.optimize(obj_tree, n_trials=max(20, args.trials // 2), show_progress_bar=False)
    tree = lgb.LGBMRegressor(n_estimators=1, learning_rate=1.0, random_state=0, verbosity=-1,
                             **st.best_params)
    tree.fit(Xtr, ytr[:, TARGET])
    tree_id = float(np.abs(tree.predict(Xte) - yte[:, TARGET]).mean())
    print(f"[opt] single tree: ID DBP {tree_id:.2f}  (leaves<= {st.best_params['num_leaves']}, "
          f"depth<= {st.best_params['max_depth']})")
    # export the tree
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = lgb.plot_tree(tree, tree_index=0, figsize=(20, 10),
                           show_info=["split_gain", "internal_value", "leaf_count"])
        ax.figure.savefig(ROOT / "figures" / "gbm_single_tree.png", dpi=140, bbox_inches="tight")
        plt.close(ax.figure)
        print("[opt] tree -> figures/gbm_single_tree.png")
    except Exception as ex:
        print(f"[opt] tree plot skipped: {ex}")
    results["single_tree"] = {"id_dbp": tree_id, "params": st.best_params, "ood": ood_row(tree)}

    (ROOT / "data" / "gbm_optuna.json").write_text(json.dumps(results, indent=2, default=float))
    print("\n[done] data/gbm_optuna.json")
    # summary
    print("\n%-18s %6s | %s" % ("model", "ID", "  ".join(f"{s:>8}" for s in ood)))
    for tag in ["optuna_full", "sfs", "single_tree"]:
        r = results[tag]
        print("%-18s %6.2f | %s" % (tag, r["id_dbp"],
              "  ".join(f"{r['ood'][s]:8.2f}" for s in ood)))


if __name__ == "__main__":
    main()
