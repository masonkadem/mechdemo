"""lightgbm_arm.py -- the interpretable reference model built from AUDIT-PASSING features.

Thesis: a faithful deep model's real contribution is DISCOVERING which physiological cues
carry BP signal. Once the audit tells us which cues a faithful model both (a) makes linearly
decodable across its layers and (b) responds to with the physiologically correct sign, we can
hand-build exactly those cues -- plus demographics (age, sex) -- into a gradient-boosted tree.
Because the tree only sees named physiology, it cannot latch onto the dataset-specific
shortcuts that sink the deep models out of distribution.

Two comparisons the paper needs:
  1. audit-selected features vs all features vs demographics-only  (does the audit help?)
  2. WITH vs WITHOUT age+sex, and SHAP by feature                 (age/sex impact, explicit)
"""
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def select_audit_features(results, r2_thresh=0.05):
    """Given one deep model's audit output (from ood_benchmark results[name]), return the cues
    that are BOTH probe-decodable (max layer R^2 > thresh) AND pass their governing-law sign
    test. This is the 'features the faithful model uses' set."""
    probe = results["probe"]                       # {layer: {cue: r2}}
    physics = results["physics"]                    # {cue: {sign_ok, ...}}
    decodable = {}
    for lyr in probe.values():
        for cue, r2 in lyr.items():
            decodable[cue] = max(decodable.get(cue, 0.0), r2)
    keep = []
    for cue, v in physics.items():
        if v["sign_ok"] is True and decodable.get(cue, 0.0) > r2_thresh:
            keep.append(cue)
    return keep, decodable


DEMO_KEYS = ("age", "sex", "bmi", "height", "weight")


def build_feature_table(scalars, demo=None, feature_list=None, n=None, demo_keys=None):
    """Assemble (N, F) design matrix + names from the per-segment cue dict `scalars` and an
    optional demographics dict (any of age/sex/bmi/height/weight). `feature_list` fixes the
    column SCHEMA: every requested cue becomes a column even if absent here (filled with NaN and
    imputed later), so the same model can score a set that lacks a cue -- e.g. PPG-only sets have
    no `pat`/`period`. Without a fixed schema the column count silently drifts and prediction
    fails."""
    if n is None:
        n = len(next(iter(scalars.values()))) if scalars else (len(demo["age"]) if demo else 0)
    keys = feature_list if feature_list is not None else list(scalars)
    cols, names = [], []
    for k in keys:
        col = np.asarray(scalars[k], float) if k in scalars else np.full(n, np.nan)
        cols.append(col); names.append(k)
    # demo_keys is a fixed schema: None = auto (use whatever `demo` has), a tuple (incl. the
    # EMPTY tuple) = exactly these keys. An empty tuple therefore means "no demographics".
    if demo_keys is not None:
        for k in demo_keys:
            col = (np.asarray(demo[k], float) if (demo and k in demo and demo[k] is not None)
                   else np.full(n, np.nan))       # fixed schema: emit NaN for a missing field
            cols.append(col); names.append(k)
    elif demo:
        for k in DEMO_KEYS:
            if k in demo and demo[k] is not None:
                cols.append(np.asarray(demo[k], float)); names.append(k)
    X = np.column_stack(cols) if cols else np.zeros((n, 0))
    return X, names


def _impute(X, medians=None):
    """Median-impute NaNs column-wise. If `medians` (from the TRAIN set) is given, use those --
    critical when a whole column is NaN at score time (e.g. demographics absent on an external
    set): imputing within-column would fall back to 0.0, an out-of-range value that silently
    misroutes trees. Pass train medians so absent columns take the training median instead."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isfinite(col)
        if medians is not None and np.isfinite(medians[j]):
            fill = medians[j]                            # train median (correct for absent cols)
        elif m.any():
            fill = np.median(col[m])
        else:
            fill = 0.0
        col[~m] = fill
        X[:, j] = col
    return X


def column_medians(X):
    """Per-column medians (ignoring NaN) to reuse as train-set impute values."""
    return np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan
                     for c in X.T])


def train_gbm(Xtr, ytr, Xva, yva, names, seed=0):
    """One LightGBM regressor per BP target (SBP, DBP), early-stopped on val MAE.
    Returns (models, val_mae). The TRAIN column medians are attached to each model as
    `_train_medians` so predict_gbm imputes score-time NaNs (incl. whole-column-absent
    features like demographics) with train values, not out-of-range within-column fallbacks."""
    med = column_medians(Xtr)
    Xtr, Xva = _impute(Xtr, med), _impute(Xva, med)
    models, vmae = [], []
    for t in range(ytr.shape[1]):
        if lgb is not None:
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, random_state=seed,
                                  verbosity=-1)
            m.fit(Xtr, ytr[:, t], eval_set=[(Xva, yva[:, t])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03,
                                              validation_fraction=0.15, random_state=seed)
            m.fit(Xtr, ytr[:, t])
        m._train_medians = med
        models.append(m)
        vmae.append(float(np.abs(m.predict(Xva) - yva[:, t]).mean()))
    return models, vmae


def predict_gbm(models, X):
    med = getattr(models[0], "_train_medians", None)
    X = _impute(X, med)
    return np.column_stack([m.predict(X) for m in models])


def feature_importance(models, names, target=1):
    """Gain-based importance for one target -> sorted [(feature, gain)]. This is the GBM's
    direct 'what it uses' readout, to line up against the deep models' probe+perturbation audit."""
    m = models[target]
    imp = m.feature_importances_ if hasattr(m, "feature_importances_") else \
        getattr(m, "feature_importances_", np.zeros(len(names)))
    return sorted(zip(names, imp.astype(float)), key=lambda t: -t[1])


def shap_by_feature(models, X, names, target=1, n=500, seed=0):
    """Mean |SHAP| per feature for one target -> {feature: mean_abs_shap}. Used to quantify
    age/sex contribution explicitly. Needs the `shap` package; returns None if unavailable."""
    try:
        import shap
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    Xi = _impute(X)
    idx = rng.choice(len(Xi), min(n, len(Xi)), replace=False)
    expl = shap.TreeExplainer(models[target])
    sv = expl.shap_values(Xi[idx])
    return {nm: float(np.abs(sv[:, j]).mean()) for j, nm in enumerate(names)}


def demographics_ablation(feat_full, names, y, split_fn, seed=0):
    """Train GBM WITH and WITHOUT demographics (age/sex/bmi/height/weight) on identical splits
    and report the MAE delta + each demographic's SHAP. `split_fn()` -> (tr_idx, va_idx, te_idx).
    Directly answers 'how much do demographics matter'."""
    tr, va, te = split_fn()
    demo_cols = [j for j, n in enumerate(names) if n in DEMO_KEYS]
    keep_no = [j for j in range(len(names)) if j not in demo_cols]
    out = {}
    for tag, cols in [("with_demo", list(range(len(names)))), ("no_demo", keep_no)]:
        nm = [names[j] for j in cols]
        mdl, _ = train_gbm(feat_full[tr][:, cols], y[tr], feat_full[va][:, cols], y[va], nm, seed)
        pred = predict_gbm(mdl, feat_full[te][:, cols])
        out[tag] = {"mae": np.abs(pred - y[te]).mean(0).tolist(), "models": mdl, "names": nm}
    if demo_cols:
        nm = out["with_demo"]["names"]
        sh = shap_by_feature(out["with_demo"]["models"], feat_full[te], nm, target=1)
        out["demo_shap"] = {k: sh[k] for k in DEMO_KEYS if sh and k in sh} if sh else None
    out["delta_dbp"] = out["no_demo"]["mae"][1] - out["with_demo"]["mae"][1]
    return out
