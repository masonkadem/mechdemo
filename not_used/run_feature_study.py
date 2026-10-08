"""run_feature_study.py -- exhaustive feature study end to end.

  1. extract the ~83-feature library (features_full) on 8k subject-diverse VitalDB segments, cache
  2. pre-filter: drop near-constant and redundant (|corr| > 0.95) features
  3. for SBP and DBP: LightGBM (all) + mlxtend SFS (forward, CV) -> selected set + MAE
  4. correlation matrix of the top-correlated features
  5. SHAP dependence plots (colored by interacting feature) for SBP and DBP

    python run_feature_study.py --n 8000 --sfs-max 15
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import lightgbm as lgb

import mechlib
import features_full as ff
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"


def get_cache(data, n, seed=0):
    cache = ROOT / "data" / f"_feat_full_{Path(data).stem}_{n}.pkl"
    if cache.exists():
        print(f"[fs] loading cached features {cache.name}")
        return pickle.load(open(cache, "rb"))
    d = mechlib.load_mini(data); fs = d["fs"]
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(d["gte"]), min(n, len(d["gte"])), replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    print(f"[fs] extracting library on {len(X)} seg / {len(np.unique(d['gte'][sel]))} subj...")
    F = ff.compute_full(X, fs, ppg_ch=1, ecg_ch=0)
    blob = {"F": F, "y": d["yte"][sel], "g": d["gte"][sel], "fs": fs}
    pickle.dump(blob, open(cache, "wb"))
    print(f"[fs] cached -> {cache.name}")
    return blob


def matrix(F, keep):
    names = keep
    M = np.column_stack([np.asarray(F[k], float) for k in names])
    for j in range(M.shape[1]):                 # median impute
        col = M[:, j]; m = np.isfinite(col)
        col[~m] = np.median(col[m]) if m.any() else 0.0
    return M, names


def prefilter(F, y, max_corr=0.95):
    """Drop near-constant; then greedily drop one of each pair with |corr|>max_corr (keep the
    one more correlated with DBP)."""
    keys = [k for k, v in F.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3
            and np.nanstd(np.asarray(v, float)) > 1e-9]
    M, names = matrix(F, keys)
    dbp = y[:, 1]
    with_bp = {k: abs(spearmanr(M[:, i], dbp).correlation) for i, k in enumerate(names)}
    C = np.corrcoef(M.T)
    drop = set()
    for i in range(len(names)):
        if names[i] in drop:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in drop:
                continue
            if abs(C[i, j]) > max_corr:
                lose = names[j] if with_bp[names[i]] >= with_bp[names[j]] else names[i]
                drop.add(lose)
    keep = [k for k in names if k not in drop]
    print(f"[fs] prefilter: {len(names)} -> {len(keep)} (dropped {len(drop)} redundant)")
    return keep


def sfs_select(M, y, names, target, sfs_max):
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.base import clone
    base = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                             random_state=0, verbosity=-1)
    sfs = SFS(clone(base), k_features=(1, sfs_max), forward=True, floating=False,
              scoring="neg_mean_absolute_error", cv=3, n_jobs=1)
    sfs.fit(M, y[:, target])
    idx = list(sfs.k_feature_idx_)
    return [names[i] for i in idx], idx, sfs.get_metric_dict()


def gbm_mae(M, y, target, tr, te):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    m.fit(M[tr], y[tr, target])
    return float(np.abs(m.predict(M[te]) - y[te, target]).mean()), m


def corr_matrix_fig(F, y, keep, path, topn=20):
    dbp = y[:, 1]; sbp = y[:, 0]
    rows = []
    for k in keep:
        v = np.asarray(F[k], float); m = np.isfinite(v)
        rows.append((k, spearmanr(v[m], sbp[m]).correlation, spearmanr(v[m], dbp[m]).correlation))
    rows.sort(key=lambda r: -max(abs(r[1]), abs(r[2])))
    rows = rows[:topn]
    M = np.array([[r[1], r[2]] for r in rows])
    fig, ax = plt.subplots(figsize=(4.4, 0.33 * len(rows) + 1.2))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4), aspect="auto")
    ax.set_xticks([0, 1], ["SBP", "DBP"], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j]) > 0.28 else "black")
    ax.set_title(f"Top {len(rows)} features by |correlation| with BP\n(Spearman)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.3, shrink=0.6, label="rho")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {path.name}")


def shap_fig(M, y, names, target, tname, path, inter_thresh=0.05):
    """SHAP dependence for the top-6 features. Color each panel by its strongest INTERACTING
    feature computed from EXACT SHAP interaction values -- but only when that interaction
    exceeds `inter_thresh` (relative to main effects); otherwise the panel is grey (no real
    interaction, so coloring would mislead)."""
    import shap
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
    m.fit(M, y[:, target])
    expl = shap.TreeExplainer(m)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(M), min(2000, len(M)), replace=False)
    Xs = M[idx]; sv = expl.shap_values(Xs)
    order = np.argsort(-np.abs(sv).mean(0))[:6]

    # exact SHAP interaction values on a subsample (O(n * F^2), so cap rows)
    isub = idx[:500]
    iv = expl.shap_interaction_values(M[isub])            # (n, F, F)
    inter = np.abs(iv).mean(0); np.fill_diagonal(inter, 0.0)
    main = np.abs(np.array([iv[:, i, i] for i in range(len(names))]).T).mean(0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, j in zip(axes.ravel(), order):
        xj = Xs[:, j]; sj = sv[:, j]
        # strongest interacting partner + its strength relative to main effects
        strength = inter[j] / (main[j] + main + 1e-9)
        c = int(np.argmax(strength)); s_val = strength[c]
        if s_val >= inter_thresh:
            cv = Xs[:, c]; lo, hi = np.nanpercentile(cv, [5, 95])
            sc = ax.scatter(xj, sj, c=cv, cmap="coolwarm", norm=Normalize(lo, hi), s=16,
                            alpha=0.8, edgecolor="none")
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(f"{names[c]}  (interaction {s_val:.2f})", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            ax.scatter(xj, sj, c="#9aa0a6", s=14, alpha=0.6, edgecolor="none")
            ax.text(0.98, 0.04, "no strong interaction", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color="#777", style="italic")
        q = np.unique(np.quantile(xj, np.linspace(0, 1, 12)))
        cx = 0.5 * (q[:-1] + q[1:])
        cy = [np.median(sj[(xj >= q[i]) & (xj < q[i + 1])]) for i in range(len(q) - 1)]
        ax.plot(cx, cy, "-", color="black", lw=2.4, zorder=4)
        ax.plot(cx, cy, "o", color="black", ms=5, zorder=5, mec="white", mew=0.8)
        ax.axhline(0, color="#444", lw=0.9, ls=":")
        ax.set_title(names[j], fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{names[j]} value", fontsize=9)
        ax.set_ylabel(f"SHAP: effect on {tname} (mmHg)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"SHAP dependence for {tname}: nonlinear effects (black line) + interactions "
                 f"(color only when interaction > {inter_thresh})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {path.name}  ({tname} top: {', '.join(names[j] for j in order)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--sfs-max", type=int, default=15)
    args = ap.parse_args()

    blob = get_cache(args.data, args.n)
    F, y, g = blob["F"], blob["y"], blob["g"]
    print(f"[fs] {len(F)} raw features, {len(y)} segments")

    keep = prefilter(F, y)
    M, names = matrix(F, keep)

    # subject-disjoint train/test for honest MAE
    subs = np.unique(g); rng = np.random.default_rng(0); rng.shuffle(subs)
    tr_s = set(subs[:int(0.7 * len(subs))].tolist())
    tr = np.isin(g, list(tr_s)); te = ~tr

    results = {"n_features_raw": len(F), "n_features_kept": len(keep), "features": names}
    for target, tname in [(0, "SBP"), (1, "DBP")]:
        all_mae, _ = gbm_mae(M, y, target, np.where(tr)[0], np.where(te)[0])
        sel, sidx, _ = sfs_select(M[np.where(tr)[0]], y[np.where(tr)[0]], names, target, args.sfs_max)
        sel_mae, _ = gbm_mae(M[:, sidx], y, target, np.where(tr)[0], np.where(te)[0])
        results[tname] = {"all_mae": all_mae, "n_all": len(names),
                          "sfs_selected": sel, "sfs_mae": sel_mae}
        print(f"[fs] {tname}: all-{len(names)} MAE {all_mae:.2f} | "
              f"SFS-{len(sel)} MAE {sel_mae:.2f}  {sel}")
        shap_fig(M, y, names, target, tname, FIG / f"fig_shap_{tname.lower()}.png")

    corr_matrix_fig(F, y, keep, FIG / "fig_corr_top.png")
    (ROOT / "data" / "feature_study.json").write_text(json.dumps(results, indent=2, default=float))
    print("[done] data/feature_study.json")


if __name__ == "__main__":
    main()
