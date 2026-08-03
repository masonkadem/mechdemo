"""feature_reference.py -- what each feature is, what it means physiologically, and how well it
actually correlates with blood pressure.

Produces three artefacts:

  1. FEATURE_REFERENCE.md / .tex  -- one row per feature: equation, plain-language meaning, the
     theoretical route by which it should relate to BP, and the MEASURED correlation, reported
     both pooled and WITHIN subject.
  2. figures/fig_feature_scatter.png -- scatter with a fitted line for the strongest features.
  3. figures/fig_shap_reference.png  -- SHAP dependence for SBP and DBP from a LightGBM model
     trained on the full feature set.

Why both pooled and within-subject correlations. Pooled correlations across subjects are
dominated by between-subject differences: anything that tracks age or body size will correlate
with BP without carrying any information about a given person's pressure changing. The
within-subject correlation is the one that matters for a device, and the two frequently disagree
in sign. Both are reported so the disagreement is visible rather than hidden.

Stiffness index note. The published SI is height / delta-T, which has velocity units and is a
direct pulse-wave-velocity proxy. PulseDB carries age, sex and BMI but NOT height, so the
canonical SI cannot be computed here. Two substitutes are evaluated and labelled as such:
delta_t alone (the timing, unscaled) and a BMI-scaled variant. Neither is the published index and
neither should be reported as SI without that caveat.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from gbm_mechanism import plain

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAVY, RED, GREY = "#2f4b7c", "#c1543b", "#9aa0a6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

# equation and the theoretical route to BP, for the features with a published basis
THEORY = {
    "aix": ("(S_peak - D_peak) / S_peak x 100%",
            "stiffer arteries return the reflected wave sooner, so it merges with the systolic "
            "peak and augments it; AIx rises with age and vascular disease"),
    "reflect_idx": ("D_peak / S_peak x 100%",
                    "size of the reflected wave relative to the forward wave; a vascular tone "
                    "and stiffness indicator"),
    "crest": ("t(systolic peak) - t(onset)",
              "upstroke duration, set by how fast the ejected volume distends the vessel; "
              "shortens with arteriosclerosis"),
    "crest_time_ratio": ("crest time / cycle time",
                         "crest time made rate-independent, which matters because heart rate is "
                         "itself a confound"),
    "delta_t": ("t(diastolic peak) - t(systolic peak)",
                "round-trip time of the reflected wave; the timing half of the stiffness index"),
    "stiffness_index": ("height / delta_t",
                        "the only index here with velocity units, so a direct pulse-wave-"
                        "velocity proxy rather than a shape descriptor"),
    "apg_b_a": ("b / a", "b/a rises with arterial stiffness"),
    "apg_c_a": ("c / a", "falls with arterial stiffness"),
    "apg_d_a": ("d / a", "falls with arterial stiffness"),
    "apg_e_a": ("e / a", "falls with arterial stiffness"),
    "takazawa": ("(b - c - d - e) / a", "composite vascular ageing index"),
    "ushiro": ("(c + d - b) / a", "ageing index variant"),
    "pat_foot": ("t(PPG foot) - t(R peak)",
                 "PAT = PEP + PTT. Moens-Korteweg predicts higher BP stiffens the artery and "
                 "shortens transit, but PEP is cardiac and contaminates the interval"),
    "hr": ("60 / RR interval",
           "rate covaries with BP through autonomic drive, not through the arterial law; the "
           "audit identified it as the dominant shortcut"),
    "rise": ("upstroke duration", "faster upstroke with stiffer vessels"),
    "notch_time": ("t(dicrotic notch) - t(onset)",
                   "notch timing tracks aortic valve closure and wave reflection"),
    "notch_depth": ("depth of the dicrotic notch",
                    "notch flattens as peripheral resistance and stiffness rise"),
    "hfd": ("Higuchi fractal dimension",
            "waveform complexity; no direct arterial-law route, included as a shape summary"),
    "age": ("--", "arteries stiffen with age; the strongest single predictor here"),
    "bmi": ("--", "body composition covaries with pressure through several routes"),
}


def within_r(x, y, g, min_seg=40):
    rs = []
    for s in np.unique(g):
        m = g == s
        if m.sum() < min_seg:
            continue
        a, b = x[m], y[m]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_seg // 2 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            continue
        rs.append(stats.spearmanr(a[ok], b[ok]).statistic)
    return (float(np.nanmedian(rs)) if rs else np.nan), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="also compute SHAP dependence")
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Fte, yte = full["Fte"], full["yte"]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    n = len(yte)
    g = d["gte"][:n]
    sbp, dbp = yte[:, 0], yte[:, 1]

    feats = dict(Fte)
    feats["age"] = np.asarray(d["age_te"], float)[:n]
    feats["bmi"] = np.asarray(d["bmi_te"], float)[:n]
    # BMI-scaled stiffness substitute -- NOT the published height/delta_t index
    if "notch_time" in feats and "crest" in feats:
        dt = np.asarray(feats["notch_time"], float) - np.asarray(feats["crest"], float)
        feats["delta_t_proxy"] = dt
        with np.errstate(divide="ignore", invalid="ignore"):
            feats["si_bmi_proxy"] = feats["bmi"] / np.where(np.abs(dt) > 1e-6, dt, np.nan)

    rows = []
    for k, v in feats.items():
        x = np.asarray(v, float)
        if np.isfinite(x).mean() < 0.3 or np.nanstd(x) < 1e-12:
            continue
        ok = np.isfinite(x)
        eq, why = THEORY.get(k, ("--", ""))
        rs_p = float(stats.spearmanr(x[ok], sbp[ok]).statistic)
        rd_p = float(stats.spearmanr(x[ok], dbp[ok]).statistic)
        rs_w, nsub = within_r(x, sbp, g)
        rd_w, _ = within_r(x, dbp, g)
        rows.append({"feature": k, "plain": plain(k), "equation": eq, "theory": why,
                     "r_sbp_pooled": rs_p, "r_dbp_pooled": rd_p,
                     "r_sbp_within": rs_w, "r_dbp_within": rd_w,
                     "n_subj": nsub, "finite": float(ok.mean())})

    rows.sort(key=lambda r: -abs(r["r_dbp_pooled"]))
    (DATA / "feature_reference.json").write_text(json.dumps(rows, indent=2, default=float))

    # ---- markdown -----------------------------------------------------------
    md = ["# Feature reference", "",
          "Spearman correlations with blood pressure on the VitalDB held-out split "
          f"({n:,} segments, {len(np.unique(g))} subjects).", "",
          "**Pooled vs within-subject.** Pooled correlations are dominated by between-subject "
          "differences: anything tracking age or body size correlates with BP without saying "
          "anything about a given person's pressure changing. The within-subject column is what "
          "a device needs. Where the two disagree in sign, the pooled value is the misleading "
          "one.", "",
          "| feature | equation | plain meaning | r SBP (pooled / within) | "
          "r DBP (pooled / within) |", "|---|---|---|---|---|"]
    for r in rows[:40]:
        md.append(f"| `{r['feature']}` | {r['equation']} | {r['plain']} | "
                  f"{r['r_sbp_pooled']:+.3f} / {r['r_sbp_within']:+.3f} | "
                  f"{r['r_dbp_pooled']:+.3f} / {r['r_dbp_within']:+.3f} |")
    md += ["", "## Theoretical route to blood pressure", "",
           "| feature | why it should relate to BP |", "|---|---|"]
    for r in rows:
        if r["theory"]:
            md.append(f"| `{r['feature']}` | {r['theory']} |")
    md += ["", "## Stiffness index", "",
           "The published stiffness index is `height / delta_t`, the one index in this set with "
           "velocity units and therefore a direct pulse-wave-velocity proxy. **PulseDB does not "
           "carry height**, so the canonical index cannot be computed here. `delta_t_proxy` "
           "(timing alone) and `si_bmi_proxy` (BMI-scaled) are substitutes and are labelled as "
           "such; neither should be reported as SI."]
    (ROOT / "FEATURE_REFERENCE.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[done] FEATURE_REFERENCE.md ({len(rows)} features)")

    # ---- scatter figure -----------------------------------------------------
    top = rows[:8]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    idx = np.random.default_rng(0).choice(n, min(4000, n), replace=False)
    for ax, r in zip(axes.ravel(), top):
        x = np.asarray(feats[r["feature"]], float)[idx]
        y = dbp[idx]
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=3, alpha=.18, color=NAVY, edgecolors="none")
        if ok.sum() > 50:
            b = np.polyfit(x[ok], y[ok], 1)
            xs = np.linspace(np.percentile(x[ok], 1), np.percentile(x[ok], 99), 30)
            ax.plot(xs, np.polyval(b, xs), color=RED, lw=1.6)
            ax.set_xlim(np.percentile(x[ok], 1), np.percentile(x[ok], 99))
        ax.set_title(f"{r['feature']}", fontsize=9, fontweight="bold", loc="left")
        ax.set_xlabel(r["plain"][:34], fontsize=7.5)
        ax.set_ylabel("DBP (mmHg)", fontsize=8)
        ax.text(.04, .06, f"pooled {r['r_dbp_pooled']:+.2f}\nwithin {r['r_dbp_within']:+.2f}",
                transform=ax.transAxes, fontsize=7.5, color=GREY)
    fig.suptitle("Strongest features against diastolic BP (line = linear fit)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_feature_scatter.{ext}", dpi=190,
                    bbox_inches="tight")
    plt.close(fig)
    print("[done] figures/fig_feature_scatter.png")

    if args.shap:
        shap_figure(full, feats, yte, n)


def shap_figure(full, feats, yte, n):
    import lightgbm as lgb
    import shap
    import lightgbm_arm as gbm
    Ftr, ytr = full["Ftr"], full["ytr"]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    Mtr = np.column_stack([np.asarray(Ftr[k], float) for k in keys]
                          + [np.asarray(d["age_tr"], float), np.asarray(d["bmi_tr"], float)])
    names = keys + ["age", "bmi"]
    med = gbm.column_medians(Mtr)
    Mte = np.column_stack([np.asarray(feats.get(k, np.full(n, np.nan)), float)[:n]
                           for k in names])
    sub = np.random.default_rng(0).choice(n, min(3000, n), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for row, (tgt, lab) in enumerate([(0, "SBP"), (1, "DBP")]):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=63,
                              random_state=0, verbosity=-1)
        m.fit(gbm._impute(Mtr, med), ytr[:, tgt])
        sv = shap.TreeExplainer(m).shap_values(gbm._impute(Mte[sub], med))
        order = np.argsort(-np.abs(sv).mean(0))[:4]
        for ax, j in zip(axes[row], order):
            xv = gbm._impute(Mte[sub], med)[:, j]
            ax.scatter(xv, sv[:, j], s=3, alpha=.2, color=NAVY, edgecolors="none")
            ax.axhline(0, color="k", lw=.6, alpha=.4)
            ax.set_xlabel(plain(names[j])[:34], fontsize=7.5)
            ax.set_ylabel(f"SHAP on {lab}", fontsize=8)
            ax.set_title(names[j], fontsize=9, fontweight="bold", loc="left")
            ax.set_xlim(np.percentile(xv, 1), np.percentile(xv, 99))
    fig.suptitle("SHAP dependence, LightGBM on the full feature set",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_shap_reference.{ext}", dpi=190,
                    bbox_inches="tight")
    plt.close(fig)
    print("[done] figures/fig_shap_reference.png")


if __name__ == "__main__":
    main()
