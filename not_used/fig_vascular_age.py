"""fig_vascular_age.py -- is vascular ageing from PPG a viable project?

Four panels, each answering one question a supervisor or reviewer would ask before committing
time to it:

  a  does PPG morphology predict age at all?          predicted vs chronological, patient-disjoint
  b  is it better than knowing nothing?               MAE against baselines and the literature band
  c  which features carry it, and is that physiology?  gain ranking, APG waves highlighted
  d  does the residual mean anything?                  the vascular-age claim, tested not assumed

Panel d is the one that decides the project. Predicting age from a waveform is only a
"vascular age" if the residual (predicted - chronological) tracks things that should accompany
accelerated vascular ageing. Otherwise it is an age regressor and the residual is just error.

    python fig_vascular_age.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LIT = (6.0, 9.0)          # published PPG age-prediction MAE range, dedicated finger sensors

C_MAIN, C_ALT, C_REF, C_BAD = "#0072B2", "#009E73", "#8A8A8A", "#C0392B"


def cv_predict(X, y, S, kind, n_splits=5):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=S):
        if kind == "ridge":
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((X[tr] - mu) / sd, y[tr])
            pred[te] = m.predict((X[te] - mu) / sd)
        else:
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                  random_state=0, verbosity=-1).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return pred


def main():
    df = pd.read_csv(DATA / "vascage_train.csv")
    meta = {"s", "age", "sex", "bmi"}
    feats = [c for c in df.columns if c not in meta]
    D = df.dropna(subset=["age"]).copy()
    D[feats] = D[feats].apply(lambda c: c.fillna(c.median()))
    X, y, S = D[feats].values, D.age.values, D.s.values
    print(f"[data] {len(y)} patients, {len(feats)} PPG features, "
          f"age {y.min():.0f}-{y.max():.0f} (SD {y.std():.1f})")

    p_ridge = cv_predict(X, y, S, "ridge")
    p_gbm = cv_predict(X, y, S, "gbm")
    mae_r, mae_g = np.abs(p_ridge - y).mean(), np.abs(p_gbm - y).mean()
    base_med = np.abs(y - np.median(y)).mean()
    # a fair "demographics only" reference: what sex+BMI alone buy
    Xd = D[["sex", "bmi"]].values
    p_demo = cv_predict(Xd, y, S, "gbm")
    mae_d = np.abs(p_demo - y).mean()
    best, p_best = ("LightGBM", p_gbm) if mae_g <= mae_r else ("ridge", p_ridge)
    mae_b = min(mae_g, mae_r)
    r_best = np.corrcoef(p_best, y)[0, 1]
    print(f"[eval] ridge {mae_r:.2f} | lightgbm {mae_g:.2f} | demo-only {mae_d:.2f} | "
          f"median {base_med:.2f}  (r={r_best:+.2f})")

    res = p_best - y
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, random_state=0, verbosity=-1).fit(X, y)
    imp = sorted(zip(feats, m.booster_.feature_importance("gain")), key=lambda t: -t[1])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.linewidth": 0.9, "legend.fontsize": 9})
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 9.4), constrained_layout=True)

    # ---- a: predicted vs actual -------------------------------------------
    a = ax[0, 0]
    a.scatter(y, p_best, s=16, alpha=0.45, color=C_MAIN, edgecolor="none")
    lo, hi = y.min() - 3, y.max() + 3
    a.plot([lo, hi], [lo, hi], color="#333", lw=1.2, ls="--", label="perfect")
    z = np.polyfit(y, p_best, 1)
    a.plot([lo, hi], np.polyval(z, [lo, hi]), color=C_BAD, lw=2,
           label=f"fit (slope {z[0]:.2f})")
    a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("chronological age (years)")
    a.set_ylabel("predicted age from PPG (years)")
    a.set_title(f"a   PPG predicts age    MAE {mae_b:.1f} yr,  r = {r_best:+.2f}",
                loc="left", fontsize=11.5)
    a.legend(frameon=False, loc="upper left")

    # ---- b: is it better than knowing nothing -----------------------------
    b = ax[0, 1]
    names = ["predict\nmedian age", "sex + BMI\nonly", "PPG\nridge", "PPG\nLightGBM"]
    vals = [base_med, mae_d, mae_r, mae_g]
    cols = [C_REF, C_REF, C_ALT, C_MAIN]
    bars = b.bar(names, vals, color=cols, width=0.6)
    b.axhspan(LIT[0], LIT[1], color="#F0E6D2", zorder=0)
    b.text(3.42, np.mean(LIT), "published PPG\nage models\n(finger sensors)",
           fontsize=8.5, color="#8A7A5A", va="center", ha="center")
    for rect, v in zip(bars, vals):
        b.annotate(f"{v:.1f}", (rect.get_x() + rect.get_width() / 2, v), (0, 3),
                   textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
    b.set_xlim(-0.6, 4.2)
    b.set_ylabel("age prediction error (MAE, years)")
    b.set_title("b   how much does the waveform add?", loc="left", fontsize=11.5)

    # ---- c: which features -------------------------------------------------
    c = ax[1, 0]
    top = imp[:10][::-1]
    labs = [t[0] for t in top]
    gains = [t[1] for t in top]
    is_apg = ["apg" in l or l in ("takazawa", "ushiro", "reflect_be") or l.startswith("t_")
              for l in labs]
    c.barh(labs, gains, color=[C_MAIN if q else C_ALT for q in is_apg], height=0.68)
    c.set_xlabel("LightGBM split gain")
    c.set_title("c   what carries the signal\n"
                "      blue = APG (2nd-derivative) stiffness family", loc="left", fontsize=11.5)
    c.tick_params(axis="y", labelsize=9)

    # ---- d: does the residual mean anything --------------------------------
    dd = ax[1, 1]
    tests = []
    for nm, lab in (("bmi", "BMI"), ("sex", "sex")):
        v = D[nm].values
        ok = np.isfinite(v)
        tests.append((lab, float(np.corrcoef(res[ok], v[ok])[0, 1])))
    # residual vs age: a good index should NOT simply be regression-to-the-mean
    tests.append(("age\n(want ~0)", float(np.corrcoef(res, y)[0, 1])))
    labs2 = [t[0] for t in tests]
    vals2 = [t[1] for t in tests]
    bars2 = dd.bar(labs2, vals2, color=[C_BAD if abs(v) > 0.2 else C_REF for v in vals2],
                   width=0.55)
    dd.axhline(0, color="#333", lw=1)
    dd.axhspan(-0.2, 0.2, color="#EEEEEE", zorder=0)
    for rect, v in zip(bars2, vals2):
        dd.annotate(f"{v:+.2f}", (rect.get_x() + rect.get_width() / 2, v),
                    (0, 4 if v > 0 else -12), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
    dd.set_ylim(-1.05, 0.55)
    dd.set_ylabel("correlation with the residual\n(predicted − chronological age)")
    dd.set_title("d   is the residual a vascular signal?", loc="left", fontsize=11.5)
    dd.text(0.5, -0.98, "hypertension and diabetes are the decisive tests —\n"
            "they need the VitalDB clinical table, not this npz",
            fontsize=8.5, color="#666", ha="center")

    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)
    out = ROOT / "figures" / "fig_vascular_age.png"
    fig.savefig(out, dpi=200); fig.savefig(out.with_suffix(".pdf"))
    json.dump({"n": int(len(y)), "mae_ridge": float(mae_r), "mae_gbm": float(mae_g),
               "mae_demo_only": float(mae_d), "mae_median": float(base_med),
               "r": float(r_best), "residual_sd": float(res.std()),
               "top_features": [(n_, float(v)) for n_, v in imp[:10]]},
              open(DATA / "vascular_age_fig.json", "w"), indent=2)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
