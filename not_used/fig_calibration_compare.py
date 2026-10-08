"""fig_calibration_compare.py -- fine-tuning vs post-hoc calibration, at equal cuff cost.

The question a device team actually has to answer: given k cuff readings from a new patient,
what should you do with them? Four options, all fitted on the SAME k segments of the SAME
held-out patients, all scored after the same gap:

  cuff average              average the k readings, ignore the waveform          1 parameter
  equation                  a + b/PTT^2, per patient                             2 parameters
  post-hoc calibration      frozen model + one per-patient offset                1 parameter
  fine-tuned                for the transformer, a per-patient linear head on its features;
                            for LightGBM, CONTINUED BOOSTING (init_model) -- trees have no
                            penultimate layer, so there is no head to swap

Two model families appear under both calibration strategies (LightGBM and a transformer), so the
figure separates WHAT the model is from HOW it is calibrated -- the two are usually confounded.

Panel b is the controlled version of the same point: one model, one feature set, only the
calibration strategy changes.

    python fig_calibration_compare.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import torch
from sklearn.linear_model import Ridge

import lightgbm_arm as gbm
import mechlib
import ood_benchmark as ob
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GAP = 50
KS = [2, 3, 5, 10, 20, 50, 100]
PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
              colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)


def fit_head(A, yc, B, alpha=10.0):
    """Per-patient ridge; alpha scales with features-per-sample so every arm is regularised
    by the same rule regardless of how many parameters it fits."""
    if len(A) < 2:
        return np.full(len(B), yc.mean())
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-6
    if not keep.any():
        return np.full(len(B), yc.mean())
    A, B, mu, sd = A[:, keep], B[:, keep], mu[keep], sd[keep]
    return Ridge(alpha=alpha * max(1.0, A.shape[1] / len(A))
                 ).fit((A - mu) / sd, yc).predict((B - mu) / sd)


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, y = np.array(d["gte"]), np.array(d["yte"])[:, 1]
    rows = {p: np.where(g == p)[0] for p in np.unique(g)}

    def offset_curve(p):
        """Post-hoc calibration: model frozen, one per-patient constant."""
        return {k: float(np.median([np.abs(p[i[k+GAP:]] + (y[i[:k]] - p[i[:k]]).mean()
                                           - y[i[k+GAP:]]).mean()
                                    for i in rows.values() if k + GAP < len(i) - 5]))
                for k in KS}

    def head_curve(F):
        """Fine-tuning: a per-patient linear head on the model's features."""
        return {k: float(np.median([np.abs(fit_head(F[i[:k]], y[i[:k]], F[i[k+GAP:]])
                                           - y[i[k+GAP:]]).mean()
                                    for i in rows.values() if k + GAP < len(i) - 5]))
                for k in KS}

    # ---- LightGBM on the 83 audited waveform features -----------------------
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte = full["Ftr"], full["ytr"][:, 1], full["Fte"]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    ntr, nte = len(ytr), len(y)
    Mtr = np.column_stack([np.asarray(Ftr[k], float)[:ntr] for k in keys])
    med = gbm.column_medians(Mtr)
    Mtr = gbm._impute(Mtr, med)
    Mte = gbm._impute(np.column_stack(
        [np.asarray(Fte.get(k, np.full(nte, np.nan)), float)[:nte] for k in keys]), med)
    src = full if "age_tr" in full else d
    Dtr = np.c_[np.asarray(src["age_tr"], float)[:ntr], np.asarray(src["sex_tr"], float)[:ntr],
                np.asarray(src["bmi_tr"], float)[:ntr]]
    Dte = np.c_[np.array(d["age_te"]), np.array(d["sex_te"]), np.array(d["bmi_te"])]
    pred_gbm = lgb.LGBMRegressor(**PARAMS).fit(np.c_[Mtr, Dtr], ytr).predict(np.c_[Mte, Dte])

    gbm_post = offset_curve(pred_gbm)

    # TRUE fine-tuning for a tree model is not a swapped head -- trees have no penultimate
    # layer. It is continued boosting from the trained model (`init_model`), adding a few
    # shallow patient-specific trees on that patient's k segments.
    base_booster = lgb.LGBMRegressor(**PARAMS).fit(np.c_[Mtr, Dtr], ytr).booster_
    Ate = np.c_[Mte, Dte]

    def gbm_finetune_curve():
        out = {}
        for k in KS:
            e = []
            for i in rows.values():
                if k + GAP >= len(i) - 5:
                    continue
                c, t = i[:k], i[k + GAP:]
                try:
                    m = lgb.LGBMRegressor(n_estimators=20, learning_rate=0.05, num_leaves=4,
                                          min_child_samples=1, random_state=0, verbosity=-1
                                          ).fit(Ate[c], y[c], init_model=base_booster)
                    p = m.predict(Ate[t])
                except Exception:
                    p = np.full(len(t), y[c].mean())
                e.append(float(np.abs(p - y[t]).mean()))
            out[k] = float(np.median(e))
        return out

    gbm_ft = gbm_finetune_curve()
    # ridge on the raw 83 features: a per-patient model that never uses the trained GBM at all
    ridge_feats = head_curve(Mte)

    # ---- transformer, same 396k training data as LightGBM -------------------
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    tck = torch.load(ROOT / "models" / "transformer_ecgppg_full.pt", map_location=DEV,
                     weights_only=False)
    net = ob.build_model("transformer", n_ch=2, L=1250)
    net.load_state_dict(tck["state_dict"]); net.to(DEV).eval()
    Xte = mechlib.normalize(np.array(d["Xte"])[:, :, [ECG, PPG]])
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    buf, out = [], []
    hk = last.register_forward_hook(lambda m_, i_, o_: buf.append(i_[0].detach().cpu().numpy()))
    with torch.no_grad():
        for i0 in range(0, len(Xte), 512):
            out.append(net(torch.tensor(Xte[i0:i0 + 512]).permute(0, 2, 1).to(DEV)).cpu().numpy())
    hk.remove()
    Ftrans = np.concatenate(buf)
    pred_tr = np.concatenate(out)[:, 1] * float(tck["sd"][1]) + float(tck["mu"][1])
    tr_post, tr_ft = offset_curve(pred_tr), head_curve(Ftrans)

    # ---- references ---------------------------------------------------------
    floor = {k: float(np.median([np.abs(y[i[:k]].mean() - y[i[k+GAP:]]).mean()
                                 for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    z = 1.0 / np.clip(np.load(DATA / "_calib_ppg_notch.npy"), 0.02, None) ** 2
    o = z.copy()
    for i in rows.values():
        ok = np.isfinite(z[i])
        o[i] = np.where(ok, z[i], np.median(z[i][ok]) if ok.any() else 0.0)
    eq = head_curve(o[:, None])

    for nm, c in (("cuff average", floor), ("equation (PTT)", eq),
                  ("GBM post-hoc", gbm_post), ("GBM fine-tuned (trees)", gbm_ft), ("ridge on 83 feats", ridge_feats),
                  ("transformer post-hoc", tr_post), ("transformer fine-tuned", tr_ft)):
        print(f"{nm:24s} " + "  ".join(f"k={k}:{c[k]:5.2f}" for k in (2, 5, 20, 100)))

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.labelsize": 11.5, "axes.linewidth": 0.9, "legend.fontsize": 9.5})
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)

    A = ax[0]
    A.axvspan(1.85, 5, color="#F2ECE2", zorder=0)
    A.text(3.05, 3.22, "realistic for a device", ha="center", fontsize=9.5, color="#8A7A5A")
    series = [
        (floor,    "#3C3C3C", "-",  "s", 1.8, "cuff average  (1 param)"),
        (eq,       "#009E73", "-",  "o", 1.8, "equation $a+b/PTT^2$  (2)"),
        (ridge_feats, "#C9C9C9", ":", "v", 1.5, "ridge on 83 features  (84 param)"),
        (gbm_ft,   "#9FC5E8", "--", "^", 1.8, "LightGBM · fine-tuned  (+20 trees)"),
        (tr_ft,    "#F4B183", "--", "^", 1.8, "transformer · fine-tuned head  (65)"),
        (tr_post,  "#D55E00", "-",  "o", 2.6, "transformer · post-hoc calib.  (1)"),
        (gbm_post, "#0072B2", "-",  "D", 3.0, "LightGBM · post-hoc calib.  (1)"),
    ]
    for c, col, ls, mk, lw, lab in series:
        A.plot(KS, [c[k] for k in KS], marker=mk, ls=ls, color=col, lw=lw, ms=6,
               mec="white", mew=0.9, label=lab, clip_on=False, zorder=3)
    A.set_xscale("log"); A.set_xticks(KS); A.set_xticklabels(KS); A.minorticks_off()
    A.set_xlim(1.85, 118); A.set_ylim(3.15, 9.4)
    A.set_xlabel("cuff readings used for calibration ($k$)")
    A.set_ylabel("diastolic BP error (mmHg)")
    A.set_title("a   four ways to spend the same cuff readings", loc="left", fontsize=12)
    A.legend(frameon=False, loc="upper right", labelspacing=0.5)

    # panel b: the controlled contrast -- same model, two strategies
    B = ax[1]
    B.axvspan(1.85, 5, color="#F2ECE2", zorder=0)
    for c, col, ls, mk, lab in ((gbm_post, "#0072B2", "-", "D", "LightGBM · post-hoc (1 param)"),
                                (gbm_ft, "#9FC5E8", "--", "^", "LightGBM · fine-tuned (+20 trees)"),
                                (tr_post, "#D55E00", "-", "o", "transformer · post-hoc (1)"),
                                (tr_ft, "#F4B183", "--", "^", "transformer · fine-tuned (65)")):
        B.plot(KS, [c[k] for k in KS], marker=mk, ls=ls, color=col, lw=2.4, ms=6,
               mec="white", mew=0.9, label=lab, clip_on=False, zorder=3)
    for c1, c2, col in ((gbm_post, gbm_ft, "#0072B2"), (tr_post, tr_ft, "#D55E00")):
        xs = [k for k in KS if (c1[k] - c2[k]) * (c1[KS[0]] - c2[KS[0]]) < 0]
        if xs:
            B.axvline(xs[0], color=col, ls=":", lw=1.2, zorder=1)
            B.annotate(f"crossover\nk≈{xs[0]}", (xs[0], 8.9), (4, 0),
                       textcoords="offset points", fontsize=8.5, color=col, va="top")
    B.set_xscale("log"); B.set_xticks(KS); B.set_xticklabels(KS); B.minorticks_off()
    B.set_xlim(1.85, 118); B.set_ylim(3.15, 9.4)
    B.set_xlabel("cuff readings used for calibration ($k$)")
    B.set_ylabel("diastolic BP error (mmHg)")
    B.set_title("b   same model, two calibration strategies", loc="left", fontsize=12)
    B.legend(frameon=False, loc="upper right", labelspacing=0.5, framealpha=0)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    out_p = ROOT / "figures" / "fig_calibration_compare.png"
    fig.savefig(out_p, dpi=300); fig.savefig(out_p.with_suffix(".pdf"))
    json.dump({"cuff_average": floor, "equation": eq, "gbm_posthoc": gbm_post,
               "gbm_finetuned_trees": gbm_ft, "ridge_83_features": ridge_feats, "transformer_posthoc": tr_post,
               "transformer_finetuned": tr_ft},
              open(DATA / "calibration_compare.json", "w"), indent=2)
    print(f"[done] {out_p}")


if __name__ == "__main__":
    main()
