"""fig_calibration_strategy.py -- which method should a device use, and at what calibration cost?

The finding this figure exists to show: at low k the winner is decided by HOW MANY PARAMETERS you
fit per patient, not by model class. A gradient-boosted model with a 1-parameter offset beats
both a physiological equation and a 129-parameter fine-tuned transformer below ~10 cuff readings;
the transformer only pays off beyond ~20.

  panel a   the methods, over the full range of k
  panel b   the same model, calibrated three ways -- isolating strategy from model class

    python fig_calibration_strategy.py
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
KS = [1, 2, 3, 5, 10, 20, 50, 100]
PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
              colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)


def fit_head(A, yc, B, alpha=10.0):
    if len(A) < 2:
        return np.full(len(B), yc.mean())
    mu, sd = A.mean(0), A.std(0)
    keep = sd > 1e-6
    if not keep.any():
        return np.full(len(B), yc.mean())
    A, B, mu, sd = A[:, keep], B[:, keep], mu[keep], sd[keep]
    return Ridge(alpha=alpha * max(1.0, A.shape[1] / len(A))
                 ).fit((A - mu) / sd, yc).predict((B - mu) / sd)


def head_curve_feats(F, y, rows):
    """Per-patient linear head on a model's penultimate features."""
    return {k: float(np.median([np.abs(fit_head(F[i[:k]], y[i[:k]], F[i[k+GAP:]])
                                       - y[i[k+GAP:]]).mean()
                                for i in rows.values() if k + GAP < len(i) - 5]))
            for k in KS}


def main():
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g, y = np.array(d["gte"]), np.array(d["yte"])[:, 1]
    rows = {p: np.where(g == p)[0] for p in np.unique(g)}

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
    Dtr = np.c_[np.array(d["age_tr"])[:ntr], np.array(d["sex_tr"])[:ntr],
                np.array(d["bmi_tr"])[:ntr]]
    Dte = np.c_[np.array(d["age_te"]), np.array(d["sex_te"]), np.array(d["bmi_te"])]

    pred_w = lgb.LGBMRegressor(**PARAMS).fit(Mtr, ytr).predict(Mte)
    pred_wd = lgb.LGBMRegressor(**PARAMS).fit(np.c_[Mtr, Dtr], ytr).predict(np.c_[Mte, Dte])

    def offset_curve(p):
        return {k: float(np.median([np.abs(p[i[k+GAP:]] + (y[i[:k]] - p[i[:k]]).mean()
                                           - y[i[k+GAP:]]).mean()
                                    for i in rows.values() if k + GAP < len(i) - 5]))
                for k in KS}

    def eqn_curve(v, invsq=True):
        """The same Bramwell-Hill 2-parameter per-patient fit, one timing feature."""
        z = 1.0 / np.clip(v, 0.02, None) ** 2 if invsq else v.astype(float)
        o = z.copy()
        for i in rows.values():
            ok = np.isfinite(z[i])
            o[i] = np.where(ok, z[i], np.median(z[i][ok]) if ok.any() else 0.0)
        o = o[:, None]
        return {k: float(np.median([np.abs(fit_head(o[i[:k]], y[i[:k]], o[i[k+GAP:]])
                                           - y[i[k+GAP:]]).mean()
                                    for i in rows.values() if k + GAP < len(i) - 5]))
                for k in KS}

    gbm_w, gbm_wd = offset_curve(pred_w), offset_curve(pred_wd)
    eq_ppg = eqn_curve(np.load(DATA / "_calib_ppg_notch.npy"))
    eq_pat = eqn_curve(np.load(DATA / "_calib_ptt_maxslope.npy"))
    eq_inv = eqn_curve(np.load(DATA / "_calib_pat_abp.npy"))
    floor = {k: float(np.median([np.abs(y[i[:k]].mean() - y[i[k+GAP:]]).mean()
                                 for i in rows.values() if k + GAP < len(i) - 5])) for k in KS}
    # Transformer trained on the SAME 396k segments / 1,100 patients as LightGBM
    # (models/transformer_ecgppg_full.pt), so the comparison is like-for-like. The MAE encoder
    # used earlier saw only 80k / 223 patients and is not a fair counterpart.
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    tck = torch.load(ROOT / "models" / "transformer_ecgppg_full.pt", map_location=DEV,
                     weights_only=False)
    net = ob.build_model("transformer", n_ch=2, L=1250)
    net.load_state_dict(tck["state_dict"]); net.to(DEV).eval()
    last = [mm for mm in net.modules() if isinstance(mm, torch.nn.Linear)][-1]
    buf = []
    hk = last.register_forward_hook(lambda m_, i_, o_: buf.append(i_[0].detach().cpu().numpy()))
    Xte_ = mechlib.normalize(np.array(d["Xte"])[:, :, [ECG, PPG]])
    with torch.no_grad():
        for i0 in range(0, len(Xte_), 512):
            net(torch.tensor(Xte_[i0:i0 + 512]).permute(0, 2, 1).to(DEV))
    hk.remove()
    Ftr_ = np.concatenate(buf)
    tr = head_curve_feats(Ftr_, y, rows)

    # k=0: every arm's uncalibrated starting point (no per-patient fit of any kind)
    def k0(p_):
        return float(np.median([np.abs(p_[i] - y[i]).mean() for i in rows.values()]))
    K0 = {"cuff average": k0(np.full(len(y), float(ytr.mean()))),
          "LightGBM": k0(pred_w), "LightGBM + demo": k0(pred_wd),
          "transformer": k0(np.array(tck["mu"])[1] + 0 * y)}
    # the transformer's own population prediction, not a constant
    with torch.no_grad():
        tp = []
        for i0 in range(0, len(Xte_), 512):
            tp.append(net(torch.tensor(Xte_[i0:i0 + 512]).permute(0, 2, 1).to(DEV)
                          ).cpu().numpy())
    tp = np.concatenate(tp)[:, 1] * float(tck["sd"][1]) + float(tck["mu"][1])
    K0["transformer"] = k0(tp)
    print("\nk=0 (uncalibrated): " + "  ".join(f"{n}={v:.2f}" for n, v in K0.items()))

    for nm, c in (("cuff average", floor), ("eq PAT ECG->PPG", eq_pat),
                  ("eq PAT invasive", eq_inv), ("eq PTT PPG-only", eq_ppg),
                  ("transformer", tr), ("LightGBM", gbm_w), ("LightGBM + demo", gbm_wd)):
        print(f"{nm:22s} " + "  ".join(f"k={k}:{c.get(k, float('nan')):5.2f}"
                                       for k in (2, 5, 20, 100)))

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.linewidth": 0.9,
        "legend.fontsize": 9.5,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)

    ax.axvspan(0.85, 5, color="#F2ECE2", zorder=0)
    ax.text(2.1, 3.16, "realistic for a device", ha="center", fontsize=9.5, color="#8A7A5A")
    ax.text(0.995, 0.015,
            "post-hoc calibration: frozen model + one per-patient offset\n"
            "fine-tuned head: per-patient linear layer on the model's features",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, color="#777")

    for c, colr, ls, mk, lw, lab in (
            (floor,  "#3C3C3C", "-",  "s", 1.7, "cuff average (no waveform)"),
            (eq_pat, "#C8C8C8", "--", "o", 1.5, "equation · PAT (ECG→PPG)"),
            (eq_inv, "#9FB6D0", "--", "^", 1.5, "equation · PAT (invasive)"),
            (eq_ppg, "#009E73", "-",  "o", 2.0, "equation · PTT (PPG only)"),
            (tr,     "#D55E00", "-",  "o", 2.6, "transformer + fine-tuned head (396k)"),
            (gbm_w,  "#7AA6D0", "-",  "D", 2.2, "LightGBM + post-hoc calibration"),
            (gbm_wd, "#0072B2", "-",  "D", 3.0, "LightGBM + demographics, post-hoc calib.")):
        xs = [k for k in KS if k in c]
        ax.plot(xs, [c[k] for k in xs], marker=mk, ls=ls, color=colr, lw=lw, ms=6,
                mec="white", mew=0.9, label=lab, clip_on=False, zorder=3)

    # k=0 lives at x=0.62 on the log axis: a visually separate "no calibration" slot
    X0 = 0.62
    for v, colr, mk in ((K0["cuff average"], "#3C3C3C", "s"), (K0["transformer"], "#D55E00", "o"),
                        (K0["LightGBM"], "#7AA6D0", "D"), (K0["LightGBM + demo"], "#0072B2", "D")):
        ax.plot([X0], [v], marker=mk, color=colr, ms=7, mec="white", mew=0.9,
                clip_on=False, zorder=4)
    ax.axvline(0.85, color="#CCCCCC", lw=0.9, ls="-", zorder=1)
    ax.text(X0, 8.42, "no calib.", ha="center", va="bottom", fontsize=9, color="#666")

    ax.set_xscale("log"); ax.set_xticks([X0] + KS); ax.set_xticklabels(["0"] + [str(k) for k in KS])
    ax.minorticks_off(); ax.set_xlim(0.5, 130); ax.set_ylim(3.1, 8.4)
    ax.set_xlabel("cuff readings used for calibration ($k$)", labelpad=8)
    ax.set_ylabel("diastolic BP error (mmHg)", labelpad=8)
    ax.legend(frameon=False, loc="upper right", labelspacing=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    out = ROOT / "figures" / "fig_calibration_strategy.png"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
