"""calib_all_models.py -- deep nets and LightGBM variants on the SAME calibration axis.

Two questions this answers.

1. How do we compare with the published benchmark?
   The PulseDB CalFree paper (AI4HealthUOL) reports SBP/DBP MAE for a median baseline and five
   architectures. Our deep nets are retrained from scratch on the same split, so a like-for-like
   row is the only honest way to say whether our pipeline reproduces theirs before drawing any
   conclusion from it.

2. Does calibration change the ranking?
   Every uncalibrated comparison in this project put all architectures within 8.1-8.8 mmHg DBP,
   i.e. indistinguishable. Calibration is the axis where models might separate, and it is the
   deployable question for a device: not "what is the error" but "how many cuff readings does
   this model need to reach a target error". Deep nets have never been put on that axis here --
   only LightGBM has.

Anchors are drawn at RANDOM per subject, never the first k. First-k anchors sit adjacent in time
to the scored segments and absorb local drift as well as per-subject offset, which flatters every
number by roughly 0.2-0.3 mmHg.

    python calib_all_models.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import lightgbm as lgb

import mechlib
import ood_benchmark as ob
import lightgbm_arm as gbm
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KS = [0, 1, 2, 3, 5, 10, 20]
MIN_SEG = 60
TARGET = 1                                    # DBP

# published PulseDB CalFree results (SBP, DBP), for the reproduction check
PAPER = {"Baseline (median)": (14.88, 9.44), "LeNet1d": (12.37, 7.89),
         "XResNet1d50": (12.40, 7.85), "XResNet1d101": (12.70, 8.05),
         "Inception1d": (14.54, 10.96), "S4": (12.39, 8.03)}
ARCH_TO_PAPER = {"lenet1d": "LeNet1d", "inception1d": "Inception1d",
                 "xresnet1d50": "XResNet1d50", "xresnet1d101": "XResNet1d101"}


def anchor_curve(pred, y, groups, ks=KS, seed=0):
    """Median per-subject MAE after fitting a one-parameter offset on k random anchors."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        errs = []
        for s in np.unique(groups):
            idx = np.where(groups == s)[0]
            if len(idx) < MIN_SEG:
                continue
            if k == 0:
                hold, off = idx, 0.0
            else:
                a = rng.choice(idx, k, replace=False)
                hold = np.setdiff1d(idx, a)
                if len(hold) < 20:
                    continue
                off = float(np.mean(y[a] - pred[a]))
            errs.append(float(np.mean(np.abs(pred[hold] + off - y[hold]))))
        out[k] = float(np.median(errs)) if errs else float("nan")
    return out


def main():
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g_all, y_all = d["gte"], d["yte"]
    res = {}

    # ---- deep nets ----------------------------------------------------------
    subs = [s for s in np.unique(g_all) if (g_all == s).sum() >= 150]
    sel = np.concatenate([np.where(g_all == s)[0][:150] for s in subs])
    Xd = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    yd, gd = y_all[sel], g_all[sel]
    print(f"[deep] {len(Xd)} segments, {len(subs)} subjects", flush=True)
    print(f"\n{'model':16s} {'paper DBP':>10s} {'ours k=0':>9s} " +
          " ".join(f"k={k:<4d}" for k in KS[1:]), flush=True)

    for mk in ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]:
        try:
            ck = torch.load(ROOT / "models" / f"{mk}_ecgppg_full.pt", map_location=DEVICE,
                            weights_only=False)
            m = ob.build_model(mk, n_ch=2, L=1250)
            m.load_state_dict(ck["state_dict"]); m.to(DEVICE).eval()
        except Exception as e:
            print(f"{mk:16s} cannot load ({str(e)[:40]})", flush=True); continue
        p = ob.predict(m, Xd, DEVICE, ck["mu"], ck["sd"])
        sbp = float(np.abs(p[:, 0] - yd[:, 0]).mean())
        curve = anchor_curve(p[:, TARGET], yd[:, TARGET], gd)
        paper = PAPER.get(ARCH_TO_PAPER.get(mk, ""), (None, None))
        res[mk] = {"kind": "deep", "sbp_mae": sbp, "curve": curve,
                   "paper_sbp": paper[0], "paper_dbp": paper[1],
                   "n_params": int(sum(x.numel() for x in m.parameters()))}
        print(f"{mk:16s} {fmt(paper[1]):>10s} {curve[0]:9.2f} " +
              " ".join(f"{curve[k]:6.2f}" for k in KS[1:]), flush=True)

    # ---- LightGBM variants --------------------------------------------------
    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    n_te = len(yte)
    gte = g_all[:n_te]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    dtr = {"age": d["age_tr"], "sex": d["sex_tr"], "bmi": d["bmi_tr"]}
    dte = {"age": d["age_te"], "sex": d["sex_te"], "bmi": d["bmi_te"]}

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo:
            for dk in ("age", "sex", "bmi"):
                v = np.asarray(demo[dk], float)
                cols.append(v[:n] if len(v) >= n else np.resize(v, n))
        return np.column_stack(cols)

    variants = {
        "gbm default (83)": dict(n_estimators=800, learning_rate=0.03, num_leaves=63),
        "gbm deep (83)": dict(n_estimators=1500, learning_rate=0.02, num_leaves=127),
        "gbm shallow (83)": dict(n_estimators=400, learning_rate=0.05, num_leaves=31),
        "gbm single tree (83)": dict(n_estimators=1, learning_rate=1.0, num_leaves=32),
    }
    print(flush=True)
    for tag, params in variants.items():
        for use_demo in (False, True):
            name = tag + (" + demo" if use_demo else "")
            Mtr = tbl(Ftr, keys, len(ytr), dtr if use_demo else None)
            med = gbm.column_medians(Mtr)
            m = lgb.LGBMRegressor(**params, subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=50, random_state=0, verbosity=-1)
            m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
            pred = m.predict(gbm._impute(tbl(Fte, keys, n_te, dte if use_demo else None), med))
            curve = anchor_curve(pred, yte[:, TARGET], gte)
            res[name] = {"kind": "gbm", "curve": curve,
                         "n_leaves": int(sum(t["num_leaves"] for t in
                                             m.booster_.dump_model()["tree_info"]))}
            print(f"{name:24s} {curve[0]:6.2f} " +
                  " ".join(f"{curve[k]:6.2f}" for k in KS[1:]), flush=True)

    # ---- summary ------------------------------------------------------------
    ref = min((v["curve"][20] for v in res.values() if np.isfinite(v["curve"][20])),
              default=np.nan)
    print(f"\n{'model':24s} {'k=0':>7s} {'k=20':>7s} {'gain':>7s} {'anchors to 5.0':>15s}")
    for k, v in sorted(res.items(), key=lambda kv: kv[1]["curve"][20]):
        c = v["curve"]
        hit = next((kk for kk in KS if c[kk] <= 5.0), None)
        print(f"{k:24s} {c[0]:7.2f} {c[20]:7.2f} {c[0]-c[20]:7.2f} "
              f"{str(hit) if hit is not None else '>20':>15s}")

    (DATA / "calib_all_models.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/calib_all_models.json  (best k=20: {ref:.2f} mmHg)")


def fmt(v):
    return "--" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    main()
