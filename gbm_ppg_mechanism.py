"""gbm_ppg_mechanism.py -- the mechanism arms for a PPG-ONLY (wearable) device.

Why this is a different question from the ECG+PPG case
------------------------------------------------------
A wrist or finger wearable has no ECG, so pulse arrival time is not measurable at all: PAT needs
an R peak to time from. What the field substitutes are single-site TIMING PROXIES read off the
PPG wave itself, on the theory that a stiffer artery returns its reflected wave sooner and
reshapes the pulse:

  reflection timing   t_b, t_c, t_d, t_e, notch_time -- when the reflected wave and the dicrotic
                      notch arrive relative to the pulse foot. The closest single-site analogue
                      of transit time.
  reflection depth    aix, apg ratios, takazawa, ushiro -- how much the reflection augments the
                      peak. Amplitude rather than timing, but the same physics.
  upstroke timing     rise, crest, sw*, vpg -- how fast the pulse climbs, which stiffness also
                      changes.
  width               dw* -- pulse width at various heights, a shape summary.

These proxies are what a wearable must rely on if the governing law is to be respected at all, so
the arms mirror gbm_mechanism.py: remove the rate shortcut, then force the sanctioned mechanism
and see whether accuracy survives.

Evaluated in-distribution (VitalDB, PPG channel only) and on the four genuinely PPG-only external
datasets (BCG, Sensors, UCI2, PPG-BP), each against its own subject-mean floor -- without which
the external numbers read far better than they are.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm
import eval_protocols as ep
from gbm_mechanism import PLAIN, plain

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = 1

# anything needing an ECG channel -- unavailable to a wearable, so excluded from every arm
ECG_ONLY = ["pat", "pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width",
            "hrv_lf", "hrv_hf", "hrv_lfhf", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv",
            "rr_mean", "qrs_dur", "qt", "pr"]
# the rate shortcut, still estimable from PPG alone
RATE = ["hr", "period", "pow_lf", "pow_hf", "amp_cv"]
# the sanctioned single-site mechanism: when the reflected wave arrives
REFL_TIME = ["t_b", "t_c", "t_d", "t_e", "notch_time", "reflect_be"]
REFL_AMP = ["aix", "reflect_idx", "notch_depth", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a",
            "takazawa", "ushiro"]
UPSTROKE = ["rise", "crest", "sw10", "sw25", "sw50", "sw75", "sw90",
            "vpg_max", "vpg_min", "vpg_ratio", "t_vpg_max", "t_vpg_min", "decay_slope"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    OOD = pickle.load(open(DATA / "_variants_ood_feats.pkl", "rb"))
    OOD = {k: v for k, v in OOD.items() if k != "MIMIC-BP"}      # PPG-only externals
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]

    usable = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
              and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    ppg_all = [k for k in usable if k not in ECG_ONLY]
    have = set(ppg_all)
    rate = [k for k in RATE if k in have]
    rtime = [k for k in REFL_TIME if k in have]
    ramp = [k for k in REFL_AMP if k in have]
    upst = [k for k in UPSTROKE if k in have]
    morph = [k for k in ppg_all if k not in rate + rtime + ramp + upst]

    print(f"[ppg] {len(ppg_all)} PPG-derivable features "
          f"({len(usable) - len(ppg_all)} ECG-dependent excluded)")
    print(f"[ppg] rate shortcut {len(rate)}, reflection timing {len(rtime)}, "
          f"reflection depth {len(ramp)}, upstroke {len(upst)}", flush=True)

    arms = {
        "all PPG + demo": (ppg_all, True),
        "all PPG": (ppg_all, False),
        "no shortcut (rate removed)": ([k for k in ppg_all if k not in rate], False),
        "reflection timing only": (rtime, False),
        "reflection (timing + depth)": (rtime + ramp, False),
        "timing forced (refl + upstroke)": (rtime + ramp + upst, False),
        "rate only (the shortcut)": (rate, False),
        "demographics only": ([], True),
    }

    dtr = [d["age_tr"], d["bmi_tr"]]
    dte = [d["age_te"], d["bmi_te"]]
    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo is not None:
            for v in demo:
                cols.append(np.asarray(v, float)[:n] if v is not None else np.full(n, np.nan))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    # subject-mean floor per external set, so the numbers are readable
    floors = {}
    for nm, (F, y, g) in OOD.items():
        v = [float(np.mean(np.abs(y[g == s, TARGET] - y[g == s, TARGET].mean())))
             for s in np.unique(g) if (g == s).sum() >= 3]
        floors[nm] = float(np.median(v)) if v else float("nan")

    res = {}
    hdr = (f"{'arm':32s} {'n':>3s} {'ID k=0':>7s} {'k=20':>6s} " +
           " ".join(f"{n:>8s}" for n in OOD))
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, (ks_, use_demo) in arms.items():
        Mtr = tbl(Ftr, ks_, len(ytr), dtr if use_demo else None)
        if Mtr.shape[1] == 0:
            continue
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
        pid = m.predict(gbm._impute(tbl(Fte, ks_, len(yte), dte if use_demo else None), med))
        curve = ep.anchor_curve(pid, yte[:, TARGET], gte, min_seg=60)
        names = ks_ + (["age", "bmi"] if use_demo else [])
        gain = m.booster_.feature_importance("gain")
        top = [(names[i], float(gain[i])) for i in np.argsort(-gain)[:4] if gain[i] > 0]
        row = {"n_feat": len(names), "curve": curve,
               "top": top, "top_plain": [plain(n) for n, _ in top]}
        cells = []
        for nm, (F, y, g) in OOD.items():
            # externals carry no demographics; train-median imputation keeps the schema fixed
            pe = m.predict(gbm._impute(
                tbl(F, ks_, len(y), [None, None] if use_demo else None), med))
            mae = float(np.median([np.mean(np.abs(pe[g == s] - y[g == s, TARGET]))
                                   for s in np.unique(g) if (g == s).sum() >= 3]))
            row[nm] = mae
            cells.append(f"{mae:8.2f}")
        res[name] = row
        print(f"{name:32s} {len(names):3d} {curve[0]:7.2f} {curve[20]:6.2f} "
              + " ".join(cells), flush=True)
        if row["top_plain"]:
            print(f"{'':32s} uses: " + "; ".join(row["top_plain"][:2]), flush=True)

    print(f"\n{'subject-mean floor':32s} {'--':>3s} {'--':>7s} {'--':>6s} "
          + " ".join(f"{floors[n]:8.2f}" for n in OOD))

    ref = res.get("all PPG", {}).get("curve", {}).get(20)
    if ref:
        print(f"\nAgainst all-PPG at 20 anchors ({ref:.2f} mmHg):")
        for name, r in res.items():
            if name == "all PPG":
                continue
            print(f"  {name:32s} {r['curve'][20] - ref:+.2f} mmHg")

    res["_floors"] = floors
    (DATA / "gbm_ppg_mechanism.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/gbm_ppg_mechanism.json")

    if args.latex:
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{PPG-only (wearable) mechanism arms. No ECG is available, so pulse "
                 r"arrival time cannot be measured; single-site reflection timing is the "
                 r"closest sanctioned proxy. DBP MAE (mmHg); $k$ is the number of per-subject "
                 r"calibration anchors.}",
                 r"\label{tab:ppgmech}",
                 r"\begin{tabular}{l r r r " + "r" * len(OOD) + r" l}", r"\toprule",
                 r"Feature set & $n$ & $k{=}0$ & $k{=}20$ & "
                 + " & ".join(OOD) + r" & What it leans on \\", r"\midrule"]
        for name, r in res.items():
            if name.startswith("_"):
                continue
            cells = " & ".join(f"{r.get(n, float('nan')):.2f}" for n in OOD)
            desc = "; ".join(r["top_plain"][:2]) if r["top_plain"] else "--"
            lines.append(f"{name} & {r['n_feat']} & {r['curve'][0]:.2f} & "
                         f"{r['curve'][20]:.2f} & {cells} & {desc} \\\\")
        lines.append(r"\midrule")
        lines.append("Predict subject mean & 0 & -- & -- & "
                     + " & ".join(f"{floors[n]:.2f}" for n in OOD) + r" & -- \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out = ROOT / "TABLE_ppg_mechanism.tex"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] {out.name}")


if __name__ == "__main__":
    main()
