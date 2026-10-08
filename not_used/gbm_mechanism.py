"""gbm_mechanism.py -- shortcut-free and arrival-time-forced LightGBM variants.

Why these arms exist
--------------------
The causal audit established that heart rate is a SHORTCUT: perturbing cardiac period moves the
deep models' predictions 1.4-2.5x more than perturbing pulse arrival time, even though the
governing law (Moens-Korteweg) runs through arrival time, not rate. Meanwhile arrival time itself
never appears in any gradient-boosted variant's top five features by split gain.

Both facts are observational. They say what the models DO use, not what they COULD use if the
shortcut were unavailable. These arms answer that:

  full            everything, the reference
  no-shortcut     rate and heart-rate-variability features removed. If accuracy survives, the
                  rate dependence was convenience rather than necessity.
  no-shortcut-demo also removes age/sex/BMI. Demographics behave like a per-subject identifier
                  (they helped CalBased 6.16 -> 4.62 while hurting AAMI 10.81 -> 11.63), so this
                  arm is the strictest "waveform physiology only" condition.
  pat-only        arrival-time features alone -- the model the field's physics implies.
  pat-forced      arrival time plus morphology, with rate and demographics removed, so the only
                  route to a timing signal is the one the governing law sanctions.

Reported on all three PulseDB protocols plus the anchor curve, against the subject-mean floor.
A restricted arm that matches the full model is the interesting outcome: it would mean the extra
features buy accounting, not mechanism.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm
import eval_protocols as ep

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARGET = 1

# the shortcut: cardiac rate and its variability
RATE = ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv",
        "hrv_lf", "hrv_hf", "hrv_lfhf", "period", "pow_lf", "pow_hf"]
# the sanctioned mechanism: pulse arrival / transit time
PAT = ["pat", "pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"]

# plain-language descriptions, so a table can say what a feature IS rather than what it is called
PLAIN = {
    "ppg_p10": "10th-percentile pulse height (how low the trough sits)",
    "ppg_p25": "25th-percentile pulse height",
    "ppg_p75": "75th-percentile pulse height",
    "ppg_p90": "90th-percentile pulse height",
    "ppg_skew_g": "pulse-shape asymmetry (lopsidedness of the wave)",
    "ppg_skew": "pulse-shape asymmetry",
    "ppg_kurt_g": "pulse peakedness (sharp vs rounded)",
    "dw10": "width near the base of the downstroke",
    "dw25": "width a quarter up the downstroke",
    "dw50": "width at half height, falling edge",
    "dw75": "width three-quarters up the downstroke",
    "sw10": "width near the base of the upstroke",
    "sw50": "width at half height, rising edge",
    "t_b": "timing of the early rebound in wave curvature",
    "t_c": "timing of the reflected-wave shoulder",
    "t_d": "timing of the late reflection",
    "t_e": "timing of the dicrotic notch",
    "apg_b_a": "early-rebound depth relative to the main peak (stiffness index)",
    "apg_c_a": "reflection shoulder relative to the main peak",
    "apg_d_a": "late reflection relative to the main peak",
    "apg_e_a": "notch depth relative to the main peak",
    "takazawa": "Takazawa stiffness index from wave curvature",
    "ushiro": "Ushiro ageing index from wave curvature",
    "aix": "augmentation index (how much reflected wave adds to the peak)",
    "reflect_idx": "strength of the reflected wave",
    "notch_depth": "depth of the dicrotic notch",
    "notch_time": "timing of the dicrotic notch",
    "rise": "time from foot to peak (upstroke speed)",
    "crest": "crest time",
    "decay_slope": "how fast pressure falls after the peak",
    "sys_area": "area under the systolic part of the pulse",
    "dia_area": "area under the diastolic part",
    "sys_dia_ratio": "systolic to diastolic area ratio",
    "vpg_max": "steepest rise rate of the pulse",
    "vpg_min": "steepest fall rate of the pulse",
    "vpg_ratio": "rise-to-fall rate ratio",
    "amp_mean": "average pulse amplitude",
    "amp_cv": "beat-to-beat amplitude variability",
    "peak_mean": "average peak height",
    "hfd": "waveform complexity (Higuchi fractal dimension)",
    "katz_fd": "waveform complexity (Katz fractal dimension)",
    "spec_ent": "spectral entropy (how spread out the frequencies are)",
    "hr": "heart rate",
    "rr_mean": "average beat-to-beat interval",
    "rr_sdnn": "beat-to-beat interval variability",
    "rr_pnn50": "proportion of large beat-interval changes",
    "pow_hf": "high-frequency power (respiratory band)",
    "pow_lf": "low-frequency power",
    "pat": "pulse arrival time (ECG to finger)",
    "pat_foot": "pulse arrival time to the pulse foot",
    "pat_peak": "pulse arrival time to the pulse peak",
    "xcorr_lag": "ECG-PPG cross-correlation lag",
    "age": "age",
    "bmi": "body mass index",
    "sex": "sex",
    # --- remaining features, so every column in every table can be named in plain words
    "apg_bd_a": "combined early-rebound and late-reflection index",
    "apg_cd_a": "combined shoulder and late-reflection index",
    "apg_ce_a": "combined shoulder and notch index",
    "reflect_be": "spacing between the early rebound and the notch",
    "dw90": "width near the top of the downstroke",
    "sw25": "width a quarter up the upstroke",
    "sw75": "width three-quarters up the upstroke",
    "sw90": "width near the top of the upstroke",
    "t_vpg_max": "time of the steepest rise",
    "t_vpg_min": "time of the steepest fall",
    "vpg_ms_area": "area under the rise-rate curve",
    "ppg_kurt": "pulse peakedness",
    "ppg_std": "pulse-height variability",
    "peak_std": "beat-to-beat peak-height variability",
    "dom_freq": "dominant frequency of the pulse train",
    "spec_centroid": "average frequency of the waveform",
    "spec_spread": "spread of the frequency content",
    "spec_rolloff": "frequency below which most power sits",
    "pow_vlf": "very-low-frequency power (slow vascular tone)",
    "pow_mf": "mid-frequency power",
    "hrv_lf": "low-frequency heart-rate variability (sympathetic tone)",
    "hrv_hf": "high-frequency heart-rate variability (respiratory)",
    "hrv_lfhf": "sympathetic-to-parasympathetic balance",
    "rr_cv": "relative beat-interval variability",
    "rr_rmssd": "short-term beat-interval variability",
    "r_count": "number of heartbeats in the window",
    "qrs_width": "width of the ECG QRS complex",
    "qrs_amp_mean": "average ECG R-wave height",
    "qrs_amp_std": "variability of ECG R-wave height",
    "ptt_var": "beat-to-beat variability of pulse arrival time",
    "xcorr_peak": "strength of ECG-PPG alignment",
    "xcorr_width": "sharpness of ECG-PPG alignment",
}


def plain(name):
    return PLAIN.get(name, name.replace("_", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seg", type=int, default=12000)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    allk = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]
    have = set(allk)

    ext = {}
    for name, fn in ep.SUBSETS.items():
        p = ep.BP / fn
        if not p.exists():
            continue
        print(f"[load] {name} ...", flush=True)
        import mechlib
        import features_full as ff
        sig, y, subj, demo = ep.load_subset(p, args.max_seg)
        X = mechlib.normalize(sig[:, :, [0, 1]])
        ext[name] = {"F": ff.compute_full(X, ep.FS), "y": y,
                     "g": np.array([abs(hash(s)) % 1000000 for s in subj]),
                     "demo": [demo.get("Age"), demo.get("BMI")]}
        ext[name]["floor"] = ep.mean_floor(y[:, TARGET], ext[name]["g"])

    rate_in = [k for k in RATE if k in have]
    pat_in = [k for k in PAT if k in have]
    morph = [k for k in allk if k not in rate_in and k not in pat_in]
    arms = {
        "full (83 + demo)": (allk, True),
        "no shortcut (rate removed)": ([k for k in allk if k not in rate_in], True),
        "no shortcut, no demo": ([k for k in allk if k not in rate_in], False),
        "PAT only": (pat_in, False),
        "PAT forced (PAT + morphology)": (pat_in + morph, False),
    }
    print(f"\n[arms] rate features removed: {len(rate_in)}  "
          f"({', '.join(rate_in[:6])}{' ...' if len(rate_in) > 6 else ''})")
    print(f"[arms] arrival-time features: {len(pat_in)} ({', '.join(pat_in)})", flush=True)

    dtr = [d["age_tr"], d["bmi_tr"]]
    dte = [d["age_te"], d["bmi_te"]]
    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50, random_state=0, verbosity=-1)

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo is not None:
            for v in demo:
                cols.append(np.asarray(v, float)[:n] if v is not None else np.full(n, np.nan))
        return np.column_stack(cols)

    res = {}
    hdr = (f"{'arm':32s} {'n':>4s} {'k=0':>6s} {'k=5':>6s} {'k=20':>6s} " +
           " ".join(f"{n:>9s}" for n in ext))
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for name, (ks_, use_demo) in arms.items():
        if not ks_:
            continue
        Mtr = tbl(Ftr, ks_, len(ytr), dtr if use_demo else None)
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
        pcf = m.predict(gbm._impute(tbl(Fte, ks_, len(yte), dte if use_demo else None), med))
        curve = ep.anchor_curve(pcf, yte[:, TARGET], gte, min_seg=60)
        names = ks_ + (["age", "bmi"] if use_demo else [])
        gain = m.booster_.feature_importance("gain")
        top = [(names[i], float(gain[i])) for i in np.argsort(-gain)[:5] if gain[i] > 0]
        row = {"n_feat": len(names), "curve": curve,
               "top": top, "top_plain": [plain(n) for n, _ in top]}
        cells = []
        for pname, e in ext.items():
            pe = m.predict(gbm._impute(
                tbl(e["F"], ks_, len(e["y"]), e["demo"] if use_demo else None), med))
            mae = float(np.median([np.mean(np.abs(pe[e["g"] == s] - e["y"][e["g"] == s, TARGET]))
                                   for s in np.unique(e["g"]) if (e["g"] == s).sum() >= 3]))
            row[pname] = mae
            cells.append(f"{mae:9.2f}")
        res[name] = row
        print(f"{name:32s} {len(names):4d} {curve[0]:6.2f} {curve[5]:6.2f} {curve[20]:6.2f} "
              + " ".join(cells), flush=True)
        print(f"{'':32s} uses: " + "; ".join(row["top_plain"][:3]), flush=True)

    print(f"\n{'subject-mean floor':32s} {'--':>4s} {'--':>6s} {'--':>6s} {'--':>6s} "
          + " ".join(f"{ext[n]['floor']:9.2f}" for n in ext))

    ref = res.get("full (83 + demo)", {}).get("curve", {}).get(20)
    if ref:
        print(f"\nAgainst the full model at 20 anchors ({ref:.2f} mmHg):")
        for name, r in res.items():
            if name.startswith("full"):
                continue
            print(f"  {name:32s} {r['curve'][20] - ref:+.2f} mmHg")

    (DATA / "gbm_mechanism.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/gbm_mechanism.json")

    if args.latex:
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{Mechanism-restricted gradient-boosted models. Rate features are "
                 r"the shortcut identified by the causal audit; arrival time is the quantity "
                 r"the governing law sanctions. DBP MAE (mmHg).}",
                 r"\label{tab:mechanism}",
                 r"\begin{tabular}{l r r r r r l}", r"\toprule",
                 r"Feature set & $n$ & $k{=}0$ & $k{=}20$ & CalBased & AAMI & "
                 r"What the model leans on \\", r"\midrule"]
        for name, r in res.items():
            desc = "; ".join(r["top_plain"][:2])
            lines.append(
                f"{name} & {r['n_feat']} & {r['curve'][0]:.2f} & {r['curve'][20]:.2f} & "
                f"{r.get('CalBased', float('nan')):.2f} & {r.get('AAMI', float('nan')):.2f} & "
                f"{desc} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out = ROOT / "TABLE_mechanism.tex"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] {out.name}")


if __name__ == "__main__":
    main()
