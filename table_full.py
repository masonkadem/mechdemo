"""table_full.py -- one table: every model, every dataset, every protocol.

Consolidates what was scattered across TABLE_protocols / TABLE_mechanism / TABLE_ppg_mechanism
into a single document, and compiles it to PDF.

Columns, left to right:
  complexity   leaves for tree models, parameters for networks
  CalFree      the honest generalisation split, at 0 / 5 / 20 per-subject calibration anchors
  CalBased     same subjects as training (100% overlap) -- an upper bound, not a deployable number
  AAMI         calibration-free, built to the AAMI standard so it samples the BP tails
  MIMIC-BP     external ECG+PPG
  BCG / Sensors / UCI2 / PPG-BP    external PPG-only
  top features what the model leans on, in plain words

Every dataset row is anchored by a "predict the subject mean" row. Without it the external
columns read far better than they are: on most of these sets a constant predictor is already
close to the best model, so differences between models there are not evidence of transferable
signal.

    python table_full.py            # build TABLE_full.tex
    python table_full.py --pdf      # and compile it
"""
import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import lightgbm as lgb
import torch

import mechlib
import lightgbm_arm as gbm
import ood_benchmark as ob
import eval_protocols as ep
from gbm_mechanism import plain, RATE, PAT

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = 1
ARCHS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
EXT = ["MIMIC-BP", "BCG", "Sensors", "UCI2", "PPG-BP"]


def subject_mae(pred, y, g, min_seg=3):
    v = [float(np.mean(np.abs(pred[g == s] - y[g == s])))
         for s in np.unique(g) if (g == s).sum() >= min_seg]
    return float(np.median(v)) if v else float("nan")


def mean_floor(y, g, min_seg=3):
    v = [float(np.mean(np.abs(y[g == s] - y[g == s].mean())))
         for s in np.unique(g) if (g == s).sum() >= min_seg]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--max-seg", type=int, default=12000)
    args = ap.parse_args()

    full = pickle.load(open(DATA / "_feat_full_ALLtrain.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    OOD = pickle.load(open(DATA / "_variants_ood_feats.pkl", "rb"))
    d = np.load(DATA / "vitaldb_full_calfree.npz")
    gte = d["gte"][:len(yte)]
    keys = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]
    have = set(keys)
    rate_in = [k for k in RATE if k in have]
    pat_in = [k for k in PAT if k in have]

    # PulseDB protocol subsets
    proto = {}
    for name, fn in ep.SUBSETS.items():
        p = ep.BP / fn
        if not p.exists():
            continue
        print(f"[load] {name} ...", flush=True)
        import features_full as ff
        sig, y, subj, demo = ep.load_subset(p, args.max_seg)
        X = mechlib.normalize(sig[:, :, [0, 1]])
        proto[name] = {"F": ff.compute_full(X, ep.FS), "y": y, "X": X,
                       "g": np.array([abs(hash(s)) % 1000000 for s in subj]),
                       "demo": [demo.get("Age"), demo.get("BMI")]}
        proto[name]["floor"] = mean_floor(y[:, TARGET], proto[name]["g"])

    floors = {n: mean_floor(v[1][:, TARGET], v[2]) for n, v in OOD.items()}
    floors.update({n: v["floor"] for n, v in proto.items()})

    dtr = [d["age_tr"], d["bmi_tr"]]
    dte = [d["age_te"], d["bmi_te"]]

    def tbl(F, ks_, n, demo=None):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in ks_]
        if demo is not None:
            for v in demo:
                cols.append(np.asarray(v, float)[:n] if v is not None else np.full(n, np.nan))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    P = dict(subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
             random_state=0, verbosity=-1)
    variants = {
        "GBM all features": (keys, False, dict(n_estimators=800, learning_rate=0.03,
                                               num_leaves=63)),
        "GBM + demographics": (keys, True, dict(n_estimators=800, learning_rate=0.03,
                                                num_leaves=63)),
        "GBM deep + demo": (keys, True, dict(n_estimators=1500, learning_rate=0.02,
                                             num_leaves=127)),
        "GBM no rate shortcut": ([k for k in keys if k not in rate_in], False,
                                 dict(n_estimators=800, learning_rate=0.03, num_leaves=63)),
        "GBM arrival time only": (pat_in, False, dict(n_estimators=800, learning_rate=0.03,
                                                      num_leaves=63)),
        "GBM single tree": (keys, False, dict(n_estimators=1, learning_rate=1.0,
                                              num_leaves=32)),
    }

    rows = []
    for name, (ks_, use_demo, params) in variants.items():
        Mtr = tbl(Ftr, ks_, len(ytr), dtr if use_demo else None)
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params, **P)
        m.fit(gbm._impute(Mtr, med), ytr[:, TARGET])
        pcf = m.predict(gbm._impute(tbl(Fte, ks_, len(yte), dte if use_demo else None), med))
        curve = ep.anchor_curve(pcf, yte[:, TARGET], gte, min_seg=60)
        names = ks_ + (["age", "bmi"] if use_demo else [])
        gain = m.booster_.feature_importance("gain")
        top = [names[i] for i in np.argsort(-gain)[:3] if gain[i] > 0]
        r = {"model": name, "kind": "gbm",
             "complexity": f"{sum(t['num_leaves'] for t in m.booster_.dump_model()['tree_info']):,} lv",
             "n_feat": len(names), "curve": curve,
             "top": [plain(t) for t in top]}
        for pn, pv in proto.items():
            pe = m.predict(gbm._impute(
                tbl(pv["F"], ks_, len(pv["y"]), pv["demo"] if use_demo else None), med))
            r[pn] = subject_mae(pe, pv["y"][:, TARGET], pv["g"])
        for en, (F, y, g) in OOD.items():
            pe = m.predict(gbm._impute(
                tbl(F, ks_, len(y), [None, None] if use_demo else None), med))
            r[en] = subject_mae(pe, y[:, TARGET], g)
        rows.append(r)
        print(f"  {name:24s} CalFree20 {curve[20]:.2f}", flush=True)

    # deep nets
    subs = [s for s in np.unique(gte) if (gte == s).sum() >= 150]
    sel = np.concatenate([np.where(gte == s)[0][:150] for s in subs])
    Xcf = mechlib.normalize(d["Xte"][sel][:, :, [mechlib.ECG, mechlib.PPG]])
    ycf, gcf = d["yte"][sel], gte[sel]
    for mk in ARCHS:
        try:
            ck = torch.load(ROOT / "models" / f"{mk}_ecgppg_full.pt", map_location=DEVICE,
                            weights_only=False)
            net = ob.build_model(mk, n_ch=2, L=1250)
            net.load_state_dict(ck["state_dict"]); net.to(DEVICE).eval()
        except Exception as e:
            print(f"  {mk}: skip ({str(e)[:40]})"); continue
        pcf = ob.predict(net, Xcf, DEVICE, ck["mu"], ck["sd"])[:, TARGET]
        curve = ep.anchor_curve(pcf, ycf[:, TARGET], gcf, min_seg=60)
        r = {"model": mk, "kind": "deep",
             "complexity": f"{sum(x.numel() for x in net.parameters()):,} par",
             "n_feat": "raw", "curve": curve, "top": ["learned from raw waveform"]}
        for pn, pv in proto.items():
            pe = ob.predict(net, pv["X"], DEVICE, ck["mu"], ck["sd"])[:, TARGET]
            r[pn] = subject_mae(pe, pv["y"][:, TARGET], pv["g"])
        for en in EXT:
            r[en] = float("nan")          # deep nets need raw waveforms, not cached features
        rows.append(r)
        print(f"  {mk:24s} CalFree20 {curve[20]:.2f}", flush=True)

    out = {"rows": rows, "floors": floors}
    (DATA / "table_full.json").write_text(json.dumps(out, indent=2, default=float))

    # ---------------- LaTeX ----------------
    def f(v):
        return "--" if v is None or not np.isfinite(v) else f"{v:.2f}"

    cols = ["CalBased", "AAMI"] + EXT
    L = [r"\documentclass[landscape,a4paper,10pt]{article}",
         r"\usepackage[margin=1cm,landscape]{geometry}",
         r"\usepackage{booktabs,graphicx}", r"\pagestyle{empty}",
         r"\begin{document}", r"\begin{table}[t]", r"\centering",
         r"\caption{Diastolic blood-pressure mean absolute error (mmHg) for every model across "
         r"every dataset and protocol. CalFree, AAMI and the five external sets are "
         r"calibration-free; CalBased shares 100\% of its subjects with training and is an upper "
         r"bound rather than a deployable figure. $k$ is the number of per-subject calibration "
         r"anchors. The final row gives the error of predicting each subject's own mean, which "
         r"most external columns barely improve on.}",
         r"\label{tab:full}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l l r r r r " + "r" * len(cols) + r" l}", r"\toprule",
         r"Model & Complexity & $n$ & \multicolumn{3}{c}{CalFree} & "
         + " & ".join(c.replace("-", "-") for c in cols) + r" & Leans on \\",
         r"\cmidrule(lr){4-6}",
         r" & & & $k{=}0$ & $k{=}5$ & $k{=}20$ & " + " & ".join([""] * len(cols))
         + r" & \\", r"\midrule"]

    for kind, label in (("gbm", r"\textit{Feature models (LightGBM)}"),
                        ("deep", r"\textit{Deep networks (raw waveform)}")):
        sel_rows = [r for r in rows if r["kind"] == kind]
        if not sel_rows:
            continue
        L.append(rf"\multicolumn{{{6+len(cols)+1}}}{{l}}{{{label}}} \\")
        for r in sel_rows:
            c = r["curve"]
            L.append(
                f"\\quad {r['model']} & {r['complexity']} & {r['n_feat']} & "
                f"{f(c.get(0))} & {f(c.get(5))} & {f(c.get(20))} & "
                + " & ".join(f(r.get(x)) for x in cols)
                + f" & {'; '.join(r['top'][:2])} \\\\")
    L.append(r"\midrule")
    L.append("Predict subject mean & 0 & -- & -- & -- & -- & "
             + " & ".join(f(floors.get(x)) for x in cols) + r" & -- \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", r"\end{document}"]

    tex = ROOT / "TABLE_full.tex"
    tex.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[done] {tex.name}")

    if args.pdf:
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                           cwd=ROOT, capture_output=True, timeout=180)
            pdf = ROOT / "TABLE_full.pdf"
            print(f"[done] {pdf.name}" if pdf.exists() else "[warn] pdflatex produced no PDF")
        except Exception as e:
            print(f"[warn] pdflatex failed: {e}")


if __name__ == "__main__":
    main()
