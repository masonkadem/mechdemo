"""make_tables.py -- emit LaTeX table bodies from the run JSONs so the paper numbers are
exact and reproducible. Writes paper/table1_datasets.tex, table2_results.tex, table3_gbm.tex.
"""
import json
from pathlib import Path

import numpy as np
import mechlib
import physics_audit as pa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper"
OUT.mkdir(exist_ok=True)
B = "C:/Users/mason/OneDrive - McMaster University/2026/BP"


def _age(v):
    v = v[np.isfinite(v)]; v = v[v >= 10]        # 0-coded missing ages excluded
    return v.mean() if len(v) else np.nan


def table1_datasets():
    rows = []
    # FULL PulseDB VitalDB (official CalFree splits) -- report train and the CalFree test set
    full = ROOT / "data" / "vitaldb_full_calfree.npz"
    if full.exists():
        d = np.load(full)
        ytrv = np.concatenate([d["ytr"], d["yva"]]); gtrv = np.concatenate([d["gtr"], d["gva"]])
        rows.append(("VitalDB-Vital (train)", len(ytrv), len(np.unique(gtrv)), ytrv,
                     "ECG+PPG", "age, sex, BMI", _age(d["age_tr"])))
        rows.append(("VitalDB-Vital (CalFree test)", len(d["yte"]), len(np.unique(d["gte"])),
                     d["yte"], "ECG+PPG", "age, sex, BMI", _age(d["age_te"])))
    else:
        dm = mechlib.load_mini("data/vitaldb_mini_deep.npz")
        y = np.concatenate([dm["ytr"], dm["yva"], dm["yte"]])
        g = np.concatenate([dm["gtr"], dm["gva"], dm["gte"]])
        rows.append(("VitalDB (train, subset)", len(y), len(np.unique(g)), y, "ECG+PPG", "--", np.nan))
    m = pa.load_mimic_bp(B, channels=("ppg",), max_patients=1524)
    # MIMIC-BP records are 30 s (3750 samp); the model sees 10 s (1250) windows -> x3, matching
    # VitalDB's 10 s segment unit so the Segments column is apples-to-apples across datasets.
    _, k = pa.window_segments(m["X"][:1], 1250)
    rows.append(("MIMIC-BP", len(m["y"]) * k, len(np.unique(m["g"])), m["y"], "ECG+PPG", "--", np.nan))
    for nm, path in [("BCG", "data/bcg_dataset"),
                     ("Sensors", "C:/Users/mason/Downloads/sensors_dataset/sensors_dataset"),
                     ("UCI2", "data/uci2_dataset/uci2_dataset"),
                     ("PPG-BP", "C:/Users/mason/Downloads/ppgbp_dataset/ppgbp_dataset")]:
        e = pa.load_bpbenchmark(path, nm)
        demo = ("age, sex, BMI" if nm == "PPG-BP" else "age, sex") if e["demo"] else "--"
        ag = _age(e["demo"]["age"]) if e["demo"] else np.nan
        rows.append((nm, len(e["y"]), len(np.unique(e["g"])), e["y"], "PPG", demo, ag))
    # KS shift vs VitalDB from whichever OOD-track json exists
    dsj = next((ROOT / "data" / f for f in ("ood_benchmark_ppg.json", "ood_benchmark_ecgppg_full.json",
               "ood_benchmark_ecgppg.json") if (ROOT / "data" / f).exists()), None)
    ds = json.loads(dsj.read_text()).get("dist_shift", {}) if dsj else {}
    key = {"MIMIC-BP": "mimic_bp", "BCG": "bcg", "Sensors": "sensors", "UCI2": "uci2", "PPG-BP": "ppgbp"}

    L = [r"\begin{tabular}{lrrccccc}", r"\toprule",
         r"Dataset & Segments & Subjects & SBP / DBP (mmHg) & Age & Channels & Demographics & KS \\",
         r"\midrule"]
    for nm, ns, nsub, yy, ch, demo, ag in rows:
        ksv = np.mean(list(ds.get(key.get(nm, ""), {}).values())) if key.get(nm) in ds else None
        ks = f"{ksv:.2f}" if ksv is not None else "--"
        agestr = f"{ag:.0f}" if np.isfinite(ag) else "--"
        L.append(f"{nm} & {ns:,} & {nsub:,} & "
                 f"{yy[:,0].mean():.0f}$\\pm${yy[:,0].std():.0f} / "
                 f"{yy[:,1].mean():.0f}$\\pm${yy[:,1].std():.0f} & {agestr} & {ch} & {demo} & {ks} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table1_datasets.tex").write_text("\n".join(L))
    print("[tab] table1_datasets.tex")


def table2_results():
    e = json.loads((ROOT / "data" / "ood_benchmark_ecgppg.json").read_text())["models"]
    p = json.loads((ROOT / "data" / "ood_benchmark_ppg.json").read_text())["models"]
    names = [n for n in p if not n.startswith("_")]
    L = [r"\begin{tabular}{lrrrrrrc}", r"\toprule",
         r"& \multicolumn{6}{c}{DBP MAE (mmHg)} & Mechanism \\",
         r"\cmidrule(lr){2-7}",
         r"Model & ID & MIMIC & BCG & Sensors & UCI2 & PPG-BP & PAT slope / \%correct \\",
         r"\midrule"]
    for n in names:
        o = p[n]["ood"]
        aud = e[n]["audit"]["dbp"]
        L.append("{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & "
                 "{:+.0f} / {:.0f}\\% \\\\".format(
                     n, o["id"]["mae_dbp"], o.get("mimic_bp", {}).get("mae_dbp", np.nan),
                     o.get("bcg", {}).get("mae_dbp", np.nan), o.get("sensors", {}).get("mae_dbp", np.nan),
                     o.get("uci2", {}).get("mae_dbp", np.nan), o.get("ppgbp", {}).get("mae_dbp", np.nan),
                     aud["dBP_dPTT"], aud["frac_correct_sign"] * 100))
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table2_results.tex").write_text("\n".join(L))
    print("[tab] table2_results.tex")


def table3_gbm():
    g = json.loads((ROOT / "data" / "lightgbm_arm.json").read_text())
    ab = g.get("demo_ablation", {})
    L = [r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule",
         f"Faithful model (feature source) & {g['faithful']} \\\\",
         f"Audit-passing features & {', '.join(g['features'])} \\\\",
         f"ID DBP MAE (mmHg) & {g['id_mae'][1]:.2f} \\\\"]
    for s, v in g["ood"].items():
        L.append(f"OOD DBP MAE -- {s.replace('_', '-')} & {v['mae_dbp']:.2f} \\\\")
    if ab:
        L += [r"\midrule",
              f"Age/sex ablation set & {ab['set']} \\\\",
              f"DBP MAE with age+sex & {ab['with_demo_dbp']:.2f} \\\\",
              f"DBP MAE without & {ab['no_demo_dbp']:.2f} \\\\",
              f"Improvement from demographics & {ab['delta_dbp']:+.2f} mmHg \\\\"]
        sh = ab.get("demo_shap") or {}
        if sh:
            L.append(f"SHAP mean$|\\cdot|$: age / sex & {sh.get('age',0):.2f} / {sh.get('sex',0):.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table3_gbm.tex").write_text("\n".join(L))
    print("[tab] table3_gbm.tex")


if __name__ == "__main__":
    table1_datasets(); table2_results(); table3_gbm()
