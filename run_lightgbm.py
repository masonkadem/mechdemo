"""run_lightgbm.py -- stage 2: the interpretable reference model.

Reads the deep-model audit (ood_benchmark_*.json) to learn WHICH cues the faithful model
uses, hand-builds exactly those (+ age/sex) into a LightGBM, and runs it through the same
ID/OOD evaluation. Answers three publication questions:

  1. Does an interpretable model built from audit-passing features generalize BETTER OOD
     than the deep models it was distilled from?
  2. How much do age and sex matter (with/without ablation + SHAP)?
  3. Which cues does the tree lean on, and does that match the deep model's mechanism?

Run AFTER ood_benchmark.py (needs its json for feature selection).
    python run_lightgbm.py --faithful xresnet1d50 --audit data/ood_benchmark_ecgppg.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import mechlib
import physics_audit as pa
import lightgbm_arm as gbm
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent


def cues_for(X, fs, has_ecg):
    """Per-segment cue dict for a batch. ECG present -> full set incl PAT; else morphology."""
    if has_ecg:
        return mechlib.compute_scalars(X, fs)
    return mechlib.compute_morphology(X, fs, ch=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_mini_deep.npz")
    ap.add_argument("--faithful", default="xresnet1d50",
                    help="deep model whose audit selects the features")
    ap.add_argument("--audit", default="data/ood_benchmark_ecgppg.json")
    ap.add_argument("--mimic", default="C:/Users/mason/OneDrive - McMaster University/2026/BP")
    ap.add_argument("--mimic-patients", type=int, default=400)
    ap.add_argument("--external", default="")
    ap.add_argument("--vitaldb-demo", action="store_true",
                    help="run the demographics ablation on VitalDB's OWN age/sex/bmi/height/weight "
                         "(0%% missing, full range) instead of only the external sets")
    ap.add_argument("--max-seg", type=int, default=6000, help="cap cue computation per set")
    ap.add_argument("--r2-thresh", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="ppg-ood-audit")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    audit = audit.get("models", audit)          # new schema nests models under "models"
    if args.faithful not in audit:
        raise SystemExit(f"{args.faithful} not in {args.audit}; have {list(audit)}")
    keep, decodable = gbm.select_audit_features(audit[args.faithful], args.r2_thresh)
    has_ecg = "pat" in decodable                 # audit came from an ECG+PPG run
    print(f"[gbm] faithful={args.faithful}  audit-passing features: {keep}")
    print(f"[gbm] (decodable r2: "
          + ", ".join(f"{k}={v:.2f}" for k, v in sorted(decodable.items(), key=lambda t: -t[1])) + ")")
    if not keep:
        print("[gbm] no cue passed both tests; falling back to all decodable cues")
        keep = [k for k, v in decodable.items() if v > args.r2_thresh]

    # ---- ID data (VitalDB, subject-disjoint) + cue features
    import ood_benchmark as ob
    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    full = "gtr" in d and len(set(d["gtr"].tolist()).intersection(set(d["gte"].tolist()))) == 0
    if full:
        sp = {"train": dict(X=d["Xtr"], y=d["ytr"], g=d["gtr"]),
              "val": dict(X=d["Xva"], y=d["yva"], g=d["gva"]),
              "test": dict(X=d["Xte"], y=d["yte"], g=d["gte"])}
        has_demo = any(k.startswith("age_") for k in d)
        vdemo = {sp_: {k: d.get(f"{k}_{sp_[:2]}") for k in ("age", "sex", "bmi")
                       if f"{k}_{sp_[:2]}" in d}
                 for sp_ in ("train", "val", "test")} if has_demo else None
    else:
        sp = ob.subject_split(d, seed=args.seed)
        vdemo = None
    chans = [ECG, PPG] if has_ecg else [PPG]

    def prep(split, cap):
        X = mechlib.normalize(sp[split]["X"][:cap, :, chans])
        y = sp[split]["y"][:cap]
        g = sp[split]["g"][:cap]
        sc = cues_for(X, fs, has_ecg)
        dm = {k: v[:cap] for k, v in vdemo[split].items()} if vdemo else None
        return sc, y, g, dm

    print("[gbm] computing ID cue features (VitalDB)...")
    sctr, ytr, gtr, dmtr = prep("train", args.max_seg)
    scva, yva, gva, dmva = prep("val", args.max_seg // 3)
    scte, yte, gte, dmte = prep("test", args.max_seg // 2)

    names_base = keep
    Xtr, names = gbm.build_feature_table(sctr, None, names_base)
    Xva, _ = gbm.build_feature_table(scva, None, names_base)
    Xte, _ = gbm.build_feature_table(scte, None, names_base)

    models, vmae = gbm.train_gbm(Xtr, ytr, Xva, yva, names, seed=args.seed)
    idmae = np.abs(gbm.predict_gbm(models, Xte) - yte).mean(0)
    print(f"[gbm] features used: {names}")
    print(f"[gbm] ID  DBP MAE {idmae[1]:.2f}  (val {vmae[1]:.2f})")

    imp = gbm.feature_importance(models, names, target=1)
    print("[gbm] DBP feature importance: "
          + ", ".join(f"{n}={v:.0f}" for n, v in imp[:8]))

    # ---- VitalDB-native demographics ablation (0% missing, full age/BMI range -> the
    # strongest place to show age/sex/BMI impact; done on the ID test split, subject-grouped)
    vdemo_abl = None
    if args.vitaldb_demo and vdemo:
        Xf, nmz = gbm.build_feature_table(scte, dmte, names_base)   # cues + all demographics

        def split_fn():
            subs = np.unique(gte); rng = np.random.default_rng(args.seed); rng.shuffle(subs)
            a, b = int(0.6 * len(subs)), int(0.8 * len(subs))
            tr = np.isin(gte, subs[:a]); va = np.isin(gte, subs[a:b]); te = np.isin(gte, subs[b:])
            return np.where(tr)[0], np.where(va)[0], np.where(te)[0]

        abl = gbm.demographics_ablation(Xf, nmz, yte, split_fn, seed=args.seed)
        vdemo_abl = {"set": "VitalDB", "with_demo_dbp": abl["with_demo"]["mae"][1],
                     "no_demo_dbp": abl["no_demo"]["mae"][1], "delta_dbp": abl["delta_dbp"],
                     "demo_shap": abl.get("demo_shap")}
        print(f"[gbm] VitalDB demographics ablation: with={abl['with_demo']['mae'][1]:.2f} "
              f"without={abl['no_demo']['mae'][1]:.2f} delta={abl['delta_dbp']:+.2f} mmHg")
        if abl.get("demo_shap"):
            print("[gbm] VitalDB demo SHAP: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in abl["demo_shap"].items()))

    # ---- OOD sets: cue features + score
    results = {"faithful": args.faithful, "features": names, "id_mae": idmae.tolist(),
               "vitaldb_demo_ablation": vdemo_abl,
               "importance_dbp": imp, "ood": {}}

    ext_specs = [s for s in args.external.split(",") if s.strip()]
    for spec in ext_specs:
        nm, _, path = spec.partition("=")
        e = pa.load_bpbenchmark(path, name=nm)
        if len(e["X"]) > args.max_seg:
            idx = np.sort(np.random.default_rng(args.seed).choice(len(e["X"]), args.max_seg, False))
            e = {**e, "X": e["X"][idx], "y": e["y"][idx], "g": e["g"][idx],
                 "demo": ({k: v[idx] for k, v in e["demo"].items()} if e["demo"] else None)}
        Xr = mechlib.normalize(pa.resample_to(e["X"], 1250))
        # external sets are PPG-only; recompute cues on their single channel
        sc = mechlib.compute_morphology(Xr, fs, ch=0)
        Xf, _ = gbm.build_feature_table(sc, None, names_base)
        p = gbm.predict_gbm(models, Xf)
        bs = pa.bootstrap_mae(p, e["y"], e["g"])
        results["ood"][nm] = {"mae_dbp": bs["mae"][1], "lo": bs["lo"][1], "hi": bs["hi"][1],
                              "has_demo": e["demo"] is not None}
        print(f"[gbm] OOD {nm:10s} DBP {bs['mae'][1]:.2f}  [{bs['lo'][1]:.2f}, {bs['hi'][1]:.2f}]")

    # MIMIC-BP OOD (ECG+PPG if that is the feature space)
    if args.mimic:
        mc = ("ecg", "ppg") if has_ecg else ("ppg",)
        m = pa.load_mimic_bp(args.mimic, channels=mc, max_patients=args.mimic_patients, seed=args.seed)
        Xm, k = pa.window_segments(m["X"], 1250)
        ym = np.repeat(m["y"], k, 0); gm = np.repeat(m["g"], k, 0)
        if len(Xm) > args.max_seg:
            idx = np.sort(np.random.default_rng(args.seed).choice(len(Xm), args.max_seg, False))
            Xm, ym, gm = Xm[idx], ym[idx], gm[idx]
        Xm = mechlib.normalize(Xm)
        sc = cues_for(Xm, fs, has_ecg)
        Xf, _ = gbm.build_feature_table(sc, None, names_base)
        p = gbm.predict_gbm(models, Xf)
        bs = pa.bootstrap_mae(p, ym, gm)
        results["ood"]["mimic_bp"] = {"mae_dbp": bs["mae"][1], "lo": bs["lo"][1], "hi": bs["hi"][1]}
        print(f"[gbm] OOD mimic_bp   DBP {bs['mae'][1]:.2f}  [{bs['lo'][1]:.2f}, {bs['hi'][1]:.2f}]")

    # ---- age/sex ablation on the external set that has demographics with the most subjects
    # anchor the age/sex ablation on the demographics set with the MOST subjects and real age
    # spread (PPG-BP: 218 subj, age~57, hypertensive/diabetic -> where demographics matter),
    # not merely the first one found.
    demo_set = None
    prefer = ["ppgbp", "sensors", "bcg"]
    cands = []
    for spec in ext_specs:
        nm, _, path = spec.partition("=")
        e = pa.load_bpbenchmark(path, name=nm)
        if e["demo"] is not None and np.isfinite(e["demo"]["age"]).any():
            cands.append((nm, e))
    if cands:
        cands.sort(key=lambda t: (prefer.index(t[0]) if t[0] in prefer else 99,
                                  -len(np.unique(t[1]["g"]))))
        demo_set = cands[0]
    if demo_set:
        nm, e = demo_set
        if len(e["X"]) > args.max_seg:
            idx = np.sort(np.random.default_rng(args.seed).choice(len(e["X"]), args.max_seg, False))
            e = {**e, "X": e["X"][idx], "y": e["y"][idx], "g": e["g"][idx],
                 "demo": {k: v[idx] for k, v in e["demo"].items()}}
        Xr = mechlib.normalize(pa.resample_to(e["X"], 1250))
        sc = mechlib.compute_morphology(Xr, fs, ch=0)
        Xf, nmz = gbm.build_feature_table(sc, e["demo"], names_base)
        g = e["g"]

        def split_fn():
            subs = np.unique(g); rng = np.random.default_rng(args.seed); rng.shuffle(subs)
            a, b = int(0.6 * len(subs)), int(0.8 * len(subs))
            tr = np.isin(g, subs[:a]); va = np.isin(g, subs[a:b]); te = np.isin(g, subs[b:])
            return np.where(tr)[0], np.where(va)[0], np.where(te)[0]

        abl = gbm.demographics_ablation(Xf, nmz, e["y"], split_fn, seed=args.seed)
        results["demo_ablation"] = {
            "set": nm, "with_demo_dbp": abl["with_demo"]["mae"][1],
            "no_demo_dbp": abl["no_demo"]["mae"][1], "delta_dbp": abl["delta_dbp"],
            "demo_shap": abl.get("demo_shap")}
        print(f"\n[gbm] AGE/SEX ablation on {nm}: "
              f"with={abl['with_demo']['mae'][1]:.2f}  without={abl['no_demo']['mae'][1]:.2f}  "
              f"delta={abl['delta_dbp']:+.2f} mmHg")
        if abl.get("demo_shap"):
            print(f"[gbm] demo SHAP (mean|impact|): "
                  + ", ".join(f"{k}={v:.2f}" for k, v in abl["demo_shap"].items()))

    out = ROOT / "data" / "lightgbm_arm.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] {out}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, name="lightgbm-audit-features",
                         group="ood-audit", reinit=True,
                         config={"faithful": args.faithful, "features": names})
        run.summary.update({"id_mae_dbp": idmae[1]})
        for nm, v in results["ood"].items():
            run.log({f"gbm_ood/{nm}/mae_dbp": v["mae_dbp"]})
        tbl = wandb.Table(columns=["feature", "gain_dbp"])
        for n, v in imp:
            tbl.add_data(n, v)
        run.log({"gbm_importance": tbl})
        if "demo_ablation" in results:
            run.summary.update({"demo_delta_dbp": results["demo_ablation"]["delta_dbp"]})
        run.finish()


if __name__ == "__main__":
    main()
