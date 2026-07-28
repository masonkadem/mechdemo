"""gbm_families.py -- train a SEPARATE LightGBM per feature family and compare ID vs OOD.

Rather than one tree on a mixed feature set, fit one tree per physiological family so we can
read how far each signal type carries BP on its own, in-distribution and out:

  pat          arrival-time cues (pat, pat_foot, pat_peak, ptt_var)     [needs ECG]
  fractal      complexity (higuchi FD, katz FD, spectral entropy)
  demographics age, sex, height, weight, bmi                            [VitalDB-native]
  morphology   rise, aix, apg, notch, decay, ...                        [reference]
  all          every cue + demographics                                 [ceiling]

Trains on VitalDB (full CalFree if available), scores ID + MIMIC-BP (ECG+PPG) and the
PPG-only external sets. Demographics-only shows the pure age/sex/BMI baseline -- the number
a model must beat to claim it is using the *waveform*.

    python gbm_families.py --data data/vitaldb_full_calfree.npz --vitaldb-demo
"""
import argparse
import json
from pathlib import Path

import numpy as np

import mechlib
import physics_audit as pa
import features_ext as fx
import lightgbm_arm as gbm
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
MIMIC = "C:/Users/mason/OneDrive - McMaster University/2026/BP"

FAMILIES = {
    "pat":          ["pat", "pat_foot", "pat_peak", "ptt_var"],
    "xcorr":        ["xcorr_lag", "xcorr_peak", "xcorr_width"],   # ECG-PPG cross-correlation PTT
    "fractal":      ["hfd", "katz_fd", "spec_ent"],
    "demographics": ["age", "sex", "height", "weight", "bmi"],
    "morphology":   ["rise", "aix", "apg", "notch", "decay", "kurt", "peak"],
}
# families sourced from the waveform (xcorr needs ECG, so it is ID/MIMIC-only)
CUE_FAMILIES = ["pat", "xcorr", "fractal", "morphology"]


def all_cues(X, fs, has_ecg):
    """Core morphology/PAT cues + the extended fractal/derivative battery, merged."""
    base = mechlib.compute_scalars(X, fs) if has_ecg else mechlib.compute_morphology(X, fs, ch=0)
    ext = fx.compute_ext(X, fs, ppg_ch=(PPG if has_ecg else 0), ecg_ch=(ECG if has_ecg else None))
    base.update(ext)
    return base


def demo_slice(demo, idx):
    return {k: v[idx] for k, v in demo.items()} if demo else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--mimic", default=MIMIC)
    ap.add_argument("--mimic-patients", type=int, default=400)
    ap.add_argument("--external", default="bcg=data/bcg_dataset,"
                    "sensors=C:/Users/mason/Downloads/sensors_dataset/sensors_dataset,"
                    "uci2=data/uci2_dataset/uci2_dataset,"
                    "ppgbp=C:/Users/mason/Downloads/ppgbp_dataset/ppgbp_dataset")
    ap.add_argument("--max-seg", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="ppg-ood-audit")
    ap.add_argument("--entity", default="mkadem")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    full = "gtr" in d and len(set(d["gtr"].tolist()).intersection(set(d["gte"].tolist()))) == 0
    if full:
        sp = {s: dict(X=d[f"X{t}"], y=d[f"y{t}"], g=d[f"g{t}"])
              for s, t in [("train", "tr"), ("val", "va"), ("test", "te")]}
        dem = {s: {k: d.get(f"{k}_{t}") for k in ("age", "sex", "bmi", "height", "weight")
                   if f"{k}_{t}" in d}
               for s, t in [("train", "tr"), ("val", "va"), ("test", "te")]}
        dem = {s: (v or None) for s, v in dem.items()}
    else:
        import ood_benchmark as ob
        sp = ob.subject_split(d, seed=args.seed)
        dem = {s: None for s in sp}
    has_ecg = True                              # VitalDB has ECG -> pat family available

    def prep(split, cap):
        # sample segments ACROSS subjects, not the first N (which come from a few subjects and
        # let the GBM memorize a tiny cohort). Random subject-diverse draw.
        g = sp[split]["g"]
        if cap < len(g):
            idx = np.sort(np.random.default_rng(args.seed).choice(len(g), cap, replace=False))
        else:
            idx = np.arange(len(g))
        X = mechlib.normalize(sp[split]["X"][idx][:, :, [ECG, PPG]])
        sc = all_cues(X, fs, has_ecg)
        dm = demo_slice(dem[split], idx) if dem[split] else None
        return sc, sp[split]["y"][idx], sp[split]["g"][idx], dm

    # cue extraction (fractal/xcorr/per-beat) is the slow part -> cache it so figure/param
    # reruns are instant. Key on the data file + max_seg.
    import pickle
    cache = ROOT / "data" / f"_fam_cues_{Path(args.data).stem}_{args.max_seg}.pkl"
    if cache.exists():
        print(f"[fam] loading cached cue features {cache.name}", flush=True)
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        (sctr, ytr, gtr, dmtr) = blob["tr"]
        (scva, yva, gva, dmva) = blob["va"]
        (scte, yte, gte, dmte) = blob["te"]
        ood = blob["ood"]
    else:
        print("[fam] computing VitalDB cue features (train/val/test)...", flush=True)
        sctr, ytr, gtr, dmtr = prep("train", args.max_seg)
        scva, yva, gva, dmva = prep("val", args.max_seg // 3)
        scte, yte, gte, dmte = prep("test", args.max_seg // 2)
        ood = None
    if ood is None:
        print("[fam] computing OOD cue features...", flush=True)
        ood = {}
        for spec in [s for s in args.external.split(",") if s.strip()]:
            nm, _, path = spec.partition("=")
            e = pa.load_bpbenchmark(path, name=nm)
            if len(e["X"]) > args.max_seg:
                ii = np.sort(np.random.default_rng(args.seed).choice(len(e["X"]), args.max_seg, False))
                e = {**e, "X": e["X"][ii], "y": e["y"][ii], "g": e["g"][ii],
                     "demo": ({k: v[ii] for k, v in e["demo"].items()} if e["demo"] else None)}
            Xr = mechlib.normalize(pa.resample_to(e["X"], 1250))
            ood[nm] = dict(sc=all_cues(Xr, fs, False), y=e["y"], g=e["g"], demo=e["demo"])
        if args.mimic:
            m = pa.load_mimic_bp(args.mimic, channels=("ecg", "ppg"),
                                 max_patients=args.mimic_patients, seed=args.seed)
            Xm, k = pa.window_segments(m["X"], 1250)
            ym, gm = np.repeat(m["y"], k, 0), np.repeat(m["g"], k, 0)
            if len(Xm) > args.max_seg:
                ii = np.sort(np.random.default_rng(args.seed).choice(len(Xm), args.max_seg, False))
                Xm, ym, gm = Xm[ii], ym[ii], gm[ii]
            ood["mimic_bp"] = dict(sc=all_cues(mechlib.normalize(Xm), fs, True), y=ym, g=gm, demo=None)
        # persist so param/figure reruns skip the slow extraction
        with open(cache, "wb") as f:
            pickle.dump({"tr": (sctr, ytr, gtr, dmtr), "va": (scva, yva, gva, dmva),
                         "te": (scte, yte, gte, dmte), "ood": ood}, f)
        print(f"[fam] cached cue features -> {cache.name}", flush=True)

    # ---- fit one GBM per family + combined models
    results = {}
    # isolated families + two combined models so the demographics contribution is explicit:
    #   waveform      = every cue family pooled, NO demographics  (pure signal ceiling)
    #   waveform+demo = waveform cues + age/sex/bmi/...           (full combined ceiling)
    # the gap between them is exactly what demographics add on top of the waveform.
    all_cue_names = sorted({c for fam2 in CUE_FAMILIES for c in FAMILIES[fam2]})
    fam_specs = list(FAMILIES.items()) + [("waveform", all_cue_names),
                                          ("waveform+demo", all_cue_names)]
    for fam, feats in fam_specs:
        # build train/val/test tables for this family's schema (+demographics where relevant)
        use_demo = fam in ("demographics", "waveform+demo")
        if fam == "demographics":
            names_base = []                      # demographics only, no cues
        else:
            names_base = feats

        # fixed demographics schema so every split emits identical columns even when a split
        # is missing a field (e.g. val has no bmi) -> fill NaN, impute later
        dkeys = tuple(k for k in ("age", "sex", "bmi", "height", "weight")
                      if dmtr and k in dmtr) if use_demo else ()
        Xtr, names = gbm.build_feature_table(sctr, dmtr if use_demo else None, names_base,
                                             demo_keys=dkeys)
        Xva, _ = gbm.build_feature_table(scva, dmva if use_demo else None, names_base,
                                         demo_keys=dkeys, n=len(yva))
        Xte, _ = gbm.build_feature_table(scte, dmte if use_demo else None, names_base,
                                         demo_keys=dkeys, n=len(yte))
        if Xtr.shape[1] == 0:
            print(f"[fam] {fam}: no features available, skip")
            continue

        models, vmae = gbm.train_gbm(Xtr, ytr, Xva, yva, names, seed=args.seed)
        idmae = np.abs(gbm.predict_gbm(models, Xte) - yte).mean(0)
        row = {"features": names, "id_dbp": float(idmae[1]), "id_sbp": float(idmae[0]), "ood": {}}
        for nm, e in ood.items():
            edemo = e["demo"] if use_demo else None
            Xf, _ = gbm.build_feature_table(e["sc"], edemo, names_base,
                                            demo_keys=dkeys, n=len(e["y"]))
            p = gbm.predict_gbm(models, Xf)
            bs = pa.bootstrap_mae(p, e["y"], e["g"])
            row["ood"][nm] = {"dbp": bs["mae"][1], "lo": bs["lo"][1], "hi": bs["hi"][1]}
        row["importance_dbp"] = gbm.feature_importance(models, names, target=1)[:8]
        results[fam] = row
        oods = "  ".join(f"{nm}={row['ood'][nm]['dbp']:.1f}" for nm in row["ood"])
        print(f"[fam] {fam:12s} ID DBP {idmae[1]:.2f}  |  {oods}", flush=True)

    out = ROOT / "data" / "gbm_families.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] {out}")

    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.project, entity=args.entity, name="gbm-feature-families",
                         group="ood-audit", reinit=True)
        oodsets = list(next(iter(results.values()))["ood"])
        tbl = wandb.Table(columns=["family", "id_dbp"] + [f"ood/{o}" for o in oodsets])
        for fam, r in results.items():
            tbl.add_data(fam, r["id_dbp"], *[r["ood"][o]["dbp"] for o in oodsets])
        run.log({"family_comparison": tbl})
        run.finish()


if __name__ == "__main__":
    main()
