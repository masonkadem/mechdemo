"""gbm_ppg_physio.py -- PPG-ONLY physiology model, evaluated ID + on the four PPG-only OOD sets.

Wearables have no ECG, so this is the deployable setting. We compare feature families that are
computable from a single PPG channel, including the complexity/fractal family that was central
to the sleep-staging work, and demographics:

  fractal      : Higuchi/Katz FD, spectral entropy      (complexity; P1 connection)
  reflection   : AIx, notch, APG a-e amplitudes+timings (single-site stiffness proxy)
  morphology   : widths, areas, rise/crest, percentiles
  demographics : age, sex, BMI                          (where available)
  combinations thereof

No ECG-derived features anywhere (no PAT, no HRV, no QRS) -- so the comparison is honest for a
PPG-only device. Demographics are present on VitalDB/BCG/Sensors/PPG-BP but NOT on UCI2, and are
imputed with TRAIN medians when absent (never within-column, which caused an earlier artifact).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm
import physics_audit as pa

ROOT = Path(__file__).resolve().parent

FRACTAL = ["hfd", "katz_fd", "spec_ent", "ppg_skew_g", "ppg_kurt_g"]
REFLECT = ["aix", "reflect_idx", "notch_depth", "notch_time", "t_b", "t_c", "t_d", "t_e",
           "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "takazawa", "ushiro"]
MORPH = ["rise", "crest", "sys_area", "dia_area", "sys_dia_ratio", "decay_slope",
         "sw10", "sw25", "sw50", "sw75", "sw90", "dw10", "dw25", "dw50", "dw75", "dw90",
         "peak_mean", "amp_mean", "amp_cv", "ppg_p10", "ppg_p25", "ppg_p75", "ppg_p90",
         "vpg_max", "vpg_min", "vpg_ratio", "t_vpg_max", "t_vpg_min"]
DEMO = ["age", "sex", "bmi"]


def main():
    full = pickle.load(open(ROOT / "data" / "_feat_full_ALLtrain.pkl", "rb"))
    OODF = pickle.load(open(ROOT / "data" / "_variants_ood_feats.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    # PPG-only OOD sets (drop MIMIC-BP: it is the ECG+PPG set and has no demographics)
    OOD = {k: v for k, v in OODF.items() if k != "MIMIC-BP"}
    have = set(Ftr)

    # demographics for VitalDB train/test come from the converted npz
    d = np.load(ROOT / "data" / "vitaldb_full_calfree.npz")
    dem_tr = {"age": d["age_tr"], "sex": d["sex_tr"], "bmi": d["bmi_tr"]}
    dem_te = {"age": d["age_te"], "sex": d["sex_te"], "bmi": d["bmi_te"]}

    sets = {
        "fractal": ([k for k in FRACTAL if k in have], False),
        "reflection": ([k for k in REFLECT if k in have], False),
        "morphology": ([k for k in MORPH if k in have], False),
        "demographics only": ([], True),
        "fractal + demo": ([k for k in FRACTAL if k in have], True),
        "reflection + fractal": ([k for k in REFLECT + FRACTAL if k in have], False),
        "reflection + fractal + demo": ([k for k in REFLECT + FRACTAL if k in have], True),
        "all PPG-only": ([k for k in REFLECT + FRACTAL + MORPH if k in have], False),
        "all PPG-only + demo": ([k for k in REFLECT + FRACTAL + MORPH if k in have], True),
    }

    def tbl(F, keys, n, demo=None, use_demo=False):
        cols = [np.asarray(F.get(k, np.full(n, np.nan)), float) for k in keys]
        if use_demo:
            for dk in DEMO:
                v = demo.get(dk) if demo else None
                cols.append(np.asarray(v, float)[:n] if v is not None else np.full(n, np.nan))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50)
    out = {}
    hdr = f"{'feature set':30s} {'n':>3s} {'ID':>6s} | " + "  ".join(f"{k:>8s}" for k in OOD)
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    for tag, (keys, use_demo) in sets.items():
        Mtr = tbl(Ftr, keys, len(ytr), dem_tr, use_demo)
        if Mtr.shape[1] == 0:
            continue
        med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params, random_state=0, verbosity=-1)
        m.fit(gbm._impute(Mtr, med), ytr[:, 1])
        Xte_ = gbm._impute(tbl(Fte, keys, len(yte), dem_te, use_demo), med)
        row = {"n_feat": Mtr.shape[1], "ID": float(np.abs(m.predict(Xte_) - yte[:, 1]).mean())}
        for nm, (F, y, g) in OOD.items():
            # external demographics are not carried in the cached OOD features -> train-median
            Mo = gbm._impute(tbl(F, keys, len(y), None, use_demo), med)
            p = m.predict(Mo)
            row[nm] = float(pa.bootstrap_mae(np.stack([p, p], 1), y, g)["mae"][1])
        out[tag] = row
        print(f"{tag:30s} {row['n_feat']:3d} {row['ID']:6.2f} | " +
              "  ".join(f"{row[nm]:8.1f}" for nm in OOD), flush=True)

    (ROOT / "data" / "gbm_ppg_physio.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n[done] data/gbm_ppg_physio.json", flush=True)


if __name__ == "__main__":
    main()
