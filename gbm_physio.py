"""gbm_physio.py -- a minimal LightGBM built ONLY from the two physiologically-sanctioned
stiffness signals: pulse arrival time (PAT family) and wave reflection (AIx family).

Motivation: the audit says models depend on heart rate more than arrival time. A model that is
*restricted* to the governing-law quantities tells us how much BP signal those quantities
actually carry -- i.e. is the field's premise even attainable from PAT + reflection alone?

Reported: ID + all OOD, versus (a) the full 83-feature model, (b) a rate-only model, and
(c) the deep nets. Accuracy claims only -- mechanism claims for this model await a specificity-
corrected reflection perturbation.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

import lightgbm_arm as gbm
import physics_audit as pa

ROOT = Path(__file__).resolve().parent

# the two governing-law families
PAT_F = ["pat", "pat_foot", "pat_peak", "ptt_var", "xcorr_lag", "xcorr_peak", "xcorr_width"]
AIX_F = ["aix", "reflect_idx", "notch_depth", "notch_time", "t_b", "t_c", "t_d", "t_e",
         "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "takazawa", "ushiro"]
# the confound family, as a contrast
RATE_F = ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv", "hrv_lf", "hrv_hf",
          "hrv_lfhf", "period"]


def main():
    full = pickle.load(open(ROOT / "data" / "_feat_full_ALLtrain.pkl", "rb"))
    OOD = pickle.load(open(ROOT / "data" / "_variants_ood_feats.pkl", "rb"))
    Ftr, ytr, Fte, yte = full["Ftr"], full["ytr"], full["Fte"], full["yte"]
    have = set(Ftr)
    allk = [k for k in Ftr if np.isfinite(np.asarray(Ftr[k], float)).mean() > 0.3
            and np.nanstd(np.asarray(Ftr[k], float)) > 1e-9]

    sets = {
        "PAT only": [k for k in PAT_F if k in have],
        "AIx/reflection only": [k for k in AIX_F if k in have],
        "PAT + AIx (governing law)": [k for k in PAT_F + AIX_F if k in have],
        "rate only (confound)": [k for k in RATE_F if k in have],
        "PAT + AIx + rate": [k for k in PAT_F + AIX_F + RATE_F if k in have],
        "all 83 features": allk,
    }

    def tbl(F, keys, n):
        return np.column_stack([np.asarray(F.get(k, np.full(n, np.nan)), float) for k in keys])

    params = dict(n_estimators=800, learning_rate=0.03, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, min_child_samples=50)
    out = {}
    hdr = f"{'feature set':28s} {'n':>3s} {'ID':>6s} | " + "  ".join(f"{k:>9s}" for k in OOD)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for tag, keys in sets.items():
        if not keys:
            continue
        Mtr = tbl(Ftr, keys, len(ytr)); med = gbm.column_medians(Mtr)
        m = lgb.LGBMRegressor(**params, random_state=0, verbosity=-1)
        m.fit(gbm._impute(Mtr, med), ytr[:, 1])
        row = {"n_feat": len(keys), "features": keys,
               "ID": float(np.abs(m.predict(gbm._impute(tbl(Fte, keys, len(yte)), med)) - yte[:, 1]).mean())}
        for nm, (F, y, g) in OOD.items():
            p = m.predict(gbm._impute(tbl(F, keys, len(y)), med))
            row[nm] = float(pa.bootstrap_mae(np.stack([p, p], 1), y, g)["mae"][1])
        out[tag] = row
        print(f"{tag:28s} {len(keys):3d} {row['ID']:6.2f} | " +
              "  ".join(f"{row[nm]:9.1f}" for nm in OOD), flush=True)

    (ROOT / "data" / "gbm_physio.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n[done] data/gbm_physio.json", flush=True)


if __name__ == "__main__":
    main()
