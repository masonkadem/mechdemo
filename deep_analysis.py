"""deep_analysis.py -- three follow-up analyses on the trained deep models:

  1. SHORTCUT PROBING: linear-probe every layer for the CONFOUNDS a model could ride instead
     of physiology (HR, cardiac period, PPG amplitude). A confound that is highly decodable in
     late layers while the roll-audit is flat = the shortcut the model actually uses.

  2. ROLL-MAGNITUDE VALIDATION: for each subject, regress their MEASURED BP on their MEASURED
     PTT across segments -> real dBP/dPTT (mmHg/s). Compare to the roll-audit's induced slope.
     Tests whether the model's slope MAGNITUDE is physiologically calibrated, not just sign-right.

  3. CALIBRATION OVERHEAD: sweep K = 1,2,3,5,10 per-subject calibration anchors, plot DBP MAE
     vs K, to find the fewest anchors a deployed model needs.

    python deep_analysis.py --data data/vitaldb_full_calfree.npz --models xresnet1d50,transformer
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name, n_ch=2, L=1250):
    import ood_benchmark as ob
    ck = torch.load(MODEL_DIR / f"{name}_ecgppg_full.pt", map_location=DEVICE, weights_only=False)
    model = ob.build_model(name, n_ch=n_ch, L=L)
    model.load_state_dict(ck["state_dict"])
    model.to(DEVICE).eval()
    return model, ck["mu"], ck["sd"]


def predict(model, X, mu, sd, bs=512):
    import ood_benchmark as ob
    return ob.predict(model, X, DEVICE, mu, sd, bs)


# -------------------------------------------------- 1. shortcut probing
CONFOUNDS = ["hr", "period", "amp", "age"]        # non-physiological / confound cues


def shortcut_probe(name, X, scalars, age, device):
    """Layer-wise probe R^2 for each confound. Returns {layer: {confound: r2}}."""
    import ood_benchmark as ob
    feats = ob.layer_features(load_model(name)[0], name, X, device)
    targets = {k: scalars[k] for k in CONFOUNDS if k in scalars and k != "age"}
    if age is not None:
        targets["age"] = age
    rows = {}
    for lname, F in feats.items():
        rows[lname] = {t: max(mechlib.linear_probe(F, v), 0.0) for t, v in targets.items()}
    return rows


# -------------------------------------------------- 2. roll magnitude validation
def real_dbp_dptt(ptt, dbp, grp, min_seg=8, ptt_spread=0.005):
    """Per-subject real dBP/dPTT slope (mmHg/s): regress DBP on measured PTT within each
    subject who has enough segments and enough PTT spread to fit a line. Returns the
    per-subject slopes and their median."""
    slopes = []
    for g in np.unique(grp):
        m = (grp == g) & np.isfinite(ptt) & np.isfinite(dbp)
        if m.sum() < min_seg:
            continue
        p, b = ptt[m], dbp[m]
        if p.std() < ptt_spread:                  # not enough natural PTT variation to fit
            continue
        slopes.append(np.polyfit(p, b, 1)[0])
    slopes = np.array(slopes)
    return slopes, (float(np.median(slopes)) if len(slopes) else np.nan)


# -------------------------------------------------- 3. calibration overhead
def calib_sweep(pred, y, grp, Ks=(1, 2, 3, 5, 10)):
    """DBP MAE after per-subject offset calibration with K anchors, for each K."""
    out = {}
    for K in Ks:
        errs = []
        for g in np.unique(grp):
            idx = np.where(grp == g)[0]
            if len(idx) <= K:
                continue
            off = (y[idx[:K], 1] - pred[idx[:K], 1]).mean()
            errs.append(np.abs((pred[idx[K:], 1] + off) - y[idx[K:], 1]))
        out[K] = float(np.concatenate(errs).mean()) if errs else np.nan
    # uncalibrated reference
    out[0] = float(np.abs(pred[:, 1] - y[:, 1]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--models", default="xresnet1d50,transformer,lenet1d,inception1d,xresnet1d101")
    ap.add_argument("--n", type=int, default=3000, help="segments for probing/PTT (subject-diverse)")
    args = ap.parse_args()

    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    # subject-diverse test sample
    g_all = d["gte"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(g_all), min(args.n, len(g_all)), replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    y = d["yte"][sel]; grp = d["gte"][sel]
    age = d["age_te"][sel] if "age_te" in d else None

    print(f"[da] {len(X)} segments, {len(np.unique(grp))} subjects")
    print("[da] computing cues + measured PTT (once)...")
    scalars = mechlib.compute_scalars(X, fs)
    ptt = mechlib.compute_ptt(X, fs)
    print(f"[da] measured PTT: median {np.nanmedian(ptt)*1000:.0f} ms, "
          f"valid {np.isfinite(ptt).mean():.0%}")

    # ---- 2. real per-subject dBP/dPTT (ground truth) ----
    slopes, med_real = real_dbp_dptt(ptt, y[:, 1], grp)
    print(f"\n[da] REAL per-subject dDBP/dPTT: median {med_real:+.1f} mmHg/s "
          f"(n={len(slopes)} subjects, IQR [{np.percentile(slopes,25):+.0f}, "
          f"{np.percentile(slopes,75):+.0f}])")

    results = {"real_dbp_dptt": {"median": med_real, "per_subject": slopes.tolist()},
               "models": {}}

    for name in [m.strip() for m in args.models.split(",")]:
        print(f"\n=== {name} ===")
        model, mu, sd = load_model(name)
        pred = predict(model, X, mu, sd)

        # ---- 1. shortcut probe ----
        rows = shortcut_probe(name, X, scalars, age, DEVICE)
        last = list(rows)[-1]
        print("  shortcut decodability (final layer): "
              + ", ".join(f"{c}={rows[last].get(c,0):.2f}" for c in CONFOUNDS if c in rows[last]))

        # ---- roll audit slope (model's induced magnitude) ----
        fn = lambda Xr: predict(model, Xr, mu, sd)
        aud = mechlib.causal_ptt_audit(None, X, fs, DEVICE, predict_fn=fn, n_max=min(1500, len(X)))
        model_slope = aud["dbp"]["dBP_dPTT"]
        ratio = model_slope / med_real if np.isfinite(med_real) and med_real != 0 else np.nan
        print(f"  roll-audit slope {model_slope:+.1f} vs REAL {med_real:+.1f} mmHg/s  "
              f"(model/real = {ratio:.2f})")

        # ---- 3. calibration overhead ----
        cal = calib_sweep(pred, y, grp)
        print("  calib DBP MAE:  " + "  ".join(f"K={k}:{cal[k]:.2f}" for k in sorted(cal)))

        results["models"][name] = {
            "shortcut_probe": rows, "roll_slope": model_slope,
            "slope_ratio_to_real": ratio, "calib": cal,
            "final_layer_confounds": rows[last],
        }

    (ROOT / "data" / "deep_analysis.json").write_text(json.dumps(results, indent=2, default=float))
    print("\n[done] data/deep_analysis.json")


if __name__ == "__main__":
    main()
