"""vascular_age.py -- can PPG morphology predict age, and does the residual mean anything?

Why this target rather than blood pressure: every negative result in this project traces to the
within-subject regime, where BP barely moves and PTT explains 4-5% of its variance. Age is a
BETWEEN-subject quantity, which is exactly where the stiffness physiology does hold here
(-27 to -36 mmHg/s, correct sign). It is also exactly labelled, unlike ABP-derived BP.

Two stages, because the second is what separates a vascular-age index from an age regressor:

  1  predict chronological age from PPG morphology alone, patient-disjoint CV
  2  the residual (predicted - actual) is only "vascular age" if it tracks things that should
     accompany accelerated vascular ageing. Here we can test BMI and sex; hypertension and
     diabetes need the VitalDB clinical table (not in the npz), which is the follow-up.

Features: the APG a/b/c/d/e family (Takazawa ageing index and friends) from features_full, plus
single-pulse timings. All are measured INSIDE the PPG channel, so the ~500 ms PLETH lag that
invalidates cross-channel PTT here cannot touch them.

    python vascular_age.py --patients 1100
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import mechlib
import features_full as ff
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FS = 125
CACHE = DATA / "vascage_train.csv"


def timing_feats(p, fs=FS):
    sm = savgol_filter(p, 11, 3)
    rng = sm.max() - sm.min()
    if rng < 1e-9:
        return {}
    pk = find_peaks(sm, distance=int(0.4 * fs), prominence=rng * 0.3)[0]
    if len(pk) < 4:
        return {}
    d2 = savgol_filter(np.gradient(np.gradient(sm)), 9, 3)
    ct, no, wd = [], [], []
    for j in range(len(pk) - 1):
        a, b = pk[j], pk[j + 1]
        lo = max(0, a - int(0.30 * fs))
        if a - lo < 4:
            continue
        ft = lo + int(np.argmin(sm[lo:a]))
        ct.append((a - ft) / fs * 1000)
        half = (sm[a] + sm[ft]) / 2
        ab = np.where(sm[ft:b] > half)[0]
        if len(ab) > 2:
            wd.append(len(ab) / fs * 1000)
        L = b - a
        w0, w1 = a + int(0.08 * L), a + int(0.60 * L)
        if w1 - w0 > 5:
            c = find_peaks(d2[w0:w1])[0]
            if len(c):
                no.append((w0 + c[int(np.argmax(d2[w0:w1][c]))] - a) / fs * 1000)
    return {"crest": np.median(ct) if ct else np.nan,
            "notch": np.median(no) if no else np.nan,
            "width": np.median(wd) if wd else np.nan,
            "hr": 60.0 / (np.median(np.diff(pk)) / fs)}


def build(n_pat, n_seg, flush_every=100):
    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    g = np.array(d["gtr"])
    age, sex, bmi = np.array(d["age_tr"]), np.array(d["sex_tr"]), np.array(d["bmi_tr"])
    subs = np.unique(g)[:n_pat]
    rows, t0 = [], time.time()
    for n, s in enumerate(subs, 1):
        i = np.where(g == s)[0][:n_seg]
        X = mechlib.normalize(np.array(d["Xtr"][i])[:, :, [ECG, PPG]])
        acc = []
        for seg in X:
            f = ff.apg_full(seg[:, 1], FS)
            f.update(timing_feats(seg[:, 1]))
            acc.append(f)
        keys = sorted(set().union(*[set(a) for a in acc])) if acc else []
        med = {k: float(np.nanmedian([a.get(k, np.nan) for a in acc])) for k in keys}
        rows.append({"s": int(s), "age": float(age[i][0]), "sex": float(sex[i][0]),
                     "bmi": float(bmi[i][0]), **med})
        if n % flush_every == 0:
            pd.DataFrame(rows).to_csv(CACHE, index=False)      # survive an interrupt
            print(f"  {n}/{len(subs)} patients ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"[built] {len(df)} patients in {time.time()-t0:.0f}s -> {CACHE.name}", flush=True)
    return df


def evaluate(df):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import RidgeCV

    meta = {"s", "age", "sex", "bmi"}
    feats = [c for c in df.columns if c not in meta]
    D = df.dropna(subset=["age"]).copy()
    D[feats] = D[feats].apply(lambda c: c.fillna(c.median()))
    X, y, S = D[feats].values, D.age.values, D.s.values
    print(f"\n[eval] {len(y)} patients, {len(feats)} PPG features, "
          f"age {y.min():.0f}-{y.max():.0f} (SD {y.std():.1f})")

    gk = GroupKFold(n_splits=5)
    out = {}
    for name in ("ridge", "lightgbm"):
        pred = np.zeros(len(y))
        for tr, te in gk.split(X, y, groups=S):
            if name == "ridge":
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((X[tr] - mu) / sd, y[tr])
                pred[te] = m.predict((X[te] - mu) / sd)
            else:
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                      min_child_samples=20, subsample=0.8,
                                      colsample_bytree=0.8, random_state=0, verbosity=-1)
                m.fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
        mae = float(np.abs(pred - y).mean())
        r = float(np.corrcoef(pred, y)[0, 1])
        out[name] = {"mae": mae, "r": r, "pred": pred.tolist()}
        print(f"  {name:9s} MAE {mae:5.2f} yr   r {r:+.2f}")
    base = float(np.abs(y - np.median(y)).mean())
    out["baseline_median"] = base
    best = min(("ridge", "lightgbm"), key=lambda k: out[k]["mae"])
    print(f"  {'baseline':9s} MAE {base:5.2f} yr   (predict the median age)")
    print(f"  -> best model beats it by {base - out[best]['mae']:+.2f} yr")

    # stage 2: is the residual a vascular signal, or just error?
    res = np.array(out[best]["pred"]) - y
    print(f"\n[residual] SD {res.std():.1f} yr  (predicted - chronological)")
    for nm in ("bmi", "sex"):
        v = D[nm].values
        ok = np.isfinite(v)
        rr = float(np.corrcoef(res[ok], v[ok])[0, 1])
        out[f"residual_vs_{nm}"] = rr
        print(f"  corr(residual, {nm}) = {rr:+.3f}")
    print("  a vascular-age index needs the residual to track hypertension/diabetes, which are")
    print("  in the VitalDB clinical table, not this npz -- that is the decisive follow-up.")

    # which features carry the signal
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          min_child_samples=20, random_state=0, verbosity=-1).fit(X, y)
    imp = sorted(zip(feats, m.booster_.feature_importance("gain")), key=lambda t: -t[1])
    out["top_features"] = [(n, float(v)) for n, v in imp[:10]]
    print("\n[features] highest gain:")
    for n, v in imp[:8]:
        print(f"  {n:14s} {v:10.0f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=1100)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.rebuild:
        df = pd.read_csv(CACHE)
        print(f"[cache] {len(df)} patients from {CACHE.name}")
    else:
        df = build(args.patients, args.segments)
    res = evaluate(df)
    (DATA / "vascular_age.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/vascular_age.json")


if __name__ == "__main__":
    main()
