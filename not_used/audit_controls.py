"""audit_controls.py -- POSITIVE and NEGATIVE controls for the roll-audit.

The audit's nulls are only interpretable if we show the audit CAN detect PTT-dependence when it
is present by construction, and does NOT report it when it is absent by construction.

  positive : BP_hat = a/PAT + b, fit on measured PAT. Uses arrival time and nothing else, so the
             roll-audit MUST return a large, correct-signed, consistent slope.
  negative : BP_hat = constant (train mean). Depends on nothing, so the audit MUST return ~0.
  amplitude: BP_hat = f(PPG amplitude). Depends on a non-timing cue -> the PAT audit should be
             ~flat while an amplitude perturbation moves it (specificity check).

Any real model's slope is then read against these bounds.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

import mechlib
import physics_audit as pa
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent


def measured_pat(X, fs):
    """Per-segment PAT (s) using the same estimator the audit's physiology refers to."""
    return mechlib.compute_ptt(X, fs)


def main():
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    rng = np.random.default_rng(0)
    # fit split (train) and audit split (test)
    ftr = np.sort(rng.choice(len(d["gtr"]), 6000, replace=False))
    fte = np.sort(rng.choice(len(d["gte"]), 1500, replace=False))
    Xtr = mechlib.normalize(d["Xtr"][ftr][:, :, [ECG, PPG]]); ytr = d["ytr"][ftr]
    Xte = mechlib.normalize(d["Xte"][fte][:, :, [ECG, PPG]]); yte = d["yte"][fte]

    print("[ctrl] measuring PAT on fit split ...", flush=True)
    pat_tr = measured_pat(Xtr, fs)
    ok = np.isfinite(pat_tr)
    # --- positive control: BP = a/PAT + b (the governing law, fit by least squares)
    A = np.column_stack([1.0 / pat_tr[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, ytr[ok, 1], rcond=None)
    a_pat, b_pat = float(coef[0]), float(coef[1])
    print(f"[ctrl] positive control fitted: DBP = {a_pat:.3f}/PAT + {b_pat:.2f}", flush=True)

    # --- amplitude control: BP = ridge(amplitude features)
    amp_tr = np.array([np.ptp(x[:, PPG]) for x in Xtr])
    amp_mod = Ridge(alpha=1.0).fit(amp_tr[:, None], ytr[:, 1])
    dbp_mean = float(ytr[:, 1].mean())

    # A cross-correlation PAT that does NOT saturate: align each segment's PPG against a fixed
    # template by full-signal xcorr. Unlike the tangent-foot estimator (which loses the foot out
    # of its search window for large positive shifts -> only 55% tracking at +48 ms), this tracks
    # a pure time-shift exactly, so a model built on it is a valid positive control.
    tmpl = np.median(Xtr[:200, :, PPG], axis=0)

    def xcorr_shift(Xr):
        out = np.empty(len(Xr))
        for i, x in enumerate(Xr):
            c = np.correlate(x[:, PPG] - x[:, PPG].mean(), tmpl - tmpl.mean(), mode="same")
            out[i] = (np.argmax(c) - len(c) // 2) / fs        # seconds of shift vs template
        return out

    sh_tr = xcorr_shift(Xtr)
    pat_proxy_tr = np.nanmedian(pat_tr[ok]) + sh_tr           # template PAT + measured shift
    A2 = np.column_stack([1.0 / np.clip(pat_proxy_tr, 0.05, None), np.ones(len(sh_tr))])
    coef2, *_ = np.linalg.lstsq(A2, ytr[:, 1], rcond=None)
    a2, b2 = float(coef2[0]), float(coef2[1])
    # force the textbook sign so this is a POSITIVE control by construction, not a data fit
    a2 = abs(a2) if a2 != 0 else 1.0
    print(f"[ctrl] xcorr-based positive control: DBP = {a2:.3f}/PAT + {b2:.2f} (sign forced +)",
          flush=True)

    def make_fn(kind):
        def fn(Xr):
            if kind == "positive":
                p = measured_pat(Xr, fs)
                p = np.where(np.isfinite(p), p, np.nanmedian(pat_tr[ok]))
                out = a_pat / p + b_pat
            elif kind == "positive_xcorr":
                p = np.clip(np.nanmedian(pat_tr[ok]) + xcorr_shift(Xr), 0.05, None)
                out = a2 / p + b2
            elif kind == "negative":
                out = np.full(len(Xr), dbp_mean)
            else:                                        # amplitude
                amp = np.array([np.ptp(x[:, PPG]) for x in Xr])
                out = amp_mod.predict(amp[:, None])
            return np.stack([out, out], 1)               # (N,2) SBP/DBP slots
        return fn

    results = {}
    for kind in ["positive", "positive_xcorr", "negative", "amplitude"]:
        fn = make_fn(kind)
        mae = float(np.abs(fn(Xte)[:, 1] - yte[:, 1]).mean())
        aud = mechlib.causal_ptt_audit(None, Xte, fs, "cpu", predict_fn=fn, n_max=1200)["dbp"]
        # specificity: does an amplitude perturbation move it?
        amp_batt = pa.run_battery(fn, Xte, fs, cues=["amp"], target=1, has_ecg=True, n_max=800)
        results[kind] = {"dbp_mae": mae, "pat_slope": aud["dBP_dPTT"],
                         "pat_range": aud["resp_range_mmHg"],
                         "frac_correct": aud["frac_correct_sign"],
                         "amp_slope": amp_batt["amp"]["slope"],
                         "amp_range": amp_batt["amp"]["resp_range"]}
        r = results[kind]
        print(f"[ctrl] {kind:10s} MAE {mae:5.2f} | PAT slope {r['pat_slope']:+8.1f} "
              f"range {r['pat_range']:5.2f} sign {r['frac_correct']:.0%} | "
              f"AMP slope {r['amp_slope']:+7.2f} range {r['amp_range']:.2f}", flush=True)

    (ROOT / "data" / "audit_controls.json").write_text(json.dumps(results, indent=2, default=float))
    print("[done] data/audit_controls.json", flush=True)


if __name__ == "__main__":
    main()
