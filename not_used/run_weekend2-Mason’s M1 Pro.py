"""run_weekend2.py -- second unattended run, ordered by what GATES what.

Design notes
------------
Stage order is a dependency order, not a wish list. Stage 1 can invalidate the whole current
hypothesis, so it runs first and everything downstream is interpreted in light of it. Stage 2
can invalidate every cross-model mechanism comparison we have (inception1d gave slope -8.0 in
one run and +18.9 in another -- a sign flip), so it runs second.

Every stage:
  * is wrapped so a failure logs a traceback and the run continues;
  * writes data/weekend2_results.json after each stage AND after each model within a stage;
  * SKIPS itself if its results are already present, so the script is resumable after a crash
    (delete the key from the json, or pass --redo, to force a re-run).

Uses the VALIDATED audit throughout (audit_subject.subject_audit / negative-arm sweep with
non-finite PAT dropped), never the legacy mechlib.causal_ptt_audit, which imputes NaN PAT and
sweeps the saturating positive arm -- both of which bias slopes toward zero.

  python run_weekend2.py                 # all stages, resuming
  python run_weekend2.py --only 2,3      # a subset
  python run_weekend2.py --redo          # ignore cached stages
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import mechlib
import ood_benchmark as ob
import within_subject as ws
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "weekend2_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
RESULTS = json.loads(OUT.read_text()) if OUT.exists() else {}


def save():
    OUT.write_text(json.dumps(RESULTS, indent=2, default=float))


def stage(name, expect=None):
    """expect: how many entries a COMPLETE stage holds. Without it, a stage that checkpointed
    after one model looks 'done' and the remaining models are silently skipped -- which is
    exactly what happened to stage 1 on the first run."""
    def deco(fn):
        def wrapped(*a, **k):
            redo = k.pop("redo", False)
            done = RESULTS.get(name)
            if done and "error" not in done and not redo:
                have = len([x for x in done if not x.startswith("var_")])
                if expect is None or have >= expect:
                    print(f"[skip] {name} already done ({have} entries)", flush=True)
                    return
                print(f"[resume] {name}: {have}/{expect} entries present, continuing", flush=True)
            t0 = time.time()
            print(f"\n{'='*72}\n[w2] STAGE {name}\n{'='*72}", flush=True)
            try:
                RESULTS[name] = fn(*a, **k)
                print(f"[w2] {name} OK in {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:
                RESULTS[name] = {"error": str(e), "traceback": traceback.format_exc()[-2500:]}
                print(f"[w2] {name} FAILED: {e}", flush=True)
                traceback.print_exc()
            save()
        return wrapped
    return deco


def test_data(per_subject=150):
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz")
    fs, g, y = d["fs"], d["gte"], d["yte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= per_subject]
    sel = np.concatenate([np.where(g == s)[0][:per_subject] for s in subs])
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    return X, y[sel], g[sel], fs


def load_deep(mk, tag="_ecgppg_full"):
    ck = torch.load(ROOT / "models" / f"{mk}{tag}.pt", map_location=DEVICE, weights_only=False)
    m = ob.build_model(mk, n_ch=2, L=1250)
    m.load_state_dict(ck["state_dict"]); m.to(DEVICE).eval()
    return (lambda Xr: ob.predict(m, Xr, DEVICE, ck["mu"], ck["sd"]))


# ------------------------------------------------------------------ stage 1
@stage("1_corrected_audit", expect=len(ARCHS))
def corrected_audit():
    """Re-run every architecture through the VALIDATED audit. The published slopes come from
    the NaN-imputing version, whose bias is toward zero -- so PAT dependence should get
    STRONGER here. Reports per-subject slopes, so faithfulness has an error bar."""
    X, Y, G, fs = test_data()
    out = RESULTS.get("1_corrected_audit", {}) or {}
    for mk in ARCHS:
        if mk in out:
            print(f"  {mk:14s} cached, skipping", flush=True); continue
        try:
            fn = load_deep(mk)
        except Exception as e:
            print(f"  {mk}: cannot load ({str(e)[:60]})", flush=True); continue
        slopes, frac_keep = ws.subject_audit_slopes(fn, X, G, fs)
        v = np.array(list(slopes.values()))
        out[mk] = {"slope_median": float(np.median(v)),
                   "slope_lo": float(np.percentile(v, 2.5)),
                   "slope_hi": float(np.percentile(v, 97.5)),
                   "frac_subj_faithful": float(np.mean(v > 0)),
                   "n_subjects": int(len(v)), "frac_segments_kept": frac_keep,
                   "per_subject": {str(k): float(x) for k, x in slopes.items()}}
        print(f"  {mk:14s} slope {out[mk]['slope_median']:+.4f} "
              f"[{out[mk]['slope_lo']:+.4f},{out[mk]['slope_hi']:+.4f}] "
              f"{out[mk]['frac_subj_faithful']:.0%} subj faithful (n={len(v)})", flush=True)
        RESULTS["1_corrected_audit"] = out; save()
    return out


# ------------------------------------------------------------------ stage 2
@stage("2_seed_variance")
def seed_variance(seeds=(0, 1, 2, 3, 4), epochs=30, train_n=60000):
    """THE GATING EXPERIMENT. inception1d gave slope -8.0 in one run and +18.9 in another.
    If the seed spread exceeds the between-architecture spread, no cross-model mechanism
    claim in this project survives. Five seeds x five architectures, each audited with the
    validated method."""
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    tr = dict(X=mechlib.normalize(d["Xtr"][:train_n][:, :, [ECG, PPG]]), y=d["ytr"][:train_n])
    va = dict(X=mechlib.normalize(d["Xva"][:8000][:, :, [ECG, PPG]]), y=d["yva"][:8000])
    Xa, Ya, Ga, _ = test_data(per_subject=60)
    out = RESULTS.get("2_seed_variance", {})
    for arch in ARCHS:
        for seed in seeds:
            key = f"{arch}_seed{seed}"
            if key in out:
                continue
            torch.manual_seed(seed); np.random.seed(seed)
            model = ob.build_model(arch, n_ch=2, L=1250)
            model, (mu, sd) = ob.train(model, tr, va, DEVICE, epochs=epochs)
            fn = lambda Xr: ob.predict(model, Xr, DEVICE, mu, sd)
            p = fn(Xa)[:, 1]
            idm = float(np.abs(p - Ya[:, 1]).mean())
            track = ws.per_subject_r(p, Ya[:, 1], Ga)
            slopes, _ = ws.subject_audit_slopes(fn, Xa, Ga, fs)
            v = np.array(list(slopes.values()))
            out[key] = {"id_dbp": idm,
                        "within_r": float(np.nanmedian(list(track.values()))),
                        "slope_median": float(np.median(v)),
                        "frac_subj_faithful": float(np.mean(v > 0))}
            print(f"  {arch:14s} seed{seed}: ID {idm:5.2f} track {out[key]['within_r']:.3f} "
                  f"slope {out[key]['slope_median']:+.4f}", flush=True)
            RESULTS["2_seed_variance"] = out; save()
    # variance decomposition: is architecture or seed the bigger source?
    for q in ("slope_median", "within_r", "id_dbp"):
        per = {a: [out[f"{a}_seed{s}"][q] for s in seeds if f"{a}_seed{s}" in out] for a in ARCHS}
        per = {a: v for a, v in per.items() if len(v) >= 2}
        if len(per) >= 2:
            within = float(np.mean([np.std(v) for v in per.values()]))
            between = float(np.std([np.mean(v) for v in per.values()]))
            out[f"var_{q}"] = {"seed_sd": within, "arch_sd": between,
                               "arch_over_seed": between / (within + 1e-9)}
            print(f"  [var] {q:14s} seed sd {within:.4f} | arch sd {between:.4f} | "
                  f"ratio {between/(within+1e-9):.2f}", flush=True)
    return out


# ------------------------------------------------------------------ stage 3
@stage("3_calbased", expect=len(ARCHS))
def calbased(per_subject=150):
    """Within-subject tracking is essentially what calibrated deployment measures, so report
    the calibrated numbers properly: subtract each subject's own mean (an offset calibration,
    the standard CalBased protocol) and re-score. Compared against the mean-predictor floor,
    which is the baseline the cross-dataset OOD work failed to beat."""
    X, Y, G, fs = test_data(per_subject)
    out = RESULTS.get("3_calbased", {}) or {}
    for mk in ARCHS:
        if mk in out:
            print(f"  {mk:14s} cached, skipping", flush=True); continue
        try:
            fn = load_deep(mk)
        except Exception as e:
            print(f"  {mk}: cannot load ({str(e)[:60]})", flush=True); continue
        p = fn(X)[:, 1]
        raw, cal, floor = [], [], []
        for s in np.unique(G):
            m = G == s
            if m.sum() < 30:
                continue
            yy, pp = Y[m, 1], p[m]
            raw.append(np.abs(pp - yy).mean())
            cal.append(np.abs((pp - pp.mean()) - (yy - yy.mean())).mean())
            floor.append(np.abs(yy - yy.mean()).mean())      # predict this subject's own mean
        out[mk] = {"mae_raw": float(np.median(raw)),
                   "mae_calibrated": float(np.median(cal)),
                   "mae_subject_mean_floor": float(np.median(floor)),
                   "beats_floor": bool(np.median(cal) < np.median(floor))}
        print(f"  {mk:14s} raw {out[mk]['mae_raw']:5.2f} | calibrated "
              f"{out[mk]['mae_calibrated']:5.2f} | floor {out[mk]['mae_subject_mean_floor']:5.2f}"
              f" | beats floor: {out[mk]['beats_floor']}", flush=True)
        RESULTS["3_calbased"] = out; save()
    return out


# ------------------------------------------------------------------ stage 4
@stage("4_tracking_vs_faithfulness", expect=len(ARCHS))
def tracking_vs_faithfulness():
    """The headline test, at n=subjects rather than n=5 architectures. Pairs each subject's
    audit slope with that subject's tracking correlation and bootstraps the correlation.
    Also pools every (model, subject) pair, which is the largest-n version of the claim."""
    X, Y, G, fs = test_data()
    aud = RESULTS.get("1_corrected_audit") or {}
    out, pooled_x, pooled_y = {}, [], []
    for mk in ARCHS:
        if mk not in aud:
            continue
        try:
            fn = load_deep(mk)
        except Exception:
            continue
        track = ws.per_subject_r(fn(X)[:, 1], Y[:, 1], G)
        sl = {int(k): v for k, v in aud[mk]["per_subject"].items()}
        common = sorted(set(track) & set(sl))
        if len(common) < 10:
            continue
        a = [sl[s] for s in common]; b = [track[s] for s in common]
        r, lo, hi = ws.boot_ci(a, b)
        out[mk] = {"r": r, "ci": [lo, hi], "n": len(common)}
        pooled_x += a; pooled_y += b
        print(f"  {mk:14s} r(slope, tracking) {r:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(common)}",
              flush=True)
    if len(pooled_x) >= 30:
        r, lo, hi = ws.boot_ci(pooled_x, pooled_y)
        out["pooled"] = {"r": r, "ci": [lo, hi], "n": len(pooled_x)}
        print(f"  {'POOLED':14s} r {r:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(pooled_x)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()} or {"1", "2", "3", "4"}
    t0 = time.time()
    print(f"[w2] start; stages {sorted(want)}; device {DEVICE}", flush=True)
    if "1" in want:
        corrected_audit(redo=args.redo)
    if "2" in want:
        seed_variance(epochs=args.epochs, redo=args.redo)
    if "3" in want:
        calbased(redo=args.redo)
    if "4" in want:
        tracking_vs_faithfulness(redo=args.redo)
    print(f"\n[w2] ALL DONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
