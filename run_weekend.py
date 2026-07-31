"""run_weekend.py -- unattended multi-stage run. Each stage is wrapped so a failure logs and
moves on; partial results are written after every stage.

Motivation: an earlier headline ("XResNet101 uses PAT backwards, slope +8.6") did NOT replicate
on a larger audit sample (-15.0). So stage 1 is now audit STABILITY -- no slope claim is safe
without error bars. The remaining stages test what the models actually depend on.

  1. audit_stability : bootstrap the roll-audit per model -> slope mean/CI, response range,
                       across audit-set sizes and sweep widths. Gates every other claim.
  2. causal_shortcut : perturb cardiac period/HR directly and compare the response magnitude to
                       the PAT response. Turns "shortcut" from a decodability label into a
                       causal claim (or refutes it).
  3. reflection_audit: audit reflection/augmentation timing -- arguably a purer single-PPG
                       stiffness proxy than PEP-contaminated PAT.
  4. deep_seeds      : 5 architectures x 3 training seeds -> error bars on ID/OOD/mechanism.

    python run_weekend.py               # everything
    python run_weekend.py --only 1,2    # a subset
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import mechlib
import physics_audit as pa
import ood_benchmark as ob
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "weekend_results.json"
MIMIC = "C:/Users/mason/OneDrive - McMaster University/2026/BP"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
RESULTS = {}


def save():
    OUT.write_text(json.dumps(RESULTS, indent=2, default=float))


def stage(name):
    def deco(fn):
        def wrapped(*a, **k):
            t0 = time.time()
            print(f"\n{'='*70}\n[weekend] STAGE {name}\n{'='*70}", flush=True)
            try:
                RESULTS[name] = fn(*a, **k)
                print(f"[weekend] {name} OK in {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:
                RESULTS[name] = {"error": str(e), "traceback": traceback.format_exc()[-2000:]}
                print(f"[weekend] {name} FAILED: {e}", flush=True)
                traceback.print_exc()
            save()
        return wrapped
    return deco


def audit_data(n=1500, seed=0):
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(d["gte"]), n, replace=False))
    return mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]]), fs


def load_deep(mk, tag="_ecgppg_full"):
    ck = torch.load(ROOT / "models" / f"{mk}{tag}.pt", map_location=DEVICE, weights_only=False)
    m = ob.build_model(mk, n_ch=2, L=1250)
    m.load_state_dict(ck["state_dict"]); m.to(DEVICE).eval()
    return m, ck["mu"], ck["sd"]


# ----------------------------------------------------------------- stage 1
@stage("1_audit_stability")
def audit_stability(n_boot=12, boot_n=800):
    """Bootstrap the roll-audit: resample the audit set n_boot times per model and report
    slope mean/std/CI + response range, for a wide (+/-48ms) and narrow (+/-16ms) sweep.
    This is the error bar the earlier non-replicating '+8.6' lacked."""
    X, fs = audit_data(3000)
    out = {}
    for mk in ARCHS:
        m, mu, sd = load_deep(mk)
        fn = lambda Xr: ob.predict(m, Xr, DEVICE, mu, sd)
        row = {}
        for tag, deltas in [("wide48", (-6, -4, -2, 0, 2, 4, 6)), ("narrow16", (-2, -1, 0, 1, 2))]:
            slopes, ranges, fracs = [], [], []
            for b in range(n_boot):
                rng = np.random.default_rng(1000 + b)
                idx = rng.choice(len(X), boot_n, replace=False)
                a = mechlib.causal_ptt_audit(None, X[idx], fs, DEVICE, predict_fn=fn,
                                             deltas=deltas, n_max=boot_n, seed=b)["dbp"]
                slopes.append(a["dBP_dPTT"]); ranges.append(a["resp_range_mmHg"])
                fracs.append(a["frac_correct_sign"])
            slopes = np.array(slopes)
            row[tag] = {"slope_mean": float(slopes.mean()), "slope_std": float(slopes.std()),
                        "slope_lo": float(np.percentile(slopes, 2.5)),
                        "slope_hi": float(np.percentile(slopes, 97.5)),
                        "range_mean": float(np.mean(ranges)),
                        "frac_correct_mean": float(np.mean(fracs))}
            print(f"  {mk:14s} {tag:9s} slope {row[tag]['slope_mean']:+7.1f} "
                  f"+/-{row[tag]['slope_std']:.1f}  [{row[tag]['slope_lo']:+.1f},"
                  f"{row[tag]['slope_hi']:+.1f}]  range {row[tag]['range_mean']:.2f}  "
                  f"sign {row[tag]['frac_correct_mean']:.0%}", flush=True)
        out[mk] = row
        save()
    return out


# ----------------------------------------------------------------- stage 2
@stage("2_causal_shortcut")
def causal_shortcut(n=1200):
    """Is cardiac period a CAUSAL driver or merely decodable? Perturb HR/period directly
    (perturb_hr, validated) and compare response magnitude with the PAT roll response."""
    X, fs = audit_data(n)
    out = {}
    for mk in ARCHS:
        m, mu, sd = load_deep(mk)
        fn = lambda Xr: ob.predict(m, Xr, DEVICE, mu, sd)
        # PAT response (roll)
        pat = mechlib.causal_ptt_audit(None, X, fs, DEVICE, predict_fn=fn, n_max=n)["dbp"]
        # period/HR response, matched sweep
        batt = pa.run_battery(fn, X, fs, cues=["hr"], target=1, has_ecg=True, n_max=n)["hr"]
        out[mk] = {"pat_slope": pat["dBP_dPTT"], "pat_range": pat["resp_range_mmHg"],
                   "hr_slope": batt["slope"], "hr_range": batt["resp_range"],
                   "hr_over_pat_range": float(batt["resp_range"] / (pat["resp_range_mmHg"] + 1e-9))}
        print(f"  {mk:14s} PAT range {pat['resp_range_mmHg']:.2f} | HR range {batt['resp_range']:.2f}"
              f" | ratio {out[mk]['hr_over_pat_range']:.2f}", flush=True)
        save()
    return out


# ----------------------------------------------------------------- stage 3
@stage("3_reflection_audit")
def reflection_audit(n=1200):
    """Reflection/augmentation timing is arguably a purer single-PPG stiffness proxy than
    PEP-contaminated PAT. Audit it (and rise time) alongside PAT for comparison."""
    X, fs = audit_data(n)
    out = {}
    for mk in ARCHS:
        m, mu, sd = load_deep(mk)
        fn = lambda Xr: ob.predict(m, Xr, DEVICE, mu, sd)
        batt = pa.run_battery(fn, X, fs, cues=["aix", "rise", "decay"], target=1,
                              has_ecg=True, n_max=n)
        out[mk] = {c: {"slope": batt[c]["slope"], "expect": batt[c]["expect"],
                       "sign_ok": batt[c]["sign_ok"], "range": batt[c]["resp_range"]}
                   for c in batt}
        print(f"  {mk:14s} " + "  ".join(
            f"{c} {batt[c]['slope']:+.1f}({'ok' if batt[c]['sign_ok'] else 'x' if batt[c]['sign_ok'] is False else '-'})"
            for c in batt), flush=True)
        save()
    return out


# ----------------------------------------------------------------- stage 4
@stage("4_deep_seeds")
def deep_seeds(seeds=(0, 1, 2), epochs=40, train_n=80000):
    """Error bars on the accuracy/mechanism dissociation across training seeds."""
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    sp = {"train": dict(X=mechlib.normalize(d["Xtr"][:train_n][:, :, [ECG, PPG]]), y=d["ytr"][:train_n]),
          "val": dict(X=mechlib.normalize(d["Xva"][:8000][:, :, [ECG, PPG]]), y=d["yva"][:8000]),
          "test": dict(X=mechlib.normalize(d["Xte"][:8000][:, :, [ECG, PPG]]), y=d["yte"][:8000])}
    Xa = sp["test"]["X"][:1200]
    m0 = pa.load_mimic_bp(MIMIC, channels=("ecg", "ppg"), max_patients=200)
    Xm, k = pa.window_segments(m0["X"], 1250)
    ym = np.repeat(m0["y"], k, 0)
    ii = np.sort(np.random.default_rng(0).choice(len(Xm), 4000, False))
    Xm, ym = mechlib.normalize(Xm[ii]), ym[ii]

    out = {}
    for arch in ARCHS:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = ob.build_model(arch, n_ch=2, L=1250)
            model, (mu, sd) = ob.train(model, sp["train"], sp["val"], DEVICE, epochs=epochs)
            idm = float(np.abs(ob.predict(model, sp["test"]["X"], DEVICE, mu, sd) - sp["test"]["y"]).mean(0)[1])
            mim = float(np.abs(ob.predict(model, Xm, DEVICE, mu, sd) - ym).mean(0)[1])
            fn = lambda Xr: ob.predict(model, Xr, DEVICE, mu, sd)
            a = mechlib.causal_ptt_audit(None, Xa, fs, DEVICE, predict_fn=fn, n_max=1200)["dbp"]
            out[f"{arch}_seed{seed}"] = {"id_dbp": idm, "mimic_dbp": mim,
                                         "slope": a["dBP_dPTT"], "range": a["resp_range_mmHg"],
                                         "frac_correct": a["frac_correct_sign"]}
            print(f"  {arch:14s} seed{seed}: ID {idm:.2f} MIMIC {mim:.2f} "
                  f"slope {a['dBP_dPTT']:+.1f} range {a['resp_range_mmHg']:.2f}", flush=True)
            save()
    for arch in ARCHS:
        rows = [v for kk, v in out.items() if kk.startswith(arch + "_seed")]
        if rows:
            for stat, f in [("mean", np.mean), ("std", np.std)]:
                out[f"{arch}_{stat}"] = {q: float(f([r[q] for r in rows]))
                                         for q in ("id_dbp", "mimic_dbp", "slope", "range")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()} or {"1", "2", "3", "4"}
    t0 = time.time()
    print(f"[weekend] start; stages {sorted(want)}; device {DEVICE}", flush=True)
    if "1" in want:
        audit_stability()
    if "2" in want:
        causal_shortcut()
    if "3" in want:
        reflection_audit()
    if "4" in want:
        deep_seeds(epochs=args.epochs)
    print(f"\n[weekend] ALL DONE in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
