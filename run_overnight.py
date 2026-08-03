"""run_overnight.py -- firm up the three numbers the project's argument now rests on.

The motivation is a measurement claim: the governing law is real but small, and PPG cannot
recover it. Three numbers carry that, and each is currently measured on a subset:

  1. the ceiling   perfect arrival time buys 0.35 mmHg over predicting a subject's own mean
                   -- from 60 subjects, no confidence interval
  2. estimator     the best ECG-PPG estimator agrees with arterial arrival time at r = 0.21
                   -- from 50 subjects
  3. CalBased      12,000 of 51,720 segments

Stage 1 matters most. That 0.35 mmHg is the number the instrumentation argument turns on, and a
point estimate from 60 subjects cannot support it. Here it is recomputed on every eligible
subject with a subject-level bootstrap, so the claim can be stated with an interval or withdrawn.

Every stage writes after each subject or block and skips work already present, so the run can be
interrupted and restarted without losing anything.
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

import mechlib
import pat_groundtruth as G
import pat_estimators as PE
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "overnight.json"
FS = 125
RES = json.loads(OUT.read_text()) if OUT.exists() else {}


def save():
    OUT.write_text(json.dumps(RES, indent=2, default=float))


def stage(name):
    def deco(fn):
        def wrapped(*a, **k):
            if name in RES and "error" not in RES[name]:
                print(f"[skip] {name} already done", flush=True)
                return
            t0 = time.time()
            print(f"\n{'='*70}\n[night] {name}\n{'='*70}", flush=True)
            try:
                RES[name] = fn(*a, **k)
                print(f"[night] {name} OK in {(time.time()-t0)/60:.1f} min", flush=True)
            except Exception as e:
                RES[name] = {"error": str(e), "tb": traceback.format_exc()[-1500:]}
                print(f"[night] {name} FAILED: {e}", flush=True)
                traceback.print_exc()
            save()
        return wrapped
    return deco


@stage("1_ceiling_full")
def ceiling_full(per_subject=80, min_seg=40):
    """How much does perfectly measured arrival time buy, across every eligible subject?

    Per subject: fit DBP on ground-truth arrival time (Q-onset to ABP foot) and compare against
    predicting that subject's own mean. Reported with a subject-level bootstrap, because a point
    estimate over 60 subjects cannot support the claim this number is being asked to carry.
    """
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g, y = d["gte"], d["yte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= min_seg]
    print(f"[ceiling] {len(subs)} subjects with >= {min_seg} segments", flush=True)

    rows = []
    for i, s in enumerate(subs):
        idx = np.where(g == s)[0][:per_subject]
        Xr = d["Xte"][idx]
        t = np.array([G.abp_pat(Xr[j, :, ECG], Xr[j, :, 2], FS) for j in range(len(Xr))]) * 1000
        ok = np.isfinite(t)
        if ok.sum() < 25:
            continue
        b = y[idx][ok, 1]
        tt = t[ok]
        if np.std(tt) < 1e-6 or np.std(b) < 1e-6:
            continue
        floor = float(np.mean(np.abs(b - b.mean())))
        pred = np.polyval(np.polyfit(tt, b, 1), tt)
        with_pat = float(np.mean(np.abs(pred - b)))
        from scipy import stats
        rows.append({"subject": int(s), "n": int(ok.sum()), "floor": floor,
                     "with_pat": with_pat, "gain": floor - with_pat,
                     "r": float(stats.spearmanr(tt, b).statistic)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(subs)} subjects", flush=True)
            RES["1_ceiling_full"] = {"rows": rows}; save()

    gains = np.array([r["gain"] for r in rows])
    rs = np.array([r["r"] for r in rows])
    rng = np.random.default_rng(0)
    boot = [np.median(gains[rng.integers(0, len(gains), len(gains))]) for _ in range(4000)]
    out = {"n_subjects": len(rows),
           "floor_median": float(np.median([r["floor"] for r in rows])),
           "with_pat_median": float(np.median([r["with_pat"] for r in rows])),
           "gain_median": float(np.median(gains)),
           "gain_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "r_median": float(np.median(rs)),
           "frac_law_consistent": float(np.mean(rs < 0)),
           "rows": rows}
    print(f"\n  floor            {out['floor_median']:.2f} mmHg")
    print(f"  + true arrival   {out['with_pat_median']:.2f} mmHg")
    print(f"  gain             {out['gain_median']:.2f} "
          f"[{out['gain_ci'][0]:.2f}, {out['gain_ci'][1]:.2f}]  (n={out['n_subjects']})")
    print(f"  r(PAT, DBP)      {out['r_median']:+.3f}, law-consistent "
          f"{100*out['frac_law_consistent']:.0f}%")
    return out


@stage("2_estimators_full")
def estimators_full(per_subject=60, n_subjects=140):
    """Rank every ECG-PPG estimator against arterial arrival time, on more subjects."""
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= per_subject][:n_subjects]
    sel = np.concatenate([np.where(g == s)[0][:per_subject] for s in subs])
    Xraw = d["Xte"][sel]
    Xn = mechlib.normalize(Xraw[:, :, [ECG, PPG]])
    gg = g[sel]
    print(f"[est] {len(Xn)} segments, {len(subs)} subjects", flush=True)

    truth = np.array([G.abp_pat(Xraw[i, :, ECG], Xraw[i, :, 2], FS)
                      for i in range(len(Xraw))]) * 1000.0
    ok_t = np.isfinite(truth)
    print(f"[est] ground truth on {100*ok_t.mean():.0f}% of segments", flush=True)

    out = {}
    for name, fn in PE.ESTIMATORS.items():
        est = PE.batch(fn, Xn, FS) * 1000.0
        rw, nsub = G.within_r(est, truth, gg)
        m = ok_t & np.isfinite(est)
        from scipy import stats
        rp = float(stats.spearmanr(est[m], truth[m]).statistic) if m.sum() > 200 else np.nan
        out[name] = {"valid": float(np.isfinite(est).mean()), "r_within": rw,
                     "r_pooled": rp, "n_subj": nsub}
        print(f"  {name:14s} valid {100*out[name]['valid']:3.0f}%  "
              f"within r {rw:+.3f}  ({nsub} subj)", flush=True)
        RES["2_estimators_full"] = out; save()
    return out


@stage("3_calbased_full")
def calbased_full():
    """CalBased on all 51,720 segments rather than the 12k subsample."""
    import eval_protocols as ep
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-u", str(ROOT / "eval_protocols.py"),
                        "--max-seg", "60000"], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=14400)
    print(r.stdout[-2500:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return json.loads((DATA / "eval_protocols.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    want = {x.strip() for x in args.only.split(",") if x.strip()} or {"1", "2", "3"}
    t0 = time.time()
    if "1" in want:
        ceiling_full()
    if "2" in want:
        estimators_full()
    if "3" in want:
        calbased_full()
    print(f"\n[night] done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
