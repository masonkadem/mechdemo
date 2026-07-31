"""rppg_compare.py -- compare two-site rPPG runs across conditions.

A single absolute neck->hand lag is not evidence: a fixed camera/ROI processing delay produces a
stable offset that looks exactly like a transit time. What IS evidence is a reproducible SHIFT
between conditions, because any fixed instrumental delay cancels in the difference.

The built-in perturbation is hydrostatic. Raising the hand well above the heart lowers local
arterial pressure, which softens the vessel and SLOWS the pulse, so PTT should LENGTHEN relative
to the hand held at heart level. That is a signed prediction from the same governing law the
whole project audits, and it costs nothing to test.

    python rppg_compare.py                       # compares rest vs hand_up
    python rppg_compare.py --a rest --b hand_up
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(tag):
    p = ROOT / "data" / f"rppg_two_site_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="rest", help="baseline condition tag")
    ap.add_argument("--b", default="hand_up", help="perturbed condition tag")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    missing = [t for t, d in ((args.a, A), (args.b, B)) if d is None]
    if missing:
        print(f"[err] no data for: {', '.join(missing)}")
        print(f"      run:  run_webcam.bat 60 {missing[0]}")
        return

    print(f"{'condition':12s} {'fps':>6s} {'HR':>6s} {'SNR':>7s} {'lag ms':>9s} {'sd':>7s} {'n':>4s}")
    for tag, d in ((args.a, A), (args.b, B)):
        w = d.get("window_lags") or []
        print(f"{tag:12s} {d['fps']:6.1f} {d['hr_bpm']:6.0f} {d['snr']:7.1f} "
              f"{d['lag_ms']:+9.1f} {np.std(w) if w else float('nan'):7.1f} {len(w):4d}")

    wa, wb = A.get("window_lags") or [], B.get("window_lags") or []
    if len(wa) < 3 or len(wb) < 3:
        print("\n[warn] too few windows for a comparison; record longer runs (>=60 s each)")
        return

    diff = float(np.median(wb) - np.median(wa))
    # Welch t-test across independent windows; small n, so report it as indicative only
    from scipy import stats
    t, p = stats.ttest_ind(wb, wa, equal_var=False)
    pooled = float(np.sqrt((np.var(wa) + np.var(wb)) / 2))
    print(f"\n[shift] {args.b} - {args.a} = {diff:+.1f} ms   (t={t:+.2f}, p={p:.3f}, "
          f"Cohen's d = {diff / (pooled + 1e-9):+.2f})")

    if min(A["snr"], B["snr"]) < 5:
        print("[warn] weak pulse signal in at least one run -- improve lighting and re-record "
              "before interpreting this shift")
    print("\nPrediction: raising the hand lowers local arterial pressure, which SLOWS the pulse,")
    print("so PTT should LENGTHEN (positive shift).")
    if p < 0.05:
        verdict = "law-consistent" if diff > 0 else "OPPOSITE to prediction"
        print(f"Result: significant, {diff:+.1f} ms -- {verdict}.")
        if diff < 0:
            print("  An opposite-signed shift most likely means the ROIs swapped roles or the")
            print("  hand ROI drifted between runs. Check the preview before believing it.")
    else:
        print(f"Result: not significant (p={p:.3f}). Either the effect is below this rig's")
        print("  resolution, or the runs differed in ROI placement more than in physiology.")


if __name__ == "__main__":
    main()
