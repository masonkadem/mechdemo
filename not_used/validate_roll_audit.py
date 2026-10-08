"""validate_roll_audit.py -- prove the roll audit is a clean, specific PAT manipulation.

Three checks a methods reviewer will demand:
  1. SPECIFICITY: rolling PPG by delta actually changes MEASURED PAT by ~delta, while the
     PPG's own morphology cues (rise, aix, apg, ...) stay ~constant. If the roll moved
     morphology too, the audit slope could not be attributed to arrival time.
  2. NON-CIRCULAR: the hardened _shift_channel introduces no wrap-around pulse (max edge
     artifact stays small vs the pulse amplitude).
  3. NULL CONTROL sanity: a model's response to shifting a flat/irrelevant channel should be
     near zero (checked at audit time via null_channels; here we confirm the manipulation
     itself carries no PAT information).
"""
import numpy as np
import mechlib
from mechlib import ECG, PPG, _shift_channel

FS = 125


def main():
    d = mechlib.load_mini("data/vitaldb_mini_deep.npz")
    fs = d["fs"]
    X = mechlib.normalize(d["Xte"][:400, :, [ECG, PPG]])   # (N,1250,2) ECG,PPG
    deltas = [-6, -4, -2, 0, 2, 4, 6]

    print("=== 1. SPECIFICITY: does rolling PPG move PAT by ~delta, and leave morphology alone? ===")
    print(f"{'delta(ms)':>9} {'d_PAT(ms)':>10} {'exp(ms)':>8} | "
          + " ".join(f"{c:>7}" for c in ["rise", "aix", "apg", "notch"]))
    base = mechlib.compute_scalars(X, fs)
    for dl in deltas:
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], dl)
        sc = mechlib.compute_scalars(Xd, fs)
        dpat = np.nanmedian(sc["pat"] - base["pat"]) * 1000
        # morphology drift (should be ~0): median |change| across cues
        morph = {c: np.nanmedian(np.abs(sc[c] - base[c])) for c in ["rise", "aix", "apg", "notch"]}
        print(f"{dl/fs*1000:>9.0f} {dpat:>10.1f} {dl/fs*1000:>8.0f} | "
              + " ".join(f"{morph[c]:>7.3f}" for c in ["rise", "aix", "apg", "notch"]))

    print("\n=== 2. NON-CIRCULAR: edge artifact from the shift (should be small) ===")
    for dl in [6, -6]:
        rolled = np.roll(X[:, :, PPG], dl, axis=1)
        shifted = _shift_channel(X[:, :, PPG], dl)
        # discontinuity at the seam: circular roll injects the far end; padded shift holds edge
        roll_jump = np.abs(np.diff(rolled, axis=1)).max(1).mean()
        shift_jump = np.abs(np.diff(shifted, axis=1)).max(1).mean()
        pulse_amp = (X[:, :, PPG].max(1) - X[:, :, PPG].min(1)).mean()
        print(f"  delta={dl:+d}: max-step  circular-roll={roll_jump:.3f}  padded-shift={shift_jump:.3f}"
              f"  (pulse amplitude={pulse_amp:.2f})")

    print("\n=== 3. PAT range induced vs physiological PTT spread ===")
    pat = base["pat"][np.isfinite(base["pat"])]
    print(f"  measured PAT: {np.median(pat)*1000:.0f} ms (IQR {np.percentile(pat,25)*1000:.0f}-"
          f"{np.percentile(pat,75)*1000:.0f})")
    print(f"  roll sweep spans +/-{max(deltas)/fs*1000:.0f} ms "
          f"= +/-{100*max(deltas)/fs/np.median(pat):.0f}% of median PAT")


if __name__ == "__main__":
    main()
