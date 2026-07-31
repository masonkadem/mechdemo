"""rppg_shutter.py -- measure the fake transit time a rolling shutter invents between two rows.

The problem this exists to catch
--------------------------------
A CMOS sensor exposes row by row, top to bottom. Two ROIs at different IMAGE ROWS are therefore
sampled at different real times inside one nominal frame, and full-frame readout spans a large
fraction of the frame period -- tens of ms at 30 fps, which is the same size as the neck->hand
pulse transit time the two-site rig is trying to measure.

As a FIXED offset that cancels in a rest-vs-condition difference, which is why the differential
design is sound. It stops being fixed the moment an ROI moves between conditions -- and the
hand_up / hand_down protocol moves the hand to a different image row BY DESIGN. A hand that
travels 180 of 480 rows buys roughly (180/480) x readout ms of pure artifact, comparable to the
hydrostatic effect under test, and it flips sign with hand height exactly as the predicted
physiology does. The "hand_down predicts the opposite sign" control does not catch this, because
rolling shutter passes that control too.

How this measures it
--------------------
Not by modelling sensor readout. The screen is flashed at a pulse-like rate (~1.2 Hz, inside the
0.7-3 Hz rPPG band) so scene illumination changes GLOBALLY -- genuinely simultaneous at every row
of the real world. Horizontal bands at different heights are then cross-correlated against the
top band with the same `lag_subframe` the real measurement uses. Any lag between bands is pure
instrument. Fitting lag against row gives ms-per-row directly, with no readout model to be wrong
about.

Why the bands are read on the GREEN channel and not through chrom()
-------------------------------------------------------------------
CHROM standardises two chrominance projections and subtracts one from the other. For a single
pure tone both projections are proportional to that tone, so the subtraction can cancel it
outright -- measured here, a white (achromatic) flash comes out at exactly zero amplitude, and a
red one does too; green survives only by a sign accident. A single-tone flash is therefore a bad
input to CHROM regardless of colour.

That costs nothing, because rolling shutter is a property of the SENSOR: it delays when a row is
sampled, before any colour maths. And neither downstream step can move a signal in time --
`chrom()` is pointwise in time (a per-frame combination of channels, scaled by global constants)
and `bandpass` is zero-phase `filtfilt`. So a row-timing offset measured on the green channel
transfers unchanged to the CHROM pipeline. Calibrate on green, apply to chrom.

    python rppg_shutter.py --seconds 40          # calibrate (needs a dim room)
    python rppg_shutter.py --selftest            # verify the math, no camera

Then in analysis:  corrected = rppg_shutter.correct(lag_ms, row_neck, row_hand)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import rppg_cam
import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CAL = DATA / "rppg_shutter_cal.json"
FLASH_HZ = 1.2          # inside BAND=(0.7,3.0), i.e. treated exactly like a 72 bpm pulse
N_BANDS = 6


def band_rows(h, n=N_BANDS):
    """Row centres of `n` horizontal bands spanning the frame, and each band's half-height."""
    edges = np.linspace(0, h, n + 1).astype(int)
    return [( (edges[i] + edges[i + 1]) // 2, edges[i], edges[i + 1]) for i in range(n)]


def capture_flash(seconds=40.0, cam=0, show=True):
    """Flash the screen while filming a scene it lights, returning per-band mean GREEN.

    The camera must see something the SCREEN illuminates (your face works; a wall works better).
    A dim room matters: the estimator needs the screen to be the dominant varying light source.
    The flash is a smooth sinusoid rather than a hard square so that everything above the
    band-pass corner is absent by construction rather than filtered away afterwards.
    """
    import cv2
    cap = rppg_cam.open_camera(cam, 640, 480, 60)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("camera returned no frame")
    h, w = frame.shape[:2]
    bands = band_rows(h)

    win = "FLASH - keep this fullscreen and let it light the scene (q to stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    B, T = [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        el = time.time() - t0
        lvl = int(127 * (1 + np.sin(2 * np.pi * FLASH_HZ * el)))
        tile = np.zeros((240, 320, 3), np.uint8)
        tile[:, :, 1] = lvl                       # green: brightest channel for skin/wall rPPG
        cv2.imshow(win, tile)
        ok, frame = cap.read()
        if not ok:
            continue
        B.append([float(frame[r0:r1, :, 1].mean()) for _, r0, r1 in bands])   # BGR -> index 1 = G
        T.append(time.time() - t0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.array(B), np.array(T), [c for c, _, _ in bands], h


def fit(B, T, centres, h):
    """Lag of each band relative to the top band, then ms-per-row by least squares.

    B: (n_frames, n_bands) mean green level per band.  Returns the calibration dict.
    """
    fs = (len(T) - 1) / (T[-1] - T[0])
    tu = np.linspace(T[0], T[-1], len(T))
    sig = [R.bandpass(np.interp(tu, T, B[:, i]), fs) for i in range(B.shape[1])]

    # Search a FEW frames, not a quarter-second. A wide window is how a weak flash produces
    # nonsense: a first attempt searched +/-250 ms and locked onto spurious peaks at -186 and
    # -234 ms, implying a negative readout five times the frame period. It cannot be tightened to
    # the one-frame physical bound either -- at 30 fps that is a single sample, which leaves the
    # parabolic interpolation nothing to fit and costs more accuracy than it buys. Three samples
    # is the compromise: far too narrow for the spurious locks, wide enough to interpolate.
    # The physical bound is enforced afterwards by validate(), where it belongs.
    max_lag_s = 3.0 / fs
    ref = sig[0]
    lags, peaks = [], []
    for s in sig:
        lg, pk = R.lag_subframe(ref, s, fs, max_lag_s=max_lag_s)
        lags.append(lg); peaks.append(pk)
    lags = np.array(lags); rows = np.array(centres, float)

    slope, intercept = np.polyfit(rows - rows[0], lags, 1)      # ms per row
    pred = slope * (rows - rows[0]) + intercept
    ss = float(1 - np.sum((lags - pred) ** 2) / (np.sum((lags - lags.mean()) ** 2) + 1e-12))
    quantum = 1000.0 / fs
    readout = float(slope * h)
    cal = {"fps": float(fs), "n_frames": int(len(T)), "frame_height": int(h),
           "band_rows": [int(c) for c in centres],
           "band_lags_ms": [float(x) for x in lags],
           "xcorr_peaks": [float(x) for x in peaks],
           "ms_per_row": float(slope), "r2": ss,
           "implied_readout_ms": readout, "frame_quantum_ms": quantum,
           "max_lag_searched_ms": float(max_lag_s * 1000)}
    cal["valid"], cal["why"] = validate(cal)
    return cal


def validate(cal):
    """Physical sanity. Rows are read top-to-bottom over a span that fits inside one frame, so a
    valid calibration has 0 <= implied readout <= frame period and a linear lag-vs-row fit.
    Anything else is not measuring readout, and applying it would corrupt every later recording
    more than leaving it uncorrected would."""
    ro, q, r2 = cal["implied_readout_ms"], cal["frame_quantum_ms"], cal["r2"]
    if r2 < 0.7:
        return False, f"lag is not linear in row (r2 {r2:.2f} < 0.7) -- flash too weak"
    if ro < -0.5:
        return False, f"negative readout ({ro:.1f} ms): rows cannot be read bottom-to-top"
    if ro > 1.05 * q:
        return False, (f"readout {ro:.1f} ms exceeds the {q:.1f} ms frame period, which is "
                       "physically impossible")
    if min(cal["xcorr_peaks"]) < 0.5:
        return False, f"weakest band correlates at {min(cal['xcorr_peaks']):.2f} -- flash too weak"
    return True, "ok"


def load():
    """Return a calibration only if it passed the physical checks. An invalid one is ignored
    rather than silently applied."""
    if not CAL.exists():
        return None
    cal = json.loads(CAL.read_text())
    return cal if cal.get("valid", False) else None


def worst_case_ms_per_row(fps, height):
    """Upper bound on the per-row delay, with NO calibration needed.

    Readout of a frame cannot take longer than the frame period (the next frame has to start),
    so ms_per_row <= (1000/fps)/height. That bound is often all you need: if the worst possible
    artifact is already smaller than the effect you are chasing, the calibration is pointless.
    """
    return (1000.0 / fps) / height


def check(tags=None, height=480):
    """Do you actually need the calibration? Compares ROI row geometry across conditions.

    Rolling shutter contributes a delay that depends only on the ROI rows. If drow = row_hand -
    row_neck is the SAME in every condition, the artifact is identical everywhere and cancels
    exactly in a rest-vs-condition difference -- no calibration, no correction, nothing to do.
    It only matters to the extent drow DRIFTS between conditions.
    """
    rows = []
    for p in sorted(DATA.glob("rppg_two_site_*.json")):
        j = json.loads(p.read_text())
        if "row_neck" not in j or (tags and j.get("tag") not in tags):
            continue
        rows.append((j.get("tag", p.stem), j["row_neck"], j["row_hand"], j.get("fps", 30.0)))
    if len(rows) < 2:
        return {"n": len(rows), "verdict": "need at least 2 recordings with logged ROI rows"}

    drows = np.array([h - n for _, n, h, _ in rows])
    spread = float(drows.max() - drows.min())
    fps = float(np.median([f for *_, f in rows]))
    cal = load()
    per_row = cal["ms_per_row"] if cal else worst_case_ms_per_row(fps, height)
    worst = abs(per_row) * spread
    return {"n": len(rows), "conditions": [t for t, *_ in rows],
            "drows": [float(d) for d in drows], "drow_spread_rows": spread,
            "ms_per_row": float(per_row), "calibrated": bool(cal),
            "max_artifact_ms": float(worst)}


def correct(lag_ms, row_a, row_b, cal=None):
    """Remove the rolling-shutter component from a measured a->b lag.

    `row_a` / `row_b` are the ROI vertical centres in pixels, in the same frame geometry the
    calibration was taken in. Returns (corrected_ms, artifact_ms). Without a calibration on disk
    it returns the lag untouched and an artifact of nan -- an uncalibrated number is not silently
    presented as a corrected one.
    """
    cal = cal or load()
    if not cal:
        return float(lag_ms), float("nan")
    artifact = cal["ms_per_row"] * (row_b - row_a)
    return float(lag_ms - artifact), float(artifact)


def selftest():
    """Inject a known per-row delay into a synthetic flash and check `fit` recovers it.

    Also checks `correct` removes exactly the artifact it should, which is the property the
    hand_up / hand_down protocol actually depends on.
    """
    fs, secs, h = 30.0, 60.0, 480
    true_ms_per_row = 0.05                      # 24 ms across a 480-row frame
    n = int(fs * secs)
    T = np.arange(n) / fs
    centres = [c for c, _, _ in band_rows(h)]
    rng = np.random.default_rng(0)
    B = np.zeros((n, len(centres)))
    for i, c in enumerate(centres):
        shift = (c - centres[0]) * true_ms_per_row / 1000.0     # rolling-shutter row delay
        B[:, i] = 128 + 20 * np.sin(2 * np.pi * FLASH_HZ * (T - shift)) + rng.normal(0, 0.5, n)

    cal = fit(B, T, centres, h)
    err = abs(cal["ms_per_row"] - true_ms_per_row)
    print(f"  true {true_ms_per_row*1000:.1f} us/row -> recovered "
          f"{cal['ms_per_row']*1000:.1f} us/row   (r2 {cal['r2']:.4f})")
    print(f"  implied readout {cal['implied_readout_ms']:.1f} ms across {h} rows "
          f"(true {true_ms_per_row*h:.1f})")

    # a hand 180 rows below the neck, with NO real transit: correction should return ~0
    faked = true_ms_per_row * 180
    got, art = correct(faked, 200, 380, cal)
    print(f"  pure-artifact lag {faked:+.2f} ms at drow=180 -> corrected {got:+.3f} ms "
          f"(removed {art:+.2f})")

    ok = err < 0.2 * true_ms_per_row and abs(got) < 0.5
    print("  PASS" if ok else "  FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="do you even need to calibrate? bounds the artifact from ROI geometry")
    a = ap.parse_args()

    if a.selftest:
        print("[selftest] synthetic known-delay recovery")
        raise SystemExit(0 if selftest() else 1)

    if a.check:
        c = check()
        if "verdict" in c:
            print(f"[check] {c['verdict']} (found {c['n']})")
            raise SystemExit(0)
        print(f"[check] {c['n']} conditions: {', '.join(c['conditions'])}")
        print(f"[check] drow (hand row - neck row): {c['drows']}")
        print(f"[check] drow spread {c['drow_spread_rows']:.0f} rows, "
              f"{'calibrated' if c['calibrated'] else 'WORST-CASE bound'} "
              f"{c['ms_per_row']*1000:.1f} us/row")
        print(f"[check] max rolling-shutter contamination: {c['max_artifact_ms']:.2f} ms")
        if c["max_artifact_ms"] < 1.0:
            print("[check] NEGLIGIBLE -- the ROIs barely moved between conditions, so the "
                  "artifact cancels in the difference. No calibration needed.")
        elif not c["calibrated"]:
            print("[check] That is an upper bound, not a measurement. If it is small next to the "
                  "shift you are claiming, ignore it. If not, run `python rppg_shutter.py` to "
                  "measure it, or re-shoot holding the hand at a constant image row.")
        else:
            print("[check] Non-negligible, but calibrated -- lag_ms_corrected already accounts "
                  "for it.")
        raise SystemExit(0)

    print(f"[cal] dim the room. Flashing at {FLASH_HZ} Hz for {a.seconds:.0f}s -- point the "
          "camera at a wall or your face that the SCREEN lights.")
    B, T, centres, h = capture_flash(a.seconds, a.cam)
    if len(T) < 200:
        raise SystemExit(f"[err] only {len(T)} frames")
    cal = fit(B, T, centres, h)
    DATA.mkdir(exist_ok=True)

    print(f"\n[cal] {cal['n_frames']} frames at {cal['fps']:.1f} fps "
          f"({cal['frame_quantum_ms']:.1f} ms quantum)")
    for r, l, pk in zip(cal["band_rows"], cal["band_lags_ms"], cal["xcorr_peaks"]):
        print(f"      row {r:4d}  lag {l:+7.2f} ms   xcorr {pk:.2f}")
    print(f"[cal] {cal['ms_per_row']*1000:.1f} us per row, r2 {cal['r2']:.3f}")
    print(f"[cal] implied readout across the frame: {cal['implied_readout_ms']:.1f} ms")

    if not cal["valid"]:
        # Write it where it cannot be picked up, so a bad number can never be applied silently.
        bad = CAL.with_suffix(".FAILED.json")
        bad.write_text(json.dumps(cal, indent=2))
        print(f"\n[FAILED] {cal['why']}")
        print(f"[FAILED] not saved as a usable calibration (kept for inspection: {bad.name})")
        print("[FAILED] Dim the room, fill the frame with a surface the SCREEN lights, retry.")
        print("         Or skip it entirely: `python rppg_shutter.py --check` bounds the "
              "artifact from ROI geometry alone, no calibration needed.")
        raise SystemExit(1)

    CAL.write_text(json.dumps(cal, indent=2))
    if abs(cal["implied_readout_ms"]) > 3:
        print(f"[!!] A hand moving 180 rows between conditions fakes "
              f"{abs(cal['ms_per_row'])*180:.1f} ms of PTT shift. Subtract it with "
              f"rppg_shutter.correct(), or hold both ROIs at a constant image row.")
    print(f"\n[done] {CAL.name}")


if __name__ == "__main__":
    main()
