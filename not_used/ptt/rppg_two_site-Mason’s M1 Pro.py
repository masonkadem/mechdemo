"""rppg_two_site.py -- two-site camera PPG (neck + hand) for a TRUE pulse transit time.

Why two sites from one camera
-----------------------------
Everything measured in this project so far uses PAT = R-peak -> finger, which is
PEP (cardiac, not vascular) plus peripheral transit. PEP dominates, the governing law does not
apply to it, and four independent instruments returned null on it.

Neck (carotid) -> hand gives a TRUE PTT: carotid is proximal/near-aortic, so carotid_foot ->
hand_foot is transit along the arterial tree with PEP excluded by construction. And because both
ROIs come from the SAME camera frame, they share a clock -- which removes the inter-channel
desynchronization that made VitalDB's PAT a near-constant instrumental offset (240/242 ms).

HONEST LIMITS, read before trusting any number this prints
----------------------------------------------------------
* This webcam delivers ~30 fps measured (it reports 60). One frame is 33 ms.
* Neck->hand distance is ~60-80 cm, so true PTT is only ~20-50 ms, i.e. UNDER two frames.
* Therefore per-beat PTT is NOT resolvable here. We recover a sub-frame estimate by
  cross-correlating the two band-passed traces with parabolic interpolation of the correlation
  peak, and by averaging over many beats. That yields a stable MEAN offset, not beat-to-beat PTT.
* A mean offset is exactly the quantity that a fixed camera/ROI processing delay would also
  produce. So treat the absolute value with suspicion and prefer CHANGES under a perturbation
  (e.g. hand raised above heart vs at heart level, which alters hydrostatic pressure and should
  shift PTT in a known direction).

What would fix it: a 120+ fps camera, or a longer path (ankle), or a reference contact PPG.

Usage
-----
    python rppg_two_site.py --seconds 60            # capture and analyse
    python rppg_two_site.py --seconds 60 --show     # with live preview

Position yourself so the camera sees your face/neck, then raise your hand to chest height so
palm and neck are both in frame. Hold still; rPPG is motion-sensitive.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import rppg_cam

ROOT = Path(__file__).resolve().parent
BAND = (0.7, 3.0)          # Hz, 42-180 bpm


def bandpass(x, fs, lo=BAND[0], hi=BAND[1], order=3):
    from scipy.signal import butter, filtfilt
    hi = min(hi, 0.45 * fs)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    return filtfilt(b, a, x)


def chrom(rgb):
    """CHROM rPPG (de Haan & Jeanne): projects RGB onto a chrominance axis that suppresses
    motion/specular artifacts far better than the green channel alone."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    rn = r / (np.mean(r) + 1e-9)
    gn = g / (np.mean(g) + 1e-9)
    bn = b / (np.mean(b) + 1e-9)
    xs = 3 * rn - 2 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    sx, sy = np.std(xs), np.std(ys)
    return xs - (sx / (sy + 1e-9)) * ys


def capture(seconds, show=False, cam=0, pick=True):
    """Grab mean RGB from a neck ROI and a hand ROI, with real per-frame timestamps.

    ROIs are placed by hand rather than by a face detector. OpenCV 5 removed
    cv2.CascadeClassifier, and manual placement is in any case more reliable for this task: the
    neck ROI must sit on bare skin over the carotid, which an automatic face-offset heuristic
    frequently misses (landing on a collar or a shadow) and which the operator can see at a
    glance. Fixed ROIs also keep the two regions in a constant position across conditions, which
    matters because the whole measurement is a comparison between runs.
    """
    import cv2
    cap = rppg_cam.open_camera(cam, 640, 480, 60)   # platform-aware; CAP_DSHOW is Windows-only
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera opened but returned no frame")
    h, w = frame.shape[:2]
    # sensible defaults: neck just right of centre, hand in the lower-left quadrant
    neck_roi = (int(0.40 * w), int(0.62 * h), int(0.20 * w), int(0.16 * h))
    hand_roi = (int(0.05 * w), int(0.55 * h), int(0.30 * w), int(0.35 * h))

    if pick:
        print("[roi] drag a box over your NECK (bare skin over the carotid), then ENTER.",
              flush=True)
        r = cv2.selectROI("select NECK then press ENTER", frame, False, False)
        if r[2] > 5 and r[3] > 5:
            neck_roi = tuple(int(v) for v in r)
        cv2.destroyAllWindows()
        ok, frame = cap.read()
        print("[roi] now drag a box over your PALM, then ENTER.", flush=True)
        r = cv2.selectROI("select HAND then press ENTER", frame, False, False)
        if r[2] > 5 and r[3] > 5:
            hand_roi = tuple(int(v) for v in r)
        cv2.destroyAllWindows()
    print(f"[roi] neck {neck_roi}  hand {hand_roi}", flush=True)

    def mean_rgb(fr, r):
        x, y, ww, hh = r
        p = fr[y:y + hh, x:x + ww]
        return p[:, :, ::-1].reshape(-1, 3).mean(0) if p.size else np.zeros(3)

    N, H, T = [], [], []
    t0 = time.time()
    print(f"[cap] recording {seconds}s -- hold still", flush=True)
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        N.append(mean_rgb(frame, neck_roi))
        H.append(mean_rgb(frame, hand_roi))
        T.append(time.time() - t0)
        if show:
            for r, c, lab in ((neck_roi, (0, 255, 0), "NECK"),
                              (hand_roi, (0, 128, 255), "HAND")):
                x, y, ww, hh = r
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), c, 2)
                cv2.putText(frame, lab, (x, max(16, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            cv2.putText(frame, f"{seconds - (time.time()-t0):4.0f}s left", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # live pulsatility: std of the recent green trace, which should VISIBLY rise once a
            # pulse is being picked up. Lets the operator confirm the ROI works before spending
            # a full recording on a bad placement.
            if len(N) > 60:
                for i, (buf, c, yy) in enumerate(((N, (0, 255, 0), 52), (H, (0, 128, 255), 76))):
                    a = np.array(buf[-60:])[:, 1]
                    amp = float(np.std(a) / (np.mean(a) + 1e-9) * 1000)
                    bar = int(min(amp * 12, 180))
                    cv2.rectangle(frame, (110, yy - 10), (110 + bar, yy - 2), c, -1)
                    cv2.putText(frame, f"pulse {amp:4.1f}", (10, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            cv2.imshow("neck (green) / hand (orange) - q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    # ROIs travel with the traces: the vertical centres are what the rolling-shutter correction
    # needs, and they are unrecoverable after the fact if not recorded here.
    return np.array(N), np.array(H), np.array(T), neck_roi, hand_roi


def lag_subframe(a, b, fs, max_lag_s=0.25):
    """Cross-correlation lag of b relative to a, with parabolic sub-sample interpolation.

    Sub-sample interpolation is essential here: the expected neck->hand PTT (20-50 ms) is under
    two frames at 30 fps, so an integer-lag estimate would quantise it to 0 or 33 ms.
    """
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    n = int(max_lag_s * fs)
    c = np.correlate(b, a, "full")
    mid = len(c) // 2
    lo, hi = mid - n, mid + n + 1
    seg = c[lo:hi]
    k = int(np.argmax(seg))
    if 0 < k < len(seg) - 1:                      # parabolic refinement
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        d = (y0 - y2) / (2 * (y0 - 2 * y1 + y2) + 1e-12)
    else:
        d = 0.0
    return ((lo + k + d) - mid) / fs * 1000.0, float(seg[k] / (len(a) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--tag", default="rest")
    ap.add_argument("--no-pick", action="store_true",
                    help="skip the ROI picker and use default boxes")
    args = ap.parse_args()

    N, H, T, neck_roi, hand_roi = capture(args.seconds, args.show, args.cam,
                                          pick=not args.no_pick)
    row_neck = neck_roi[1] + neck_roi[3] / 2.0        # ROI vertical centres, for shutter correction
    row_hand = hand_roi[1] + hand_roi[3] / 2.0
    if len(T) < 100:
        print("[err] too few frames"); return
    fs = (len(T) - 1) / (T[-1] - T[0])
    print(f"[cap] {len(T)} frames, {fs:.1f} fps effective, {T[-1]:.1f}s", flush=True)
    if fs < 45:
        print(f"[warn] {fs:.0f} fps => {1000/fs:.0f} ms per frame. Expected neck->hand PTT is "
              f"20-50 ms, i.e. under two frames. Per-beat PTT is NOT resolvable; only a "
              f"beat-averaged sub-frame offset is.", flush=True)

    # resample onto a uniform grid (webcam timestamps jitter)
    tu = np.linspace(T[0], T[-1], len(T))
    nb = bandpass(np.interp(tu, T, chrom(N)), fs)
    hb = bandpass(np.interp(tu, T, chrom(H)), fs)

    from scipy.signal import find_peaks, welch
    f, P = welch(nb, fs, nperseg=min(len(nb), int(8 * fs)))
    band = (f > BAND[0]) & (f < BAND[1])
    hr = float(f[band][np.argmax(P[band])] * 60)
    snr = float(P[band].max() / (np.median(P[band]) + 1e-12))
    print(f"[sig] neck HR {hr:.0f} bpm, spectral SNR {snr:.1f}", flush=True)
    if snr < 5:
        print("[warn] weak pulse signal -- improve lighting, hold still, retry", flush=True)

    lag, peak = lag_subframe(nb, hb, fs)
    # bootstrap over 10 s windows for an honest spread
    W = int(10 * fs)
    lags = [lag_subframe(nb[i:i + W], hb[i:i + W], fs)[0]
            for i in range(0, len(nb) - W, W // 2)] if len(nb) > W else []
    print(f"\n[ptt] neck -> hand lag {lag:+.1f} ms   (xcorr peak {peak:.3f})")
    if lags:
        print(f"[ptt] per-window: median {np.median(lags):+.1f} ms, "
              f"sd {np.std(lags):.1f} ms, n={len(lags)}")
        print(f"[ptt] frame quantum is {1000/fs:.0f} ms -- an sd below that means the estimate "
              f"is interpolation-limited, not physiology-limited")
    plausible = 5.0 <= abs(lag) <= 120.0
    print(f"[ptt] physiologically plausible (5-120 ms): {plausible}")
    if not plausible:
        print("      A value outside that range is a processing/ROI artifact, not a transit time.")

    # Rolling shutter: the two ROIs sit at different image rows, so part of this lag is the
    # sensor sampling them at different times. See rppg_shutter.py.
    import rppg_shutter
    lag_corr, artifact = rppg_shutter.correct(lag, row_neck, row_hand)
    print(f"\n[shut] ROI rows: neck {row_neck:.0f}, hand {row_hand:.0f} "
          f"(drow {row_hand - row_neck:+.0f})")
    if np.isnan(artifact):
        print("[shut] NO CALIBRATION -- run `python rppg_shutter.py` first. Until then part of "
              "the lag above is sensor row timing, and any hand_up/hand_down shift is "
              "uninterpretable because moving the hand moves its image row.")
    else:
        print(f"[shut] rolling-shutter artifact {artifact:+.1f} ms -> corrected lag "
              f"{lag_corr:+.1f} ms")

    out = {"tag": args.tag, "fps": fs, "n_frames": len(T), "hr_bpm": hr, "snr": snr,
           "lag_ms": lag, "xcorr_peak": peak,
           "row_neck": float(row_neck), "row_hand": float(row_hand),
           "shutter_artifact_ms": artifact, "lag_ms_corrected": lag_corr,
           "window_lags": [float(x) for x in lags], "plausible": bool(plausible)}
    p = ROOT / "data" / f"rppg_two_site_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    np.savez(ROOT / "data" / f"rppg_two_site_{args.tag}.npz", neck=nb, hand=hb, t=tu, fs=fs)
    print(f"\n[done] {p.name}")
    print("\nNext: rerun with --tag hand_up (hand raised well above heart). Hydrostatic pressure "
          "falls, so PTT should LENGTHEN. A reproducible shift between conditions is far stronger "
          "evidence than any single absolute number.")


if __name__ == "__main__":
    main()
