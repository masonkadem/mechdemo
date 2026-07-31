"""rppg_pose.py -- pose-tracked, densely-sampled rPPG along the neck-to-hand arterial path.

What this adds over the hand-drawn boxes
----------------------------------------
MediaPipe Pose gives 33 body landmarks per frame, so the sampling sites FOLLOW the body instead
of being fixed rectangles the subject has to hold still inside. Patches are laid out along the
anatomical arterial path -- neck (carotid) -> shoulder -> upper arm (brachial) -> elbow ->
forearm (radial) -> wrist -> hand -- with several patches per segment, giving on the order of a
hundred sampling points whose anatomical distance from the neck is known per frame.

That distance axis is the point. A two-site lag cannot distinguish a transit time from a fixed
processing offset, but arrival time that grows LINEARLY WITH DISTANCE along the arm can only be
propagation, and its slope is pulse wave velocity, which has a known upper-limb range (4-12 m/s)
that serves as an external check.

The signal pipeline, staged so every step is inspectable
--------------------------------------------------------
  raw        per-patch mean of skin-masked RGB
  detrended  the first WARMUP_S seconds dropped (auto-exposure and white balance settle over
             the first few seconds and produce a large low-frequency swing that no band-pass
             fully removes), then a moving-average detrend
  chrom      chrominance projection to isolate the pulsatile component
  filtered   band-pass 0.7-3.0 Hz
  accepted   patches that pass physiological plausibility: a dominant spectral peak in band,
             enough of the in-band power at that peak, an HR within 40-180 bpm consistent with
             the consensus across patches, and inter-beat-interval scatter in a physiological
             range

All five stages are saved, so a bad result can be traced to the stage that caused it.

    python rppg_pose.py --seconds 60 --tag rest
    python rppg_pose.py --seconds 60 --tag rest --stages    # save the stage figure too
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import rppg_two_site as R
import rppg_multi as M

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "models" / "pose_landmarker.task"

WARMUP_S = 3.0            # discarded: camera auto-exposure/AWB settle
MIN_PEAK_FRAC = 0.12      # min share of in-band power at the dominant peak
MIN_SNR = 6.0             # dominant peak vs MEDIAN in-band power -- the gate that works.
# Peak-fraction alone accepts pure noise: band-passed white noise scores 0.132, above the
# 0.12 threshold, because filtering concentrates power in band whether or not a pulse is
# present. Peak/median separates cleanly instead: measured 1.9 for noise against 48.6 for
# pulse-in-noise and ~1e5 for a clean pulse.
HR_TOL_BPM = 12.0         # a patch must agree with the consensus HR within this
IBI_SD_MAX_MS = 220.0     # implausible beat-to-beat scatter => not a pulse

# (landmark_a, landmark_b, n_patches, cumulative distance in cm at a and b along the path)
# Landmarks are MediaPipe Pose indices. Distances are nominal adult values, overridable.
SEGMENTS = [
    ("neck",      (11, 12), 6,  (0.0, 0.0)),      # between the shoulders, up toward the carotid
    ("shoulder",  (12, 14), 6,  (8.0, 20.0)),     # r_shoulder -> r_elbow (upper arm)
    ("upper_arm", (12, 14), 6,  (20.0, 32.0)),
    ("forearm",   (14, 16), 8,  (32.0, 56.0)),    # r_elbow -> r_wrist
    ("hand",      (16, 20), 6,  (56.0, 70.0)),    # r_wrist -> r_index
]


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if not MODEL.exists():
        raise RuntimeError(f"missing pose model at {MODEL}")
    opts = vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.PoseLandmarker.create_from_options(opts), mp


def sample_points(lms, w, h):
    """Patch centres along the arterial path, with each point's distance from the neck."""
    pts, dist, seg = [], [], []
    for name, (a, b), n, (d0, d1) in SEGMENTS:
        if a >= len(lms) or b >= len(lms):
            continue
        pa, pb = lms[a], lms[b]
        if min(getattr(pa, "visibility", 1.0), getattr(pb, "visibility", 1.0)) < 0.5:
            continue
        for k in range(n):
            f = (k + 0.5) / n
            x = int((pa.x + f * (pb.x - pa.x)) * w)
            y = int((pa.y + f * (pb.y - pa.y)) * h)
            if name == "neck":                    # shift up from the shoulder line to the throat
                y -= int(0.06 * h)
            pts.append((x, y)); dist.append(d0 + f * (d1 - d0)); seg.append(name)
    return pts, np.array(dist), seg


def detrend(x, fs, win_s=1.5):
    """Moving-average detrend: removes slow illumination drift the band-pass leaves behind."""
    k = max(int(win_s * fs) | 1, 3)
    pad = np.pad(x, k // 2, mode="edge")
    base = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(x)]
    return x - base


def plausible(x, fs, consensus_hr=None):
    """Physiological acceptance test. Returns (ok, hr_bpm, snr, ibi_sd_ms).

    Gates, in order of how much work they do: spectral SNR (peak vs median in-band power),
    peak fraction, HR within 40-180 bpm, inter-beat-interval scatter, and agreement with
    the consensus HR across points.
    """
    from scipy.signal import welch
    if len(x) < int(6 * fs) or np.std(x) < 1e-12:
        return False, np.nan, 0.0, np.nan
    f, P = welch(x, fs, nperseg=min(len(x), int(6 * fs)))
    m = (f > R.BAND[0]) & (f < R.BAND[1])
    if not m.any() or P[m].sum() <= 0:
        return False, np.nan, 0.0, np.nan
    k = int(np.argmax(P[m]))
    hr = float(f[m][k] * 60)
    frac = float(P[m][k] / P[m].sum())
    from rppg_sota import instantaneous_hr
    _, _, hr_beat, ibisd = instantaneous_hr(x, fs)
    snr = float(P[m][k] / (np.median(P[m]) + 1e-15))
    ok = (snr >= MIN_SNR and frac >= MIN_PEAK_FRAC and 40 <= hr <= 180
          and (not np.isfinite(ibisd) or ibisd <= IBI_SD_MAX_MS))
    if ok and consensus_hr is not None:
        ok = abs(hr - consensus_hr) <= HR_TOL_BPM
    return bool(ok), hr, snr, ibisd


def capture(seconds, cam=0, show=True):
    import cv2
    lmk, mp = make_landmarker()
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    acc, T, dist_ref, seg_ref, npts = [], [], None, None, None
    t0 = time.time()
    print(f"[cap] recording {seconds}s -- keep your right arm and neck in frame", flush=True)
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        el = time.time() - t0
        rgb = frame[:, :, ::-1]
        res = lmk.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
            int(el * 1000))
        if not res.pose_landmarks:
            continue
        h, w = frame.shape[:2]
        pts, dist, seg = sample_points(res.pose_landmarks[0], w, h)
        if not pts:
            continue
        if npts is None:
            npts, dist_ref, seg_ref = len(pts), dist, seg
        if len(pts) != npts:                      # landmark set changed; skip for consistency
            continue
        msk = M.skin_mask(frame)
        row, r = [], 9
        for (x, y) in pts:
            x0, x1 = max(0, x - r), min(w, x + r)
            y0, y1 = max(0, y - r), min(h, y + r)
            p = frame[y0:y1, x0:x1]
            mm = msk[y0:y1, x0:x1]
            if p.size == 0 or (mm > 0).mean() < 0.3:
                row.append((np.nan, np.nan, np.nan))
            else:
                row.append(tuple(p[mm > 0][:, ::-1].mean(0)))
        acc.append(row); T.append(el)
        if show:
            vis = frame.copy()
            vis[msk == 0] = (vis[msk == 0] * .4).astype(np.uint8)
            cols = {"neck": (0, 255, 0), "shoulder": (0, 220, 120), "upper_arm": (0, 200, 220),
                    "forearm": (0, 150, 255), "hand": (0, 100, 255)}
            for (x, y), s in zip(pts, seg):
                cv2.circle(vis, (x, y), 5, cols.get(s, (255, 255, 255)), -1)
            cv2.putText(vis, f"{seconds-el:4.0f}s  fps {len(T)/max(el,1e-3):4.1f}  "
                        f"{len(pts)} pts", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (255, 255, 255), 2)
            if el < WARMUP_S:
                cv2.putText(vis, "warm-up (discarded)", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
            cv2.imshow("pose-tracked rPPG - q to stop", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return np.array(acc, float), np.array(T), dist_ref, seg_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tag", default="rest")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--stages", action="store_true", help="save the stage-by-stage figure")
    args = ap.parse_args()

    acc, T, dist, seg = capture(args.seconds, args.cam)
    if len(T) < 100:
        print("[err] too few frames with a detected pose"); return
    fs = (len(T) - 1) / (T[-1] - T[0])
    print(f"[cap] {len(T)} frames, {fs:.1f} fps, {acc.shape[1]} points", flush=True)

    keep = T >= WARMUP_S
    print(f"[pre] dropping {(~keep).sum()} warm-up frames (<{WARMUP_S:.0f}s)", flush=True)
    acc, T = acc[keep], T[keep]
    tu = np.linspace(T[0], T[-1], len(T))

    stages = {}
    filtered, quals, hrs = [], [], []
    for i in range(acc.shape[1]):
        rgb = acc[:, i, :]
        good = np.isfinite(rgb).all(1)
        if good.mean() < 0.6:
            filtered.append(None); quals.append(0.0); hrs.append(np.nan); continue
        fill = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
        det = np.stack([detrend(fill[:, c], fs) for c in range(3)], 1)
        ch = R.chrom(fill)
        bp = R.bandpass(ch, fs)
        if i == 0:
            stages = {"raw": fill[:, 1], "detrended": det[:, 1], "chrom": ch, "filtered": bp}
        _, hr, qm, _ = plausible(bp, fs)
        filtered.append(bp); quals.append(qm); hrs.append(hr)
    quals, hrs = np.array(quals), np.array(hrs)

    cons = float(np.median(hrs[np.isfinite(hrs) & (quals > MIN_SNR)])) \
        if np.isfinite(hrs).any() else np.nan
    acc_ok = np.zeros(len(filtered), bool)
    for i, x in enumerate(filtered):
        if x is not None:
            acc_ok[i], _, _, _ = plausible(x, fs, cons)
    print(f"\n[qc] consensus HR {cons:.1f} bpm")
    print(f"[qc] accepted {acc_ok.sum()}/{len(filtered)} points "
          f"({100*acc_ok.mean():.0f}%) as physiologically plausible", flush=True)
    if acc_ok.sum() < 6:
        print("[qc] too few accepted points -- improve lighting, keep the arm still and in "
              "frame, and retry"); return

    # ---- arrival time vs anatomical distance --------------------------------
    ref = int(np.argmax(np.where(acc_ok, quals, -1)))
    lags, dd = [], []
    for i, x in enumerate(filtered):
        if not acc_ok[i] or i == ref:
            continue
        lag, _ = R.lag_subframe(filtered[ref], x, fs)
        lags.append(lag); dd.append(dist[i] - dist[ref])
    lags, dd = np.array(lags), np.array(dd)
    out = {"tag": args.tag, "fps": fs, "n_frames": len(T), "n_points": int(acc.shape[1]),
           "n_accepted": int(acc_ok.sum()), "consensus_hr": cons,
           "distances_cm": dd.tolist(), "lags_ms": lags.tolist()}
    if len(dd) >= 5 and np.ptp(dd) > 5:
        sl, ic = np.polyfit(dd, lags, 1)
        r = float(np.corrcoef(dd, lags)[0, 1])
        pwv = 0.01 / (sl / 1000.0) if abs(sl) > 1e-9 else float("inf")
        out.update({"slope_ms_per_cm": float(sl), "r": r, "pwv_m_s": float(pwv)})
        print(f"\n[chain] arrival vs distance: slope {sl:+.3f} ms/cm, r = {r:+.3f}, "
              f"n = {len(dd)}")
        print(f"[chain] implied PWV = {pwv:.1f} m/s")
        if 4.0 <= abs(pwv) <= 12.0 and abs(r) > 0.5:
            print("[chain] PLAUSIBLE -- monotonic in distance and inside the 4-12 m/s "
                  "upper-limb range. This is propagation, not a fixed offset.")
        else:
            print("[chain] NOT plausible. Outside 4-12 m/s or weak distance correlation means "
                  "the lags are artifact-dominated. At 30 fps the true arm transit (~10-20 ms) "
                  "is under one frame, so this is the expected hard case.")

    (DATA / f"rppg_pose_{args.tag}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(DATA / f"rppg_pose_{args.tag}.npz",
             sigs=np.stack([x if x is not None else np.zeros(len(tu)) for x in filtered]),
             accepted=acc_ok, quals=quals, hrs=hrs, dist=dist, fs=fs, **stages)
    print(f"\n[done] data/rppg_pose_{args.tag}.json")

    if args.stages and stages:
        stage_figure(stages, fs, args.tag, cons, acc_ok, quals)


def stage_figure(stages, fs, tag, cons, acc_ok, quals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5, 1, figsize=(9, 8.4), sharex=False)
    order = [("raw", "raw skin RGB (green)"), ("detrended", "detrended"),
             ("chrom", "chrominance projection"), ("filtered", "band-pass 0.7-3 Hz")]
    n = int(min(len(stages["filtered"]), 12 * fs))
    t = np.arange(n) / fs
    for a, (k, lab) in zip(ax, order):
        a.plot(t, stages[k][:n], lw=1.0, color="#2f4b7c")
        a.set_ylabel(lab, fontsize=8)
        a.set_yticks([])
        a.spines[["top", "right", "left"]].set_visible(False)
    ax[-1].hist(quals[np.isfinite(quals)], bins=24, color="#9a9a9a")
    ax[-1].axvline(MIN_SNR, color="#c1543b", lw=1.4, label=f"accept ≥ {MIN_SNR}")
    ax[-1].set_xscale("log")
    ax[-1].set_xlabel("spectral SNR (cardiac peak / median in-band)", fontsize=9)
    ax[-1].set_ylabel("points", fontsize=8)
    ax[-1].legend(fontsize=8, frameon=False)
    ax[-1].spines[["top", "right"]].set_visible(False)
    ax[0].set_title(f"{tag}: signal stages, consensus HR {cons:.0f} bpm, "
                    f"{acc_ok.sum()}/{len(acc_ok)} points accepted",
                    loc="left", fontsize=10, fontweight="bold")
    ax[-2].set_xlabel("s", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_rppg_stages_{tag}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] figures/fig_rppg_stages_{tag}.png")


if __name__ == "__main__":
    main()
