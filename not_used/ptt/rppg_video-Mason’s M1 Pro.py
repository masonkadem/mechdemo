"""rppg_video.py -- offline rPPG from a recorded video file (e.g. 240 fps phone slow-motion).

Why a file instead of the webcam
--------------------------------
The webcam delivers 30 fps measured, so one frame is 33 ms while the arm transit we want to
resolve is ~10-20 ms -- under a single frame. A modern phone shoots 240 fps slow-motion, i.e.
4.2 ms per frame, which turns arm PTT from sub-frame into something spanning several frames. That
is the difference between arguing about interpolation and measuring the quantity.

Frame timing comes from the container's frame count and FPS rather than wall-clock arrival, so
there is no capture jitter -- another advantage over live capture.

    python rppg_video.py --video clip.mov --tag rest --stages

The video should show the neck and the whole arm down to the hand, in bright even light, with the
phone braced. Record 20-30 s: at 240 fps that is already 5,000-7,000 frames.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rppg_two_site as R
import rppg_multi as M
import rppg_pose as P

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def probe(path):
    """Container metadata, without decoding the whole file."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS),
            "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    info["duration_s"] = info["n_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()
    return info


def extract(path, max_frames=0, scale=0.5, progress=None):
    """Per-frame skin-masked RGB at pose-tracked points along the arterial path.

    scale downsamples before pose detection and sampling: a 1080p 240 fps clip is mostly
    redundant spatially, and half-resolution roughly quarters the decode+detect cost without
    changing the mean colour of a patch.
    """
    import cv2
    lmk, mp = P.make_landmarker()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    acc, idx, dist_ref, seg_ref, npts = [], [], None, None, None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        res = lmk.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            int(i / fps * 1000))
        i += 1
        if progress and i % 200 == 0:
            progress(i / max(total, 1))
        if not res.pose_landmarks:
            continue
        h, w = frame.shape[:2]
        pts, dist, seg = P.sample_points(res.pose_landmarks[0], w, h)
        if not pts:
            continue
        if npts is None:
            npts, dist_ref, seg_ref = len(pts), dist, seg
        if len(pts) != npts:
            continue
        msk = M.skin_mask(frame)
        row, r = [], max(4, int(9 * scale * 2))
        for (x, y) in pts:
            x0, x1 = max(0, x - r), min(w, x + r)
            y0, y1 = max(0, y - r), min(h, y + r)
            p = frame[y0:y1, x0:x1]
            mm = msk[y0:y1, x0:x1]
            if p.size == 0 or (mm > 0).mean() < 0.3:
                row.append((np.nan, np.nan, np.nan))
            else:
                row.append(tuple(p[mm > 0][:, ::-1].mean(0)))
        acc.append(row); idx.append(i - 1)
    cap.release()
    # container timing: exact, unlike wall-clock capture
    return np.array(acc, float), np.array(idx) / fps, fps, dist_ref, seg_ref


def analyse(acc, T, fps, dist, tag, make_stages=False):
    if len(T) < 100:
        raise RuntimeError(f"only {len(T)} frames with a detected pose")
    keep = T >= P.WARMUP_S
    if keep.sum() < 100:                      # short clip: keep everything but the first second
        keep = T >= min(1.0, T[-1] * 0.1)
    acc, T = acc[keep], T[keep]
    tu = np.linspace(T[0], T[-1], len(T))
    fs = (len(T) - 1) / (T[-1] - T[0])

    stages, filtered, quals, hrs = {}, [], [], []
    for i in range(acc.shape[1]):
        rgb = acc[:, i, :]
        good = np.isfinite(rgb).all(1)
        if good.mean() < 0.6:
            filtered.append(None); quals.append(0.0); hrs.append(np.nan); continue
        fill = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
        bp = R.bandpass(R.chrom(fill), fs)
        if not stages:
            stages = {"raw": fill[:, 1],
                      "detrended": P.detrend(fill[:, 1], fs),
                      "chrom": R.chrom(fill), "filtered": bp}
        _, hr, q, _ = P.plausible(bp, fs)
        filtered.append(bp); quals.append(q); hrs.append(hr)
    quals, hrs = np.array(quals), np.array(hrs)

    m = np.isfinite(hrs) & (quals > P.MIN_SNR)
    cons = float(np.median(hrs[m])) if m.any() else float("nan")
    ok = np.zeros(len(filtered), bool)
    for i, x in enumerate(filtered):
        if x is not None:
            ok[i], _, _, _ = P.plausible(x, fs, cons)

    out = {"tag": tag, "fps_video": fps, "fps_effective": fs, "n_frames": int(len(T)),
           "n_points": int(acc.shape[1]), "n_accepted": int(ok.sum()),
           "consensus_hr": cons, "frame_ms": 1000.0 / fs}
    if ok.sum() >= 5:
        ref = int(np.argmax(np.where(ok, quals, -1)))
        lags, dd = [], []
        for i, x in enumerate(filtered):
            if ok[i] and i != ref:
                lag, _ = R.lag_subframe(filtered[ref], x, fs)
                lags.append(lag); dd.append(dist[i] - dist[ref])
        lags, dd = np.array(lags), np.array(dd)
        out["lags_ms"] = lags.tolist(); out["distances_cm"] = dd.tolist()
        if len(dd) >= 5 and np.ptp(dd) > 5:
            sl = float(np.polyfit(dd, lags, 1)[0])
            r = float(np.corrcoef(dd, lags)[0, 1])
            pwv = 0.01 / (sl / 1000.0) if abs(sl) > 1e-9 else float("inf")
            out.update({"slope_ms_per_cm": sl, "r": r, "pwv_m_s": pwv,
                        "pwv_plausible": bool(4.0 <= abs(pwv) <= 12.0 and abs(r) > 0.5)})
    (DATA / f"rppg_video_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(DATA / f"rppg_video_{tag}.npz",
             sigs=np.stack([x if x is not None else np.zeros(len(tu)) for x in filtered]),
             accepted=ok, quals=quals, hrs=hrs, dist=dist, fs=fs, **stages)
    if make_stages and stages:
        P.stage_figure(stages, fs, tag, cons, ok, quals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", default="video")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    info = probe(args.video)
    print(f"[vid] {info['w']}x{info['h']}  {info['fps']:.1f} fps  "
          f"{info['n_frames']} frames  {info['duration_s']:.1f}s", flush=True)
    print(f"[vid] frame quantum {1000/max(info['fps'],1e-9):.1f} ms "
          f"({'good -- arm transit spans several frames' if info['fps'] >= 120 else 'marginal: at this rate arm transit is under one frame'})",
          flush=True)

    acc, T, fps, dist, seg = extract(args.video, args.max_frames, args.scale,
                                     progress=lambda f: print(f"  {100*f:4.0f}%", flush=True))
    res = analyse(acc, T, fps, dist, args.tag, args.stages)
    print(f"\n[qc] consensus HR {res['consensus_hr']:.1f} bpm")
    print(f"[qc] accepted {res['n_accepted']}/{res['n_points']} points")
    if "pwv_m_s" in res:
        print(f"[chain] slope {res['slope_ms_per_cm']:+.3f} ms/cm, r = {res['r']:+.3f}")
        print(f"[chain] implied PWV {res['pwv_m_s']:.1f} m/s -- "
              f"{'PLAUSIBLE' if res['pwv_plausible'] else 'NOT plausible (artifact-dominated)'}")
    print(f"\n[done] data/rppg_video_{args.tag}.json")


if __name__ == "__main__":
    main()
