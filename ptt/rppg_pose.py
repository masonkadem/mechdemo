"""rppg_pose.py -- pose-tracked, densely-sampled rPPG along the neck-to-hand arterial path.

What this adds over the hand-drawn boxes
----------------------------------------
MediaPipe Pose gives 33 body landmarks per frame, so the sampling sites FOLLOW the body instead
of being fixed rectangles the subject has to hold still inside. Patches are laid out along the
anatomical arterial path -- face and neck (carotid) -> upper arm (brachial) -> forearm (radial)
-> hand -- with each point's distance along that path known per frame.

Patch counts follow bare skin, not anatomy: on a clothed person the shoulder and upper arm are
under a sleeve, so patches there return nan through the skin mask. Face, neck, forearm and hand
are the sites that actually survive, and the face gets the largest share because facial rPPG has
by far the best SNR while sitting at essentially the same arterial distance as the carotid.

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

import rppg_cam
import rppg_two_site as R
import rppg_multi as M
import rppg_live as LIVE

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

# (name, (landmark_a, landmark_b), n_patches, (dist_a_cm, dist_b_cm), y_offset_frac)
# Landmarks are MediaPipe Pose indices. Distances are nominal adult values, overridable.
# y_offset_frac shifts the whole segment up the image as a fraction of frame height, to reach
# skin that has no landmark of its own (the throat above the shoulder line, the forehead above
# the ear line).
#
# Patch counts follow what is actually BARE SKIN on a clothed person. Shoulder and upper arm are
# under a sleeve for most people, so patches spent there return nan through the skin mask; face,
# neck, forearm and hand are the sites that survive. The face earns the largest share because
# facial rPPG is by far the highest-SNR site, and it sits at essentially the same arterial
# distance as the carotid, so it is a proximal reference rather than a point on the arm.
#
# An earlier version had "shoulder" and "upper_arm" BOTH interpolating (12, 14) with n=6, so they
# placed patches on identical pixels while labelling them 8-20 cm and 20-32 cm. Every point
# entered the lag-vs-distance regression twice, 12 cm apart, with the same lag -- flattening the
# slope and inflating the PWV that the 4-12 m/s check is meant to police.
SEGMENTS = [
    # Face split across three sites rather than one ear-to-ear line: a single line lifted to the
    # forehead put every facial patch on the forehead, which throws away the cheeks -- well
    # perfused, usually unobscured, and a hedge against a forehead lost to hair or a fringe.
    ("forehead", (7, 8),   4, (0.0, 0.0),    0.10),   # l_ear -> r_ear, lifted above the brows
    ("cheek_l",  (2, 7),   3, (0.0, 0.0),   -0.05),   # l_eye -> l_ear, dropped onto the cheek
    ("cheek_r",  (5, 8),   3, (0.0, 0.0),   -0.05),   # r_eye -> r_ear
    # Neck narrowed from 6 patches to 2. Interpolating the full shoulder line put the outer
    # patches on collar and shirt, which contributes clothing reflectance rather than pulse; the
    # two central points sit over the carotid triangles either side of the midline.
    ("neck",     (11, 12), 2, (0.0, 0.0),    0.06),
    # upper_arm removed: on a clothed subject it lands on a sleeve, and a sleeve patch still
    # enters the lag-vs-distance fit at its nominal 8-32 cm, dragging the slope toward zero and
    # inflating the PWV that the 4-12 m/s plausibility check exists to police.
    ("forearm",  (14, 16), 8, (32.0, 56.0),  0.0),    # r_elbow -> r_wrist
    ("hand",     (16, 20), 6, (56.0, 70.0),  0.0),    # r_wrist -> r_index
]
PROXIMAL = ("forehead", "cheek_l", "cheek_r", "neck")   # reference sites for distal transit
DISTAL = ("hand",)


def schema():
    """Canonical (segment, distance) for every patch SLOT, independent of what is visible.

    Fixed length by construction. The capture loop used to key on len(pts) and drop any frame
    whose count differed from the first, so a single segment dipping under the 0.5 visibility
    threshold -- a hand turning, a fringe over the forehead -- discarded the entire frame. With
    seven segments that fires constantly, and it is why no recording ever reached the point of
    saving anything. Slots that are not visible now yield nan instead, which the per-point
    coverage test downstream already knows how to reject.
    """
    out = []
    for name, _, n, (d0, d1), _ in SEGMENTS:
        for k in range(n):
            out.append((name, d0 + ((k + 0.5) / n) * (d1 - d0)))
    return out


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
    """Patch centres for every slot in `schema()`, or None where that slot is not visible.

    Always the same length, so a frame is never discarded because one segment dropped out.
    """
    pts = []
    for name, (a, b), n, (d0, d1), off in SEGMENTS:
        ok = a < len(lms) and b < len(lms)
        if ok:
            pa, pb = lms[a], lms[b]
            ok = min(getattr(pa, "visibility", 1.0), getattr(pb, "visibility", 1.0)) >= 0.5
        if not ok:
            pts.extend([None] * n)
            continue
        dy = int(off * h)
        for k in range(n):
            f = (k + 0.5) / n
            x = int((pa.x + f * (pb.x - pa.x)) * w)
            y = int((pa.y + f * (pb.y - pa.y)) * h) - dy
            pts.append((x, y) if 0 <= x < w and 0 <= y < h else None)
    return pts


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


WIN = "pose-tracked rPPG"


def capture(seconds, cam=0, show=True, arm=True):
    """Preview first, record on demand.

    With `arm`, the window opens in a preview state and records nothing until the on-screen
    button is clicked (or SPACE pressed). Recording blind from the terminal wasted whole runs:
    you could not check that the sites had locked, that the arm was lit, or that a pulse was
    actually coming through until the 60 s were already spent. The preview runs the full
    pipeline, including the live panel, so all of that is visible BEFORE the clock starts.
    """
    import cv2
    lmk, mp = make_landmarker()
    cap = rppg_cam.open_camera(cam, 640, 480, 60)   # platform-aware; CAP_DSHOW is Windows-only
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    armed = bool(show and arm)                 # armed == waiting for the user to press record
    state = {"rec": not armed, "quit": False, "btn": (0, 0, 0, 0)}

    def on_mouse(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bx, by, bw, bh = state["btn"]
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            if not state["rec"]:
                state["rec"] = True
            else:
                state["quit"] = True

    if show:
        cv2.namedWindow(WIN)
        cv2.setMouseCallback(WIN, on_mouse)

    SCHEMA = schema()
    seg_ref = np.array([s for s, _ in SCHEMA])
    dist_ref = np.array([d for _, d in SCHEMA])
    PM, DM = np.isin(seg_ref, PROXIMAL), np.isin(seg_ref, DISTAL)
    acc, T = [], []
    nseen, nkept = 0, 0
    COLS = {"forehead": (150, 255, 150), "cheek_l": (110, 240, 110), "cheek_r": (110, 240, 110),
            "neck": (0, 255, 0), "upper_arm": (0, 200, 220),
            "forearm": (0, 150, 255), "hand": (0, 100, 255)}
    ALL = [s[0] for s in SEGMENTS]
    skin = M.SkinModel()                       # learns this person's skin, incl. in shadow
    panel = LIVE.LivePanel() if show else None
    t_wall = time.time()
    t0 = t_wall                                # reset to the moment recording actually starts
    print(f"[cap] {'preview -- press the button (or SPACE) to record' if armed else 'recording'} "
          f"{seconds:.0f}s -- keep your face, neck and right arm in frame", flush=True)
    while True:
        if state["quit"]:
            break
        if state["rec"] and time.time() - t0 >= seconds:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        if not state["rec"]:                   # preview: keep the clock pinned at zero
            t0 = time.time()
        elif nseen == 0:
            print("[cap] recording started", flush=True)
        nseen += 1 if state["rec"] else 0
        el = time.time() - t0
        rgb = frame[:, :, ::-1]
        res = lmk.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
            int(el * 1000))
        h, w = frame.shape[:2]
        pts = sample_points(res.pose_landmarks[0], w, h) if res.pose_landmarks \
            else [None] * len(SCHEMA)
        vis_pts = [p for p in pts if p is not None]

        msk = skin.mask(frame, vis_pts) if vis_pts else None
        if vis_pts:
            # Every frame with any visible slot is kept. Slots that are not visible become nan
            # rather than costing the whole frame, which is what discarded 98% of a real 60 s
            # recording (32 of 1690 frames) and left nothing to save.
            row, r = [], 9
            for p in pts:
                if p is None:
                    row.append((np.nan, np.nan, np.nan)); continue
                x, y = p
                x0, x1 = max(0, x - r), min(w, x + r)
                y0, y1 = max(0, y - r), min(h, y + r)
                pa = frame[y0:y1, x0:x1]
                mm = msk[y0:y1, x0:x1]
                if pa.size == 0 or (mm > 0).mean() < 0.3:
                    row.append((np.nan, np.nan, np.nan))
                else:
                    row.append(tuple(pa[mm > 0][:, ::-1].mean(0)))
            if state["rec"]:                      # preview computes everything but stores nothing
                acc.append(row); T.append(el); nkept += 1
            if panel is not None:                 # panel runs in preview too, that is the point
                A = np.array(row, float)
                with np.errstate(invalid="ignore"):
                    pr = np.nanmean(A[PM], 0) if PM.any() else np.full(3, np.nan)
                    ds = np.nanmean(A[DM], 0) if DM.any() else np.full(3, np.nan)
                panel.push(pr, ds, time.time() - t_wall)

        # Draw EVERY frame, including ones rejected above. Previously the consistency gate
        # `continue`d before this block, so a segment dipping under the visibility threshold
        # froze the preview and looked like the tracker had died -- with no clue as to which
        # part of the body had dropped out.
        if show:
            if msk is None:
                vis = cv2.convertScaleAbs(frame, alpha=0.4)
            else:
                # cv2 ops instead of numpy fancy-indexing: measured 9.5 ms/frame -> under 1 ms,
                # which was 28% of the frame budget spent on cosmetics.
                vis = cv2.convertScaleAbs(frame, alpha=0.4)
                cv2.copyTo(frame, msk, vis)
            have = set()
            for p, s in zip(pts, seg_ref):
                if p is None:
                    continue
                cv2.circle(vis, p, 5, COLS.get(s, (255, 255, 255)), -1)
                have.add(s)
            hdr = (f"REC  {max(0, seconds-el):4.0f}s left   {nkept/max(el,1e-3):4.1f} fps kept"
                   if state["rec"] else "PREVIEW  (nothing is being saved yet)")
            cv2.putText(vis, f"{hdr}   {len(vis_pts)}/{len(SCHEMA)} pts", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (60, 60, 255) if state["rec"] else (255, 255, 255), 2)
            for i, s in enumerate(ALL):           # per-segment lock indicator
                on = s in have
                cv2.putText(vis, f"{'OK ' if on else '-- '}{s}", (10, 48 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, .5,
                            COLS.get(s, (255, 255, 255)) if on else (90, 90, 90), 1)
            # clickable button, bottom-left of the video half
            bw, bh = 210, 42
            bx, by = 10, h - bh - 10
            state["btn"] = (bx, by, bw, bh)
            rec = state["rec"]
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                          (40, 40, 190) if rec else (60, 150, 60), -1)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (235, 235, 235), 1)
            cv2.circle(vis, (bx + 22, by + bh // 2), 8, (255, 255, 255), -1 if rec else 2)
            cv2.putText(vis, "STOP" if rec else "START RECORDING",
                        (bx + 40, by + bh // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (255, 255, 255), 2 if rec else 1, cv2.LINE_AA)
            cv2.putText(vis, "click, or SPACE" + ("" if rec else "   /   q to quit"),
                        (bx + bw + 12, by + bh // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        (190, 190, 190), 1, cv2.LINE_AA)
            if not vis_pts:
                cv2.putText(vis, "NO POSE - step back so face, neck and right arm are in frame",
                            (10, by - 12), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 165, 255), 1)
            elif rec and el < WARMUP_S:
                cv2.putText(vis, "warm-up (discarded)", (10, by - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)
            vis = np.hstack([vis, panel.render(h)])
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if state["rec"]:
                    break
                state["rec"] = True
            elif key in (ord("q"), 27):
                state["quit"] = True
    cap.release()
    if show:
        cv2.destroyAllWindows()
    if nseen and nkept < 0.5 * nseen:
        print(f"[warn] kept only {nkept}/{nseen} frames -- the pose or a segment kept dropping "
              f"out. Effective rate {nkept/max(seconds,1e-3):.1f} fps, which is what the timing "
              f"actually rests on.", flush=True)
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

    all_stages = {}
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
        all_stages[i] = {"raw": fill[:, 1], "detrended": det[:, 1], "chrom": ch, "filtered": bp}
        _, hr, qm, _ = plausible(bp, fs)
        filtered.append(bp); quals.append(qm); hrs.append(hr)
    quals, hrs = np.array(quals), np.array(hrs)

    # Illustrate the stages with the BEST point, not point 0. Point 0 is the first forehead
    # patch, which is as likely as any other to be the one obscured by hair -- so the figure
    # meant to show that the pipeline works could show it failing, for an unrelated reason.
    best = int(np.argmax(quals)) if len(quals) and quals.max() > 0 else None
    stages = all_stages.get(best, {})
    if best is not None:
        print(f"[fig] stage figure uses point {best} ({seg[best]}, SNR {quals[best]:.1f})",
              flush=True)

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
             accepted=acc_ok, quals=quals, hrs=hrs, dist=dist, seg=np.asarray(seg), fs=fs,
             **stages)
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
