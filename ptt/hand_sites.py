"""hand_sites.py -- fingertip sampling via MediaPipe Hands, and the proximal null control.

Why a second model
------------------
Pose landmark 20 is the index MCP -- the knuckle -- not the tip, and pose has no other finger
detail. Fingertips are the best rPPG target on the body: dense capillary beds, no clothing, and
arteriovenous anastomoses that make the pulsatile component unusually large. Sampling knuckles
while calling them "hand" gives up most of that.

MediaPipe Hands provides 21 landmarks per hand, including all five tips (4, 8, 12, 16, 20), so
the distal site becomes the tips themselves plus the distal phalanges just behind them.

The null control
----------------
Forehead, left cheek and right cheek are all proximal: the pulse reaches them at essentially the
same time, so their MUTUAL lag should be about zero. That is a free negative control which the
rig currently does not display. If forehead-to-cheek reads 30 ms, the timing pipeline is broken
and any face-to-hand number it produces is meaningless -- exactly the check that separates a real
transit from a processing artifact, and it costs one extra cross-correlation.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "models" / "hand_landmarker.task"

# MediaPipe Hands landmark indices
TIPS = [4, 8, 12, 16, 20]           # thumb, index, middle, ring, pinky
DIPS = [3, 7, 11, 15, 19]           # the joint just proximal to each tip
PALM = [0, 5, 9, 13, 17]            # wrist and the MCP row


def make_hand_landmarker(num_hands=1):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    if not HAND_MODEL.exists():
        raise RuntimeError(f"missing hand model at {HAND_MODEL}")
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
        min_hand_detection_confidence=0.4, min_tracking_confidence=0.4)
    return vision.HandLandmarker.create_from_options(opts), mp


def hand_points(result, w, h, groups=("tip", "dip")):
    """Pixel coordinates for the requested finger groups, with their distance labels.

    Distances continue the pose chain: the pose 'hand' segment ends at 70 cm (wrist to index
    MCP), so the palm sits at 70, the distal phalanges at 74 and the tips at 77. Those are
    nominal adult values -- what matters for the PWV fit is that they are ordered and roughly
    correct, since the slope is what carries the physiology.
    """
    if not result.hand_landmarks:
        return [], [], []
    lms = result.hand_landmarks[0]
    spec = {"palm": (PALM, 70.0), "dip": (DIPS, 74.0), "tip": (TIPS, 77.0)}
    pts, dist, seg = [], [], []
    for g in groups:
        idxs, dd = spec[g]
        for i in idxs:
            if i >= len(lms):
                continue
            pts.append((int(lms[i].x * w), int(lms[i].y * h)))
            dist.append(dd)
            seg.append(f"finger_{g}")
    return pts, np.array(dist), seg


def null_control(sigs, fs, sites=("forehead", "cheek_l", "cheek_r")):
    """Pairwise lag among proximal sites, which should all be near zero.

    Returns (median_abs_lag_ms, per_pair) or (nan, {}) if fewer than two sites are usable. A
    median well above the frame quantum means the timing pipeline is producing offsets where the
    physiology says there are none, and the face-to-hand number cannot be trusted either.
    """
    import rppg_two_site as R
    have = [s for s in sites if s in sigs and sigs[s] is not None
            and np.isfinite(sigs[s]).all() and np.std(sigs[s]) > 1e-9]
    if len(have) < 2:
        return float("nan"), {}
    pairs = {}
    for i in range(len(have)):
        for j in range(i + 1, len(have)):
            lag, _ = R.lag_subframe(sigs[have[i]], sigs[have[j]], fs,
                                    max_lag_s=min(0.25, 6.0 / fs))
            pairs[f"{have[i]}-{have[j]}"] = float(lag)
    return float(np.median(np.abs(list(pairs.values())))), pairs


def verdict(null_ms, fs):
    """Plain reading of the null control, against the frame quantum."""
    q = 1000.0 / max(fs, 1e-6)
    if not np.isfinite(null_ms):
        return "no control", (150, 150, 150)
    if null_ms <= 0.5 * q:
        return f"null {null_ms:.0f} ms  (ok)", (140, 245, 140)
    if null_ms <= q:
        return f"null {null_ms:.0f} ms  (marginal)", (90, 200, 255)
    return f"null {null_ms:.0f} ms  (timing unreliable)", (80, 165, 235)
