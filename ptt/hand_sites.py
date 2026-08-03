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


GROUPS = ("tip", "dip")
SPEC = {"palm": (PALM, 70.0), "dip": (DIPS, 74.0), "tip": (TIPS, 77.0)}


def schema(groups=GROUPS):
    """Canonical (segment, distance) for every fingertip SLOT, independent of detection.

    Fixed length by construction, mirroring rppg_pose.schema(). The capture loop builds one row
    per frame and indexes it with masks derived from the schema, so a row whose width depends on
    whether a hand happened to be detected would desynchronise those masks -- which is exactly
    the IndexError the first version produced.
    """
    out = []
    for g in groups:
        idxs, dd = SPEC[g]
        out += [(f"finger_{g}", dd)] * len(idxs)
    return out


def hand_points(result, w, h, groups=GROUPS):
    """Pixel coordinates for the requested finger groups, with their distance labels.

    Distances continue the pose chain: the pose 'hand' segment ends at 70 cm (wrist to index
    MCP), so the palm sits at 70, the distal phalanges at 74 and the tips at 77. Those are
    nominal adult values -- what matters for the PWV fit is that they are ordered and roughly
    correct, since the slope is what carries the physiology.
    """
    sch = schema(groups)
    if not result.hand_landmarks:
        # None rather than an empty list: the row must keep its width so the schema-derived
        # masks stay aligned. A frame with no hand contributes nan at these slots.
        return [None] * len(sch), np.array([d for _, d in sch]), [s for s, _ in sch]
    lms = result.hand_landmarks[0]
    pts = []
    for g in groups:
        idxs, _ = SPEC[g]
        for i in idxs:
            pts.append((int(lms[i].x * w), int(lms[i].y * h)) if i < len(lms) else None)
    return pts, np.array([d for _, d in sch]), [s for s, _ in sch]


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


def arm_path_cm(world_lms, side="right"):
    """Length of the arterial path from the shoulder to the index fingertip, in cm.

    MediaPipe returns pose_world_landmarks in METRES, roughly hip-centred, so segment lengths
    are metric without any calibration object in the scene. Summing shoulder->elbow->wrist
    gives this subject's own arm rather than a nominal adult value, which matters because arm
    length varies by 20% across adults and enters the pulse-wave-velocity estimate linearly.

    The hand beyond the wrist is added as a fixed 18 cm: pose has no reliable finger landmarks,
    and wrist-to-fingertip varies far less between adults than the arm does.

    Returns nan when the landmarks are missing or implausible, so a bad frame drops out rather
    than contributing a wrong length.
    """
    idx = {"right": (12, 14, 16), "left": (11, 13, 15)}[side]
    try:
        pts = [np.array([world_lms[i].x, world_lms[i].y, world_lms[i].z]) for i in idx]
    except (IndexError, TypeError):
        return float("nan")
    upper = float(np.linalg.norm(pts[1] - pts[0]))
    fore = float(np.linalg.norm(pts[2] - pts[1]))
    if not (0.15 < upper < 0.50 and 0.15 < fore < 0.45):     # metres; anything else is a bad fit
        return float("nan")
    return (upper + fore) * 100.0 + 18.0


def head_to_hand_cm(world_lms, side="right"):
    """Face-to-fingertip path length: neck to shoulder, then down the arm.

    The proximal reference is the face, so the path the pulse takes from there to the fingertip
    runs back down the neck before it reaches the shoulder. Neck length is taken as the distance
    from the shoulder midpoint to the ear, which pose does provide.
    """
    arm = arm_path_cm(world_lms, side)
    if not np.isfinite(arm):
        return float("nan")
    try:
        sh = (np.array([world_lms[11].x, world_lms[11].y, world_lms[11].z])
              + np.array([world_lms[12].x, world_lms[12].y, world_lms[12].z])) / 2.0
        ear = np.array([world_lms[8].x, world_lms[8].y, world_lms[8].z])
        neck = float(np.linalg.norm(ear - sh)) * 100.0
    except (IndexError, TypeError):
        return float("nan")
    if not (5.0 < neck < 40.0):
        return float("nan")
    return neck + arm
