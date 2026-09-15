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


def hand_side(hand_result, pose_lms, w, h):
    """Which arm the detected hand belongs to: 'left', 'right', or None.

    Decided by matching the hand's WRIST to the nearer of the two pose wrists (landmark 15 left,
    16 right) in pixels -- NOT by MediaPipe's handedness flag. That flag is documented as
    assuming a mirrored, selfie-view input, while this pipeline feeds the raw unflipped frame, so
    reading it directly gives the wrong side and reading it inverted breaks the moment someone
    adds a flip to the preview. Nearest-wrist is invariant to that whole question.

    Why it matters: the path length is measured along ONE arm. Sampling the left fingertips while
    measuring the right arm's length silently mixes two limbs, and if the unsampled arm is
    partly occluded its world landmarks can still return a plausible-looking length -- a wrong
    number with nothing on screen to flag it. Arm lengths are near-symmetric so the error is
    small, but it is avoidable and it is free to avoid.
    """
    if not getattr(hand_result, "hand_landmarks", None) or pose_lms is None:
        return None
    try:
        wrist = hand_result.hand_landmarks[0][0]
        hx, hy = wrist.x * w, wrist.y * h
        d = {}
        for side, i in (("left", 15), ("right", 16)):
            p = pose_lms[i]
            d[side] = (p.x * w - hx) ** 2 + (p.y * h - hy) ** 2
    except (IndexError, TypeError, AttributeError):
        return None
    side = min(d, key=d.get)
    # A hand that is far from BOTH pose wrists is not a match for either; better to report no
    # side and keep the previous path than to attach the fingertips to an arbitrary arm.
    return side if d[side] <= (0.25 * max(w, h)) ** 2 else None


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


def differential_path_cm(world_lms, side="right"):
    """The path length the face-to-fingertip LAG actually corresponds to, in cm.

    This is the correction head_to_hand_cm does not make, and it matters for any wave speed.

    The pulse does not travel from the face to the hand. It leaves the heart once and arrives at
    both sites independently, so the measured lag is a DIFFERENCE of two arrival times:

        lag = (heart -> fingertip) / PWV_arm  -  (heart -> face) / PWV_head

    Dividing the anatomical face-to-hand distance by that lag therefore asks the wave to cover a
    route it never took, and inflates the speed roughly twofold -- which is why the panel reported
    ~25 m/s against a physiological 4-12. The right numerator is the DIFFERENCE in path lengths:

        (shoulder-midpoint -> fingertip)  -  (shoulder-midpoint -> ear)

    The shoulder midpoint stands in for the aortic arch. It is a few cm high and slightly
    anterior, but it is the same reference on BOTH sides of the subtraction, so most of the error
    cancels rather than accumulating.

    One honest caveat this cannot fix: the two routes have different stiffnesses -- the head route
    is largely elastic aorta and carotid (~5-7 m/s), the arm route muscular brachial and radial
    (~8-12 m/s) -- so a single PWV from this ratio is a path-weighted blend, not the arm's own
    wave speed. Treat it as an index that should MOVE with pressure, not as a regional PWV.

    Returns nan when the landmarks are missing or implausible.
    """
    arm = arm_path_cm(world_lms, side)
    if not np.isfinite(arm):
        return float("nan")
    try:
        sh_l = np.array([world_lms[11].x, world_lms[11].y, world_lms[11].z])
        sh_r = np.array([world_lms[12].x, world_lms[12].y, world_lms[12].z])
        ear = np.array([world_lms[8].x, world_lms[8].y, world_lms[8].z])
    except (IndexError, TypeError):
        return float("nan")
    mid = (sh_l + sh_r) / 2.0
    # The arm chain starts at the shoulder JOINT, so reaching it from the midpoint adds half the
    # biacromial width. Without this the hand route is short by ~18 cm while the head route is not.
    half_shoulder = float(np.linalg.norm(sh_r - sh_l)) / 2.0 * 100.0
    to_face = float(np.linalg.norm(ear - mid)) * 100.0
    if not (5.0 < half_shoulder < 30.0 and 5.0 < to_face < 40.0):
        return float("nan")
    d = half_shoulder + arm - to_face
    return d if 15.0 < d < 110.0 else float("nan")
