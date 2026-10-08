"""rppg_cam.py -- one place that knows how to open a webcam on this machine.

Why this exists
---------------
Every capture script here opened the camera with ``cv2.VideoCapture(cam, cv2.CAP_DSHOW)``.
DirectShow is a Windows-only backend: on macOS it does not fail loudly, it returns a capture
object whose ``isOpened()`` is False, so the scripts died with "cannot open camera" on a machine
whose camera was fine. Measured on an M1 Pro: CAP_DSHOW -> isOpened False, CAP_AVFOUNDATION ->
isOpened True, 29.8 fps.

Backend choice is therefore per-platform, and the requested frame rate is NOT assumed to be the
delivered one -- ``open_camera`` returns the rate it actually measures, because every timing
claim downstream depends on the real interval between frames rather than on what the driver
reports. (The FaceTime camera reports 30 and delivers 29.8; other cameras report 60 and deliver
30, which is the discrepancy the original module docstrings ran into.)
"""
import sys
import time

BACKENDS = {"darwin": "CAP_AVFOUNDATION",   # macOS: AVFoundation
            "win32": "CAP_DSHOW",           # Windows: DirectShow
            "linux": "CAP_V4L2"}            # Linux: Video4Linux2


def open_camera(cam=0, width=640, height=480, fps=60):
    """Open `cam` with the backend this platform actually supports.

    Returns the cv2.VideoCapture. Raises RuntimeError if no backend yields a readable frame,
    which on macOS is usually a camera-permission problem rather than a code fault -- grant the
    terminal/IDE camera access in System Settings > Privacy & Security > Camera.
    """
    import cv2
    name = BACKENDS.get(sys.platform)
    order = ([getattr(cv2, name)] if name and hasattr(cv2, name) else []) + [0]  # 0 = auto
    last = None
    for backend in order:
        cap = cv2.VideoCapture(cam, backend) if backend else cv2.VideoCapture(cam)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, _ = cap.read()
            if ok:
                return cap
            last = "opened but returned no frame"
        else:
            last = "isOpened() False"
        cap.release()
    raise RuntimeError(
        f"cannot open camera {cam} ({last}). On macOS grant camera access to your terminal in "
        "System Settings > Privacy & Security > Camera.")


def measure_fps(cap, seconds=3.0):
    """Delivered frame rate and per-frame interval stats -- what the timing actually rests on.

    Timestamps are frame-ARRIVAL times, not exposure times, so the jitter reported here includes
    USB transfer and buffering. That jitter is common to every ROI in the same frame, so it
    cancels in a two-site lag; it is reported to judge heart-rate quality, not transit timing.
    """
    import numpy as np
    T = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            T.append(time.time() - t0)
    if len(T) < 10:
        raise RuntimeError(f"only {len(T)} frames in {seconds}s")
    T = np.array(T)
    dt = np.diff(T) * 1000.0
    return {"fps": (len(T) - 1) / (T[-1] - T[0]), "quantum_ms": float(np.median(dt)),
            "jitter_sd_ms": float(dt.std()), "n_frames": len(T)}


if __name__ == "__main__":
    cap = open_camera()
    try:
        print(f"backend ok for {sys.platform}")
        for k, v in measure_fps(cap).items():
            print(f"  {k:12s} {v}")
    finally:
        cap.release()
