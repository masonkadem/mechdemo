"""app_ptt.py -- desktop console for camera pulse transit time.

    python app_ptt.py          (or double-click run_app.command)

Everything the terminal scripts do, with the state visible before it costs you a recording.
Capture runs on a worker thread so the interface stays responsive; Qt objects are never touched
from that thread -- frames and metrics cross back as signals, which is the only thread-safe way
to drive widgets from a capture loop.

The layout follows the order the work actually happens in:

  Capture   arm the shot, watch the sites lock, confirm a pulse, THEN record
  Results   the three plots that decide whether the measurement is real

Design bias worth stating: this interface is built to make a bad recording obvious, not to look
confident. The transit number is shown with its spread, and greyed out when the spread exceeds
the value, because at 30 fps face-to-hand transit sits under one frame and only the CHANGE
between conditions carries evidence.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

import bp_model as BP
import rppg_methods as MTH
# Imported at module level, not inside Worker.run(), because the UI needs the gate thresholds to
# build the quality-profile controls -- the worker's local `import rppg_pose as P` left the
# window unable to see them. Cheap: rppg_methods already pulls it in.
import rppg_pose as P

ROOT = Path(__file__).resolve().parent
DATA, FIGS = ROOT / "data", ROOT / "figures"

CONDITIONS = [
    ("rest",      "Hand at heart level. Record this first, and record it twice."),
    ("rest2",     "A REPEAT of rest. The gap between the two rests is this rig's noise floor."),
    ("hand_up",   "Hand 30-40 cm ABOVE the heart.  Prediction: transit LENGTHENS."),
    ("hand_down", "Hand hanging BELOW the heart.   Prediction: transit SHORTENS."),
    ("post_exer", "Straight after ~30 s of effort.  Prediction: transit SHORTENS."),
]

CSS = """
QWidget       { background:#16181d; color:#e8e8ea; font-size:13px; }
QGroupBox     { border:1px solid #2c3038; border-radius:8px; margin-top:16px; padding-top:10px;
                font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#9aa0a6; }
QPushButton   { background:#2a2f38; border:1px solid #3a414d; border-radius:7px; padding:9px 16px; }
QPushButton:hover:!disabled { background:#333944; }
QPushButton:disabled { color:#6b7280; border-color:#2a2f38; }
QPushButton#rec  { background:#2e7d46; border-color:#3c9c58; font-weight:700; }
QPushButton#rec:hover { background:#359b53; }
QPushButton#stop { background:#a3342a; border-color:#c2453a; font-weight:700; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background:#1e2229; border:1px solid #333a45;
                      border-radius:6px; padding:6px 8px; }
QLabel#hint   { color:#9aa0a6; }
QLabel#big    { font-size:34px; font-weight:700; }
QLabel#unit   { color:#9aa0a6; }
QPushButton#sync { background:#1d3a5c; border-color:#2f6096; font-weight:700; font-size:15px;
                   padding:14px 16px; }
QPushButton#sync:hover { background:#24487093; }
QFrame#card   { background:#1b1f26; border:1px solid #2c3038; border-radius:8px; }
QLabel#rlabel { color:#8a9099; font-size:10px; font-weight:600; }
QLabel#rval   { font-size:27px; font-weight:700; color:#e8e8ea; }
QLabel#rvaldim{ font-size:27px; font-weight:700; color:#5f6570; }
QLabel#rsub   { color:#8a9099; font-size:10px; }
QListWidget   { background:#1e2229; border:1px solid #333a45; border-radius:6px; font-size:11px; }
QTabBar::tab  { background:#1b1f26; padding:9px 20px; border-top-left-radius:7px;
                border-top-right-radius:7px; }
QTabBar::tab:selected { background:#262c36; }
QTabWidget::pane { border:1px solid #2c3038; border-radius:8px; }
QProgressBar  { background:#1e2229; border:1px solid #333a45; border-radius:6px; height:8px;
                text-align:center; }
QProgressBar::chunk { background:#3c9c58; border-radius:5px; }
"""


class Chip(QtWidgets.QLabel):
    """Per-site lock indicator: lit when that segment currently has visible landmarks."""

    def __init__(self, name, label=None):
        super().__init__(label or name)
        self.name = name                    # the schema segment, which may differ from the label
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFixedHeight(24)
        self.set_on(False)

    def set_on(self, on):
        self.setStyleSheet(
            "border-radius:11px; padding:2px 11px; font-size:11px; "
            + ("background:#1d4a2c; color:#8fe0a6; border:1px solid #2e7d46;" if on
               else "background:#22262e; color:#606673; border:1px solid #2c3038;"))


class Readout(QtWidgets.QFrame):
    """One live number, its unit, and a sub-line for spread or provenance.

    The sub-line is not decoration. Every number this rig produces is either qualified by its
    scatter or not worth reading, so there is nowhere in this widget to display a bare value --
    and `dim` greys the number out whenever the qualifier says it cannot be trusted.
    """

    def __init__(self, label, unit, sub="--"):
        super().__init__()
        self.setObjectName("card")
        self.setMinimumWidth(148)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(13, 9, 13, 9)
        lay.setSpacing(1)
        self._lab = QtWidgets.QLabel(label.upper()); self._lab.setObjectName("rlabel")
        row = QtWidgets.QHBoxLayout(); row.setSpacing(4); row.setContentsMargins(0, 0, 0, 0)
        self._val = QtWidgets.QLabel("--"); self._val.setObjectName("rval")
        self._unit = QtWidgets.QLabel(unit); self._unit.setObjectName("unit")
        self._unit.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        row.addWidget(self._val); row.addWidget(self._unit); row.addStretch()
        self._sub = QtWidgets.QLabel(sub); self._sub.setObjectName("rsub")
        self._sub.setWordWrap(True); self._sub.setMinimumHeight(26)
        lay.addWidget(self._lab); lay.addLayout(row); lay.addWidget(self._sub)

    def set(self, value, sub=None, dim=False):
        self._val.setText(value)
        self._val.setObjectName("rvaldim" if dim else "rval")
        # Re-polish, or the objectName swap does not repaint: Qt resolves stylesheet rules when
        # the widget is polished, not on every paint.
        self._val.style().unpolish(self._val); self._val.style().polish(self._val)
        if sub is not None:
            self._sub.setText(sub)


class Worker(QtCore.QThread):
    """Runs the capture pipeline. Emits frames and metrics; never touches widgets."""

    frame = QtCore.Signal(object, object, float, int, int)   # bgr, live-panel bgr, el, kept, seen
    metrics = QtCore.Signal(dict)                            # live HR / PTT / path, for readouts
    status = QtCore.Signal(str)
    finished_run = QtCore.Signal(str, bool, str)             # tag, saved, message

    def __init__(self, seconds, tag, session=None, parent=None):
        super().__init__(parent)
        self.seconds, self.tag = seconds, tag
        self.session = session
        self.subject = ""
        self._recording = False
        self._abort = False
        self.sites = set()
        # Wall clock at which the capture loop's own timebase starts, and the wall clock at which
        # RECORDING starts. Sync marks are stamped by the GUI thread against time.time(), so these
        # are what convert a mark into an offset into the saved data. Written once by run()
        # before any mark can plausibly arrive, and only ever read afterwards.
        self.t_wall = float("nan")
        self.t_rec_wall = float("nan")
        self.path_manual = float("nan")
        self.method = MTH.DEFAULT
        self.propagation = True
        self.side = "right"                 # updated per frame from which wrist the hand matches
        self.profile = P.PROFILE
        self.live = {}

    def start_recording(self):
        # Stamped here rather than in the loop so the anchor is the instant the operator asked
        # for, not the top of the next frame up to 33 ms later.
        self.t_rec_wall = time.time()
        self._recording = True

    def abort(self):
        self._abort = True

    def run(self):
        try:
            import cv2
            import rppg_cam, rppg_multi as M, rppg_live as LIVE
            import rppg_pose as P
        except Exception as e:                      # noqa: BLE001
            self.finished_run.emit(self.tag, False, f"import failed: {e}")
            return
        try:
            import hand_sites as HS
            lmk, mp = P.make_landmarker()
            # Second model for the distal site. Pose landmark 20 is the index KNUCKLE, so
            # sampling "hand" from pose alone misses the fingertips entirely -- the densest
            # capillary bed on the body and the strongest rPPG signal available.
            try:
                hlm, _ = HS.make_hand_landmarker()
            except Exception as e:                  # noqa: BLE001
                # Do NOT fail silently. A bare `except: hlm = None` here meant a missing
                # hand_landmarker.task -- which is gitignored, so absent on every fresh clone --
                # quietly dropped the fingertip sites, the strongest signal in the whole rig,
                # with nothing on screen to say why the distal trace never appeared.
                hlm = None
                self.status.emit(f"fingertips unavailable: {e}")
            cap = rppg_cam.open_camera(0, 640, 480, 60)
        except Exception as e:                      # noqa: BLE001
            self.finished_run.emit(self.tag, False, str(e))
            return

        SCHEMA = P.schema()
        # Fingertip slots extend the schema, so seg/dist/PM/DM cover every column of a row.
        # Appending the tips to `pts` without extending the schema made the rows 34 wide while
        # the masks stayed 24, which raised an IndexError on the first frame with a hand in it.
        # The slots are fixed-length whether or not a hand is detected: a missing hand fills them
        # with nan rather than shortening the row.
        HAND_SCHEMA = HS.schema()
        seg = np.array([s for s, _ in SCHEMA] + [s for s, _ in HAND_SCHEMA])
        dist = np.array([d for _, d in SCHEMA] + [d for _, d in HAND_SCHEMA])
        PM = np.isin(seg, P.PROXIMAL)
        DM = np.isin(seg, P.DISTAL) | np.char.startswith(seg.astype(str), "finger_")
        panel = LIVE.LivePanel()
        acc, T = [], []
        t_wall = time.time(); t0 = t_wall
        self.t_wall = t_wall
        nseen = nkept = 0

        while not self._abort:
            if self._recording and time.time() - t0 >= self.seconds:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            if not self._recording:
                t0 = time.time()
            el = time.time() - t0
            h, w = frame.shape[:2]
            res = lmk.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame[:, :, ::-1])), int((time.time()-t_wall)*1000))
            pts = P.sample_points(res.pose_landmarks[0], w, h) if res.pose_landmarks \
                else [None] * len(SCHEMA)
            # Fingertips are appended AFTER the fixed pose schema, so the schema length that the
            # capture loop keys on is unchanged and no frame is dropped for a count mismatch.
            tip_pts = [None] * len(HAND_SCHEMA)
            if hlm is not None:
                hres = hlm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame[:, :, ::-1])),
                    int((time.time() - t_wall) * 1000))
                tip_pts, _td, _ts = HS.hand_points(hres, w, h)
                # Measure the path along the arm the fingertips actually belong to, instead of
                # assuming the right one. Sticky: a frame that cannot decide keeps the last
                # decision rather than flipping the path length mid-recording.
                sd = HS.hand_side(hres, res.pose_landmarks[0] if res.pose_landmarks else None,
                                  w, h)
                if sd:
                    self.side = sd
            pts = list(pts) + list(tip_pts)
            # Path length from pose world landmarks (metres), so the wave speed uses THIS
            # subject's arm rather than a nominal one -- arm length varies about 20% across
            # adults and enters the velocity linearly.
            panel.path_manual = self.path_manual
            if res.pose_world_landmarks:
                # The DIFFERENTIAL path, not the anatomical face-to-hand route: the lag is a
                # difference of two arrival times from the heart, so dividing the full
                # face-to-hand distance by it inflated the wave speed to ~25 m/s.
                panel.set_path(HS.differential_path_cm(res.pose_world_landmarks[0], self.side))
            elif np.isfinite(self.path_manual):
                panel.set_path(None)               # let a manual value stand with no pose fit
            vis_pts = [p for p in pts if p is not None]

            if vis_pts:
                row, r = [], 9
                for p in pts:
                    if p is None:
                        row.append((np.nan,)*3); continue
                    x, y = p
                    y0, y1 = max(0, y-r), min(h, y+r); x0, x1 = max(0, x-r), min(w, x+r)
                    pa = frame[y0:y1, x0:x1]
                    # No skin mask. It cost 23 ms of a 33 ms frame budget -- more than both
                    # landmarkers combined, dominated by an arctan2 over every pixel -- while
                    # the patches are placed by pose and hand landmarks, so the mask was only
                    # deciding which pixels INSIDE an already-anatomical patch to average. The
                    # per-patch checks below do the part that mattered: reject a patch that is
                    # clipped, black, or too uniform to be skin, all on 18x18 pixels rather
                    # than 307k.
                    if pa.size == 0:
                        row.append((np.nan,)*3); continue
                    px = pa.reshape(-1, 3).astype(np.float32)
                    v = px.max(1)
                    keep = (v > 12) & (v < 250)
                    if keep.mean() < .5:
                        row.append((np.nan,)*3); continue
                    row.append(tuple(px[keep][:, ::-1].mean(0)))
                if self._recording:
                    acc.append(row); T.append(el); nkept += 1
                A = np.array(row, float)
                # An all-nan group is the normal case, not an error: the hand is simply out of
                # frame or unlit, and the panel is built to show that. np.errstate does not cover
                # it -- nanmean raises a RuntimeWarning through the warnings module, so a hand
                # off camera spammed stderr once per frame.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pr = np.nanmean(A[PM], 0) if PM.any() else np.full(3, np.nan)
                    ds = np.nanmean(A[DM], 0) if DM.any() else np.full(3, np.nan)
                tnow = time.time() - t_wall
                panel.push(pr, ds, tnow)
                if self.propagation:
                    panel.push_all(row, tnow, dist, seg, self.method)
            if self._recording:
                nseen += 1

            vis = frame.copy()
            live = set()
            # Ringed markers rather than filled discs: a 4 px solid dot on a dim background
            # reads as speckle, and at 24 patches the frame looked noisy. A dark outline plus a
            # small bright core stays legible over both skin and shadow.
            # seg covers the pose schema only; the appended fingertips extend past it, and a
            # plain zip would silently drop them from the overlay while still sampling them.
            seg_all = list(seg) + ["finger"] * (len(pts) - len(seg))
            lagmap = panel.site_lag if self.propagation else {}
            span = max((abs(v) for v in lagmap.values()), default=0.0)
            for i, (p, s) in enumerate(zip(pts, seg_all)):
                if p is not None:
                    if i in lagmap and span > 1e-6:
                        # Colour BY ARRIVAL TIME, not by anatomy: green at the reference through
                        # to red at the latest patch. A real pulse paints a smooth
                        # proximal-to-distal gradient; noise paints confetti, and that is the
                        # whole point of showing it.
                        u = float(np.clip(lagmap[i] / span, 0.0, 1.0))
                        col = (int(60 + 40 * (1 - u)), int(245 * (1 - u) + 60 * u),
                               int(90 * (1 - u) + 245 * u))
                    else:
                        col = (90, 200, 255) if (s in P.DISTAL or s.startswith("finger")) \
                            else (140, 245, 140)
                    cv2.circle(vis, p, 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(vis, p, 5, col, 1, cv2.LINE_AA)
                    cv2.circle(vis, p, 1, col, -1, cv2.LINE_AA)
                    live.add(s)
            if self.propagation and panel.site_fit:
                sf = panel.site_fit
                good = abs(sf.get("r", 0.0)) >= 0.5 and 4.0 <= sf.get("pwv_ms", 0) <= 12.0
                cv2.putText(vis, f"arrival vs distance: r={sf['r']:+.2f}  "
                            f"{sf['pwv_ms']:.1f} m/s  n={sf['n']}", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .48,
                            (140, 245, 140) if good else (90, 165, 235), 1, cv2.LINE_AA)
            self.sites = live
            self.frame.emit(vis, panel.render(h), el, nkept, nseen)
            # The medians, not the instantaneous values: a single window's lag is sub-frame noise
            # at 30 fps, and the readout must show the quantity the model is actually fed.
            self.live = {
                "hr": panel.hr,
                "ptt_ms": float(np.median(panel.hist)) if panel.hist else float("nan"),
                "spread_ms": float(np.std(panel.hist)) if len(panel.hist) >= 3 else float("nan"),
                "null_ms": float(np.median(panel.null_hist)) if panel.null_hist else float("nan"),
                "path_cm": panel.path_cm,
                "fs": panel.fs,
                "n_lag": len(panel.hist),
                "prop_r": panel.site_fit.get("r", float("nan")),
                "prop_pwv": panel.site_fit.get("pwv_ms", float("nan")),
                "prop_n": panel.site_fit.get("n", 0),
            }
            self.metrics.emit(self.live)

        cap.release()
        if self._abort and not acc:
            self.finished_run.emit(self.tag, False, "cancelled"); return
        if len(T) < 100:
            self.finished_run.emit(self.tag, False,
                                   f"only {len(T)} usable frames -- nothing saved"); return
        self.status.emit("analysing ...")
        try:
            msg, saved = self._analyse(np.array(acc, float), np.array(T), dist, seg)
        except Exception as e:                      # noqa: BLE001
            msg, saved = f"analysis failed: {e}", False
        self.finished_run.emit(self.tag, saved, msg)

    @staticmethod
    def _iso(t, ms=False):
        """Local clock time, or None. Blank-safe so a missing anchor cannot invent a time."""
        if t is None or not np.isfinite(t):
            return None
        # Round to milliseconds FIRST, then split. Rounding the fraction separately is what
        # creates the .9996 trap: it rounds up to 1000, has to be wrapped to 000 to stay
        # three digits, and the seconds field never learns it should have advanced -- a
        # one-second error, in the one component that is hardest to notice is wrong. Rounding
        # the whole timestamp lets localtime carry into the next second by itself.
        #
        # Rounded rather than truncated so this agrees exactly with _stem() and
        # export_csv._clock_ms(); truncating here made the filename say ...-527 while
        # clock_record said ...526 for the same instant.
        t = round(float(t), 3)
        s = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))
        return s + f".{int(round((t % 1) * 1000)):03d}" if ms else s

    def _stem(self):
        """Filename stem: subject, condition, and the WALL-CLOCK TIME the recording started.

        The timestamp is the date and local time of day, e.g. s01_rest_20260915-133045, because
        the first thing you need when reconciling this against a cuff log, a lab notebook or
        another instrument is "which recording was running at 13:30". Reading that off a unix
        float inside the json is possible but not something you can do while a subject waits.

        It also removes the two silent-overwrite paths the old `rppg_pose_{tag}` naming had --
        subject 2's 'rest' replacing subject 1's, and a re-take destroying the original -- since
        two recordings cannot start in the same second. The take counter stays as a backstop for
        exactly that case rather than as the main mechanism.

        Local time, not UTC: it has to match the wall clock on the lab wall. The unix timestamps
        in the json remain the unambiguous record for anything automated.
        """
        t = self.t_rec_wall if np.isfinite(self.t_rec_wall) else time.time()
        # Milliseconds included: two recordings started in the same second would otherwise
        # collide, and more importantly the filename is then the same instant, to the same
        # resolution, as the sync marks and the per-sample timestamps in the export.
        t = round(float(t), 3)          # round before splitting; see _iso for why
        when = (time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
                + f"-{int(round((t % 1) * 1000)):03d}")
        base = f"{self.subject}_{self.tag}_{when}" if self.subject else f"{self.tag}_{when}"
        if not (DATA / f"rppg_pose_{base}.json").exists():
            return base
        k = 2
        while (DATA / f"rppg_pose_{base}_take{k}.json").exists():
            k += 1
        return f"{base}_take{k}"

    def _analyse(self, acc, T, dist, seg):
        """Same pipeline as rppg_pose.main, reused so the app and CLI cannot diverge."""
        import rppg_pose as P
        import rppg_two_site as R
        import json
        fs = (len(T) - 1) / (T[-1] - T[0])
        keep = T >= P.WARMUP_S
        acc, T = acc[keep], T[keep]
        tu = np.linspace(T[0], T[-1], len(T))
        filtered, quals, hrs, stages_all = [], [], [], {}
        for i in range(acc.shape[1]):
            rgb = acc[:, i, :]; good = np.isfinite(rgb).all(1)
            if good.mean() < P.MIN_FRAME_FRAC:
                filtered.append(None); quals.append(0.); hrs.append(np.nan); continue
            fill = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
            bp = MTH.extract(fill, fs, self.method)
            if bp is None:
                filtered.append(None); quals.append(0.); hrs.append(np.nan); continue
            ch = R.chrom(fill)          # kept for the stage figure, whatever method was used
            stages_all[i] = {"raw": fill[:, 1],
                             "detrended": P.detrend(fill[:, 1], fs), "chrom": ch, "filtered": bp}
            _, hr, qm, _ = P.plausible(bp, fs)
            filtered.append(bp); quals.append(qm); hrs.append(hr)
        quals, hrs = np.array(quals), np.array(hrs)
        # Shared helper, so this cannot drift from the CLI. It returns None -- not nan -- when no
        # site clears the SNR gate, because nan propagated into the agreement gate rejected every
        # site including flawless ones.
        cons = P.consensus_hr(hrs, quals)
        ok = np.zeros(len(filtered), bool)
        ok_strict = np.zeros(len(filtered), bool)
        details = []
        for i, x in enumerate(filtered):
            if x is None:
                details.append({"n": 0, "snr": 0.0, "frac": 0.0, "hr": np.nan,
                                "ibi_sd": np.nan, "fail": ["short"]})
                continue
            ok[i], _, _, _ = P.plausible(x, fs, cons)
            d = P.gate_detail(x, fs, cons)
            details.append(d)
            # The strict verdict is recorded even under a relaxed profile, so a permissive
            # recording can still be analysed honestly afterwards instead of having to be redone.
            ok_strict[i] = P.passes_strict(d["hr"], d["snr"], d["frac"], d["ibi_sd"], cons)
        if ok.sum() < P.MIN_SITES:
            _, lines = P.gate_report(details, np.asarray(seg).astype(str), P.MIN_SITES)
            return "<br>".join(lines), False
        cons = cons if cons is not None else float("nan")
        out = {"tag": self.tag, "fps": fs, "n_frames": len(T), "n_points": int(acc.shape[1]),
               "n_accepted": int(ok.sum()), "consensus_hr": cons,
               "gate_profile": P.PROFILE, "n_accepted_strict": int(ok_strict.sum()),
               # The thresholds actually in force, not just their preset name: the SNR gate can
               # be dialled at the bench, and a preset name alone would not recover the number.
               "gates": {"min_snr": P.MIN_SNR, "min_peak_frac": P.MIN_PEAK_FRAC,
                         "hr_tol_bpm": P.HR_TOL_BPM, "ibi_sd_max_ms": P.IBI_SD_MAX_MS,
                         "min_sites": P.MIN_SITES, "min_frame_frac": P.MIN_FRAME_FRAC},
               "method": self.method,
               "path_cm": self.live.get("path_cm", float("nan")),
               "t_wall_capture": self.t_wall, "t_wall_record": self.t_rec_wall,
               # Local clock times beside the unix floats: these are what you actually compare
               # against a cuff printout or a lab notebook. Milliseconds kept on the record
               # start, since that is the instant every sync mark is measured from.
               "clock_record": self._iso(self.t_rec_wall, ms=True),
               "clock_end": self._iso(self.t_rec_wall + float(T[-1] - T[0])
                                      if np.isfinite(self.t_rec_wall) else float("nan")),
               "clock_capture": self._iso(self.t_wall),
               "duration_s": round(float(T[-1] - T[0]), 3),
               "timezone": time.strftime("%Z%z")}
        # Marks and cuff readings are re-expressed as offsets into THIS recording's own clock, so
        # the saved file can be aligned offline without also needing the session file. They are
        # copied, not moved: the session file remains the authority.
        if self.session is not None:
            t_rec = self.t_rec_wall
            def _off(ev):
                e = dict(ev)
                e["t_in_record_s"] = (round(ev["t_wall"] - t_rec, 4)
                                      if np.isfinite(t_rec) else None)
                return e
            out["marks"] = [_off(m) for m in self.session.marks]
            out["calib"] = [_off(c) for c in self.session.calib]
            out["session"] = self.session.name
        out["subject"] = self.subject
        stem = self._stem()
        out["file"] = f"rppg_pose_{stem}"
        (DATA / f"rppg_pose_{stem}.json").write_text(json.dumps(out, indent=2, default=float))
        best = int(np.argmax(quals))
        np.savez(DATA / f"rppg_pose_{stem}.npz",
                 sigs=np.stack([x if x is not None else np.zeros(len(tu)) for x in filtered]),
                 accepted=ok, accepted_strict=ok_strict, gate_profile=P.PROFILE,
                 quals=quals, hrs=hrs, dist=dist, seg=np.asarray(seg), fs=fs,
                 # The per-channel means BEFORE any extraction, plus their timestamps. Without
                 # these the npz holds only one method's output, so a recording could never be
                 # re-analysed with POS or ICA afterwards -- a whole session would have to be
                 # repeated to change that choice. ~1.5 MB for a 60 s run, which is nothing.
                 raw_rgb=acc, t=T, tu=tu, method=self.method,
                 # Wall-clock anchors travel WITH the signals, not only in the sibling json, so
                 # an exporter can put an absolute millisecond timestamp on every sample without
                 # having to find and parse a second file.
                 t_wall_capture=self.t_wall, t_wall_record=self.t_rec_wall,
                 warmup_s=P.WARMUP_S,
                 **stages_all.get(best, {}))
        msg = (f"saved as <b>rppg_pose_{stem}</b><br>HR {cons:.0f} bpm, "
               f"{ok.sum()}/{len(ok)} sites accepted at {fs:.0f} fps")
        if P.PROFILE != "strict":
            # Never let a permissive recording read like a clean one. The strict count is the
            # number that answers "is this a measurement"; the permissive count only answers
            # "did anything come through at all".
            msg += (f"<br><b>gate profile: {P.PROFILE}</b> -- only "
                    f"<b>{ok_strict.sum()}/{len(ok)}</b> sites would pass STRICT. "
                    f"Use this to inspect the signal, not to claim a result.")
        return msg, True


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Pulse Transit Time")
        self.resize(1340, 900)
        self.worker = None
        self.live = {}                       # last metrics dict from the worker
        self.session = BP.Session()
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._capture_tab(), "Capture")
        tabs.addTab(self._results_tab(), "Results")
        self.tabs = tabs
        self.setCentralWidget(tabs)

        # Space marks the cuff; ctrl+enter logs a reading. Both are application-wide so neither
        # depends on what happens to hold focus while you are looking at the subject, not the
        # screen. The buttons take no focus, so space cannot be captured by one of them.
        for keys, slot in ((QtCore.Qt.Key_Space, self._sync),
                           ("Ctrl+Return", self._log_bp)):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self).activated.connect(slot)

        self._refresh_cal()
        self.statusBar().showMessage(f"Ready -- session {self.session.name}")

    # ------------------------------------------------------------- capture tab
    def _capture_tab(self):
        page = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(page)

        self.video = QtWidgets.QLabel("Camera preview")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(640, 480)
        self.video.setStyleSheet("background:#0e1013; border-radius:8px; color:#5a6069;")
        self.panel = QtWidgets.QLabel()
        self.panel.setFixedWidth(430)
        self.panel.setAlignment(QtCore.Qt.AlignTop)
        self.panel.setStyleSheet("background:#0e1013; border-radius:8px;")
        left = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.video, 1); row.addWidget(self.panel)
        left.addLayout(row)

        self.chips = {}
        chiprow = QtWidgets.QHBoxLayout(); chiprow.setSpacing(6)
        # finger_tip earns a chip of its own: it is the strongest rPPG signal on the body and the
        # site the whole transit measurement depends on, so "is it locked" must be answerable
        # before a recording rather than from the panel's "fingertips not visible" afterwards.
        for s, lab in (("forehead", "forehead"), ("cheek_l", "cheek L"), ("cheek_r", "cheek R"),
                       ("forearm", "forearm"), ("hand", "hand"), ("finger_tip", "fingertips")):
            c = Chip(s, lab); self.chips[s] = c; chiprow.addWidget(c)
        chiprow.addStretch()
        left.addLayout(chiprow)

        # The four numbers the session is actually about, big enough to read from arm's length
        # while the subject is in the chair and you are holding a cuff bulb.
        self.ro = {
            "hr":   Readout("Heart rate", "bpm"),
            "ptt":  Readout("Face -> fingertips", "ms"),
            "pwv":  Readout("Wave speed", "m/s"),
            "bp":   Readout("Blood pressure", "mmHg"),
        }
        rr = QtWidgets.QHBoxLayout(); rr.setSpacing(9)
        for k in ("hr", "ptt", "pwv", "bp"):
            rr.addWidget(self.ro[k], 1)
        left.addLayout(rr)
        lay.addLayout(left, 1)

        side = QtWidgets.QVBoxLayout(); side.setSpacing(12)
        gb = QtWidgets.QGroupBox("Recording"); f = QtWidgets.QVBoxLayout(gb)
        # Subject id goes into every filename. Without it the condition alone named the file, so
        # the next subject's 'rest' overwrote the last one's.
        self.subject = QtWidgets.QLineEdit()
        self.subject.setPlaceholderText("e.g. s01   (used in every filename)")
        self.subject.setMaxLength(24)
        self.subject.editingFinished.connect(self._set_subject)
        f.addWidget(QtWidgets.QLabel("Subject")); f.addWidget(self.subject)
        self.cond = QtWidgets.QComboBox()
        for tag, _ in CONDITIONS:
            self.cond.addItem(tag)
        self.cond.currentIndexChanged.connect(self._hint)
        f.addWidget(QtWidgets.QLabel("Condition")); f.addWidget(self.cond)
        self.hint = QtWidgets.QLabel(); self.hint.setObjectName("hint")
        self.hint.setWordWrap(True); self.hint.setMinimumHeight(46)
        f.addWidget(self.hint)
        f.addSpacing(6)
        self.secs = QtWidgets.QSpinBox(); self.secs.setRange(15, 300); self.secs.setValue(60)
        self.secs.setSuffix("  seconds")
        f.addWidget(QtWidgets.QLabel("Duration")); f.addWidget(self.secs)
        f.addSpacing(6)
        self.meth = QtWidgets.QComboBox()
        for m in MTH.METHODS:
            self.meth.addItem(m)
        self.meth.setCurrentText(MTH.DEFAULT)
        self.meth.currentTextChanged.connect(self._set_method)
        self.meth.setToolTip(
            "pos   plane-orthogonal-to-skin (Wang 2017) -- projects out the specular component\n"
            "chrom chrominance ratio (de Haan 2013) -- what this pipeline used to use\n"
            "ica   FastICA on RGB (Poh 2010), best component by in-band SNR\n"
            "pca   orthogonal cousin of ICA; a useful disagreement check\n"
            "green raw green channel -- baseline only, fooled by in-band light flicker")
        f.addWidget(QtWidgets.QLabel("Extraction method")); f.addWidget(self.meth)
        f.addSpacing(6)
        self.prof = QtWidgets.QComboBox()
        for p in ("strict", "relaxed", "off"):
            self.prof.addItem(p)
        self.prof.setCurrentText(P.PROFILE)
        self.prof.currentTextChanged.connect(self._set_profile)
        self.prof.setToolTip(
            "strict   SNR>=6, 6 sites -- the only setting whose output is a measurement\n"
            "relaxed  SNR>=2.5, 3 sites -- see marginal signal while it is still recognisable\n"
            "off      SNR>=1.2, 1 site -- shows almost anything, INCLUDING NOISE (noise scores\n"
            "         1.9-2.7 on this metric), for checking lighting and ROI placement\n\n"
            "The strict verdict is saved either way, as accepted_strict in the npz.")
        f.addWidget(QtWidgets.QLabel("Quality gates")); f.addWidget(self.prof)
        srow = QtWidgets.QHBoxLayout()
        self.snr_box = QtWidgets.QDoubleSpinBox()
        self.snr_box.setRange(0.0, 60.0); self.snr_box.setDecimals(1)
        self.snr_box.setSingleStep(0.1); self.snr_box.setValue(P.MIN_SNR)
        self.snr_box.valueChanged.connect(self._set_snr)
        self.snr_box.setToolTip(
            "The SNR gate, directly. Picking a profile sets this; you can then dial it.\n"
            "For scale, measured on this pipeline: band-passed noise 1.9-2.7,\n"
            "weak-but-real pulse 2.6+, pulse in noise ~48, clean pulse ~440.\n"
            "The value used is saved with every recording.")
        srow.addWidget(QtWidgets.QLabel("min SNR")); srow.addWidget(self.snr_box)
        f.addLayout(srow)
        self.lbl_prof = QtWidgets.QLabel(); self.lbl_prof.setObjectName("hint")
        self.lbl_prof.setWordWrap(True); self.lbl_prof.setTextFormat(QtCore.Qt.RichText)
        f.addWidget(self.lbl_prof)
        # The raw per-channel means are saved with every recording, so this choice is not
        # destructive -- any recording can be re-extracted with any method afterwards.
        mnote = QtWidgets.QLabel(
            "Raw RGB is saved too, so this is reversible. Run "
            "<code>python rppg_methods.py</code> to rank the methods on your own recordings.")
        mnote.setObjectName("hint"); mnote.setWordWrap(True)
        mnote.setTextFormat(QtCore.Qt.RichText)
        f.addWidget(mnote)
        side.addWidget(gb)

        self.btn_prev = QtWidgets.QPushButton("Start camera preview")
        self.btn_prev.clicked.connect(self._preview)
        self.btn_rec = QtWidgets.QPushButton("Record"); self.btn_rec.setObjectName("rec")
        self.btn_rec.setEnabled(False); self.btn_rec.clicked.connect(self._record)
        self.btn_stop = QtWidgets.QPushButton("Stop"); self.btn_stop.setObjectName("stop")
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop)
        for b in (self.btn_prev, self.btn_rec, self.btn_stop):
            side.addWidget(b)
        self.prog = QtWidgets.QProgressBar(); self.prog.setRange(0, 100); self.prog.setValue(0)
        self.prog.setTextVisible(False)
        side.addWidget(self.prog)

        gb2 = QtWidgets.QGroupBox("Live"); g2 = QtWidgets.QGridLayout(gb2)
        self.lbl_state = QtWidgets.QLabel("idle"); self.lbl_state.setObjectName("hint")
        self.lbl_fps = QtWidgets.QLabel("-"); self.lbl_fps.setObjectName("hint")
        self.lbl_path = QtWidgets.QLabel("-"); self.lbl_path.setObjectName("hint")
        self.lbl_null = QtWidgets.QLabel("-"); self.lbl_null.setObjectName("hint")
        g2.addWidget(QtWidgets.QLabel("State"), 0, 0); g2.addWidget(self.lbl_state, 0, 1)
        g2.addWidget(QtWidgets.QLabel("Kept"), 1, 0);  g2.addWidget(self.lbl_fps, 1, 1)
        g2.addWidget(QtWidgets.QLabel("Path"), 2, 0);  g2.addWidget(self.lbl_path, 2, 1)
        g2.addWidget(QtWidgets.QLabel("Control"), 3, 0); g2.addWidget(self.lbl_null, 3, 1)
        self.lbl_prop = QtWidgets.QLabel("-"); self.lbl_prop.setObjectName("hint")
        self.lbl_prop.setWordWrap(True)
        g2.addWidget(QtWidgets.QLabel("Propagation"), 4, 0); g2.addWidget(self.lbl_prop, 4, 1)
        self.cb_prop = QtWidgets.QCheckBox("Colour patches by arrival time")
        self.cb_prop.setChecked(True)
        self.cb_prop.setToolTip(
            "Regress arrival time on arterial distance across all patches, every 0.4 s.\n"
            "A real pulse gives a smooth green->red proximal-to-distal gradient and r > 0.5;\n"
            "noise gives confetti. This is the check a single face-to-hand lag cannot provide.")
        self.cb_prop.toggled.connect(self._set_prop)
        g2.addWidget(self.cb_prop, 5, 0, 1, 2)
        side.addWidget(gb2)

        # ---------------------------------------------------------------- cuff synchronisation
        gs = QtWidgets.QGroupBox("Cuff sync"); fs_ = QtWidgets.QVBoxLayout(gs)
        self.btn_sync = QtWidgets.QPushButton("SYNC MARK\nspace")
        self.btn_sync.setObjectName("sync")
        self.btn_sync.setFocusPolicy(QtCore.Qt.NoFocus)   # else space re-triggers the button
        self.btn_sync.clicked.connect(self._sync)
        fs_.addWidget(self.btn_sync)
        self.lbl_marks = QtWidgets.QLabel("no marks yet"); self.lbl_marks.setObjectName("hint")
        self.lbl_marks.setWordWrap(True)
        fs_.addWidget(self.lbl_marks)
        sync_note = QtWidgets.QLabel(
            "Press as the cuff starts inflating. The stamp is taken in the keypress handler, so "
            "software adds well under a millisecond -- but <b>your reaction time does not</b>, "
            "and that is 200-300 ms. Press on the same cue every time: a repeatable offset "
            "cancels when you compare conditions, a variable one does not.")
        sync_note.setObjectName("hint"); sync_note.setWordWrap(True)
        sync_note.setTextFormat(QtCore.Qt.RichText)
        fs_.addWidget(sync_note)
        side.addWidget(gs)

        # ------------------------------------------------------------------------ calibration
        gc = QtWidgets.QGroupBox("Calibration"); fc = QtWidgets.QVBoxLayout(gc)
        ent = QtWidgets.QHBoxLayout()
        self.sbp = QtWidgets.QSpinBox(); self.sbp.setRange(40, 300); self.sbp.setValue(120)
        self.dbp = QtWidgets.QSpinBox(); self.dbp.setRange(20, 200); self.dbp.setValue(80)
        ent.addWidget(QtWidgets.QLabel("SBP")); ent.addWidget(self.sbp)
        ent.addWidget(QtWidgets.QLabel("DBP")); ent.addWidget(self.dbp)
        fc.addLayout(ent)
        self.btn_cal = QtWidgets.QPushButton("Log cuff reading   (ctrl+enter)")
        self.btn_cal.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_cal.clicked.connect(self._log_bp)
        fc.addWidget(self.btn_cal)
        self.cal_list = QtWidgets.QListWidget(); self.cal_list.setFixedHeight(96)
        fc.addWidget(self.cal_list)
        crow = QtWidgets.QHBoxLayout()
        b_del = QtWidgets.QPushButton("Remove selected"); b_del.setFocusPolicy(QtCore.Qt.NoFocus)
        b_del.clicked.connect(self._drop_bp)
        crow.addWidget(b_del)
        fc.addLayout(crow)
        self.lbl_fit = QtWidgets.QLabel(); self.lbl_fit.setObjectName("hint")
        self.lbl_fit.setWordWrap(True); self.lbl_fit.setTextFormat(QtCore.Qt.RichText)
        fc.addWidget(self.lbl_fit)
        cal_note = QtWidgets.QLabel(
            "A reading is only usable if a transit time is on screen when you log it, so wait "
            "for the PTT number before pressing. Vary the pressure between readings -- five "
            "readings all at rest calibrate the intercept and say nothing about the slope.")
        cal_note.setObjectName("hint"); cal_note.setWordWrap(True)
        fc.addWidget(cal_note)
        side.addWidget(gc)

        # --------------------------------------------------------------------- path override
        gp = QtWidgets.QGroupBox("Path length"); fp = QtWidgets.QVBoxLayout(gp)
        prow = QtWidgets.QHBoxLayout()
        self.path_box = QtWidgets.QDoubleSpinBox()
        self.path_box.setRange(0.0, 150.0); self.path_box.setDecimals(1)
        self.path_box.setSpecialValueText("pose estimate")   # 0 means "use the model"
        self.path_box.setValue(0.0); self.path_box.setSuffix(" cm")
        self.path_box.valueChanged.connect(self._set_path)
        prow.addWidget(self.path_box)
        fp.addLayout(prow)
        pnote = QtWidgets.QLabel(
            "Pose world landmarks give this subject's own ear-shoulder-elbow-wrist chain plus "
            "18 cm of hand, in metres, with no calibration object in the scene. Wave speed "
            "scales linearly with it, so a tape measure over the same route beats the estimate "
            "if you have one -- enter it here and it overrides.")
        pnote.setObjectName("hint"); pnote.setWordWrap(True)
        fp.addWidget(pnote)
        side.addWidget(gp)

        note = QtWidgets.QLabel(
            "Preview runs the whole pipeline and saves nothing. Wait until the sites you need "
            "are lit and a pulse is visible, then Record.\n\nRecord <b>rest</b> and <b>rest2</b> "
            "before anything else: the gap between two identical recordings is the noise floor "
            "every other result has to beat.")
        note.setObjectName("hint"); note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.RichText)          # else the <b> tags render literally
        side.addWidget(note); side.addStretch()
        holder = QtWidgets.QWidget(); holder.setLayout(side)
        holder.setFixedWidth(320)                       # stop the hints being clipped mid-word
        # Scrollable: the sidebar now carries sync, calibration and path as well as recording,
        # which is taller than a laptop screen with the video at a usable size.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(holder); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        lay.addWidget(scroll)
        self._hint()
        return page

    def _hint(self):
        self.hint.setText(dict(CONDITIONS)[self.cond.currentText()])

    # ------------------------------------------------------------- results tab
    def _results_tab(self):
        page = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(page)
        bar = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Build robustness plots from saved recordings")
        b.clicked.connect(self._build_plots)
        bx = QtWidgets.QPushButton("Export CSV")
        bx.clicked.connect(lambda: self._export(waveforms=True))
        bt = QtWidgets.QPushButton("Export CSV (tables only)")
        bt.setToolTip("Skip the per-sample waveform files, which are much the largest")
        bt.clicked.connect(lambda: self._export(waveforms=False))
        bo = QtWidgets.QPushButton("Reveal in Finder")
        bo.clicked.connect(self._reveal)
        for x in (b, bx, bt, bo):
            bar.addWidget(x)
        bar.addStretch()
        lay.addLayout(bar)
        self.res_msg = QtWidgets.QLabel("No plots yet."); self.res_msg.setObjectName("hint")
        self.res_msg.setWordWrap(True)
        lay.addWidget(self.res_msg)
        self.gallery = QtWidgets.QTabWidget()
        lay.addWidget(self.gallery, 1)
        return page

    def _export(self, waveforms=True):
        """Write every session and recording out as CSV, and say exactly what was written."""
        import export_csv as EX
        self.session.save()                  # flush the live session before exporting it
        self.res_msg.setText("exporting ...")
        QtWidgets.QApplication.processEvents()
        try:
            wrote = EX.export_all(EX.EXPORT, waveforms=waveforms)
        except Exception as e:               # noqa: BLE001 -- surface it, never fail silently
            self.res_msg.setText(f"<b>export failed:</b> {e}")
            return
        if not wrote:
            self.res_msg.setText(
                "Nothing to export yet -- no completed recordings and no cuff readings.")
            return
        rows = "".join(
            f"<tr><td style='padding-right:14px'>{p.stat().st_size:,d}</td>"
            f"<td>{p.name}</td></tr>" for p in wrote)
        self.res_msg.setText(
            f"<b>{len(wrote)} file(s)</b> -> {EX.EXPORT}<br>"
            f"<table style='font-size:11px'>{rows}</table>")
        self.statusBar().showMessage(f"exported {len(wrote)} file(s) to {EX.EXPORT}", 6000)

    def _reveal(self):
        """Open the export folder in Finder, so the files can be dragged straight out."""
        import export_csv as EX
        EX.EXPORT.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(EX.EXPORT)))

    def _build_plots(self):
        import subprocess
        self.res_msg.setText("building ...")
        QtWidgets.QApplication.processEvents()
        r = subprocess.run([sys.executable, str(ROOT / "rppg_robust.py")],
                           capture_output=True, text=True, cwd=str(ROOT))
        self.res_msg.setText(f"<pre style='font-size:11px'>{(r.stdout or r.stderr)[-1800:]}</pre>")
        self.gallery.clear()
        for p in sorted(FIGS.glob("fig_robust_*.png")):
            lbl = QtWidgets.QLabel(); lbl.setAlignment(QtCore.Qt.AlignCenter)
            pm = QtGui.QPixmap(str(p))
            lbl.setPixmap(pm.scaled(1000, 620, QtCore.Qt.KeepAspectRatio,
                                    QtCore.Qt.SmoothTransformation))
            sc = QtWidgets.QScrollArea(); sc.setWidget(lbl); sc.setWidgetResizable(True)
            self.gallery.addTab(sc, p.stem.replace("fig_robust_", ""))
        if not self.gallery.count():
            self.res_msg.setText(self.res_msg.text() + "<br>No figures produced yet.")

    # ---------------------------------------------------------------- controls
    def _set_snr(self, v):
        """Dial the SNR gate directly, without leaving the current profile."""
        P.set_min_snr(v)
        if self.worker:
            self.worker.profile = f"{self.prof.currentText()}(snr{v:g})"
        self.statusBar().showMessage(f"min SNR = {v:g}", 4000)

    def _set_profile(self, name):
        """Switch quality-gate thresholds, and say plainly what that does and does not buy."""
        p = P.set_profile(name)
        # Reflect the profile's SNR in the box without re-triggering _set_snr through the signal.
        if hasattr(self, "snr_box"):
            self.snr_box.blockSignals(True)
            self.snr_box.setValue(p["MIN_SNR"])
            self.snr_box.blockSignals(False)
        if self.worker:
            self.worker.profile = name
        if name == "strict":
            self.lbl_prof.setText(
                f"SNR&ge;{p['MIN_SNR']:.0f}, {p['MIN_SITES']} sites. Output is a measurement.")
        else:
            self.lbl_prof.setText(
                f"SNR&ge;{p['MIN_SNR']:.1f}, {p['MIN_SITES']} site(s). "
                f"<b>Noise scores 1.9-2.7 on this metric</b>, so this shows signal that is not "
                f"necessarily a pulse. The strict verdict is still saved as "
                f"<code>accepted_strict</code>, and recordings are tagged "
                f"<code>{name}</code>.")
        self.statusBar().showMessage(f"quality gates: {name}", 5000)
        self._refresh_bp()

    def _set_prop(self, on):
        self._prop = bool(on)
        if self.worker:
            self.worker.propagation = bool(on)

    def _set_method(self, m):
        """Switch extraction method. Applies to the next analysis and to the live panel."""
        self._method = m
        if self.worker:
            self.worker.method = m
        self.statusBar().showMessage(f"extraction method: {m}", 4000)

    def _set_subject(self):
        """Fold the typed subject id into the session name and every future filename."""
        s = BP.slug(self.subject.text())
        if s != self.subject.text():
            self.subject.setText(s)          # show what will actually be written
        self.session.set_subject(s)
        if self.worker:
            self.worker.subject = s
        self.statusBar().showMessage(f"session {self.session.name}", 4000)

    # -------------------------------------------------------------- sync and calibration
    def _sync(self):
        """Stamp a synchronisation mark at the keypress.

        perf_counter is read FIRST, before any bookkeeping, because everything after it adds to
        the offset this mark exists to pin down. The wall clock rides along so the mark can be
        matched against the cuff's own printout.
        """
        t = time.perf_counter()
        tag = self.cond.currentText()
        rec = bool(self.worker and self.worker._recording)
        m = self.session.mark(
            label="cuff", t_mono=t, note=tag, recording=rec,
            ptt_ms=self.live.get("ptt_ms"), hr=self.live.get("hr"))
        into = ""
        if rec and np.isfinite(getattr(self.worker, "t_rec_wall", float("nan"))):
            into = f"  ({m['t_wall'] - self.worker.t_rec_wall:+.2f} s into '{tag}')"
        self.lbl_marks.setText(
            f"<b>{len(self.session.marks)} mark(s)</b><br>last {m['iso'].split('T')[1]}{into}"
            + ("" if rec else "<br><span style='color:#d8a657'>not recording</span>"))
        self.lbl_marks.setTextFormat(QtCore.Qt.RichText)
        # Brief flash, so a press that did register is unmistakable without looking away.
        self.btn_sync.setText(f"MARKED  #{len(self.session.marks)}")
        QtCore.QTimer.singleShot(450, lambda: self.btn_sync.setText("SYNC MARK\nspace"))
        self.statusBar().showMessage(f"mark {len(self.session.marks)} at {m['iso']}", 4000)

    def _log_bp(self):
        """Pair the cuff reading now on the spinboxes with the live features."""
        t = time.perf_counter()
        ptt = self.live.get("ptt_ms", float("nan"))
        p = self.session.add_calib(
            self.sbp.value(), self.dbp.value(), ptt_ms=ptt, hr=self.live.get("hr"),
            path_cm=self.live.get("path_cm"), spread_ms=self.live.get("spread_ms"),
            label=self.cond.currentText(), t_mono=t)
        self._refresh_cal()
        if BP.features(p["ptt_ms"], p["hr"], p["path_cm"]) is None:
            # Kept, not discarded -- the cuff reading is still a real observation, and the
            # operator may want it in the record. But it cannot enter a fit, and saying so now
            # is the difference between 5 usable points and a surprise at analysis time.
            self.statusBar().showMessage(
                "logged, but NOT usable for calibration: no valid transit time at that moment",
                8000)
        else:
            self.statusBar().showMessage(
                f"logged {p['sbp']:.0f}/{p['dbp']:.0f} at PTT {p['ptt_ms']:.1f} ms", 5000)

    def _drop_bp(self):
        i = self.cal_list.currentRow()
        if i >= 0:
            self.session.drop_calib(i)
            self._refresh_cal()

    def _set_path(self, v):
        """0 means 'use the pose estimate'; anything else is a tape measure and overrides."""
        cm = float("nan") if v <= 0.0 else float(v)
        if self.worker:
            self.worker.path_manual = cm
        self._manual_path = cm

    def _refresh_cal(self):
        """Rebuild the point list and the fit status. Called on every change to the set."""
        self.cal_list.clear()
        for p in self.session.calib:
            usable = BP.features(p.get("ptt_ms"), p.get("hr"), p.get("path_cm")) is not None
            ptt = p.get("ptt_ms")
            txt = (f"{p['sbp']:.0f}/{p['dbp']:.0f}  "
                   f"{'PTT %.1f ms' % ptt if ptt else 'no PTT'}  {p.get('label', '')}")
            it = QtWidgets.QListWidgetItem(("  " if usable else "x ") + txt)
            if not usable:
                it.setForeground(QtGui.QColor("#8a6a3a"))
            self.cal_list.addItem(it)
        self.fits = self.session.fits()
        n = len(self.session.usable("sbp"))
        need = BP.MODE_MIN
        s, d = self.fits["sbp"], self.fits["dbp"]
        if not s.ok:
            msg = ("<b>No calibration.</b> No blood pressure will be shown -- a population "
                   "average is not a measurement of this subject.")
        else:
            msg = f"<b>SBP</b> {s.describe()}<br><b>DBP</b> {d.describe()}"
            if s.mode == "offset":
                msg += (f"<br>The slope is ASSUMED from physiology. {need['slope'] - n} more "
                        "reading(s) at a different pressure will start fitting it.")
            elif s.mode == "slope":
                msg += (f"<br>{need['full'] - n} more to fit the rate term too.")
        self.lbl_fit.setText(msg)
        self._refresh_bp()

    def _refresh_bp(self):
        """Push the current estimate into the readout, or refuse to."""
        ptt, hr = self.live.get("ptt_ms", float("nan")), self.live.get("hr", float("nan"))
        path = self.live.get("path_cm", float("nan"))
        fits = getattr(self, "fits", None)
        if not fits or not fits["sbp"].ok:
            self.ro["bp"].set("--", "log a cuff reading to calibrate", dim=True)
            return
        sv, ss = fits["sbp"].predict(ptt, hr, path)
        dv, _ = fits["dbp"].predict(ptt, hr, path)
        if not np.isfinite(sv):
            self.ro["bp"].set("--", "no valid transit time right now", dim=True)
            return
        spread = self.live.get("spread_ms", float("nan"))
        # Dim unless the calibration fitted its own slope AND the transit time feeding it is
        # bigger than its own scatter. Either failure makes the number decorative.
        solid = (fits["sbp"].mode != "offset" and np.isfinite(spread)
                 and np.isfinite(ptt) and abs(ptt) > spread)
        self.ro["bp"].set(f"{sv:.0f}/{dv:.0f}",
                          f"+/-{ss:.0f} mmHg   n={fits['sbp'].n}"
                          + ("" if solid else "   indicative only"), dim=not solid)

    # ---------------------------------------------------------------- live metrics
    def _on_metrics(self, m):
        self.live = m
        hr, ptt = m.get("hr", float("nan")), m.get("ptt_ms", float("nan"))
        spread, path = m.get("spread_ms", float("nan")), m.get("path_cm", float("nan"))
        null, fs = m.get("null_ms", float("nan")), m.get("fs", float("nan"))

        self.ro["hr"].set(f"{hr:.0f}" if np.isfinite(hr) else "--",
                          "spectral peak, several seconds" if np.isfinite(hr) else "collecting")
        if np.isfinite(ptt):
            noisy = not np.isfinite(spread) or abs(ptt) <= spread
            self.ro["ptt"].set(f"{ptt:+.1f}",
                               (f"+/-{spread:.1f} over {m.get('n_lag', 0)} windows"
                                if np.isfinite(spread) else "spread not yet estimable")
                               + ("   below its own scatter" if noisy else ""), dim=noisy)
        else:
            self.ro["ptt"].set("--", "no distal signal yet", dim=True)

        pwv = BP.pwv_ms(ptt, path)
        if np.isfinite(pwv):
            # 4-12 m/s is the physiological range; outside it the timing, not the subject, is
            # what the number is describing.
            bad = not (4.0 <= pwv <= 12.0)
            self.ro["pwv"].set(f"{pwv:.1f}",
                               f"over {path:.0f} cm" + ("   outside 4-12 m/s" if bad else ""),
                               dim=bad)
        else:
            self.ro["pwv"].set("--", "needs a transit time", dim=True)

        self.lbl_path.setText(
            f"{path:.0f} cm" + ("  (manual)" if np.isfinite(getattr(self, "_manual_path",
                                                                    float("nan"))) else "")
            if np.isfinite(path) else "-")
        # The null control, against the frame quantum: if two sites at the SAME arterial distance
        # read a lag comparable to the face-to-hand one, the timing pipeline is what is being
        # measured. Kept next to the number it invalidates.
        if np.isfinite(null):
            q = 1000.0 / max(fs, 1e-6) if np.isfinite(fs) else float("inf")
            self.lbl_null.setText(f"{null:+.1f} ms" + ("" if abs(null) <= 0.5 * q else "  HIGH"))
        else:
            self.lbl_null.setText("-")
        # Arrival-vs-distance: r says the ordering is real, the slope says how fast. Both or
        # neither -- a good r with an absurd speed means the patches are ordered but the timing
        # is scaled wrong, which is a different fault from noise.
        pr, ppwv, pn = m.get("prop_r"), m.get("prop_pwv"), m.get("prop_n", 0)
        if pr is not None and np.isfinite(pr) and pn:
            verdict = ("real" if abs(pr) >= 0.5 and np.isfinite(ppwv) and 4 <= ppwv <= 12
                       else "not ordered")
            self.lbl_prop.setText(f"r={pr:+.2f}  {ppwv:.1f} m/s  n={pn}  ({verdict})")
        else:
            self.lbl_prop.setText("-")
        self._refresh_bp()

    def _preview(self):
        if self.worker:
            return
        self.worker = Worker(self.secs.value(), self.cond.currentText(), self.session, self)
        self.worker.path_manual = getattr(self, "_manual_path", float("nan"))
        self.worker.subject = BP.slug(self.subject.text())
        self.worker.method = self.meth.currentText()
        self.worker.propagation = self.cb_prop.isChecked()
        # Re-assert the profile on the worker's thread-visible state: set_profile mutates module
        # globals, but the worker also records which profile the run was made under.
        self.worker.profile = self.prof.currentText()
        P.set_profile(self.prof.currentText())
        self.worker.frame.connect(self._on_frame)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.finished_run.connect(self._on_done)
        self.worker.start()
        self.btn_prev.setEnabled(False); self.btn_rec.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.lbl_state.setText("preview -- nothing saved")
        self.statusBar().showMessage("Preview running")

    def _record(self):
        if not self.worker:
            return
        self.worker.seconds = self.secs.value()
        self.worker.tag = self.cond.currentText()
        # Re-read the subject here too: preview is often started before the id is typed in, and
        # the filename is decided at save time, not at preview time.
        self.worker.subject = BP.slug(self.subject.text())
        self.worker.start_recording()
        self.btn_rec.setEnabled(False); self.cond.setEnabled(False); self.secs.setEnabled(False)
        self.lbl_state.setText(f"RECORDING '{self.worker.tag}'")
        self.statusBar().showMessage("Recording")

    def _stop(self):
        if self.worker:
            self.worker.abort()

    def _on_frame(self, bgr, panel_bgr, el, kept, seen):
        # Frames are queued across the thread boundary, so one can land after the worker has
        # finished and been cleared. Dereferencing it here then raised mid-session.
        if not self.worker:
            return
        for name, chip in self.chips.items():
            chip.set_on(name in self.worker.sites)
        self.video.setPixmap(self._pix(bgr, self.video.width(), self.video.height()))
        self.panel.setPixmap(self._pix(panel_bgr, 430, panel_bgr.shape[0]))
        if self.worker and self.worker._recording:
            self.prog.setValue(int(100 * min(1.0, el / max(self.worker.seconds, 1e-9))))
            self.lbl_fps.setText(f"{kept} frames  ({kept/max(el,1e-3):.1f}/s)")

    @staticmethod
    def _pix(bgr, w, h):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                           QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(img).scaled(w, h, QtCore.Qt.KeepAspectRatio,
                                                   QtCore.Qt.SmoothTransformation)

    def _on_done(self, tag, saved, msg):
        self.worker = None
        self.btn_prev.setEnabled(True); self.btn_rec.setEnabled(False)
        self.btn_stop.setEnabled(False); self.cond.setEnabled(True); self.secs.setEnabled(True)
        self.prog.setValue(0); self.lbl_state.setText("idle")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Saved" if saved else "Not saved")
        box.setIcon(QtWidgets.QMessageBox.Information if saved
                    else QtWidgets.QMessageBox.Warning)
        box.setText(f"<b>{tag}</b><br>{msg}")
        if not saved:
            box.setInformativeText("Nothing was written. A clean exit is not evidence of a "
                                   "good recording, so this says so explicitly.")
        box.exec()
        self.statusBar().showMessage(msg)

    def closeEvent(self, e):
        if self.worker:
            self.worker.abort(); self.worker.wait(2000)
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(CSS)
    w = Main(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
