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
from pathlib import Path

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

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
QComboBox, QSpinBox { background:#1e2229; border:1px solid #333a45; border-radius:6px;
                      padding:6px 8px; }
QLabel#hint   { color:#9aa0a6; }
QLabel#big    { font-size:34px; font-weight:700; }
QLabel#unit   { color:#9aa0a6; }
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

    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFixedHeight(24)
        self.set_on(False)

    def set_on(self, on):
        self.setStyleSheet(
            "border-radius:11px; padding:2px 11px; font-size:11px; "
            + ("background:#1d4a2c; color:#8fe0a6; border:1px solid #2e7d46;" if on
               else "background:#22262e; color:#606673; border:1px solid #2c3038;"))


class Worker(QtCore.QThread):
    """Runs the capture pipeline. Emits frames and metrics; never touches widgets."""

    frame = QtCore.Signal(object, object, float, int, int)   # bgr, live-panel bgr, el, kept, seen
    status = QtCore.Signal(str)
    finished_run = QtCore.Signal(str, bool, str)             # tag, saved, message

    def __init__(self, seconds, tag, parent=None):
        super().__init__(parent)
        self.seconds, self.tag = seconds, tag
        self._recording = False
        self._abort = False
        self.sites = set()

    def start_recording(self):
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
            except Exception:
                hlm = None
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
            pts = list(pts) + list(tip_pts)
            # Path length from pose world landmarks (metres), so the wave speed uses THIS
            # subject's arm rather than a nominal one -- arm length varies about 20% across
            # adults and enters the velocity linearly.
            if res.pose_world_landmarks:
                pc = HS.head_to_hand_cm(res.pose_world_landmarks[0])
                if np.isfinite(pc):
                    panel.path_cm = pc
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
                with np.errstate(invalid="ignore"):
                    pr = np.nanmean(A[PM], 0) if PM.any() else np.full(3, np.nan)
                    ds = np.nanmean(A[DM], 0) if DM.any() else np.full(3, np.nan)
                panel.push(pr, ds, time.time() - t_wall)
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
            for p, s in zip(pts, seg_all):
                if p is not None:
                    col = (90, 200, 255) if (s in P.DISTAL or s.startswith("finger")) \
                        else (140, 245, 140)
                    cv2.circle(vis, p, 5, (20, 20, 20), 2, cv2.LINE_AA)
                    cv2.circle(vis, p, 5, col, 1, cv2.LINE_AA)
                    cv2.circle(vis, p, 1, col, -1, cv2.LINE_AA)
                    live.add(s)
            self.sites = live
            self.frame.emit(vis, panel.render(h), el, nkept, nseen)

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
            if good.mean() < .6:
                filtered.append(None); quals.append(0.); hrs.append(np.nan); continue
            fill = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
            ch = R.chrom(fill); bp = R.bandpass(ch, fs)
            stages_all[i] = {"raw": fill[:, 1],
                             "detrended": P.detrend(fill[:, 1], fs), "chrom": ch, "filtered": bp}
            _, hr, qm, _ = P.plausible(bp, fs)
            filtered.append(bp); quals.append(qm); hrs.append(hr)
        quals, hrs = np.array(quals), np.array(hrs)
        cons = float(np.median(hrs[np.isfinite(hrs) & (quals > P.MIN_SNR)])) \
            if np.isfinite(hrs).any() else np.nan
        ok = np.zeros(len(filtered), bool)
        for i, x in enumerate(filtered):
            if x is not None:
                ok[i], _, _, _ = P.plausible(x, fs, cons)
        if ok.sum() < 6:
            return (f"only {ok.sum()} sites passed quality gates (need 6). "
                    "More light on face and forearm, hold still."), False
        out = {"tag": self.tag, "fps": fs, "n_frames": len(T), "n_points": int(acc.shape[1]),
               "n_accepted": int(ok.sum()), "consensus_hr": cons}
        (DATA / f"rppg_pose_{self.tag}.json").write_text(json.dumps(out, indent=2, default=float))
        best = int(np.argmax(quals))
        np.savez(DATA / f"rppg_pose_{self.tag}.npz",
                 sigs=np.stack([x if x is not None else np.zeros(len(tu)) for x in filtered]),
                 accepted=ok, quals=quals, hrs=hrs, dist=dist, seg=np.asarray(seg), fs=fs,
                 **stages_all.get(best, {}))
        return (f"saved -- HR {cons:.0f} bpm, {ok.sum()}/{len(ok)} sites accepted "
                f"at {fs:.0f} fps"), True


class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Pulse Transit Time")
        self.resize(1180, 760)
        self.worker = None
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._capture_tab(), "Capture")
        tabs.addTab(self._results_tab(), "Results")
        self.tabs = tabs
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")

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
        for s in ("forehead", "cheek_l", "cheek_r", "forearm", "hand"):
            c = Chip(s); self.chips[s] = c; chiprow.addWidget(c)
        chiprow.addStretch()
        left.addLayout(chiprow)
        lay.addLayout(left, 1)

        side = QtWidgets.QVBoxLayout(); side.setSpacing(12)
        gb = QtWidgets.QGroupBox("Recording"); f = QtWidgets.QVBoxLayout(gb)
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
        g2.addWidget(QtWidgets.QLabel("State"), 0, 0); g2.addWidget(self.lbl_state, 0, 1)
        g2.addWidget(QtWidgets.QLabel("Kept"), 1, 0);  g2.addWidget(self.lbl_fps, 1, 1)
        side.addWidget(gb2)

        note = QtWidgets.QLabel(
            "Preview runs the whole pipeline and saves nothing. Wait until the sites you need "
            "are lit and a pulse is visible, then Record.\n\nRecord <b>rest</b> and <b>rest2</b> "
            "before anything else: the gap between two identical recordings is the noise floor "
            "every other result has to beat.")
        note.setObjectName("hint"); note.setWordWrap(True)
        note.setTextFormat(QtCore.Qt.RichText)          # else the <b> tags render literally
        side.addWidget(note); side.addStretch()
        holder = QtWidgets.QWidget(); holder.setLayout(side)
        holder.setFixedWidth(290)                       # stop the hints being clipped mid-word
        lay.addWidget(holder)
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
        bar.addWidget(b); bar.addStretch()
        lay.addLayout(bar)
        self.res_msg = QtWidgets.QLabel("No plots yet."); self.res_msg.setObjectName("hint")
        self.res_msg.setWordWrap(True)
        lay.addWidget(self.res_msg)
        self.gallery = QtWidgets.QTabWidget()
        lay.addWidget(self.gallery, 1)
        return page

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
    def _preview(self):
        if self.worker:
            return
        self.worker = Worker(self.secs.value(), self.cond.currentText(), self)
        self.worker.frame.connect(self._on_frame)
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
        self.worker.start_recording()
        self.btn_rec.setEnabled(False); self.cond.setEnabled(False); self.secs.setEnabled(False)
        self.lbl_state.setText(f"RECORDING '{self.worker.tag}'")
        self.statusBar().showMessage("Recording")

    def _stop(self):
        if self.worker:
            self.worker.abort()

    def _on_frame(self, bgr, panel_bgr, el, kept, seen):
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
