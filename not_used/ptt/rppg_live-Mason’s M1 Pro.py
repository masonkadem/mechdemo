"""rppg_live.py -- live pulse waveforms, heart rate and transit time drawn beside the video.

The point is to fail fast. A 60 s recording that turns out to have no pulse in it is 60 s
wasted, and the stage figures only reveal that afterwards. This panel shows the proximal and
distal pulse traces as they accumulate, so a bad ROI, a dark arm or a moving subject is obvious
within a few seconds.

Reading the numbers honestly
---------------------------
HR is trustworthy. It comes from a spectral peak over several seconds and the camera samples the
0.7-3 Hz band roughly ten times over.

The live PTT is NOT a per-beat transit time and must not be read as one. At 30 fps a frame is
33 ms while face-to-hand transit is 20-50 ms, so a single window cannot resolve it; what is shown
is a cross-correlation offset over the whole buffer, refined below the sample interval. It is
displayed with the spread across recent windows precisely so the spread can be judged against the
value -- a median of 30 ms with a 40 ms spread is noise, and the panel says so rather than
printing a confident number. The absolute value also still contains any fixed instrument delay,
including the rolling-shutter row offset (see rppg_shutter.py), so what carries evidence is the
CHANGE between conditions, not the number itself.
"""
import numpy as np

import rppg_two_site as R

NAVY, RED, GREY, GREEN = (124, 75, 47), (59, 84, 193), (166, 160, 154), (90, 140, 59)  # BGR
PANEL_W = 430
MIN_BUF_S = 6.0            # below this there is not enough signal to say anything
HIST = 40                  # recent lag estimates kept for the spread and the sparkline


class LivePanel:
    """Rolling buffers -> HR, transit-time estimate and a rendered panel."""

    def __init__(self, fs_guess=30.0, buf_s=10.0, update_every=0.4):
        self.fs = fs_guess
        self.buf_s = buf_s
        self.update_every = update_every
        # prox2 is a second, independent proximal region (the other cheek). Forehead and
        # cheeks sit at the same arterial distance, so the lag between them is the rig's own
        # noise -- the control that says whether a face-to-hand number means anything.
        self.prox, self.prox2, self.dist, self.T = [], [], [], []
        self.null = np.nan
        self.null_hist = []
        self.path_cm = np.nan          # face-to-fingertip path, from pose world landmarks
        self.hr = self.snr = np.nan
        self.lag = np.nan
        self.hist = []
        self._last = -1e9

    def push(self, prox_rgb, dist_rgb, t, prox2_rgb=None):
        """Add one frame's proximal (face) and distal (hand) mean RGB.

        prox2_rgb is an optional second proximal region. When supplied, the lag between the two
        proximal signals is measured alongside the face-to-hand lag and shown as the null.
        """
        self.prox.append(prox_rgb); self.dist.append(dist_rgb); self.T.append(t)
        self.prox2.append(prox2_rgb if prox2_rgb is not None else prox_rgb)
        while self.T and t - self.T[0] > self.buf_s:
            self.prox.pop(0); self.prox2.pop(0); self.dist.pop(0); self.T.pop(0)
        if t - self._last >= self.update_every:
            self._last = t
            self._recompute()

    def _raw_traces(self):
        """Unfiltered mean-green for each site, on the same window as _traces.

        Green alone rather than the chrominance projection: the point of the raw panel is to show
        what the sensor delivered before any processing, and green is the channel a reader can
        reason about.
        """
        if len(self.T) < 30:
            return None
        Pm = np.asarray(self.prox, float)
        Dm = np.asarray(self.dist, float)
        okp = np.isfinite(Pm).all(1)
        if okp.sum() < 30:
            return None
        rp = Pm[okp][:, 1]
        okd = np.isfinite(Dm).all(1)
        rd = Dm[okd][:, 1] if okd.sum() >= 30 else None
        return rp, rd

    def _traces(self):
        """Buffers -> band-passed proximal signal, and the distal one when the hand is visible.

        The distal site is optional on purpose. Requiring both meant a hand out of frame -- or
        merely unlit -- suppressed the heart rate too, so the panel sat on "collecting" with no
        indication that the face signal was perfectly good. Returns (prox, dist_or_None, fs).
        """
        if len(self.T) < 30 or self.T[-1] - self.T[0] < MIN_BUF_S:
            return None
        T = np.asarray(self.T)
        P = np.asarray(self.prox, float); D = np.asarray(self.dist, float)
        okp = np.isfinite(P).all(1)
        if okp.sum() < 30:
            return None
        Tp, P = T[okp], P[okp]
        fs = (len(Tp) - 1) / (Tp[-1] - Tp[0])
        if not np.isfinite(fs) or fs < 5:
            return None
        self.fs = fs
        tu = np.linspace(Tp[0], Tp[-1], len(Tp))
        p = R.bandpass(np.interp(tu, Tp, R.chrom(P)), fs)

        both = okp & np.isfinite(D).all(1)
        d = None
        if both.sum() >= 30 and T[both][-1] - T[both][0] >= MIN_BUF_S:
            # resample BOTH onto one grid so the lag between them stays meaningful
            Tb = T[both]
            tb = np.linspace(Tb[0], Tb[-1], len(Tb))
            p_b = R.bandpass(np.interp(tb, Tb, R.chrom(P[both[okp]])), fs)
            d = R.bandpass(np.interp(tb, Tb, R.chrom(D[both])), fs)
            p, tu = p_b, tb
        return p, d, fs

    def _recompute(self):
        tr = self._traces()
        if tr is None:
            return
        p, d, fs = tr
        from scipy.signal import welch
        f, S = welch(p, fs, nperseg=min(len(p), int(4 * fs)))
        m = (f > R.BAND[0]) & (f < R.BAND[1])
        if m.any() and S[m].sum() > 0:
            # Interpolate the peak instead of taking the bin centre. A 4 s window gives 0.25 Hz
            # bins = 15 bpm, so the raw bin quantises HR badly (measured 75 bpm for a true 69).
            # A parabola through the log-power neighbours recovers the peak between bins.
            #
            # The peak is located within the band but interpolated against the FULL spectrum.
            # Interpolating on the sliced band leaves no neighbour when the peak lands in the
            # first in-band bin, which silently disabled refinement for low heart rates: a true
            # 52 bpm returned exactly 45.0 bpm, the 0.75 Hz bin centre.
            k = int(np.flatnonzero(m)[np.argmax(S[m])])
            if 0 < k < len(S) - 1:
                y0, y1, y2 = np.log(S[k - 1:k + 2] + 1e-30)
                shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
                shift = float(np.clip(shift, -0.5, 0.5))
            else:
                shift = 0.0
            df = f[1] - f[0] if len(f) > 1 else 0.0
            self.hr = float((f[k] + shift * df) * 60)
            self.snr = float(S[k] / (np.median(S[m]) + 1e-15))
        if d is None:
            return
        # search only a few samples: wide windows let a weak signal lock onto nonsense
        lag, _ = R.lag_subframe(p, d, fs, max_lag_s=min(0.25, 4.0 / fs))
        if np.isfinite(lag):
            self.lag = float(lag)
            self.hist.append(self.lag)
            del self.hist[:-HIST]
        # the null, on the same window and the same estimator as the measurement
        P2 = np.asarray(self.prox2, float)
        okq = np.isfinite(P2).all(1)
        if okq.sum() >= 30:
            Tq = np.asarray(self.T)[okq]
            fq = (len(Tq) - 1) / max(Tq[-1] - Tq[0], 1e-6)
            if fq >= 5:
                tq = np.linspace(Tq[0], Tq[-1], len(Tq))
                q = R.bandpass(np.interp(tq, Tq, R.chrom(P2[okq])), fq)
                n = min(len(p), len(q))
                if n > 30 and np.std(q[:n]) > 1e-9:
                    nl, _ = R.lag_subframe(p[:n], q[:n], fq, max_lag_s=min(0.25, 4.0 / fq))
                    if np.isfinite(nl):
                        self.null = float(nl)
                        self.null_hist.append(self.null)
                        del self.null_hist[:-HIST]

    # ------------------------------------------------------------------ render
    def render(self, h):
        """Three numbers and the traces that produced them.

        The panel used to carry a beat overlay, a sparkline, an SNR readout and a PWV verdict.
        None of that is needed to answer the question the rig exists for -- is the face-to-hand
        offset larger than the offset between two points that are at the SAME distance -- and
        every extra element cost frame time and space that the traces wanted.
        """
        import cv2
        img = np.full((h, PANEL_W, 3), 28, np.uint8)

        tr = self._traces()
        if tr is None:
            cv2.putText(img, "collecting ...", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        GREY, 1, cv2.LINE_AA)
            need = max(0.0, MIN_BUF_S - (self.T[-1] - self.T[0] if len(self.T) > 1 else 0))
            cv2.putText(img, f"{need:.0f}s more", (14, 64), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        GREY, 1, cv2.LINE_AA)
            return img
        p, d, fs = tr

        # --- the two traces -------------------------------------------------------------
        y = 30
        for sig, col, name in [(p, GREEN, "face"), (d, RED, "fingertips")]:
            if sig is None:
                continue
            cv2.putText(img, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
            z = sig[-int(min(len(sig), 6 * fs)):]
            z = z / (np.std(z) + 1e-9)
            xs = np.linspace(14, PANEL_W - 14, len(z))
            ys = y + 34 - np.clip(z, -2.6, 2.6) * 11.0
            cv2.polylines(img, [np.int32(np.stack([xs, ys], 1))], False, col, 1, cv2.LINE_AA)
            y += 72
        if d is None:
            cv2.putText(img, "fingertips not visible", (14, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 165, 235), 1, cv2.LINE_AA)

        # --- the three numbers ----------------------------------------------------------
        y += 14
        cv2.line(img, (14, y), (PANEL_W - 14, y), (52, 54, 58), 1)
        y += 30

        def row(label, val, spread, colour, note=""):
            nonlocal y
            cv2.putText(img, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, .44, GREY, 1,
                        cv2.LINE_AA)
            txt = "--" if not np.isfinite(val) else f"{val:+.1f}"
            cv2.putText(img, txt, (14, y + 34), cv2.FONT_HERSHEY_SIMPLEX, .95, colour, 2,
                        cv2.LINE_AA)
            cv2.putText(img, "ms", (96, y + 34), cv2.FONT_HERSHEY_SIMPLEX, .45, GREY, 1,
                        cv2.LINE_AA)
            if np.isfinite(spread):
                cv2.putText(img, f"+/- {spread:.1f}", (PANEL_W - 96, y + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, GREY, 1, cv2.LINE_AA)
            if note:
                cv2.putText(img, note, (14, y + 54), cv2.FONT_HERSHEY_SIMPLEX, .40, GREY, 1,
                            cv2.LINE_AA)
            y += 76 if note else 60

        nsd = float(np.std(self.null_hist)) if len(self.null_hist) >= 3 else np.nan
        msd = float(np.std(self.hist)) if len(self.hist) >= 3 else np.nan
        nmed = float(np.median(self.null_hist)) if self.null_hist else np.nan
        mmed = float(np.median(self.hist)) if self.hist else np.nan

        row("FACE -> FACE   (control, expect 0)", nmed, nsd, (170, 175, 180))
        # The measurement is only white when it clears the control; otherwise it is grey,
        # because a difference smaller than the rig's own scatter is not a measurement.
        beats = (np.isfinite(mmed) and np.isfinite(nmed)
                 and abs(mmed) > max(abs(nmed), nsd if np.isfinite(nsd) else 0) * 2)
        row("FACE -> FINGERTIPS", mmed, msd,
            (235, 235, 235) if beats else GREY,
            "" if beats else "not yet above the control")

        # Path length and, when both it and a measurement exist, the implied wave speed. This
        # is the number the 4-12 m/s physiological range can be checked against; the lag alone
        # cannot be, since it scales with how long the subject's arm happens to be.
        if np.isfinite(self.path_cm):
            txt = f"{self.path_cm:.0f} cm"
            if np.isfinite(mmed) and abs(mmed) > 1e-6:
                txt += f"   {self.path_cm / 100.0 / (abs(mmed) / 1000.0):.1f} m/s"
            cv2.putText(img, txt, (PANEL_W - 168, h - 40), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        GREY, 1, cv2.LINE_AA)
        cv2.putText(img, f"{self.hr:.0f} bpm" if np.isfinite(self.hr) else "-- bpm",
                    (14, h - 40), cv2.FONT_HERSHEY_SIMPLEX, .55, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, f"{fs:.0f} fps   {1000/fs:.0f} ms/frame", (14, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, GREY, 1, cv2.LINE_AA)
        return img
