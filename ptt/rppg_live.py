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
        self.prox, self.dist, self.T = [], [], []
        self.hr = self.snr = np.nan
        self.lag = np.nan
        self.hist = []
        self._last = -1e9

    def push(self, prox_rgb, dist_rgb, t):
        """Add one frame's proximal (face) and distal (hand) mean RGB."""
        self.prox.append(prox_rgb); self.dist.append(dist_rgb); self.T.append(t)
        while self.T and t - self.T[0] > self.buf_s:
            self.prox.pop(0); self.dist.pop(0); self.T.pop(0)
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

    # ------------------------------------------------------------------ render
    def render(self, h):
        import cv2
        img = np.full((h, PANEL_W, 3), 28, np.uint8)
        put = lambda s, y, c=(235, 235, 235), sc=.5, th=1: cv2.putText(
            img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

        tr = self._traces()
        put("PULSE", 26, GREY, .45)
        if tr is None:
            put("collecting ...", 60, GREY)
            need = max(0.0, MIN_BUF_S - (self.T[-1] - self.T[0] if len(self.T) > 1 else 0))
            put(f"{need:.0f}s more", 82, GREY)
            return img
        p, d, fs = tr
        # Raw beside cleaned, because "the filter is inventing this" is the first thing anyone
        # asks of an rPPG trace. The raw mean-green trace carries the pulse buried under
        # illumination drift; showing both makes clear the band-pass is removing drift rather
        # than synthesising a rhythm.
        raw = self._raw_traces()
        traces = [(p, GREEN, "face", raw[0] if raw else None)]
        if d is not None:
            traces.append((d, RED, "hand", raw[1] if raw else None))
        for k, (sig, col, name, rw) in enumerate(traces):
            y0, hh = 40 + k * 92, 34
            if rw is not None and len(rw) > 8:
                r = rw[-int(min(len(rw), 6 * fs)):].astype(float)
                r = (r - r.mean()) / (np.std(r) + 1e-9)
                xs = np.linspace(14, PANEL_W - 14, len(r))
                ys = y0 + 14 - np.clip(r, -2.2, 2.2) * 6.0
                cv2.polylines(img, [np.int32(np.stack([xs, ys], 1))], False, (105, 105, 112),
                              1, cv2.LINE_AA)
                cv2.putText(img, "raw", (PANEL_W - 34, y0 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                            .34, (105, 105, 112), 1, cv2.LINE_AA)
            s = sig[-int(min(len(sig), 6 * fs)):]
            s = s / (np.std(s) + 1e-9)
            xs = np.linspace(14, PANEL_W - 14, len(s))
            ys = y0 + 34 + hh / 2 - np.clip(s, -2.6, 2.6) * (hh / 5.6)
            cv2.polylines(img, [np.int32(np.stack([xs, ys], 1))], False, col, 1, cv2.LINE_AA)
            cv2.putText(img, name, (PANEL_W - 62, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        col, 1, cv2.LINE_AA)

        # Beat overlay: both sites averaged over the last few beats and drawn on one axis. If the
        # lag is a transit time the two curves keep a fixed offset beat after beat; if it is
        # noise they wander. This is the check a reader can make by eye, which no single lag
        # number supports on its own.
        if d is not None and np.isfinite(self.hr) and self.hr > 30:
            yb = 40 + len(traces) * 92
            cv2.putText(img, "beat overlay", (14, yb - 6), cv2.FONT_HERSHEY_SIMPLEX, .38,
                        GREY, 1, cv2.LINE_AA)
            per = int(round(fs * 60.0 / self.hr))
            if per > 6 and len(p) > 3 * per:
                nb = min(6, len(p) // per)
                stack = lambda z: np.mean([z[-(i + 1) * per:len(z) - i * per]
                                           for i in range(nb)], 0)
                try:
                    ap, ad = stack(p), stack(d)
                    xs = np.linspace(14, PANEL_W - 14, per)
                    for z, c in ((ap, GREEN), (ad, RED)):
                        zz = (z - z.mean()) / (np.std(z) + 1e-9)
                        ys = yb + 26 - np.clip(zz, -2.4, 2.4) * 9.0
                        cv2.polylines(img, [np.int32(np.stack([xs, ys], 1))], False, c, 1,
                                      cv2.LINE_AA)
                    # mark the two upstroke peaks; the gap between them is the lag
                    ip, idd = int(np.argmax(np.gradient(ap))), int(np.argmax(np.gradient(ad)))
                    for ii, c in ((ip, GREEN), (idd, RED)):
                        x = 14 + (PANEL_W - 28) * ii / max(per - 1, 1)
                        cv2.line(img, (int(x), yb + 6), (int(x), yb + 46), c, 1)
                    if np.isfinite(self.lag):
                        cv2.putText(img, f"{self.lag:+.0f} ms", (PANEL_W - 70, yb + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 235, 235), 1,
                                    cv2.LINE_AA)
                except Exception:
                    pass

        y = 268
        if np.isfinite(self.hr):
            good = self.snr >= 5
            cv2.putText(img, f"{self.hr:.0f}", (14, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 1.15,
                        (235, 235, 235) if good else GREY, 2, cv2.LINE_AA)
            cv2.putText(img, "bpm", (96, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .5, GREY, 1,
                        cv2.LINE_AA)
            cv2.putText(img, f"SNR {self.snr:.1f}" + ("" if good else "  weak signal"),
                        (150, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        GREY if good else (80, 165, 235), 1, cv2.LINE_AA)
        y = 262
        put("PROXIMAL -> HAND OFFSET", y, GREY, .45)
        if d is None:
            put("hand not visible", y + 30, (80, 165, 235))
            put("bring the hand into frame and light it", y + 50, GREY, .42)
        elif len(self.hist) >= 4:
            med = float(np.median(self.hist)); sd = float(np.std(self.hist))
            # An offset means nothing if it is smaller than its own scatter.
            trust = sd < abs(med) and 5.0 <= abs(med) <= 120.0
            cv2.putText(img, f"{med:+.0f}", (14, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (235, 235, 235) if trust else GREY, 2, cv2.LINE_AA)
            cv2.putText(img, f"+/- {sd:.0f} ms", (104, y + 34), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        GREY, 1, cv2.LINE_AA)
            msg = ("unstable: spread exceeds the value" if sd >= abs(med) else
                   "outside 5-120 ms: artifact" if not (5.0 <= abs(med) <= 120.0) else
                   "stable  (still contains fixed delay)")
            cv2.putText(img, msg, (14, y + 56), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        GREY if trust else (80, 165, 235), 1, cv2.LINE_AA)
            hs = np.array(self.hist[-HIST:], float)     # sparkline of recent estimates
            if len(hs) > 2 and np.ptp(hs) > 1e-9:
                x0, x1, yb, hh2 = 14, PANEL_W - 14, y + 106, 34
                xs = np.linspace(x0, x1, len(hs))
                ys = yb - (hs - hs.min()) / np.ptp(hs) * hh2
                cv2.polylines(img, [np.int32(np.stack([xs, ys], 1))], False, NAVY, 1,
                              cv2.LINE_AA)
                cv2.putText(img, f"last {len(hs)} windows", (x0, yb + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, GREY, 1, cv2.LINE_AA)
        else:
            put("collecting ...", y + 30, GREY)

        cv2.putText(img, f"{fs:.1f} fps   {1000/fs:.0f} ms/frame", (14, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, "transit is sub-frame: read CHANGES, not the value", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, GREY, 1, cv2.LINE_AA)
        return img
