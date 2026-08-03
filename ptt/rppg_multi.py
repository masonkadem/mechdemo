"""rppg_multi.py -- skin-masked multi-patch rPPG with a spatio-temporal pulse map.

Two problems with the single-box approach this replaces
-------------------------------------------------------
1. A hand-drawn box contains skin AND background AND clothing. Every non-skin pixel adds noise
   that is uncorrelated with the pulse but strongly correlated with motion, so ROI drift shows up
   as signal. Here every patch is masked to skin pixels (YCrCb + HSV chrominance gates, which are
   far more illumination-robust than RGB thresholds) and a patch is discarded outright if it is
   less than a set fraction skin.
2. One patch per site is fragile: a shadow, a specular highlight or a moment of motion kills the
   whole trace. Here each site is tiled into a GRID of small patches, each patch yields its own
   trace, and the site signal is the quality-weighted combination of the good ones. Patch quality
   is spectral: the fraction of band-limited power sitting at the cardiac peak.

Sites, chosen for physiology rather than convenience:
  forehead   -- strong, stable perfusion; the standard rPPG reference site
  neck_right -- over the right carotid/jugular, the most proximal accessible pulse
  neck_left  -- the contralateral control: a real carotid signal should appear on both
  hand       -- the distal site whose delay from the neck is the quantity of interest

The spatio-temporal map
-----------------------
Because every patch has its own trace, the per-patch phase at the cardiac frequency gives the
arrival time of the pulse AT EACH POINT on the skin. Rendering that as arrows over the face shows
the wave propagating -- proximal (neck) leading, distal (forehead periphery, hand) lagging. This
is a genuinely informative visualisation: it turns "PTT" from a single scalar into a measured
propagation field, and it is a direct visual check that the timing being extracted is
physiological rather than an artifact of ROI placement (an artifact has no coherent spatial
gradient).

    python rppg_multi.py --seconds 60 --tag rest
    python rppg_multi.py --seconds 60 --tag rest --map      # + spatio-temporal figure
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import rppg_cam
import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GRID = 3                      # each site is tiled GRID x GRID
MIN_SKIN = 0.45               # a patch must be at least this fraction skin to be used


def skin_mask(bgr):
    """Skin mask from chrominance, which is much more illumination-robust than RGB gates.

    YCrCb: skin clusters tightly in Cr/Cb regardless of luminance.
    HSV:   rejects the desaturated greys (walls, clothing) that survive the YCrCb gate.
    """
    import cv2
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    m2 = cv2.inRange(hsv, (0, 30, 50), (25, 255, 255))
    m3 = cv2.inRange(hsv, (160, 30, 50), (180, 255, 255))     # hue wraps at red
    m = cv2.bitwise_and(m1, cv2.bitwise_or(m2, m3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


class SkinModel:
    """Skin mask that learns THIS person under THIS light, seeded from the pose landmarks.

    The fixed-threshold `skin_mask` above rejects skin it should keep. Measured on one skin
    colour dimmed progressively: identical chrominance (S = 97 throughout) passes at 35%
    illumination and fails at 25%, purely because of the V >= 50 floor. A forearm in its own
    shadow therefore drops out of the mask, and the patches sitting on it go to nan. The fixed
    Cr/Cb box has the same problem across skin tones -- it is centred on one of them.

    Since the tracked points are on the body by construction, they are free labelled samples of
    the right answer. Take the median and MAD over them and gate relative to that, so the mask
    follows both skin tone and illumination. Median/MAD rather than mean/sd because some seeds
    legitimately land on clothing, and a robust centre ignores a minority of them.

    Colour space matters more than the fitting. YCrCb is the obvious choice and it is the wrong
    one here: scaling RGB by f scales (R - Y) by f, so Cr = 0.713 (R - Y) + 128 collapses toward
    128 as a pixel darkens. Shadowed skin drifts to neutral, away from the lit skin the model was
    fitted on, and gets cut -- the exact failure being fixed. Normalised rg-chromaticity,
    r = R/(R+G+B), is invariant to that scaling because the factor cancels, so the same skin in
    and out of shadow lands in the same place.
    """

    def __init__(self, k=3.0, refresh=15, v_floor=12, s_floor=42, v_ceil=248):
        # k lowered from 5.0: at five MADs the ellipse routinely swallowed the background.
        # s_floor is the fix that matters. rg-chromaticity cannot separate skin from a white or
        # grey wall -- every neutral colour sits at r = g = 1/3, and light skin is only ~0.006
        # away in g, well inside the 0.02 tolerance floor. Skin is always chromatic; unsaturated
        # pixels are not skin whatever their rg. v_ceil drops blown-out highlights, which are
        # colourless for the same reason.
        self.k, self.refresh, self.v_floor = k, refresh, v_floor
        self.s_floor, self.v_ceil = s_floor, v_ceil
        self.med = self.tol = None
        self._n = 0

    @staticmethod
    def _rg(bgr):
        """Normalised rg-chromaticity, invariant to illumination scale."""
        f = bgr.astype(np.float32)
        s = f.sum(2) + 1e-6
        return f[:, :, 2] / s, f[:, :, 1] / s          # BGR: index 2 = R, 1 = G

    def update(self, bgr, pts, r=9):
        """Re-fit from seed points. Cheap, so it runs every `refresh` frames, not every frame."""
        if not pts:
            return
        rr, gg = self._rg(bgr)
        h, w = bgr.shape[:2]
        samp = []
        for (x, y) in pts:
            y0, y1 = max(0, y - r), min(h, y + r)
            x0, x1 = max(0, x - r), min(w, x + r)
            if y1 > y0 and x1 > x0:
                samp.append(np.stack([rr[y0:y1, x0:x1].ravel(), gg[y0:y1, x0:x1].ravel()], 1))
        if not samp:
            return
        samp = np.concatenate(samp, 0)
        if len(samp) < 50:
            return
        med = np.median(samp, 0)
        mad = np.median(np.abs(samp - med), 0) * 1.4826
        self.med = med
        self.tol = np.maximum(self.k * mad, 0.02)      # floor: never collapse to a point
    def mask(self, bgr, pts=None):
        import cv2
        if pts is not None and (self._n % self.refresh == 0 or self.med is None):
            self.update(bgr, pts)
        self._n += 1
        if self.med is None:                           # not seeded yet: fixed fallback
            return skin_mask(bgr)
        rr, gg = self._rg(bgr)
        m = ((np.abs(rr - self.med[0]) < self.tol[0]) &
             (np.abs(gg - self.med[1]) < self.tol[1])).astype(np.uint8) * 255
        # Saturation and value gates, applied AFTER the learned rg test. Below s_floor a pixel
        # is neutral -- wall, paper, white shirt -- and rg cannot tell it from skin. Above
        # v_ceil it is clipped and equally colourless.
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v, sat = hsv[:, :, 2], hsv[:, :, 1]
        m = cv2.bitwise_and(m, ((v > self.v_floor) & (v < self.v_ceil) &
                                (sat > self.s_floor)).astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def patch_boxes(roi, grid=GRID):
    x, y, w, h = roi
    pw, ph = w // grid, h // grid
    return [(x + i * pw, y + j * ph, pw, ph)
            for j in range(grid) for i in range(grid) if pw > 4 and ph > 4]


def patch_quality(sig, fs, lo=R.BAND[0], hi=R.BAND[1]):
    """Fraction of in-band power at the dominant peak. High = a clean single-frequency pulse."""
    from scipy.signal import welch
    if len(sig) < int(4 * fs) or np.std(sig) < 1e-9:
        return 0.0, np.nan
    f, P = welch(sig, fs, nperseg=min(len(sig), int(6 * fs)))
    m = (f > lo) & (f < hi)
    if not m.any() or P[m].sum() <= 0:
        return 0.0, np.nan
    k = int(np.argmax(P[m]))
    return float(P[m][k] / P[m].sum()), float(f[m][k])


def phase_at(sig, fs, f0):
    """Phase of the trace at the cardiac frequency, via a single-bin DFT.

    Sign convention, verified against synthetic delays: a signal DELAYED by d seconds has phase
    LOWER by 2*pi*f0*d, so arrival time = -(phase difference) / (2*pi*f0). Callers must negate;
    `arrival_ms` below does it so the sign lives in exactly one place.
    """
    t = np.arange(len(sig)) / fs
    z = np.sum(sig * np.exp(-2j * np.pi * f0 * t))
    return float(np.angle(z)), float(np.abs(z))


def arrival_ms(phase, ref_phase, f0):
    """Arrival time (ms) of a patch relative to the reference. Positive = arrives LATER."""
    d = np.angle(np.exp(1j * (phase - ref_phase)))
    return -d / (2 * np.pi * f0) * 1000.0


def capture(seconds, cam=0, show=True):
    """Record mean skin-masked RGB for every patch of every site."""
    import cv2
    cap = rppg_cam.open_camera(cam, 640, 480, 60)   # platform-aware; CAP_DSHOW is Windows-only
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    sites = {}
    for name in ("forehead", "neck_right", "neck_left", "hand"):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        prev = frame.copy()
        m = skin_mask(frame)
        prev[m == 0] = (prev[m == 0] * 0.25).astype(np.uint8)   # dim non-skin so the gate is visible
        cv2.putText(prev, f"drag {name.upper()} (bright = detected skin), ENTER; ESC to skip",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        r = cv2.selectROI(f"select {name}", prev, False, False)
        cv2.destroyAllWindows()
        if r[2] > 8 and r[3] > 8:
            sites[name] = tuple(int(v) for v in r)
    if "neck_right" not in sites or "hand" not in sites:
        raise RuntimeError("need at least neck_right and hand")

    boxes = {s: patch_boxes(r) for s, r in sites.items()}
    acc = {s: [[] for _ in b] for s, b in boxes.items()}
    T = []
    t0 = time.time()
    print(f"[cap] recording {seconds}s over "
          f"{sum(len(b) for b in boxes.values())} patches", flush=True)
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        msk = skin_mask(frame)
        for s, bs in boxes.items():
            for i, (x, y, w, h) in enumerate(bs):
                p = frame[y:y + h, x:x + w]
                mm = msk[y:y + h, x:x + w]
                if p.size == 0:
                    acc[s][i].append((np.nan, np.nan, np.nan, 0.0)); continue
                frac = float((mm > 0).mean())
                if frac < MIN_SKIN:
                    acc[s][i].append((np.nan, np.nan, np.nan, frac)); continue
                px = p[mm > 0][:, ::-1].mean(0)            # BGR -> RGB, skin pixels only
                acc[s][i].append((px[0], px[1], px[2], frac))
        T.append(time.time() - t0)
        if show:
            vis = frame.copy()
            vis[msk == 0] = (vis[msk == 0] * 0.35).astype(np.uint8)
            for s, bs in boxes.items():
                col = {"forehead": (0, 255, 255), "neck_right": (0, 255, 0),
                       "neck_left": (0, 200, 120), "hand": (0, 128, 255)}[s]
                for (x, y, w, h) in bs:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
                x, y, w, h = sites[s]
                cv2.putText(vis, s, (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
            cv2.putText(vis, f"{seconds-(time.time()-t0):4.0f}s", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.imshow("skin-masked multi-patch - q to stop", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return {s: np.array(v, float) for s, v in acc.items()}, np.array(T), sites, boxes


def analyse(acc, T, tag, sites, boxes, make_map=False):
    fs = (len(T) - 1) / (T[-1] - T[0])
    tu = np.linspace(T[0], T[-1], len(T))
    print(f"[cap] {len(T)} frames, {fs:.1f} fps", flush=True)

    traces, quals, freqs = {}, {}, {}
    for s, arr in acc.items():
        tr, q, f0s = [], [], []
        for i in range(arr.shape[0]):
            rgb = arr[i][:, :3]
            good = np.isfinite(rgb).all(1)
            if good.mean() < 0.6:
                tr.append(None); q.append(0.0); f0s.append(np.nan); continue
            filled = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
            x = R.bandpass(R.chrom(filled), fs)
            qq, ff = patch_quality(x, fs)
            tr.append(x); q.append(qq); f0s.append(ff)
        traces[s], quals[s], freqs[s] = tr, np.array(q), np.array(f0s)

    # global cardiac frequency: the quality-weighted median across every usable patch
    allq = np.concatenate([quals[s] for s in quals])
    allf = np.concatenate([freqs[s] for s in freqs])
    ok = np.isfinite(allf) & (allq > 0.1)
    f0 = float(np.median(allf[ok])) if ok.any() else np.nan
    print(f"[sig] cardiac frequency {f0:.2f} Hz ({f0*60:.0f} bpm) from {ok.sum()} patches",
          flush=True)

    combined = {}
    print(f"\n{'site':12s} {'patches':>8s} {'used':>5s} {'mean q':>7s}")
    for s in traces:
        w = quals[s].copy()
        w[w < 0.10] = 0.0                      # drop patches with no coherent pulse
        used = int((w > 0).sum())
        if used == 0:
            print(f"{s:12s} {len(w):8d} {used:5d}      --   (no usable patch)"); continue
        stack = np.stack([t / (np.std(t) + 1e-9) for t, ww in zip(traces[s], w) if ww > 0])
        ww = w[w > 0]
        combined[s] = np.average(stack, axis=0, weights=ww)
        print(f"{s:12s} {len(w):8d} {used:5d} {ww.mean():7.3f}")

    ref = "neck_right" if "neck_right" in combined else list(combined)[0]
    out = {"tag": tag, "fps": fs, "n_frames": len(T), "hr_bpm": f0 * 60 if f0 == f0 else None,
           "reference_site": ref, "lags_ms": {}, "window_lags": {}}
    print(f"\n{'site':12s} {'lag vs '+ref:>16s} {'sd':>7s}")
    for s, x in combined.items():
        if s == ref:
            continue
        lag, _ = R.lag_subframe(combined[ref], x, fs)
        W = int(10 * fs)
        wl = [R.lag_subframe(combined[ref][i:i + W], x[i:i + W], fs)[0]
              for i in range(0, len(x) - W, W // 2)] if len(x) > W else []
        out["lags_ms"][s] = lag
        out["window_lags"][s] = [float(v) for v in wl]
        print(f"{s:12s} {lag:+16.1f} {np.std(wl) if wl else float('nan'):7.1f}")

    (DATA / f"rppg_multi_{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(DATA / f"rppg_multi_{tag}.npz",
             **{f"sig_{s}": v for s, v in combined.items()}, fs=fs, t=tu)

    if make_map and f0 == f0:
        spatial_map(traces, quals, boxes, sites, fs, f0, tag)
    print(f"\n[done] data/rppg_multi_{tag}.json")
    return out


def spatial_map(traces, quals, boxes, sites, fs, f0, tag):
    """Per-patch arrival time rendered as a propagation field."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, lag_ms, q = [], [], []
    ref_phase = None
    for s in traces:
        for i, tr in enumerate(traces[s]):
            if tr is None or quals[s][i] < 0.10:
                continue
            ph, amp = phase_at(tr, fs, f0)
            x, y, w, h = boxes[s][i]
            if ref_phase is None:
                ref_phase = ph
            pts.append((x + w / 2, y + h / 2))
            lag_ms.append(arrival_ms(ph, ref_phase, f0))
            q.append(quals[s][i])
    if len(pts) < 4:
        print("[map] too few good patches for a map"); return
    pts = np.array(pts); lag_ms = np.array(lag_ms); q = np.array(q)
    lag_ms -= np.median(lag_ms)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = ax[0].scatter(pts[:, 0], -pts[:, 1], c=lag_ms, s=90 + 400 * q, cmap="coolwarm",
                       edgecolors="k", linewidths=.4)
    for s, r in sites.items():
        x, y, w, h = r
        ax[0].add_patch(plt.Rectangle((x, -y - h), w, h, fill=False, ec="#555", lw=.8))
        ax[0].text(x, -y + 6, s, fontsize=7.5, color="#333")
    plt.colorbar(sc, ax=ax[0], label="pulse arrival, relative (ms)")
    # propagation arrows: fit a plane to arrival(x, y), so the gradient points from EARLY to
    # LATE, i.e. along the direction the wave travels. A coherent field gives consistent arrows;
    # noise gives none, which is the visual check that the timing is physiological.
    if len(pts) >= 6:
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        coef, *_ = np.linalg.lstsq(A * q[:, None], lag_ms * q, rcond=None)
        gx, gy = float(coef[0]), float(coef[1])
        nrm = np.hypot(gx, gy)
        if nrm > 1e-9:
            span = max(pts[:, 0].ptp(), pts[:, 1].ptp())
            sc_ = 0.22 * span / nrm
            for px, py in pts[::max(1, len(pts) // 12)]:
                ax[0].arrow(px, -py, gx * sc_, -gy * sc_, head_width=.02 * span,
                            color="k", alpha=.55, length_includes_head=True)
            pred = A @ coef
            r2 = 1 - np.sum((lag_ms - pred) ** 2) / (np.sum((lag_ms - lag_ms.mean()) ** 2) + 1e-9)
            ax[0].text(.02, .02, f"propagation R² = {r2:.2f}", transform=ax[0].transAxes,
                       fontsize=8.5, fontweight="bold")
    ax[0].set_title("a  spatio-temporal pulse arrival (arrows: early → late)", loc="left",
                    fontsize=10, fontweight="bold")
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_aspect("equal")

    order = np.argsort(lag_ms)
    ax[1].scatter(lag_ms[order], np.arange(len(order)), c=lag_ms[order], cmap="coolwarm", s=28)
    ax[1].set_xlabel("relative arrival (ms)", fontsize=9)
    ax[1].set_ylabel("patch (sorted)", fontsize=9)
    ax[1].set_title(f"b  spread {lag_ms.max()-lag_ms.min():.0f} ms across {len(lag_ms)} patches",
                    loc="left", fontsize=10, fontweight="bold")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_rppg_map_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] figures/fig_rppg_map_{tag}.png")
    print("[map] NOTE: phase is wrapped, so arrivals are only interpretable modulo one cardiac")
    print("      cycle (~%.0f ms). A coherent proximal->distal gradient is the thing to look"
          % (1000 / f0))
    print("      for; isolated outliers are usually low-quality patches.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tag", default="rest")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--map", action="store_true")
    args = ap.parse_args()
    acc, T, sites, boxes = capture(args.seconds, args.cam)
    if len(T) < 100:
        print("[err] too few frames"); return
    analyse(acc, T, args.tag, sites, boxes, make_map=args.map)


if __name__ == "__main__":
    main()
