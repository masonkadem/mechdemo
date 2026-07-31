"""rppg_sota.py -- POS extraction, an anatomical arm chain, and instantaneous HR.

Three upgrades over rppg_multi, each motivated by what the first real recordings showed.

1. POS available alongside CHROM -- NOT asserted as better here
   The first recordings gave excellent HR (67.6 / 67.7 / 67.5 bpm across three conditions,
   matching a wrist reference) but unusable timing: per-window lag scatter of 44-119 ms against
   an effect of 20-50 ms, and one run at SNR 2.2. POS (Wang 2017, Plane-Orthogonal-to-Skin) is
   the usual recommendation over CHROM (de Haan 2013) for motion robustness, and is implemented
   here with the paper's temporal normalisation and overlap-add windowing.

   HOWEVER, on our own synthetic benchmarks CHROM beat POS: median SNR 2420 vs 779 with identical
   HR error, both for shared additive motion and for channel-dependent gain motion. That most
   likely reflects the benchmark rather than the methods -- POS's advantage is documented for
   real motion with specular reflection and varying illumination spectra, which simple synthetic
   models do not reproduce. So `--method` selects between them and defaults to CHROM, the one
   that measurably wins on the evidence we actually have. Decide between them on YOUR recordings,
   not on a claim from the literature or from this docstring.

2. The arm chain -- the validation that matters
   A two-point lag cannot distinguish transit time from a fixed processing offset. Sampling a
   CHAIN of sites down the arm (neck -> shoulder -> upper arm -> forearm -> hand) gives a
   monotonic anatomical distance axis. If arrival time grows linearly with distance, the
   measurement is propagation; if it does not, it is artifact. The slope is pulse wave velocity
   in m/s, which is the physiological quantity the whole project is about, and PWV has a known
   range (4-12 m/s for the upper limb) that provides an external sanity check no two-point
   measurement can offer.

3. Instantaneous HR
   Per-beat HR from peak detection on the POS trace, plus a rolling spectral estimate, so the
   number can be compared against a watch in real time rather than after the fact.

Deliberately NOT included: pupil diameter. It indexes autonomic arousal, not pulse arrival, and
at 30 fps a webcam cannot resolve it reliably -- it would add a confound, not information.

    python rppg_sota.py --seconds 60 --tag rest --chain
    python rppg_sota.py --hr-only          # live HR readout, for watch comparison
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import rppg_cam
import rppg_two_site as R
import rppg_multi as M

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# anatomical chain, proximal -> distal, with rough distances along the arterial path (cm).
# Distances are nominal; --measure lets the operator enter their own.
CHAIN = [("neck", 0.0), ("shoulder", 18.0), ("upper_arm", 33.0),
         ("forearm", 55.0), ("hand", 75.0)]


def pos(rgb, fs, win_s=1.6):
    """Plane-Orthogonal-to-Skin rPPG (Wang et al. 2017).

    Per sliding window: temporally normalise RGB by its own mean, project onto the two
    plane-orthogonal directions, combine them with a ratio of standard deviations, and overlap-add
    the mean-removed result. This is the step that buys motion robustness over CHROM.
    """
    n = len(rgb)
    L = max(int(win_s * fs), 16)
    H = np.zeros(n)
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for i in range(0, n - L + 1):
        win = rgb[i:i + L]
        mu = win.mean(0)
        if np.any(mu <= 0):
            continue
        Cn = win / mu                       # temporal normalisation
        S = P @ Cn.T                        # (2, L)
        s1, s2 = S[0], S[1]
        a = np.std(s1) / (np.std(s2) + 1e-9)
        h = s1 + a * s2
        H[i:i + L] += (h - h.mean()) / (np.std(h) + 1e-9)     # overlap-add
    return H


def instantaneous_hr(sig, fs, min_bpm=40, max_bpm=180):
    """Per-beat HR from peak detection, plus a robust summary.

    Returns (times, bpm_per_beat, median_bpm, ibi_sd_ms). The IBI scatter is reported because a
    physiological trace has modest beat-to-beat variability while a noise-driven one does not --
    it is a cheap check that peaks are beats.
    """
    from scipy.signal import find_peaks
    x = (sig - sig.mean()) / (sig.std() + 1e-9)
    d = int(fs * 60.0 / max_bpm)
    pk, _ = find_peaks(x, distance=max(d, 2), prominence=0.4)
    if len(pk) < 4:
        return np.array([]), np.array([]), float("nan"), float("nan")
    ibi = np.diff(pk) / fs
    ok = (ibi > 60.0 / max_bpm) & (ibi < 60.0 / min_bpm)
    ibi = ibi[ok]
    if len(ibi) < 3:
        return np.array([]), np.array([]), float("nan"), float("nan")
    bpm = 60.0 / ibi
    return pk[1:][ok] / fs, bpm, float(np.median(bpm)), float(np.std(ibi) * 1000)


def rolling_hr(sig, fs, win_s=8.0):
    """Spectral HR in a sliding window -- steadier than per-beat, good for a live readout."""
    from scipy.signal import welch
    L = int(win_s * fs)
    out = []
    for i in range(0, max(len(sig) - L, 1), max(int(fs), 1)):
        w = sig[i:i + L]
        if len(w) < L // 2:
            break
        f, P = welch(w, fs, nperseg=min(len(w), int(6 * fs)))
        m = (f > R.BAND[0]) & (f < R.BAND[1])
        if m.any():
            out.append(float(f[m][np.argmax(P[m])] * 60))
    return np.array(out)


def capture_chain(seconds, sites, cam=0, show=True, grid=2):
    """Record skin-masked RGB for a list of named sites, each tiled into a small grid."""
    import cv2
    cap = rppg_cam.open_camera(cam, 640, 480, 60)   # platform-aware; CAP_DSHOW is Windows-only
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    boxes = {}
    for name in sites:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        prev = frame.copy()
        msk = M.skin_mask(frame)
        prev[msk == 0] = (prev[msk == 0] * 0.25).astype(np.uint8)
        cv2.putText(prev, f"drag {name.upper()}  (bright = skin)  ENTER, or ESC to skip",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        r = cv2.selectROI(f"select {name}", prev, False, False)
        cv2.destroyAllWindows()
        if r[2] > 8 and r[3] > 8:
            boxes[name] = M.patch_boxes(tuple(int(v) for v in r), grid)
    if len(boxes) < 2:
        raise RuntimeError("need at least two sites")

    acc = {s: [[] for _ in b] for s, b in boxes.items()}
    T = []
    t0 = time.time()
    print(f"[cap] {len(boxes)} sites, {sum(len(b) for b in boxes.values())} patches", flush=True)
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        msk = M.skin_mask(frame)
        for s, bs in boxes.items():
            for i, (x, y, w, h) in enumerate(bs):
                p = frame[y:y + h, x:x + w]
                mm = msk[y:y + h, x:x + w]
                if p.size == 0 or (mm > 0).mean() < M.MIN_SKIN:
                    acc[s][i].append((np.nan, np.nan, np.nan))
                else:
                    acc[s][i].append(tuple(p[mm > 0][:, ::-1].mean(0)))
        T.append(time.time() - t0)
        if show:
            vis = frame.copy()
            vis[msk == 0] = (vis[msk == 0] * .35).astype(np.uint8)
            for k, (s, bs) in enumerate(boxes.items()):
                col = [(0, 255, 0), (0, 220, 120), (0, 200, 220), (0, 150, 255),
                       (0, 100, 255)][k % 5]
                for (x, y, w, h) in bs:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
                cv2.putText(vis, s, (bs[0][0], max(14, bs[0][1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1)
            el = time.time() - t0
            cv2.putText(vis, f"{seconds-el:4.0f}s   fps {len(T)/max(el,1e-3):4.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.imshow("chain capture - q to stop", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return {s: np.array(v, float) for s, v in acc.items()}, np.array(T), boxes


def site_signal(arr, T, tu, fs, method="chrom"):
    """Quality-weighted rPPG signal for one site, combining its patches.

    `method` selects the chrominance projection: "chrom" (de Haan 2013) or "pos" (Wang 2017).
    CHROM is the default because it won on our synthetic benchmarks; see the module docstring.
    """
    proj = (lambda a: pos(a, fs)) if method == "pos" else R.chrom
    sigs, qs = [], []
    for i in range(arr.shape[0]):
        rgb = arr[i]
        good = np.isfinite(rgb).all(1)
        if good.mean() < 0.6:
            continue
        filled = np.stack([np.interp(tu, T[good], rgb[good, c]) for c in range(3)], 1)
        x = R.bandpass(proj(filled), fs)
        q, _ = M.patch_quality(x, fs)
        if q > 0.10:
            sigs.append(x / (np.std(x) + 1e-9)); qs.append(q)
    if not sigs:
        return None, 0.0, 0
    w = np.array(qs)
    return np.average(np.stack(sigs), axis=0, weights=w), float(w.mean()), len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--tag", default="rest")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--chain", action="store_true", help="record the full arm chain")
    ap.add_argument("--hr-only", action="store_true", help="live HR readout only")
    ap.add_argument("--method", default="chrom", choices=["chrom", "pos"],
                    help="chrominance projection; chrom won our benchmarks, pos is the "
                         "literature default for motion")
    ap.add_argument("--distances", default="",
                    help="comma-separated cm along the arterial path, matching the sites used")
    args = ap.parse_args()

    if args.hr_only:
        sites = ["forehead"]
    elif args.chain:
        sites = [s for s, _ in CHAIN]
    else:
        sites = ["neck", "hand"]

    acc, T, boxes = capture_chain(args.seconds, sites, args.cam)
    if len(T) < 100:
        print("[err] too few frames"); return
    fs = (len(T) - 1) / (T[-1] - T[0])
    tu = np.linspace(T[0], T[-1], len(T))
    print(f"[cap] {len(T)} frames, {fs:.1f} fps", flush=True)
    if fs < 20:
        print(f"[warn] {fs:.0f} fps is too low for reliable timing -- close other camera apps, "
              f"lower the site count, or improve lighting", flush=True)

    sig, qual, npatch = {}, {}, {}
    for s, arr in acc.items():
        x, q, n = site_signal(arr, T, tu, fs, args.method)
        if x is not None:
            sig[s], qual[s], npatch[s] = x, q, n

    print(f"\n{'site':12s} {'patches':>8s} {'quality':>8s} {'HR bpm':>8s} {'IBI sd ms':>10s}")
    hrs = {}
    for s, x in sig.items():
        _, _, hr, ibisd = instantaneous_hr(x, fs)
        hrs[s] = hr
        print(f"{s:12s} {npatch[s]:8d} {qual[s]:8.3f} {hr:8.1f} {ibisd:10.1f}")
    good_hr = [v for v in hrs.values() if v == v]
    if good_hr:
        print(f"\n[HR] median across sites {np.median(good_hr):.1f} bpm  "
              f"(spread {np.ptp(good_hr):.1f}) -- compare against your watch")

    out = {"tag": args.tag, "fps": fs, "n_frames": len(T), "method": args.method,
           "hr_bpm": float(np.median(good_hr)) if good_hr else None,
           "hr_by_site": hrs, "quality": qual, "n_patches": npatch}

    # ---- the arm chain: arrival vs anatomical distance ----------------------
    if args.chain and len(sig) >= 3:
        dist = dict(CHAIN)
        if args.distances:
            vals = [float(v) for v in args.distances.split(",")]
            dist = {s: v for s, v in zip(sites, vals)}
        ref = "neck" if "neck" in sig else list(sig)[0]
        D, Lg = [], []
        print(f"\n{'site':12s} {'distance cm':>12s} {'lag vs '+ref+' ms':>18s}")
        for s, x in sig.items():
            if s == ref or s not in dist:
                continue
            lag, _ = R.lag_subframe(sig[ref], x, fs)
            D.append(dist[s] - dist[ref]); Lg.append(lag)
            print(f"{s:12s} {dist[s]-dist[ref]:12.1f} {lag:18.1f}")
        if len(D) >= 3:
            D, Lg = np.array(D), np.array(Lg)
            sl, ic = np.polyfit(D, Lg, 1)
            r = float(np.corrcoef(D, Lg)[0, 1])
            pwv = (0.01 / (sl / 1000.0)) if abs(sl) > 1e-9 else float("inf")
            out["chain"] = {"distances_cm": D.tolist(), "lags_ms": Lg.tolist(),
                            "slope_ms_per_cm": float(sl), "r": r, "pwv_m_s": float(pwv)}
            print(f"\n[chain] arrival vs distance: slope {sl:+.3f} ms/cm, r = {r:+.3f}")
            print(f"[chain] implied PWV = {pwv:.1f} m/s")
            if 4.0 <= pwv <= 12.0 and r > 0.8:
                print("[chain] PLAUSIBLE: within the 4-12 m/s upper-limb range and monotonic in "
                      "distance -- this is propagation, not a fixed offset.")
            else:
                print("[chain] NOT plausible. A PWV outside 4-12 m/s or a weak distance "
                      "correlation means the lags are dominated by artifact, not transit.")

    (DATA / f"rppg_sota_{args.tag}.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez(DATA / f"rppg_sota_{args.tag}.npz", **{f"sig_{s}": v for s, v in sig.items()},
             fs=fs, t=tu)
    print(f"\n[done] data/rppg_sota_{args.tag}.json")


if __name__ == "__main__":
    main()
