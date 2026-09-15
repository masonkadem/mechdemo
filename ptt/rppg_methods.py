"""rppg_methods.py -- the classical rPPG extractors, on one interface, with a way to compare them.

    python rppg_methods.py                       # rank every method on the saved recordings
    python rppg_methods.py --file data/rppg_pose_s01_rest.npz
    python rppg_methods.py --selftest            # synthetic ground truth

What each method is for, and why FFT is not one of them
-------------------------------------------------------
Worth separating two jobs that are easy to conflate:

  SEPARATION   recovering the pulse from an RGB trace dominated by motion and illumination
               change. This is what GREEN/CHROM/POS/ICA/PCA do, and it is where the quality of
               an rPPG signal is won or lost.
  MEASUREMENT  reading a rate or a lag off the recovered pulse. This is what the FFT/Welch
               periodogram and the cross-correlation do.

An FFT cannot do the separation job. A periodogram of a raw green trace shows the pulse peak AND
the motion peak, and it cannot tell you which is which -- if anything it flatters the artefact,
because motion at 1-2 Hz sits squarely inside the 0.7-3 Hz pulse band. So the methods below all
run BEFORE the FFT that already exists in the pipeline.

The methods
-----------
GREEN   the raw green channel. Haemoglobin absorbs green strongly, so it carries the most
        pulsatile signal of the three -- but also every lighting change, at full strength.
        Kept as the baseline that any real method must beat.

CHROM   (de Haan & Jeanne 2013) two colour-difference signals, X = 3R-2G and Y = 1.5R+G-1.5B,
        combined as X/sd(X) - Y/sd(Y). The chrominance ratio cancels illumination intensity to
        first order, because a brightness change scales all three channels together. This is
        what the pipeline currently uses.

POS     (Wang et al. 2017, "Algorithmic Principles of Remote PPG") projects onto the plane
        ORTHOGONAL to the skin-tone direction in temporally-normalised RGB space. The insight is
        that specular reflection -- the part that carries the lighting and motion artefact -- lies
        along the skin-tone axis, so projecting it out removes the artefact rather than trying to
        cancel it. Generally the strongest unsupervised method in the literature and the sensible
        default; on the benchmarks it beats CHROM most clearly exactly where this rig is weakest,
        namely low light and small head motion.

ICA     (Poh et al. 2010) treats R, G, B as three sensors observing a mixture of independent
        sources and runs FastICA, then picks the component whose spectrum is most pulse-like.
        Strong when the artefact is genuinely independent of the pulse; its weakness is that the
        component order is arbitrary, so the selection rule matters as much as the unmixing, and
        with only 3 channels it is separating 3 sources at most.

PCA     the same idea with an orthogonality assumption instead of independence. Cheaper, and
        usually a little worse, but a useful check: if PCA and ICA disagree a lot, the mixture
        is not well described by 3 linear sources and neither should be trusted much.

All return a band-passed 1-D signal on the same time base, so they are drop-in interchangeable
and `compare` can score them against each other on your own recordings rather than on a paper's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

METHODS = ("green", "chrom", "pos", "ica", "pca")
DEFAULT = "pos"

# Methods that cancel ILLUMINATION INTENSITY by construction, because they work on ratios between
# colour channels rather than on the channels themselves. A brightness change scales R, G and B
# together and so drops out of a ratio.
#
# The others (green, pca, ica) are intensity-based: they see a light flicker as signal. PCA and ICA
# can in principle learn to reject it, but only if it is orthogonal to / independent of the pulse,
# and a 1.4 Hz flicker inside the 0.7-3 Hz pulse band is neither.
CHROMINANCE = ("chrom", "pos")
INTENSITY = ("green", "pca", "ica")


# ------------------------------------------------------------------------------ helpers
def _finite(rgb):
    """Interpolate over nan gaps so a dropped frame does not end the trace.

    Every method below is a linear operation across channels and time; a single nan would
    otherwise propagate through the filter and destroy the whole signal rather than one sample.
    """
    rgb = np.asarray(rgb, float).copy()
    n = len(rgb)
    idx = np.arange(n)
    for c in range(rgb.shape[1]):
        g = np.isfinite(rgb[:, c])
        if g.sum() < 2:
            return None
        if not g.all():
            rgb[~g, c] = np.interp(idx[~g], idx[g], rgb[g, c])
    return rgb


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _pulse_score(x, fs):
    """Peak-to-median in-band power: the same SNR the quality gate uses.

    Reused deliberately -- a method that scores well here is a method whose output will actually
    clear the pipeline's gate, which is the only thing that matters operationally.
    """
    from scipy.signal import welch
    if x is None or len(x) < int(4 * fs) or not np.isfinite(x).all() or np.std(x) < 1e-12:
        return 0.0, np.nan
    f, P = welch(x, fs, nperseg=min(len(x), int(6 * fs)))
    m = (f > R.BAND[0]) & (f < R.BAND[1])
    if not m.any() or P[m].sum() <= 0:
        return 0.0, np.nan
    k = int(np.argmax(P[m]))
    return float(P[m][k] / (np.median(P[m]) + 1e-15)), float(f[m][k] * 60)


# ------------------------------------------------------------------------------ methods
def green(rgb, fs):
    rgb = _finite(rgb)
    return None if rgb is None else R.bandpass(_norm(rgb[:, 1]), fs)


def chrom(rgb, fs):
    rgb = _finite(rgb)
    return None if rgb is None else R.bandpass(R.chrom(rgb), fs)


def pos(rgb, fs, win_s=1.6):
    """Plane-Orthogonal-to-Skin (Wang et al. 2017).

    Sliding window, because the skin-tone direction drifts as illumination and pose change; a
    single global projection would be optimal only at the mean appearance of the whole recording.
    Windows are overlap-added after being individually mean-removed, which is what makes the
    result insensitive to slow baseline drift without a separate detrend.
    """
    rgb = _finite(rgb)
    if rgb is None:
        return None
    n = len(rgb)
    L = max(int(win_s * fs), 20)
    if n < L + 2:
        L = n                                  # short trace: one window, still well defined
    # The projection that annihilates the skin-tone axis in temporally-normalised space.
    S = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(n)
    for i in range(0, n - L + 1):
        w = rgb[i:i + L]
        mu = w.mean(0)
        if np.any(np.abs(mu) < 1e-12):
            continue
        Cn = (w / mu).T                        # temporal normalisation, per channel
        Z = S @ Cn                             # 2 x L
        s1, s2 = Z[0], Z[1]
        a = np.std(s1) / (np.std(s2) + 1e-12)
        h = s1 + a * s2                        # alpha-tuning: equalise the two projections
        out[i:i + L] += h - h.mean()           # overlap-add, each window zero-mean
    if not np.any(out):
        return None
    return R.bandpass(_norm(out), fs)


def _pick_component(comps, fs):
    """Choose the component that looks most like a pulse, by in-band SNR.

    ICA and PCA both return components in an order that carries no physiological meaning, so the
    selection rule is part of the method, not an afterthought. Ranking by the same SNR the quality
    gate uses keeps the choice consistent with how the signal will later be judged.
    """
    best, bs = None, -1.0
    for c in comps:
        s, _ = _pulse_score(R.bandpass(_norm(c), fs), fs)
        if s > bs:
            best, bs = c, s
    return best


def ica(rgb, fs, seed=0):
    """FastICA on the three colour channels (Poh et al. 2010)."""
    rgb = _finite(rgb)
    if rgb is None:
        return None
    X = np.stack([_norm(rgb[:, c]) for c in range(3)], 1)
    try:
        import warnings as _w
        from sklearn.decomposition import FastICA
        with _w.catch_warnings():
            # Non-convergence is common on 3 near-collinear colour channels and is not fatal --
            # the returned unmixing is still usable, and the component is chosen by SNR anyway.
            # Left unsuppressed it printed once per site, i.e. 34 times per recording.
            _w.simplefilter("ignore")
            S = FastICA(n_components=3, random_state=seed, max_iter=1000, tol=1e-3,
                        whiten="unit-variance").fit_transform(X).T
    except Exception:                          # noqa: BLE001 -- convergence or missing sklearn
        return pca(rgb, fs)                    # degrade to the orthogonal cousin, never to None
    c = _pick_component(S, fs)
    return None if c is None else R.bandpass(_norm(c), fs)


def pca(rgb, fs):
    rgb = _finite(rgb)
    if rgb is None:
        return None
    X = np.stack([_norm(rgb[:, c]) for c in range(3)], 1)
    X = X - X.mean(0)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    c = _pick_component((X @ Vt.T).T, fs)
    return None if c is None else R.bandpass(_norm(c), fs)


FUNCS = {"green": green, "chrom": chrom, "pos": pos, "ica": ica, "pca": pca}


def extract(rgb, fs, method=DEFAULT):
    """One RGB trace -> one band-passed pulse signal, by the named method."""
    if method not in FUNCS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(FUNCS)}")
    return FUNCS[method](np.asarray(rgb, float), float(fs))


def compare(rgb, fs, methods=METHODS):
    """{method: (snr, hr_bpm)} for one trace -- the per-site view."""
    out = {}
    for m in methods:
        try:
            out[m] = _pulse_score(extract(rgb, fs, m), fs)
        except Exception as e:                 # noqa: BLE001 -- one bad method must not stop the rest
            out[m] = (0.0, np.nan)
            print(f"[warn] {m}: {e}")
    return out


def compare_sites(acc, fs, methods=METHODS):
    """Rank methods across every site of a recording.

    Ranked by CROSS-SITE AGREEMENT first, not by SNR. The selftest shows why: on a trace with a
    1.4 Hz illumination artefact the raw green channel scores the HIGHEST in-band SNR of any
    method while reporting a heart rate 11 bpm wrong, because peak-to-median power cannot tell a
    strong pulse from a strong artefact sitting in the same band.

    Agreement is the artefact-resistant criterion available without ground truth: the pulse is
    the one oscillation every site on the body shares, so a method whose sites converge on a
    single rate has found the pulse, and one whose sites scatter has found their local noise.
    SNR remains as the tie-break, since among methods that agree the cleaner one is better.
    """
    import rppg_pose as P
    snrs = {m: [] for m in methods}
    hrs = {m: [] for m in methods}
    for i in range(acc.shape[1]):
        c = compare(acc[:, i, :], fs, methods)
        for m in methods:
            snrs[m].append(c[m][0])
            hrs[m].append(c[m][1])
    rows = []
    for m in methods:
        s, h = np.array(snrs[m], float), np.array(hrs[m], float)
        strong = s >= P.MIN_SNR
        hh = h[strong & np.isfinite(h)]
        med = float(np.median(hh)) if hh.size else np.nan
        agree = int(np.sum(np.abs(hh - med) <= P.HR_TOL_BPM)) if hh.size else 0
        rows.append({"method": m, "median_snr": float(np.median(s)),
                     "max_snr": float(s.max()) if s.size else 0.0,
                     "n_pass": int(strong.sum()), "n_sites": int(s.size),
                     "hr_med": med, "n_agree": agree})
    rows.sort(key=lambda r: (r["n_agree"], r["median_snr"]), reverse=True)
    return rows


def diagnose(rows):
    """Decide which method to trust, from physics rather than from the scores.

    Neither of the obvious data-driven criteria survives contact with a real recording:

      SNR              rewards locking onto a strong artefact. On the synthetic test the raw green
                       channel scores the HIGHEST in-band SNR of all five methods while reporting
                       a rate 11 bpm wrong.
      cross-site       does not help when the artefact is GLOBAL, and illumination artefacts are
      agreement        global almost by definition -- one lamp lights every site. In that test all
                       five methods agree perfectly, three of them on the artefact.

    So this does not pick a winner by score. It compares what the chrominance methods say against
    what the intensity methods say, and uses the DISAGREEMENT as the diagnostic: if they part
    company, something that is not the pulse is modulating brightness, and only the methods that
    cancel intensity can be believed. That inference does not need ground truth.
    """
    import rppg_pose as P
    by = {r["method"]: r for r in rows}
    ch = [by[m]["hr_med"] for m in CHROMINANCE if m in by and np.isfinite(by[m]["hr_med"])]
    it = [by[m]["hr_med"] for m in INTENSITY if m in by and np.isfinite(by[m]["hr_med"])]
    if not ch:
        return DEFAULT, ["no chrominance method produced a rate -- signal is very weak"]
    hc = float(np.median(ch))
    notes = [f"chrominance methods ({'/'.join(CHROMINANCE)}) agree on {hc:.0f} bpm"]
    if not it:
        return DEFAULT, notes
    hi = float(np.median(it))
    notes.append(f"intensity methods ({'/'.join(INTENSITY)}) say {hi:.0f} bpm")
    if abs(hc - hi) > P.HR_TOL_BPM:
        notes += [
            f"they DISAGREE by {abs(hc-hi):.0f} bpm -> an illumination or motion artefact is "
            f"modulating brightness in band.",
            f"Trust the chrominance result ({hc:.0f} bpm). The intensity methods are locked to "
            f"the artefact, which is also why their SNR looks better.",
            "Fix at source if you can: steady DC lighting, no window, no screen glow on the "
            "subject, and check for mains flicker at 50/60 Hz aliasing down into band.",
        ]
        best = "pos" if "pos" in by else "chrom"
    else:
        notes.append("they agree -> no strong intensity artefact; any method is usable, and the "
                     "highest-SNR one is a reasonable choice.")
        best = max((r for r in rows if r["method"] in CHROMINANCE),
                   key=lambda r: r["median_snr"])["method"]
    return best, notes


# ----------------------------------------------------------------------------- reporting
def _print_rows(rows, title):
    print(f"\n{title}")
    print(f"  {'method':8s} {'sites agree':>11s} {'median SNR':>11s} {'best SNR':>9s} "
          f"{'passing':>9s} {'HR':>7s}")
    best, notes = diagnose(rows)
    for r in rows:
        kind = "chrom-based" if r["method"] in CHROMINANCE else "intensity"
        star = "  <-- USE THIS" if r["method"] == best else ""
        print(f"  {r['method']:8s} {r['n_agree']:>7d}/{r['n_sites']:<3d} "
              f"{r['median_snr']:11.2f} {r['max_snr']:9.1f} "
              f"{r['n_pass']:>4d}/{r['n_sites']:<4d} {r['hr_med']:7.1f}  {kind:11s}{star}")
    # Printed after the table on purpose: the recommendation is NOT the top row of the ranking,
    # and presenting it as though it were would re-introduce the mistake the ranking makes.
    for n in notes:
        print(f"  * {n}")


def _selftest():
    """Synthetic ground truth: a pulse plus a strong illumination artefact.

    The artefact is applied to all three channels together, which is what a light flicker or a
    move toward a window actually does. That is the regime CHROM and POS are designed for, so a
    method that cannot beat GREEN here is not implemented correctly.
    """
    rng = np.random.default_rng(0)
    fs, dur, hr = 30.0, 30.0, 1.15
    t = np.arange(0, dur, 1 / fs)
    pulse = np.sin(2 * np.pi * hr * t) + 0.3 * np.sin(4 * np.pi * hr * t)
    # skin tone, pulse amplitude per channel (green strongest), shared multiplicative artefact
    tone = np.array([0.78, 0.55, 0.45])
    amp = np.array([0.30, 1.00, 0.20]) * 0.012
    art = (1.0 + 0.09 * np.sin(2 * np.pi * 0.33 * t) + 0.05 * np.sin(2 * np.pi * 1.4 * t)
           + 0.02 * rng.normal(size=t.size))
    rgb = (tone[None, :] * art[:, None]) * (1.0 + amp[None, :] * pulse[:, None]) * 255.0
    rgb += rng.normal(0, 0.45, rgb.shape)

    true_bpm = hr * 60
    print(f"synthetic: {dur:.0f}s at {fs:.0f} fps, true HR {true_bpm:.1f} bpm, "
          f"9% illumination drift + a 1.4 Hz artefact INSIDE the pulse band")
    rows = []
    for m in METHODS:
        snr, hrb = compare(rgb, fs, [m])[m]
        err = abs(hrb - true_bpm) if np.isfinite(hrb) else np.inf
        rows.append({"method": m, "snr": snr, "hr": hrb, "err": err})
    # Ranked by HR ERROR, not SNR. With ground truth available, accuracy is the only criterion
    # that means anything -- and ranking by SNR here put the raw green channel first while it was
    # reporting a rate 11 bpm wrong off the illumination artefact.
    rows.sort(key=lambda r: r["err"])
    print(f"\n  {'method':8s} {'HR':>7s} {'HR err':>8s} {'SNR':>8s}")
    for r in rows:
        print(f"  {r['method']:8s} {r['hr']:7.1f} {r['err']:8.2f} {r['snr']:8.1f}"
              + ("   <-- most ACCURATE" if r is rows[0] else ""))

    gr = next(r for r in rows if r["method"] == "green")
    good = [r for r in rows if r["err"] <= 2.0]
    assert gr["err"] > 5.0, "green should be FOOLED by an in-band illumination artefact"
    assert {"pos", "chrom"} <= {r["method"] for r in good}, \
        f"pos and chrom must both recover the true rate, got {[(r['method'], r['err']) for r in rows]}"
    # The point of the whole exercise: the highest SNR is NOT the right answer.
    by_snr = max(rows, key=lambda r: r["snr"])
    assert by_snr["method"] == "green" and by_snr["err"] > 5.0
    print(f"\nselftest ok")
    print(f"  {len(good)}/{len(rows)} methods within 2 bpm: "
          f"{', '.join(r['method'] for r in good)}")
    print(f"  green is fooled by {gr['err']:.0f} bpm while scoring the HIGHEST SNR "
          f"({gr['snr']:.0f}) -- which is why compare_sites ranks by cross-site")
    print(f"  agreement rather than by SNR, and why MIN_SNR alone cannot gate out an")
    print(f"  artefact that sits inside the 0.7-3 Hz band.")


def main():
    ap = argparse.ArgumentParser(description="Compare rPPG extraction methods.")
    ap.add_argument("--file", help="one rppg_pose_*.npz; default is every one in data/")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    files = [Path(a.file)] if a.file else sorted(DATA.glob("rppg_pose_*.npz"))
    if not files:
        print("no recordings in data/ -- run a capture first, or try --selftest")
        return
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if "raw_rgb" not in z:
                # The npz stores PROCESSED signals, not the raw per-channel means, so the other
                # methods cannot be re-run on it. Saying so beats silently comparing nothing.
                print(f"\n{f.name}: no raw RGB stored -- cannot re-extract. "
                      f"Re-record with the current build to enable method comparison.")
                continue
            _print_rows(compare_sites(z["raw_rgb"], float(z["fs"])), f.name)


if __name__ == "__main__":
    main()
