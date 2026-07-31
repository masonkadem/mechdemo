"""rppg_robust.py -- the three plots that decide whether a camera PTT measurement is real.

Accuracy plots flatter a measurement; these are built to expose it.

1. ARRIVAL vs DISTANCE.  The discriminator. A fixed instrument delay -- camera pipeline, ROI
   processing, rolling-shutter row offset -- is CONSTANT in distance. Propagation is LINEAR in
   it. So the slope, not the offset, is the physiological quantity, and its reciprocal is pulse
   wave velocity with a known upper-limb range (4-12 m/s) to check against. Plotted with a
   bootstrap confidence band, because a slope fitted to scattered points with no interval is an
   opinion.

2. TEST-RETEST.  The cheapest decisive control, and the one usually skipped. Record the SAME
   condition twice. Whatever separation appears between two identical conditions is the noise
   floor of the whole rig. If rest-vs-rest moves as much as rest-vs-hand_up, there is no effect
   to report, no matter what a p-value says.

3. CONDITION DISTRIBUTIONS.  Per-window estimates as a strip/box, never bars of means. Two
   conditions can differ significantly in mean and overlap almost completely, and an overlapping
   measurement cannot classify a single future recording -- which is what a blood-pressure claim
   ultimately needs.

    python rppg_robust.py                 # every plot the saved data supports
    python rppg_robust.py --list          # what has been recorded so far
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a"
PROXIMAL = ("forehead", "cheek_l", "cheek_r", "face", "neck")
DISTAL = ("hand",)
WIN_S, STEP_FRAC = 10.0, 0.5


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return plt


def runs():
    """Every saved pose recording as {tag: (json, npz)}."""
    out = {}
    for j in sorted(DATA.glob("rppg_pose_*.json")):
        tag = j.stem.replace("rppg_pose_", "")
        n = DATA / f"rppg_pose_{tag}.npz"
        if n.exists():
            out[tag] = (json.loads(j.read_text()), np.load(n, allow_pickle=True))
    return out


def window_lags(npz, win_s=WIN_S):
    """Per-window proximal->distal lag (ms). The spread across windows IS the uncertainty."""
    sigs, acc = npz["sigs"], npz["accepted"]
    seg = npz["seg"].astype(str) if "seg" in npz else np.array([""] * len(sigs))
    fs = float(npz["fs"])
    pm = np.isin(seg, PROXIMAL) & acc
    dm = np.isin(seg, DISTAL) & acc
    if not pm.any() or not dm.any():
        return np.array([]), pm.sum(), dm.sum()
    p, d = sigs[pm].mean(0), sigs[dm].mean(0)
    W = int(win_s * fs)
    if len(p) <= W:
        lag, _ = R.lag_subframe(p, d, fs, max_lag_s=min(0.25, 6.0 / fs))
        return np.array([lag]), pm.sum(), dm.sum()
    step = max(int(W * STEP_FRAC), 1)
    out = [R.lag_subframe(p[i:i + W], d[i:i + W], fs, max_lag_s=min(0.25, 6.0 / fs))[0]
           for i in range(0, len(p) - W, step)]
    return np.array(out), pm.sum(), dm.sum()


# ------------------------------------------------------------------ 1. distance
def fig_distance(tag, npz, n_boot=2000, seed=0):
    plt = _mpl()
    sigs, acc, dist = npz["sigs"], npz["accepted"], npz["dist"]
    seg = npz["seg"].astype(str) if "seg" in npz else np.array([""] * len(sigs))
    fs, quals = float(npz["fs"]), npz["quals"]
    idx = np.flatnonzero(acc)
    if len(idx) < 5:
        return None
    ref = idx[np.argmax(quals[idx])]
    dd, lg, sg = [], [], []
    for i in idx:
        if i == ref:
            continue
        lag, _ = R.lag_subframe(sigs[ref], sigs[i], fs, max_lag_s=min(0.25, 6.0 / fs))
        dd.append(dist[i] - dist[ref]); lg.append(lag); sg.append(seg[i])
    dd, lg, sg = np.array(dd), np.array(lg), np.array(sg)
    if len(dd) < 5 or np.ptp(dd) < 5:
        return None

    sl, ic = np.polyfit(dd, lg, 1)
    rng = np.random.default_rng(seed)
    bs = np.array([np.polyfit(*[a[k] for a in (dd, lg)], 1)
                   for k in (rng.integers(0, len(dd), len(dd)) for _ in range(n_boot))])
    lo_s, hi_s = np.percentile(bs[:, 0], [2.5, 97.5])
    xs = np.linspace(dd.min(), dd.max(), 60)
    band = np.array([b[0] * xs + b[1] for b in bs])
    blo, bhi = np.percentile(band, [2.5, 97.5], axis=0)

    to_pwv = lambda s: 0.01 / (s / 1000.0) if abs(s) > 1e-9 else np.inf
    pwv, pwv_lo, pwv_hi = to_pwv(sl), to_pwv(hi_s), to_pwv(lo_s)
    r = float(np.corrcoef(dd, lg)[0, 1])
    slope_sig = (lo_s > 0) or (hi_s < 0)          # does the CI exclude a flat line?

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(xs, blo, bhi, color=NAVY, alpha=.16, lw=0, label="95% CI")
    ax.plot(xs, sl * xs + ic, color=NAVY, lw=1.6)
    for s, c in (("upper_arm", "#7aa6c2"), ("forearm", RED), ("hand", "#e08a30"),
                 ("neck", GREEN), ("forehead", "#6cc24a"), ("cheek_l", "#8fd17a"),
                 ("cheek_r", "#8fd17a")):
        m = sg == s
        if m.any():
            ax.scatter(dd[m], lg[m], s=26, color=c, edgecolor="none", alpha=.85, label=s)
    ax.axhline(0, color="#ccc", lw=.8)
    ax.set_xlabel("anatomical distance from reference site (cm)")
    ax.set_ylabel("arrival lag (ms)")
    verdict = ("propagation" if slope_sig and 4 <= abs(pwv) <= 12 and abs(r) > .5
               else "NOT established")
    ax.set_title(f"{tag}: arrival vs distance -- {verdict}\n"
                 f"PWV {pwv:.1f} m/s  [{min(pwv_lo, pwv_hi):.1f}, {max(pwv_lo, pwv_hi):.1f}]"
                 f"   r = {r:+.2f}   n = {len(dd)}", loc="left", fontsize=9.5,
                 fontweight="bold")
    ax.legend(fontsize=7, frameon=False, ncol=3)
    note = ("Slope CI excludes zero: lag grows with distance, which a fixed delay cannot do."
            if slope_sig else
            "Slope CI includes zero: consistent with a pure fixed offset. No propagation shown.")
    ax.text(0, -0.22, note, transform=ax.transAxes, fontsize=7.5, color="#444")
    fig.tight_layout()
    p = FIGS / f"fig_robust_distance_{tag}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"path": p, "pwv": pwv, "ci": (pwv_lo, pwv_hi), "r": r, "slope_sig": bool(slope_sig)}


# --------------------------------------------------------------- 2. test-retest
def fig_retest(rep, others):
    """`rep`: >=2 tags that are the SAME condition. `others`: the contrast conditions."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rng = np.random.default_rng(0)
    labels, groups = [], []
    for tag, lags in rep + others:
        labels.append(tag); groups.append(lags)
    for i, (lab, g) in enumerate(zip(labels, groups)):
        if not len(g):
            continue
        c = GREY if i < len(rep) else NAVY
        ax.scatter(np.full(len(g), i) + rng.normal(0, .06, len(g)), g, s=16, color=c,
                   alpha=.7, edgecolor="none")
        ax.hlines(np.median(g), i - .28, i + .28, color=RED, lw=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("per-window proximal->hand lag (ms)")

    noise = abs(np.median(rep[0][1]) - np.median(rep[1][1])) if len(rep) >= 2 else np.nan
    lines = []
    if np.isfinite(noise):
        ax.axhspan(np.median(rep[0][1]) - noise, np.median(rep[0][1]) + noise,
                   color=GREY, alpha=.15, lw=0)
        lines.append(f"repeat-vs-repeat gap = {noise:.1f} ms  <- the rig's noise floor")
        for tag, g in others:
            if len(g):
                eff = abs(np.median(g) - np.median(rep[0][1]))
                verdict = "EXCEEDS noise floor" if eff > noise else "within noise -- not real"
                lines.append(f"{tag}: {eff:.1f} ms  ({verdict})")
    ax.set_title("Test-retest: is any condition effect bigger than repeating the same one?\n"
                 + "   ".join(lines[:1]), loc="left", fontsize=9.5, fontweight="bold")
    if len(lines) > 1:
        ax.text(0, -0.20, "   |   ".join(lines[1:]), transform=ax.transAxes, fontsize=7.5,
                color="#444")
    fig.tight_layout()
    p = FIGS / "fig_robust_retest.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"path": p, "noise_floor_ms": noise}


# ---------------------------------------------------------- 3. condition spread
def fig_conditions(per_tag):
    plt = _mpl()
    tags = [t for t, g in per_tag if len(g)]
    if len(tags) < 2:
        return None
    plt_data = [g for t, g in per_tag if len(g)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bp = ax.boxplot(plt_data, positions=range(len(tags)), widths=.5, showfliers=False,
                    patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor="#dfe6ef", edgecolor=NAVY, lw=1.1)
    for k in ("whiskers", "caps", "medians"):
        for e in bp[k]:
            e.set(color=NAVY, lw=1.1)
    rng = np.random.default_rng(0)
    for i, g in enumerate(plt_data):
        ax.scatter(np.full(len(g), i) + rng.normal(0, .07, len(g)), g, s=14, color=RED,
                   alpha=.55, edgecolor="none", zorder=3)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, fontsize=8)
    ax.set_ylabel("per-window proximal->hand lag (ms)")

    # Overlap is the property that matters for ever classifying a single new recording.
    ov = ""
    if len(plt_data) >= 2:
        a, b = plt_data[0], plt_data[1]
        lo = max(np.percentile(a, 25), np.percentile(b, 25))
        hi = min(np.percentile(a, 75), np.percentile(b, 75))
        ov = ("IQRs overlap" if hi > lo else "IQRs separate")
    ax.set_title(f"Per-window spread by condition ({ov})", loc="left", fontsize=9.5,
                 fontweight="bold")
    ax.text(0, -0.20, "Distributions, not bars of means: a difference in means with overlapping "
            "spreads cannot classify one new recording.", transform=ax.transAxes, fontsize=7.5,
            color="#444")
    fig.tight_layout()
    p = FIGS / "fig_robust_conditions.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"path": p, "overlap": ov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--repeat-tags", nargs="*", default=None,
                    help="tags that are the SAME condition, for test-retest (e.g. rest rest2)")
    a = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    got = runs()
    if not got:
        print("[err] no completed pose recordings in data/ (rppg_pose_*.json + .npz).")
        print("      Record at least one with run_rppg.command first.")
        return
    per_tag = []
    print(f"{'tag':<14}{'HR':>7}{'acc pts':>9}{'windows':>9}{'median lag':>12}{'sd':>7}")
    for tag, (j, npz) in got.items():
        lags, npx, ndx = window_lags(npz)
        per_tag.append((tag, lags))
        med = np.median(lags) if len(lags) else np.nan
        sd = np.std(lags) if len(lags) else np.nan
        print(f"{tag:<14}{j.get('consensus_hr', np.nan):7.1f}{j.get('n_accepted', 0):9d}"
              f"{len(lags):9d}{med:12.1f}{sd:7.1f}")
        if not len(lags):
            print(f"{'':14}  (no proximal/distal pair accepted: {npx} proximal, {ndx} distal)")
    if a.list:
        return

    for tag, (j, npz) in got.items():
        r = fig_distance(tag, npz)
        if r:
            print(f"\n[1] {r['path'].name}: PWV {r['pwv']:.1f} m/s "
                  f"[{min(r['ci']):.1f}, {max(r['ci']):.1f}], r={r['r']:+.2f}, "
                  f"slope CI excludes zero: {r['slope_sig']}")
        else:
            print(f"\n[1] {tag}: too few accepted points spread over distance to fit.")

    rep = a.repeat_tags or [t for t in got if t.startswith("rest")]
    rep_pairs = [(t, dict(per_tag)[t]) for t in rep if t in dict(per_tag)]
    others = [(t, g) for t, g in per_tag if t not in rep]
    if len(rep_pairs) >= 2:
        r = fig_retest(rep_pairs, others)
        print(f"[2] {r['path'].name}: noise floor {r['noise_floor_ms']:.1f} ms")
    else:
        print(f"[2] test-retest needs the SAME condition recorded twice. Have {rep}. "
              "Record 'rest' again (tag it rest2) -- this is the control that decides "
              "whether any condition effect is real.")

    r = fig_conditions(per_tag)
    print(f"[3] {r['path'].name}: {r['overlap']}" if r else
          "[3] condition spread needs at least two conditions with an accepted proximal/distal "
          "pair.")


if __name__ == "__main__":
    main()
