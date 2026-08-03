"""monotonicity.py -- does arrival time actually grow along the arm?

The check this provides
-----------------------
A single face-to-hand lag can be produced by anything: a fixed camera delay, a filter phase
shift, or noise. What cannot be faked is ORDER. If the number is a transit time then arrival must
increase monotonically along the arterial path -- face, then forearm, then hand, then fingertips
-- and the slope of arrival against distance must land in the physiological range for pulse wave
velocity, 4-12 m/s in the upper limb.

That gives three independent verdicts from one plot:

  monotonic     does arrival increase with distance, site by site
  linear        does it increase at a constant rate (Spearman against distance)
  plausible     does the fitted slope imply a PWV inside 4-12 m/s

The proximal sites double as a null control. Forehead and both cheeks are at the same arterial
distance, so their spread is the rig's own noise floor. A face-to-hand lag smaller than that
spread is not a measurement, whatever its value, and the plot marks it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import rppg_two_site as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAVY, RED, GREEN, GREY = "#2f4b7c", "#c1543b", "#3b8c5a", "#9aa0a6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
PWV_LO, PWV_HI = 4.0, 12.0


def site_lags(sigs, dists, fs, ref=None):
    """Lag of every site against the proximal reference, in ms."""
    names = [k for k in sigs if sigs[k] is not None and np.std(sigs[k]) > 1e-9]
    if len(names) < 2:
        return {}, {}
    if ref is None:
        ref = min(names, key=lambda k: dists.get(k, 1e9))
    out = {}
    for k in names:
        if k == ref:
            out[k] = 0.0
            continue
        lag, _ = R.lag_subframe(sigs[ref], sigs[k], fs, max_lag_s=min(0.25, 8.0 / fs))
        out[k] = float(lag)
    return out, {k: dists.get(k, np.nan) for k in out}


def assess(lags, dists, noise_ms):
    """Monotonicity, linearity and PWV plausibility from per-site lags."""
    ks = [k for k in lags if np.isfinite(dists.get(k, np.nan))]
    if len(ks) < 3:
        return {"ok": False, "why": "need at least three sites at different distances"}
    ks.sort(key=lambda k: dists[k])
    d = np.array([dists[k] for k in ks], float)
    y = np.array([lags[k] for k in ks], float)
    if np.ptp(d) < 5:
        return {"ok": False, "why": "sites span too little distance"}

    steps = np.diff(y)
    mono = bool(np.all(steps >= -noise_ms))          # allow one noise floor of wobble
    rho = float(stats.spearmanr(d, y).statistic)
    slope = float(np.polyfit(d, y, 1)[0])            # ms per cm
    pwv = 0.01 / (slope / 1000.0) if abs(slope) > 1e-9 else np.inf
    plaus = bool(PWV_LO <= pwv <= PWV_HI)
    return {"ok": True, "sites": ks, "dist_cm": d.tolist(), "lag_ms": y.tolist(),
            "monotonic": mono, "spearman": rho, "slope_ms_per_cm": slope,
            "pwv_m_s": float(pwv), "pwv_plausible": plaus, "noise_ms": float(noise_ms),
            "verdict": ("propagation" if (mono and plaus and rho > 0.5) else
                        "not propagation")}


def figure(res, tag, out=None):
    if not res.get("ok"):
        print(f"[mono] {res.get('why')}"); return
    d = np.array(res["dist_cm"]); y = np.array(res["lag_ms"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.axhspan(-res["noise_ms"], res["noise_ms"], color=GREY, alpha=0.18)
    ax.text(d.min(), res["noise_ms"], "  rig noise floor (proximal spread)", fontsize=7.5,
            color=GREY, va="bottom")
    xs = np.linspace(d.min(), d.max(), 30)
    ax.plot(xs, np.polyval(np.polyfit(d, y, 1), xs), "--", color=RED, lw=1.3,
            label=f"PWV {res['pwv_m_s']:.1f} m/s")
    ax.plot(d, y, "-o", color=NAVY, lw=1.6, ms=6)
    for x, v, k in zip(d, y, res["sites"]):
        ax.annotate(k, (x, v), fontsize=7.5, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("distance along the arterial path (cm)", fontsize=9)
    ax.set_ylabel("pulse arrival, relative to face (ms)", fontsize=9)
    ok = res["verdict"] == "propagation"
    ax.set_title(f"{tag}: {res['verdict']}", loc="left", fontsize=10, fontweight="bold",
                 color=(GREEN if ok else RED))
    ax.text(0.02, 0.95,
            f"monotonic: {'yes' if res['monotonic'] else 'no'}\n"
            f"rho = {res['spearman']:+.2f}\n"
            f"PWV {'inside' if res['pwv_plausible'] else 'OUTSIDE'} 4-12 m/s",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    p = out or (ROOT.parent / "figures" / f"fig_monotonicity_{tag}.png")
    fig.savefig(p, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"[mono] {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rest")
    args = ap.parse_args()
    f = DATA / f"rppg_pose_{args.tag}.npz"
    if not f.exists():
        print(f"[mono] no recording at {f}"); return
    z = np.load(f, allow_pickle=True)
    fs = float(z["fs"])
    sigs = {k[4:]: z[k] for k in z.files if k.startswith("sig_")}
    dists = json.loads(str(z["dists"])) if "dists" in z.files else {}
    lags, dd = site_lags(sigs, dists, fs)
    prox = [v for k, v in lags.items() if k in ("forehead", "cheek_l", "cheek_r")]
    noise = float(np.std(prox)) if len(prox) >= 2 else 1000.0 / fs
    res = assess(lags, dd, noise)
    print(json.dumps({k: v for k, v in res.items() if k != "sites"}, indent=2, default=float))
    figure(res, args.tag)


if __name__ == "__main__":
    main()
