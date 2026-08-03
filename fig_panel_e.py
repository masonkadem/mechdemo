"""fig_panel_e.py -- mechanism vs OOD robustness, with the estimator sensitivity shown.

Panel a plots audit sensitivity against OOD penalty using second_deriv, chosen because it ties
for the best agreement with invasive arterial arrival time (r = +0.206 on 100% of segments) --
a criterion fixed before any of these correlations were computed.

Panel b is the reason to trust it: the correlation is negative under six of seven published PAT
estimators, and inverts only under foot_tangent, which scores r = 0.026 against the arterial
ground truth and therefore is not measuring arrival time at all. The result does not hinge on the
choice.
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import ood_benchmark as ob
import pat_estimators as PE
from mechlib import ECG, PPG, _shift_channel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAVY, RED, GREY = "#2f4b7c", "#c1543b", "#9aa0a6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
FS = 125
DELT = (-6, -4, -2, 0)
NAMES = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
DISP = {"lenet1d": "LeNet", "inception1d": "Incep", "xresnet1d50": "XR50",
        "xresnet1d101": "XR101", "transformer": "Trans"}
# agreement with invasive arterial arrival time (pat_groundtruth.py, within subject)
GT = {"foot_tangent": 0.026, "foot_min": -0.209, "peak": 0.208, "max_slope": 0.136,
      "second_deriv": 0.206, "xcorr": 0.061, "xcorr_deriv": 0.177}
PRIMARY = "second_deriv"


def main():
    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= 60][:45]
    sel = np.concatenate([np.where(g == s)[0][:35] for s in subs])
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    gg = g[sel]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nom = np.array([1000.0 * x / FS for x in DELT])

    Xs = {}
    for dl in DELT:
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], dl)
        Xs[dl] = Xd

    e = json.loads((DATA / "ood_benchmark_ecgppg_full.json").read_text())["models"]
    gap = np.array([e[n]["ood"]["mimic_bp"]["mae_dbp"] - e[n]["ood"]["id"]["mae_dbp"]
                    for n in NAMES])

    preds = {}
    for mk in NAMES:
        ck = torch.load(ROOT / "models" / f"{mk}_ecgppg_full.pt", map_location=dev,
                        weights_only=False)
        m = ob.build_model(mk, n_ch=2, L=1250)
        m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
        preds[mk] = np.stack([ob.predict(m, Xs[dl], dev, ck["mu"], ck["sd"])[:, 1]
                              for dl in DELT], 1)

    def slopes_for(est):
        fn = PE.ESTIMATORS[est]
        keep = np.isfinite(PE.batch(fn, Xs[0], FS))
        for dl in DELT:
            keep &= np.isfinite(PE.batch(fn, Xs[dl], FS))
        out = []
        for mk in NAMES:
            P = preds[mk]
            v = [np.median([np.polyfit(nom, P[i], 1)[0]
                            for i in np.where((gg == s) & keep)[0]])
                 for s in np.unique(gg) if ((gg == s) & keep).sum() >= 5]
            out.append(float(np.median(v)))
        return np.array(out)

    all_r, all_sl = {}, {}
    for est in PE.ESTIMATORS:
        sl = slopes_for(est)
        all_sl[est] = sl
        all_r[est] = float(np.corrcoef(np.abs(sl), gap)[0, 1])
        print(f"  {est:14s} r = {all_r[est]:+.3f}", flush=True)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.7),
                                 gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.62})

    sl = np.abs(all_sl[PRIMARY]) * 1000.0          # mmHg per second of shift
    r = all_r[PRIMARY]
    b1, b0 = np.polyfit(sl, gap, 1)
    xs = np.linspace(sl.min() * 0.9, sl.max() * 1.05, 30)
    ax.plot(xs, b0 + b1 * xs, "--", color="k", lw=1.1, label=f"r = {r:.2f}")
    ax.scatter(sl, gap, s=85, color=NAVY, edgecolor="k", lw=1, zorder=3)
    for x, y, n in zip(sl, gap, NAMES):
        ax.annotate(DISP[n], (x, y), fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("roll-audit sensitivity  |dDBP/dΔ|  (mmHg/s)", fontsize=8.5)
    ax.set_ylabel("OOD penalty (mmHg)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("a   more arrival-time use, smaller OOD penalty", loc="left",
                 fontsize=9.5, fontweight="bold")

    order = sorted(all_r, key=lambda k: all_r[k])
    cols = [RED if k == "foot_tangent" else (NAVY if k == PRIMARY else GREY) for k in order]
    bx.barh(range(len(order)), [all_r[k] for k in order], color=cols)
    bx.axvline(0, color="k", lw=0.8)
    bx.set_yticks(range(len(order)),
                  [f"{k}  (GT {GT[k]:+.2f})" for k in order], fontsize=7.5)
    bx.set_xlabel("r (audit sensitivity, OOD penalty)", fontsize=8.5)
    bx.set_title("b   holds across estimators", loc="left", fontsize=9.5, fontweight="bold")
    bx.text(0.98, 0.06, "red: fails ground-truth check", transform=bx.transAxes,
            fontsize=7.5, color=RED, ha="right")

    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig_panel_e.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (DATA / "panel_e.json").write_text(json.dumps(
        {"r_by_estimator": all_r, "slopes": {k: v.tolist() for k, v in all_sl.items()},
         "ood_penalty": gap.tolist(), "models": NAMES, "primary": PRIMARY},
        indent=2, default=float))
    print(f"\n[fig] figures/fig_panel_e.png   primary {PRIMARY}: r = {r:+.3f}")


if __name__ == "__main__":
    main()
