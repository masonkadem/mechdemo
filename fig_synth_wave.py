"""fig_synth_wave.py -- the third synthetic task: audit validation on RAW WAVEFORMS.

Why this row is needed
----------------------
fig_app_tabs shows two synthetic settings, and neither exercises the step that actually fails on
real data. The scalar sandbox (row a-d) hands PTT to the model as a number, so it validates the
audit's logic but never touches foot detection. The real-waveform row (e-h) exercises detection
but has no ground truth, so a null there is uninterpretable.

This row closes the gap: ECG and PPG waveforms are GENERATED with a known PTT-to-BP law, a CNN is
trained on them, and the identical pipeline used on VitalDB is run end to end -- foot detection,
negative-arm roll, per-subject slope. Because alpha sets how much of BP is routed through arrival
time, the ground-truth faithfulness of every model is known.

Panels:
  a  the generated waveforms, so a reader can judge whether they are believable
  b  the injected law: BP against the PTT that was written into each segment
  c  detector fidelity -- measured PTT against injected PTT, which bounds everything downstream
  d  audit slope against alpha, the validation itself

The honest caveat, shown in panel c rather than buried: the foot detector recovers injected PTT
at only r ~ 0.56 even on clean synthetic signals, so the audit operates through a lossy estimator.
That is a property of the estimator, not of the audit, and it is why panel d matters -- the audit
still tracks alpha at r = -0.996 despite it.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import synth_waveform_audit as S

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREEN, GREY = "#2f4b7c", "#c1543b", "#3b8c5a", "#9aa0a6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def main():
    res = json.loads((ROOT / "data" / "synth_waveform_audit.json").read_text())
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(11.5, 3.1))
    gs = fig.add_gridspec(1, 4, wspace=0.42, width_ratios=[1.25, 1, 1, 1.05])

    # ---- a: are the generated waveforms believable? -------------------------
    ax = fig.add_subplot(gs[0, 0])
    seg = S.make_segment(rng, hr=68, ptt_ms=210)
    t = np.arange(S.L) / S.FS
    n = int(4 * S.FS)
    ax.plot(t[:n], seg[:n, 0] / seg[:n, 0].std() + 3.2, color=NAVY, lw=0.9, label="ECG")
    ax.plot(t[:n], seg[:n, 1] / seg[:n, 1].std(), color=GREEN, lw=0.9, label="PPG")
    ax.set_xlabel("time (s)", fontsize=8.5)
    ax.set_yticks([])
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    ax.set_title("a   generated ECG + PPG", loc="left", fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.02, "HR 68 bpm, PTT 210 ms", transform=ax.transAxes, fontsize=7,
            color=GREY)

    # ---- b: the injected law -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _, y, ptt_true, _ = S.make_dataset(600, 1.0, np.random.default_rng(11))
    ax.scatter(ptt_true, y, s=4, alpha=0.35, color=NAVY, edgecolors="none")
    xs = np.linspace(ptt_true.min(), ptt_true.max(), 50)
    ax.plot(xs, 130.0 - 0.22 * (xs - 190.0), color=RED, lw=1.4)
    ax.set_xlabel("injected PTT (ms)", fontsize=8.5)
    ax.set_ylabel("BP (mmHg)", fontsize=8.5)
    ax.set_title("b   the law we wrote in", loc="left", fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.06, r"$-0.22$ mmHg/ms", transform=ax.transAxes, fontsize=8, color=RED)

    # ---- c: detector fidelity ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    Xc, _, pt, _ = S.make_dataset(400, 1.0, np.random.default_rng(99))
    est = mechlib.compute_ptt(Xc, S.FS) * 1000.0
    ok = np.isfinite(est)
    ax.scatter(pt[ok], est[ok], s=5, alpha=0.35, color=NAVY, edgecolors="none")
    lim = [min(pt[ok].min(), est[ok].min()), max(pt[ok].max(), est[ok].max())]
    ax.plot(lim, lim, color="k", lw=0.9, ls="--")
    r = float(np.corrcoef(est[ok], pt[ok])[0, 1])
    ax.set_xlabel("injected PTT (ms)", fontsize=8.5)
    ax.set_ylabel("measured PTT (ms)", fontsize=8.5)
    ax.set_title("c   the detector is lossy", loc="left", fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.90, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold")

    # ---- d: the validation ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    al = [0.0, 0.25, 0.5, 0.75, 1.0]
    sl = [res["alphas"][str(a)]["slope"] for a in al]
    ax.plot(al, sl, "o-", color=NAVY, lw=1.8, ms=5.5)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel(r"$\alpha$  (BP routed through PTT)", fontsize=8.5)
    ax.set_ylabel("audit slope", fontsize=8.5)
    ax.set_title("d   the audit tracks it", loc="left", fontsize=9.5, fontweight="bold")
    ax.text(0.04, 0.10, f"r = {res['slope_vs_alpha_r']:+.3f}", transform=ax.transAxes,
            fontsize=9, fontweight="bold")
    ax.text(0.04, 0.78, "faithful", transform=ax.transAxes, fontsize=7.5, color=GREY)
    ax.annotate("", xy=(0.10, 0.62), xytext=(0.10, 0.76), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1))

    fig.savefig(ROOT / "figures" / "fig_synth_wave.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_synth_wave.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_synth_wave.png / .pdf")
    print(f"[fig] detector r = {r:+.3f}, audit r(alpha, slope) = "
          f"{res['slope_vs_alpha_r']:+.3f}")


if __name__ == "__main__":
    main()
