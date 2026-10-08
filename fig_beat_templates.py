"""fig_beat_templates.py -- supplementary figure: the simulator's average heartbeat against
VitalDB's. ECG aligned on the R peak, PPG on the pulse foot; each beat min-max scaled within its
window, median and interquartile band shown. Same heart-rate band (RR 0.75-0.95 s) for both.
The simulator's beat shapes are fitted to these real beats by fit_beat_templates.py.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import synth_waveform_audit as SW
from fit_beat_templates import E_WIN, FS, P_WIN, collect

ROOT = Path(__file__).resolve().parent


def main():
    d = mechlib.load_mini(str(ROOT / "data" / "vitaldb_mini.npz"))
    Er, Pr = collect(d["Xte"][:, :, :2].astype(float))
    rng = np.random.default_rng(0)
    sim = np.array([SW.make_segment(rng, 60 / rng.uniform(0.75, 0.95), rng.uniform(120, 260))
                    for _ in range(150)], float)
    Es, Ps = collect(sim)

    te = np.arange(Er.shape[1]) / FS + E_WIN[0]; tp = np.arange(Pr.shape[1]) / FS + P_WIN[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a_, t, R_, S_, title in ((ax[0], te, Er, Es, "ECG, aligned on R peak"),
                                 (ax[1], tp, Pr, Ps, "PPG, aligned on pulse foot")):
        for M, col, lab in ((R_, "k", f"VitalDB ({len(R_)} beats)"),
                            (S_, "#c1543b", f"Simulated ({len(S_)} beats)")):
            lo, hi = np.percentile(M, [25, 75], 0)
            a_.fill_between(t, lo, hi, color=col, alpha=.12, lw=0)
            a_.plot(t, np.median(M, 0), color=col, lw=1.6, label=lab)
        a_.set_title(title, loc="left", fontsize=10)
        a_.set_xlabel("Time (s)"); a_.set_ylabel("Scaled amplitude")
        a_.legend(frameon=False, fontsize=8)
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_beat_templates.png", dpi=170)
    fig.savefig(ROOT / "figures" / "fig_beat_templates.pdf")
    print("[fig] figures/fig_beat_templates.png / .pdf")


if __name__ == "__main__":
    main()
