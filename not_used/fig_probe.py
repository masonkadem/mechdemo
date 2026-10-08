"""fig_probe.py -- concise linear-probing figure: across-layer decodability of key physiological
cues in the trained deep models, computed FRESH from the saved checkpoints (the values stored in
the benchmark json used a different feature normalization and are unreliable).

Shows cues are linearly DECODABLE in the representations -- which, paired with the flat roll-audit
sensitivity, is the 'decodable is not causally used' point. Cardiac period is the most decodable
(the shortcut); PAT is only weakly decodable.
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import ood_benchmark as ob
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
NAVY, RED, GREEN, ORANGE = "#2f4b7c", "#c1543b", "#3b8c5a", "#d98c3f"

CUES = [("period", "cardiac period", RED),
        ("pat", "PAT (arrival time)", NAVY),
        ("aix", "augmentation index", GREEN),
        ("apg", "APG stiffness", ORANGE)]
MODELS = [("xresnet1d50", "XResNet50"), ("inception1d", "Inception1d"),
          ("transformer", "Transformer")]


def main():
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["gte"]), 1500, replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    sc = mechlib.compute_scalars(X, fs)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 3.6), sharey=True)
    for ax, (mkey, mname) in zip(axes, MODELS):
        ck = torch.load(ROOT / "models" / f"{mkey}_ecgppg_full.pt", map_location="cpu",
                        weights_only=False)
        m = ob.build_model(mkey, n_ch=2, L=1250); m.load_state_dict(ck["state_dict"]); m.eval()
        feats = ob.layer_features(m, mkey, X, "cpu")
        layers = list(feats)
        x = range(len(layers))
        for cue, lab, col in CUES:
            r2 = [max(mechlib.linear_probe(feats[l], sc[cue]), 0.0) for l in layers]
            ax.plot(x, r2, "-o", ms=4, lw=1.6, color=col, label=lab)
        ax.set_xticks(x, [f"L{i}" for i in x], fontsize=8)
        ax.set_xlabel("layer", fontsize=9); ax.set_ylim(-0.02, 0.85)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("linear-probe $R^2$", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Physiological cues are linearly decodable across layers "
                 "(decodable is not causally used; cf. flat roll-audit)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig_probe.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "fig_probe.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_probe.png / .pdf")


if __name__ == "__main__":
    main()
