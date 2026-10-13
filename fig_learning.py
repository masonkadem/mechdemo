"""fig_learning.py -- supplementary: learning curves of the eight focus models (focus_models.py).

Rows: SBP, DBP. Columns: 1 layer / 3 layers, each on 12k and 140k training segments. Train and
validation MAE per epoch; dashed: predicting the training mean on the validation patients.
Shows every model trained, none collapsed to a constant, and how far each gets below the mean.
Seed 0 of each configuration. Training MAE is the average over each epoch's batches, so it starts
above validation MAE, which is measured after the epoch.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parent
NAVY, RED, GREY = "#2f4b7c", "#c1543b", "#9aa0a6"


def main():
    res = json.loads((ROOT / "data" / "focus_models.json").read_text())
    cols = [(1, "12k"), (1, "140k"), (3, "12k"), (3, "140k")]
    fig, ax = plt.subplots(2, 4, figsize=(14, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
    for i, t in enumerate(("sbp", "dbp")):
        for j, (dep, dn) in enumerate(cols):
            a_ = ax[i, j]; a_.spines[["top", "right"]].set_visible(False)
            m = res.get(f"{t}_{dep}L_{dn}")
            if m is None:
                a_.axis("off"); continue
            h = m["history"]; ep = range(1, len(h["train_mae"]) + 1)
            a_.plot(ep, h["train_mae"], color=NAVY, lw=1.3, label="Train")
            a_.plot(ep, h["val_mae"], color=RED, lw=1.3, ls="--", label="Validation")
            a_.axhline(m["val_mean_mae"], color=GREY, lw=1, ls=":", label="Mean predictor")
            lo = min(min(h["val_mae"]), min(h["train_mae"]), m["val_mean_mae"])
            a_.set_ylim(lo - 1, m["val_mean_mae"] + 4)
            a_.set_title(f"{t.upper()}, {dep} layer{'s' if dep > 1 else ''}, {dn}", fontsize=9.5, loc="left")
            a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            a_.set_xlabel("Epoch"); a_.set_ylabel("MAE (mmHg)" if j == 0 else "")
            if i == 0 and j == 0:
                a_.legend(fontsize=8, frameon=False)
    fig.savefig(ROOT / "figures" / "fig_learning.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_learning.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_learning.png / .pdf")


if __name__ == "__main__":
    main()
