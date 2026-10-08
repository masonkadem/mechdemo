"""fig_importance_bars.py -- clean top-15 LightGBM feature-importance bar plots for SBP & DBP,
with novel features highlighted so the reader sees whether they rank.
"""
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAVY, RED = "#2f4b7c", "#c1543b"
# APG landmark TIMINGS + alternative APG indices. NOT claimed as novel: T_a..T_e already appear
# in the published BP-Benchmark feature set (verified in data/bcg_dataset/feat_fold_0.csv).
# Highlighted here because they are the features that REPRODUCE on a held-out subject split,
# whereas the amplitude-ratio combinations do not (see apg_discovery.py).
VALIDATED = {"t_b", "t_c", "t_d", "t_e", "apg_cd_a", "apg_bd_a", "apg_ce_a",
             "takazawa", "ushiro", "reflect_be"}
NOVEL = VALIDATED          # kept as an alias so the plotting code below is unchanged


def main():
    b = pickle.load(open("data/_feat_full_vitaldb_full_calfree_8000.pkl", "rb"))
    F, y = b["F"], b["y"]
    keys = [k for k, v in F.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3
            and np.nanstd(np.asarray(v, float)) > 1e-9]
    M = np.column_stack([np.nan_to_num(np.asarray(F[k], float),
                                       nan=np.nanmedian(np.asarray(F[k], float))) for k in keys])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (t, tname) in zip(axes, [(0, "SBP"), (1, "DBP")]):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=0, verbosity=-1)
        m.fit(M, y[:, t])
        imp = m.feature_importances_.astype(float)
        order = np.argsort(-imp)[:15][::-1]
        vals = imp[order]; labs = [keys[i] for i in order]
        cols = [RED if keys[i] in NOVEL else NAVY for i in order]
        ax.barh(range(len(order)), vals, color=cols)
        ax.set_yticks(range(len(order)), labs, fontsize=9)
        ax.set_xlabel("LightGBM gain importance", fontsize=10)
        ax.set_title(f"{tname}", fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    # legend
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=NAVY, label="other features"),
                            Patch(color=RED, label="APG timing/index (reproduces on held-out split)")],
                   fontsize=8.5, loc="lower right", frameon=False)
    fig.suptitle("Top-15 feature importance for blood-pressure prediction",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_importance_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_importance_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_importance_bars.png / .pdf")


if __name__ == "__main__":
    main()
