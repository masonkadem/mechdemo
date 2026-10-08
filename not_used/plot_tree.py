"""plot_tree.py -- render the tuned single LightGBM tree WITHOUT the graphviz binary,
by parsing dump_model() and drawing nodes/edges in matplotlib. Shows the human-readable
decision logic (split feature, threshold) and leaf DBP values.
"""
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

import lightgbm_arm as gbm

ROOT = Path(__file__).resolve().parent
READABLE = {  # abbreviated names for the plot
    "apg_d_a": "APG d/a", "apg_b_a": "APG b/a", "apg_c_a": "APG c/a", "apg_e_a": "APG e/a",
    "pat_peak": "PAT", "age": "Age", "bmi": "BMI", "sex": "Sex", "hr": "HR", "period": "Period",
    "rise": "RiseT", "aix": "AIx", "notch": "Notch", "decay": "Decay", "kurt": "Kurt",
    "peak": "PeakH", "amp": "Amp", "hfd": "FracD", "sys_area": "SysArea", "crest": "Crest",
    "pw25": "PW25", "pw50": "PW50", "rr_mean": "RRmean", "qrs_amp": "QRSamp", "vpg_max": "VPGmax",
    "spec_ent": "SpecEnt", "ptt_var": "PTTvar", "xcorr_peak": "xcorrPk",
}


def load_tree(cache, params, max_depth_plot=3):
    with open(cache, "rb") as f:
        blob = pickle.load(f)
    sctr, ytr, gtr, dmtr = blob["tr"]
    cue = [k for k, v in sctr.items() if np.isfinite(np.asarray(v, float)).mean() > 0.3]
    dkeys = tuple(k for k in ("age", "sex", "bmi") if dmtr and k in dmtr)
    X, names = gbm.build_feature_table(sctr, dmtr, cue, n=len(ytr), demo_keys=dkeys)
    X = gbm._impute(X)
    m = lgb.LGBMRegressor(n_estimators=1, learning_rate=1.0, random_state=0, verbosity=-1,
                          num_leaves=params["num_leaves"], max_depth=min(params["max_depth"], max_depth_plot),
                          min_child_samples=params["min_child_samples"])
    m.fit(X, ytr[:, 1])
    return m.booster_.dump_model(), names


def draw(node, ax, x, y, dx, dy, names, depth, maxd):
    if "leaf_value" in node or depth >= maxd:
        val = node.get("leaf_value", node.get("internal_value", 0))
        ax.text(x, y, f"DBP\n{val:.0f}", ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8e8e8", ec="black"))
        return
    f = names[node["split_feature"]]
    thr = node["threshold"]
    ax.text(x, y, f"{READABLE.get(f,f)}\n<= {thr:.2f}", ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.2))
    for child, sign, side in [(node["left_child"], "yes", -1), (node["right_child"], "no", +1)]:
        cx, cy = x + side * dx, y - dy
        ax.plot([x, cx], [y - 0.03, cy + 0.05], color="#888", lw=1)
        ax.text((x + cx) / 2, (y + cy) / 2, sign, fontsize=7, color="#666", ha="center")
        draw(child, ax, cx, cy, dx * 0.55, dy, names, depth + 1, maxd)


def main():
    import json
    params = json.loads((ROOT / "data" / "gbm_optuna.json").read_text())["single_tree"]["params"]
    dump, names = load_tree("data/_fam_cues_vitaldb_full_calfree_8000.pkl", params, max_depth_plot=3)
    tree = dump["tree_info"][0]["tree_structure"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off"); ax.set_xlim(-1, 1); ax.set_ylim(-0.1, 1.05)
    draw(tree, ax, 0.0, 1.0, 0.5, 0.32, names, 0, 3)
    ax.set_title("Interpretable single decision tree for DBP  (top 3 levels)\n"
                 "each split is a physiological threshold; leaves are predicted DBP (mmHg)",
                 fontsize=11)
    fig.savefig(ROOT / "figures" / "gbm_single_tree.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "gbm_single_tree.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[tree] figures/gbm_single_tree.png / .pdf")


if __name__ == "__main__":
    main()
