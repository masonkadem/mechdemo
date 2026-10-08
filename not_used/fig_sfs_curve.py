"""fig_sfs_curve.py -- SFS: how many features, and what do you get for each one.

Plots the forward-selection curve (CV MAE vs number of features, feature added at each step
labelled) for both targets and both capacities, against the importance-ranked-4 baseline.
Marks where each run's own plateau stop fired.

Reads whichever sfs_selection_trees*_{sbp,dbp}.json files are present, so it fills in as runs
land.

    python fig_sfs_curve.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COLORS = {1: "#0072B2", 150: "#D55E00"}
TARGET_LABEL = {"dbp": "diastolic", "sbp": "systolic"}


def plot_target(ax, target):
    runs = sorted(DATA.glob(f"sfs_selection_trees*_{target}.json"))
    table_rows = []
    for path in runs:
        d = json.loads(path.read_text())
        trees = d.get("trees", 1)
        ns = [h["n"] for h in d["history"]]
        cv = [h["cv_mae"] for h in d["history"]]
        col = COLORS.get(trees, "#555")

        ax.plot(ns, cv, "o-", color=col, lw=2.2, ms=6, mec="white", mew=1,
                label=f"{trees} tree{'s' if trees != 1 else ''}")
        for h in d["history"]:
            ax.annotate(h["added"], (h["n"], h["cv_mae"]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color=col, rotation=40)

        tol = d.get("tol"); patience = d.get("patience")
        cap = d.get("max_feat", None)
        stopped_at_cap = cap is not None and len(ns) == cap
        if not stopped_at_cap and tol is not None:
            ax.plot(ns[-1], cv[-1], "*", color=col, ms=16, mec="white", mew=0.8, zorder=5)

        ax.axhline(d["ranked4_cv"], color=col, lw=1.1, ls="--", alpha=0.55)

        table_rows.append((trees, d))

    ax.set_xlabel("number of features")
    ax.set_ylabel(f"4-fold subject-disjoint CV MAE, {TARGET_LABEL[target]} BP (mmHg)")
    ax.set_title(f"{TARGET_LABEL[target].upper()[:1]}BP", fontsize=12.5, loc="left")
    if runs:
        ax.legend(frameon=False, loc="upper right", title="SFS capacity")
    ax.spines[["top", "right"]].set_visible(False)
    return table_rows


def main():
    targets = [t for t in ("dbp", "sbp") if list(DATA.glob(f"sfs_selection_trees*_{t}.json"))]
    if not targets:
        raise SystemExit("no sfs_selection_trees*_{sbp,dbp}.json yet")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2 * len(targets), 5.6),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Forward feature selection: what each additional feature buys\n"
                 "★ marks where the search stopped itself (plateau reached)",
                 fontsize=12)

    all_rows = {}
    for ax, target in zip(axes[0], targets):
        all_rows[target] = plot_target(ax, target)

    out = ROOT / "figures" / "fig_sfs_curve.png"
    fig.savefig(out, dpi=300); fig.savefig(out.with_suffix(".pdf"))
    print(f"[done] {out}\n")

    for target, rows in all_rows.items():
        for trees, d in rows:
            stop_reason = ("plateau" if len(d["history"]) < d.get("max_feat", 999)
                          else "hit max-feat cap")
            print(f"=== target={target} | {trees} tree(s) | stopped: {stop_reason} at "
                  f"n={len(d['history'])} | ranked-4 baseline {d['ranked4_cv']:.4f} ===")
            print(f"{'n':>3} {'added':16} {'cv_mae':>9} {'Δ vs prev':>10} {'Δ vs ranked-4':>14}")
            prev = None
            for h in d["history"]:
                dprev = f"{h['cv_mae'] - prev:+.4f}" if prev is not None else "     --"
                dr4 = f"{h['cv_mae'] - d['ranked4_cv']:+.4f}"
                print(f"{h['n']:>3} {h['added']:16} {h['cv_mae']:9.4f} {dprev:>10} {dr4:>14}")
                prev = h["cv_mae"]
            print()


if __name__ == "__main__":
    main()
