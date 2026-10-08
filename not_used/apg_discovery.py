"""apg_discovery.py -- discover novel APG/PPG features and validate their BP correlation on a
DISJOINT subject split so spurious (multiple-comparison) hits are exposed.

Method:
  1. Compute the full extended battery + the new literature APG timing/index features.
  2. Split subjects into DISCOVERY and VALIDATION halves (disjoint).
  3. Rank features by |Spearman rho| with DBP on discovery.
  4. Report the SAME feature's correlation on validation. Real features hold; spurious ones
     shrink toward 0. The shrinkage column is the honesty check.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import features_ext as fx
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
READABLE = {
    "t_b": "APG t_b (foot->b)", "t_c": "APG t_c", "t_d": "APG t_d", "t_e": "APG t_e",
    "apg_ratio_cd_a": "(c-d)/a", "apg_ushiro": "(c+d-b)/a Ushiro", "apg_reflect": "(b-e)/a",
    "vpg_ms_ratio": "VPG up/down ratio", "apg_ba_over_t": "(b/a)/t_b", "apg_area_sys": "APG sys energy",
    "apg_b_a": "APG b/a", "apg_c_a": "APG c/a", "apg_d_a": "APG d/a", "apg_e_a": "APG e/a",
    "aging_idx": "Takazawa index", "vpg_max": "VPG max",
}


def all_features(X, fs):
    """Extended battery + the new APG timing/index features, merged."""
    base = fx.compute_ext(X, fs, ppg_ch=PPG, ecg_ch=ECG)
    # new APG features (per-segment loop)
    nov = {k: [] for k in ["t_b", "t_c", "t_d", "t_e", "apg_ratio_cd_a", "apg_ushiro",
                           "apg_reflect", "vpg_ms_ratio", "apg_ba_over_t", "apg_area_sys"]}
    for i in range(len(X)):
        r = fx.apg_novel_cues(X[i, :, PPG], fs)
        for k in nov:
            nov[k].append(r.get(k, np.nan))
    base.update({k: np.array(v) for k, v in nov.items()})
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--n", type=int, default=6000)
    args = ap.parse_args()

    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["gte"]), min(args.n, len(d["gte"])), replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    y = d["yte"][sel]; grp = d["gte"][sel]
    print(f"[apg] {len(X)} seg / {len(np.unique(grp))} subjects; computing features...")
    feats = all_features(X, fs)

    # disjoint subject split
    subs = np.unique(grp); rng.shuffle(subs)
    disc_s = set(subs[:len(subs) // 2].tolist())
    disc = np.isin(grp, list(disc_s)); val = ~disc
    print(f"[apg] discovery {disc.sum()} seg / {len(disc_s)} subj | "
          f"validation {val.sum()} seg / {len(subs) - len(disc_s)} subj")

    def corr(v, mask, target):
        m = mask & np.isfinite(v) & np.isfinite(target)
        if m.sum() < 30:
            return np.nan
        return spearmanr(v[m], target[m]).correlation

    rows = []
    for k, v in feats.items():
        rd_disc = corr(v, disc, y[:, 1]); rd_val = corr(v, val, y[:, 1])
        rs_disc = corr(v, disc, y[:, 0]); rs_val = corr(v, val, y[:, 0])
        if np.isfinite(rd_disc):
            rows.append((k, rs_disc, rs_val, rd_disc, rd_val))
    # rank by |DBP discovery correlation|
    rows.sort(key=lambda r: -abs(r[3]))

    print(f"\n{'feature':22s} {'SBP_disc':>8} {'SBP_val':>8} {'DBP_disc':>8} {'DBP_val':>8} {'holds?':>7}")
    novel = {"t_b", "t_c", "t_d", "t_e", "apg_ratio_cd_a", "apg_ushiro", "apg_reflect",
             "vpg_ms_ratio", "apg_ba_over_t", "apg_area_sys"}
    keep = []
    for k, sd, sv, dd, dv in rows[:20]:
        holds = "yes" if (np.isfinite(dv) and np.sign(dd) == np.sign(dv)
                          and abs(dv) > 0.5 * abs(dd)) else "shrinks"
        star = "*" if k in novel else " "
        print(f"{star}{READABLE.get(k,k):21s} {sd:+8.2f} {sv:+8.2f} {dd:+8.2f} {dv:+8.2f} {holds:>7}")
        if k in novel and holds == "yes":
            keep.append(k)
    print(f"\n[apg] novel features that HOLD on validation: {[READABLE.get(k,k) for k in keep]}")

    # figure: discovery vs validation DBP correlation (real features on the diagonal)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k, sd, sv, dd, dv in rows:
        if not np.isfinite(dv):
            continue
        isnov = k in novel
        ax.scatter(dd, dv, s=70 if isnov else 35, c=("black" if isnov else "#bbbbbb"),
                   edgecolor="black", linewidth=0.8, zorder=3 if isnov else 2,
                   marker=("D" if isnov else "o"))
        if isnov and abs(dd) > 0.08:
            ax.annotate(READABLE.get(k, k), (dd, dv), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
    lim = 0.35
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5, zorder=1)
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("DBP correlation (discovery split)")
    ax.set_ylabel("DBP correlation (validation split)")
    ax.set_title("Feature reproducibility: on-diagonal = real, off = spurious\n"
                 "(diamonds = novel APG features)", fontsize=10)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_apg_discovery.png", dpi=200, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_apg_discovery.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_apg_discovery.png")


if __name__ == "__main__":
    main()
