"""make_figures.py -- publication centerpiece figures from the run outputs.

Reads data/ood_benchmark_ecgppg.json, data/ood_benchmark_ppg.json, data/lightgbm_arm.json
and builds:

  fig1_id_vs_ood.png    ID error identical, OOD error diverges (the failure).
  fig2_mechanism_gap.png causal PAT slope / law-correctness predicts the OOD gap (the why).
  fig3_probe_layers.png  across-layer probe R^2 for the faithful model (mech-interp).
  fig4_gbm.png           LightGBM vs deep OOD + feature importance + age/sex ablation.
  fig5_dist_shift.png    KS distribution-shift heatmap (OOD is measured, not assumed).

Safe to run anytime; skips a figure whose inputs are missing.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def _models(blob):
    return blob.get("models", blob) if blob else {}


def fig_id_vs_ood(ecg, ppg):
    """Grouped bars: ID DBP MAE (near-identical) vs the worst external OOD DBP MAE per model."""
    m = _models(ppg) or _models(ecg)
    if not m:
        return
    names = [n for n in m if not n.startswith("_")]
    ext_sets = [c for c in m[names[0]]["ood"] if c in
                ("mimic_bp", "bcg", "sensors", "uci2", "ppgbp")]
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 4.6))
    x = np.arange(len(names))
    w = 0.8 / (len(ext_sets) + 1)
    ax.bar(x, [m[n]["ood"]["id"]["mae_dbp"] for n in names], w, label="ID (VitalDB)",
           color="tab:green")
    for i, s in enumerate(ext_sets):
        ax.bar(x + (i + 1) * w, [m[n]["ood"].get(s, {}).get("mae_dbp", np.nan) for n in names],
               w, label=f"OOD {s}")
    ax.set_xticks(x + 0.4 - w / 2, names, rotation=20, ha="right")
    ax.set_ylabel("DBP MAE (mmHg)")
    ax.set_title("Same in-distribution error, divergent out-of-distribution error", fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig1_id_vs_ood.png", dpi=150); plt.close(fig)
    print("[fig] fig1_id_vs_ood.png")


def fig_mechanism_gap(ecg):
    """PAT causal slope (mechanism) vs OOD penalty on MIMIC-BP (the ECG+PPG track)."""
    m = _models(ecg)
    if not m:
        return
    names = [n for n in m if not n.startswith("_")]
    if not np.isfinite(m[names[0]]["audit"]["dbp"]["dBP_dPTT"]):
        return
    slope = [m[n]["audit"]["dbp"]["dBP_dPTT"] for n in names]
    base = [m[n]["ood"]["id"]["mae_dbp"] for n in names]
    gap = [m[n]["ood"].get("mimic_bp", {}).get("mae_dbp", np.nan) - b for n, b in zip(names, base)]
    frac = [m[n]["audit"]["dbp"]["frac_correct_sign"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(slope, gap, c=frac, cmap="RdYlGn", s=90, vmin=0.3, vmax=0.9,
                    edgecolor="k", zorder=3)
    for n, s, g in zip(names, slope, gap):
        ax.annotate(n, (s, g), fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel("causal PAT slope  dDBP/dPTT (mmHg/s)  -- negative = physiological")
    ax.set_ylabel("OOD penalty on MIMIC-BP (mmHg)")
    ax.set_title("Mechanism predicts out-of-distribution fragility", fontsize=11)
    fig.colorbar(sc, label="fraction of segments correct sign")
    fig.tight_layout(); fig.savefig(FIG / "fig2_mechanism_gap.png", dpi=150); plt.close(fig)
    print("[fig] fig2_mechanism_gap.png")


def fig_probe_layers(ecg, faithful="xresnet1d50"):
    """Across-layer probe R^2 heatmap for the faithful model -- the mech-interp panel."""
    m = _models(ecg)
    if not m or faithful not in m:
        faithful = next((n for n in _models(ecg) if not n.startswith("_")), None)
        if faithful is None:
            return
    rows = m[faithful]["probe"]
    layers = list(rows)
    cues = list(rows[layers[0]])
    # order cues by peak decodability so the structure reads top-to-bottom
    peak = {c: max(rows[l].get(c, 0) for l in layers) for c in cues}
    cues = sorted(cues, key=lambda c: -peak[c])
    M = np.array([[max(rows[l].get(c, 0.0), 0.0) for l in layers] for c in cues])
    fig, ax = plt.subplots(figsize=(1.0 * len(layers) + 3, 0.3 * len(cues) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)), layers, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cues)), cues, fontsize=7)
    ax.set_title(f"{faithful}: what is decodable, and where (probe $R^2$ by layer)", fontsize=10)
    fig.colorbar(im, ax=ax, label="held-out $R^2$")
    fig.tight_layout(); fig.savefig(FIG / "fig3_probe_layers.png", dpi=150); plt.close(fig)
    print("[fig] fig3_probe_layers.png")


def fig_gbm(gbmres, ecg, ppg):
    """LightGBM OOD vs deep models + its feature importance + age/sex ablation."""
    if not gbmres:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) OOD DBP: GBM vs mean deep
    ood = gbmres["ood"]
    sets = list(ood)
    ax[0].bar(range(len(sets)), [ood[s]["mae_dbp"] for s in sets], color="tab:orange",
              label="LightGBM (audit features)")
    dm = _models(ppg) or _models(ecg)
    if dm:
        names = [n for n in dm if not n.startswith("_")]
        deep = [np.nanmean([dm[n]["ood"].get(s, {}).get("mae_dbp", np.nan) for n in names])
                for s in sets]
        ax[0].plot(range(len(sets)), deep, "k--o", label="deep mean", zorder=3)
    ax[0].set_xticks(range(len(sets)), sets, rotation=30, ha="right", fontsize=8)
    ax[0].set_ylabel("DBP MAE (mmHg)"); ax[0].set_title("Interpretable model OOD", fontsize=10)
    ax[0].legend(fontsize=8)

    # (b) feature importance
    imp = gbmres.get("importance_dbp", [])[:12]
    if imp:
        ax[1].barh([n for n, _ in imp][::-1], [v for _, v in imp][::-1], color="tab:purple")
        ax[1].set_xlabel("gain"); ax[1].set_title("What the tree uses (DBP)", fontsize=10)

    # (c) age/sex ablation
    ab = gbmres.get("demo_ablation")
    if ab:
        ax[2].bar(["with\nage+sex", "without"], [ab["with_demo_dbp"], ab["no_demo_dbp"]],
                  color=["tab:blue", "tab:gray"])
        ax[2].set_ylabel("DBP MAE (mmHg)")
        ax[2].set_title(f"Age/sex impact on {ab['set']}\ndelta = {ab['delta_dbp']:+.2f} mmHg",
                        fontsize=10)
        if ab.get("demo_shap"):
            txt = "  ".join(f"{k}:{v:.2f}" for k, v in ab["demo_shap"].items())
            ax[2].text(0.5, 0.9, "mean|SHAP| " + txt, transform=ax[2].transAxes,
                       ha="center", fontsize=8)
    fig.suptitle("LightGBM built from the faithful model's audit-passing features", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "fig4_gbm.png", dpi=150); plt.close(fig)
    print("[fig] fig4_gbm.png")


def fig_dist_shift(ecg, ppg):
    """KS distribution-shift per cue per OOD set -- quantifies 'this is really OOD'."""
    blob = ppg or ecg
    ds = blob.get("dist_shift") if blob else None
    if not ds:
        return
    sets = list(ds)
    cues = sorted({c for s in ds.values() for c in s})
    M = np.array([[ds[s].get(c, np.nan) for s in sets] for c in cues])
    fig, ax = plt.subplots(figsize=(1.1 * len(sets) + 3, 0.32 * len(cues) + 2))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(sets)), sets, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(cues)), cues, fontsize=7)
    ax.set_title("Distribution shift vs VitalDB (KS per cue; 1 = disjoint)", fontsize=10)
    fig.colorbar(im, ax=ax, label="KS statistic")
    fig.tight_layout(); fig.savefig(FIG / "fig5_dist_shift.png", dpi=150); plt.close(fig)
    print("[fig] fig5_dist_shift.png")


def push_wandb(project, entity):
    """Upload the centerpiece figures + a headline results table to a W&B summary run."""
    import wandb
    run = wandb.init(project=project, entity=entity, name="figures-summary",
                     group="ood-audit", reinit=True)
    for f in sorted(FIG.glob("fig*.png")):
        run.log({f"figure/{f.stem}": wandb.Image(str(f))})
    ecg = load("ood_benchmark_ecgppg.json"); ppg = load("ood_benchmark_ppg.json")
    m = _models(ecg) or _models(ppg)
    if m:
        names = [n for n in m if not n.startswith("_")]
        conds = list(m[names[0]]["ood"])
        tbl = wandb.Table(columns=["model", "id_dbp", "audit_slope", "frac_correct"]
                          + [f"ood/{c}" for c in conds])
        for n in names:
            r = m[n]
            tbl.add_data(n, r["ood"]["id"]["mae_dbp"], r["audit"]["dbp"]["dBP_dPTT"],
                         r["audit"]["dbp"]["frac_correct_sign"],
                         *[r["ood"][c]["mae_dbp"] for c in conds])
        run.log({"headline_results": tbl})
    run.finish()
    print("[fig] pushed figures + table to W&B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--project", default="ppg-ood-audit")
    ap.add_argument("--entity", default="mkadem")
    args = ap.parse_args()

    ecg = load("ood_benchmark_ecgppg.json")
    ppg = load("ood_benchmark_ppg.json")
    gbmres = load("lightgbm_arm.json")
    fig_id_vs_ood(ecg, ppg)
    fig_mechanism_gap(ecg)
    fig_probe_layers(ecg)
    fig_gbm(gbmres, ecg, ppg)
    fig_dist_shift(ecg, ppg)
    print("[fig] done ->", FIG)
    if not args.no_wandb:
        try:
            push_wandb(args.project, args.entity)
        except Exception as e:
            print(f"[fig] W&B upload skipped: {e}")


if __name__ == "__main__":
    main()
