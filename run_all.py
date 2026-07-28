"""run_all.py -- one-shot orchestrator for the full publication pipeline. Start it and leave.

Runs, in order, skipping any stage whose output already exists (so it is resumable if a
stage crashes or the machine reboots):

  1. ECG+PPG track   : 5 deep models @ --epochs, MIMIC-BP OOD + PAT roll-audit + ext cues
  2. PPG-only track  : 5 deep models, all 4 BP-Benchmark OOD sets + MIMIC-BP + ext cues
  3. LightGBM arm    : features the FAITHFUL model passes (probe + governing-law) + age/sex,
                       ID/OOD eval, age/sex ablation + SHAP
  4. Figures         : centerpiece publication figures from the two tracks + GBM

Everything logs to Weights & Biases (project ppg-ood-audit, entity mkadem) and writes
data/*.json + figures/*.png. Per-stage stdout is tee'd to logs/run_all_<stage>.log.

    python run_all.py                     # full weekend run (epochs=60, all MIMIC patients)
    python run_all.py --epochs 30 --quick # faster sanity pass
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOGS = ROOT / "logs"
DATA = ROOT / "data"

MIMIC = "C:/Users/mason/OneDrive - McMaster University/2026/BP"
EXTERNALS = {
    "bcg": "data/bcg_dataset",
    "sensors": "C:/Users/mason/Downloads/sensors_dataset/sensors_dataset",
    "uci2": "data/uci2_dataset/uci2_dataset",
    "ppgbp": "C:/Users/mason/Downloads/ppgbp_dataset/ppgbp_dataset",
}


def run(name, cmd, skip_if=None):
    """Run one stage, tee stdout to logs/run_all_<name>.log. Skip if its output exists."""
    if skip_if and Path(skip_if).exists():
        print(f"[run_all] SKIP {name} (found {skip_if})", flush=True)
        return True
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"run_all_{name}.log"
    print(f"\n[run_all] === {name} ===\n[run_all] {' '.join(cmd)}\n[run_all] log -> {log}",
          flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    dt = time.time() - t0
    ok = p.returncode == 0
    print(f"[run_all] {name} {'OK' if ok else 'FAILED (exit %d)' % p.returncode} in {dt/60:.1f} min",
          flush=True)
    if not ok:
        print(f"[run_all] see {log} -- continuing to next stage", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--faithful", default="xresnet1d50",
                    help="model whose audit selects the LightGBM features")
    ap.add_argument("--project", default="ppg-ood-audit")
    ap.add_argument("--entity", default="mkadem")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="small MIMIC + fewer segments for a fast end-to-end check")
    ap.add_argument("--force", action="store_true", help="re-run stages even if output exists")
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz",
                    help="training npz; full PulseDB CalFree by default (falls back to demo cache)")
    args = ap.parse_args()

    wb = ["--no-wandb"] if args.no_wandb else ["--project", args.project, "--entity", args.entity]
    mimic_pat = "150" if args.quick else "0"        # 0 = all 1524
    ext_arg = ",".join(f"{k}={v}" for k, v in EXTERNALS.items())
    data = args.data if (ROOT / args.data).exists() else "data/vitaldb_mini_deep.npz"
    if data != args.data:
        print(f"[run_all] WARNING: {args.data} missing -> falling back to {data}", flush=True)
    dd = ["--data", data]
    t_start = time.time()

    # ---- stage 1: ECG+PPG track (headline PAT audit on MIMIC-BP)
    j1 = DATA / "ood_benchmark_ecgppg.json"
    run("ecgppg", [PY, "ood_benchmark.py", "--models", "all", "--epochs", str(args.epochs), *dd,
                   "--ext-cues", "--run-tag", "ecgppg", "--mimic", MIMIC,
                   "--mimic-patients", mimic_pat, *wb],
        skip_if=None if args.force else j1)

    # ---- stage 2: PPG-only track (all external OOD sets)
    j2 = DATA / "ood_benchmark_ppg.json"
    run("ppgonly", [PY, "ood_benchmark.py", "--models", "all", "--epochs", str(args.epochs), *dd,
                    "--ppg-only", "--ext-cues", "--run-tag", "ppg", "--external", ext_arg,
                    "--mimic", MIMIC, "--mimic-patients", mimic_pat, *wb],
        skip_if=None if args.force else j2)

    # ---- stage 3: LightGBM from faithful features (needs stage-1 json)
    if j1.exists():
        run("lightgbm", [PY, "run_lightgbm.py", "--faithful", args.faithful, *dd,
                         "--audit", str(j1), "--mimic", MIMIC, "--vitaldb-demo",
                         "--mimic-patients", ("150" if args.quick else "400"),
                         "--external", ext_arg, *wb],
            skip_if=None if args.force else (DATA / "lightgbm_arm.json"))
    else:
        print("[run_all] SKIP lightgbm: stage-1 audit json missing", flush=True)

    # ---- stage 4: publication figures (also pushes figures + headline table to W&B)
    run("figures", [PY, "make_figures.py", *wb], skip_if=None)

    print(f"\n[run_all] ALL DONE in {(time.time()-t_start)/60:.1f} min", flush=True)
    print("[run_all] outputs: data/ood_benchmark_*.json, data/lightgbm_arm.json, figures/*.png")
    print(f"[run_all] W&B: https://wandb.ai/{args.entity}/{args.project}")


if __name__ == "__main__":
    main()
