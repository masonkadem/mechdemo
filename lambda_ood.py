"""lambda_ood.py -- does MAKING a model faithful make it generalise better?

Why this replaces the r = -0.71 result
---------------------------------------
The original claim correlated audit slope against OOD penalty across five architectures. It
collapsed for three reasons: it was anchored by an XResNet101 slope of +8.6 that did not
replicate (-18.3 on a larger sample), the audit was itself buggy at the time (NaN imputation
pinned slopes toward zero), and n = 5 meant one moving point swung r from -0.71 to -0.57 to
nothing. Corrected slopes are -0.009 to -0.003 with every CI spanning zero, so there is no
reliable variation left to correlate against.

This asks a better question. Rather than "do models that HAPPEN to be more faithful generalise
better?" -- observational, and confounded by everything else that differs between architectures --
it asks "does MAKING a model more faithful make it generalise better?" The PTT-supervision weight
lambda is a knob that demonstrably moves faithfulness (66% -> 76% in the earlier sweep), so
sweeping it while holding architecture, data and seed fixed gives an interventional test.

Each lambda is trained with several seeds, because the earlier single-seed sweep produced
differences (0.08 mmHg) smaller than the measured seed spread (sd 0.24 mmHg) and could not
support an ordering.

Three outcomes, all reportable:
  faithfulness up, OOD error down  -> the causal version of the original claim
  faithfulness up, OOD flat        -> faithfulness is achievable but does not buy generalisation
  faithfulness does not move       -> the knob failed to replicate, and the earlier sweep was noise
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mechlib
import physics_audit as pa
from mechlib import ECG, PPG
from ptt_supervised import DualHead, audit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIMIC = "C:/Users/mason/OneDrive - McMaster University/2026/BP"
FS = 125


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", default="0,0.25,1.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--train-n", type=int, default=40000)
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.train_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.train_n]
    print("[data] measuring arrival time on the training set ...", flush=True)
    pat_tr = mechlib.compute_ptt(Xtr, FS) * 1000.0
    mask_tr = np.isfinite(pat_tr)
    print(f"[data] measurable on {100*mask_tr.mean():.0f}% of segments (masked, not imputed)",
          flush=True)

    gte = d["gte"]
    subs = [s for s in np.unique(gte) if (gte == s).sum() >= 100]
    sel = np.concatenate([np.where(gte == s)[0][:100] for s in subs[:70]])
    Xid, yid, gid = (mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]]),
                     d["yte"][sel], gte[sel])

    # OOD: MIMIC-BP raw waveforms, so the deep models can be scored on them
    print("[data] loading MIMIC-BP ...", flush=True)
    m0 = pa.load_mimic_bp(MIMIC, channels=("ecg", "ppg"), max_patients=150)
    Xm, k = pa.window_segments(m0["X"], 1250)
    ym = np.repeat(m0["y"], k, 0)
    ii = np.sort(np.random.default_rng(0).choice(len(Xm), min(4000, len(Xm)), False))
    Xood, yood = mechlib.normalize(Xm[ii]), ym[ii]
    print(f"[data] OOD {len(Xood)} segments", flush=True)

    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    pm, ps = float(np.nanmean(pat_tr[mask_tr])), float(np.nanstd(pat_tr[mask_tr]) + 1e-9)
    Xt = torch.tensor(Xtr)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    pt = torch.tensor(np.where(mask_tr, (pat_tr - pm) / ps, 0.0), dtype=torch.float32)
    mt = torch.tensor(mask_tr.astype(np.float32))

    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = []
    print(f"\n{'lambda':>7s} {'seed':>5s} {'ID MAE':>8s} {'OOD MAE':>9s} "
          f"{'slope':>10s} {'faithful':>9s}")
    print("-" * 54)
    for lam in lambdas:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            net = DualHead().to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), 2e-3)
            for _ in range(args.epochs):
                net.train()
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), 256):
                    j = perm[i:i + 256]
                    opt.zero_grad()
                    bp, pp = net(Xt[j].to(DEVICE))
                    loss = nn.functional.mse_loss(bp, yt[j].to(DEVICE))
                    if lam > 0:
                        mj = mt[j].to(DEVICE)
                        if mj.sum() > 0:
                            loss = loss + lam * ((((pp - pt[j].to(DEVICE)) ** 2) * mj).sum()
                                                 / mj.sum())
                    loss.backward(); opt.step()

            net.eval()

            def pf(Xr, tgt=1):
                with torch.no_grad():
                    out = []
                    for i in range(0, len(Xr), 512):
                        b, _ = net(torch.tensor(Xr[i:i + 512]).to(DEVICE))
                        out.append(b.cpu().numpy())
                return np.concatenate(out)[:, tgt] * sd[tgt] + mu[tgt]

            id_mae = float(np.abs(pf(Xid) - yid[:, 1]).mean())
            ood_mae = float(np.abs(pf(Xood) - yood[:, 1]).mean())
            slope, frac = audit(pf, Xid, gid)
            rows.append({"lambda": lam, "seed": seed, "id_mae": id_mae, "ood_mae": ood_mae,
                         "slope": slope, "frac_faithful": frac})
            print(f"{lam:7.2f} {seed:5d} {id_mae:8.2f} {ood_mae:9.2f} {slope:+10.4f} "
                  f"{frac:9.0%}", flush=True)
            (DATA / "lambda_ood.json").write_text(json.dumps(rows, indent=2, default=float))

    # ---- summary: did the knob move faithfulness, and did OOD follow? --------
    print(f"\n{'lambda':>7s} {'ID MAE':>14s} {'OOD MAE':>15s} {'faithful':>15s}")
    agg = {}
    for lam in lambdas:
        r = [x for x in rows if x["lambda"] == lam]
        agg[lam] = {q: (float(np.mean([x[q] for x in r])), float(np.std([x[q] for x in r])))
                    for q in ("id_mae", "ood_mae", "frac_faithful", "slope")}
        a = agg[lam]
        print(f"{lam:7.2f} {a['id_mae'][0]:8.2f} +/-{a['id_mae'][1]:4.2f} "
              f"{a['ood_mae'][0]:9.2f} +/-{a['ood_mae'][1]:4.2f} "
              f"{a['frac_faithful'][0]:10.0%} +/-{a['frac_faithful'][1]:4.0%}")

    fa = [x["frac_faithful"] for x in rows]
    oo = [x["ood_mae"] for x in rows]
    moved = max(agg[l]["frac_faithful"][0] for l in lambdas) - \
        min(agg[l]["frac_faithful"][0] for l in lambdas)
    spread = float(np.mean([agg[l]["frac_faithful"][1] for l in lambdas]))
    print(f"\n[knob] faithfulness moved {moved:.0%} across lambda, "
          f"against a within-lambda seed spread of {spread:.0%}")
    if moved <= spread:
        print("       The knob did NOT move faithfulness beyond seed noise, so the")
        print("       faithfulness-versus-OOD question cannot be answered from this run.")
    else:
        r = float(np.corrcoef(fa, oo)[0, 1])
        print(f"[test] r(faithfulness, OOD error) = {r:+.3f} over {len(rows)} runs")
        print("       Negative means more faithful generalises better -- the causal version")
        print("       of the retracted r = -0.71. Near zero means faithfulness is achievable")
        print("       but does not buy generalisation.")
        agg["r_faithful_vs_ood"] = r

    (DATA / "lambda_ood.json").write_text(
        json.dumps({"runs": rows, "summary": agg}, indent=2, default=float))
    print(f"\n[done] data/lambda_ood.json")


if __name__ == "__main__":
    main()
