"""mae_pretrain_full.py -- pretrain the MAE on ALL 396k training segments, not 80k.

Why this exists: every model comparison in this project has been unfair to the transformer. The
MAE encoder was pretrained on Xtr[:80000] (223 patients) while LightGBM was fitted on all 396,000
segments (1,100 patients). LightGBM's better uncalibrated starting point (6.90 vs 8.19 mmHg DBP)
is at least partly that 5x data advantage, so the two cannot be compared until the transformer
gets the same budget.

Xtr is 5.9 GB in float32, so it is streamed in chunks rather than held in memory: each epoch
walks the file in CHUNK-sized blocks, shuffling within a block. Block shuffling rather than
global shuffling is the one compromise -- the alternative is a 6 GB resident array.

    python mae_pretrain_full.py                     # 30 epochs, ~25 min
    python mae_pretrain_full.py --epochs 60         # better converged, ~50 min
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mechlib
from mae_probe import MAE
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=40000, help="segments held in RAM at once")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/mae_probe_full.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    n = d["Xtr"].shape[0]
    n_pat = len(np.unique(np.array(d["gtr"])))
    print(f"[data] {n:,} segments, {n_pat} patients (was 80,000 / 223)", flush=True)

    m = MAE().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    rng = np.random.default_rng(args.seed)
    starts = np.arange(0, n, args.chunk)
    t0, hist = time.time(), []
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = cnt = 0
        for s in rng.permutation(starts):                 # chunk order varies each epoch
            e = min(s + args.chunk, n)
            X = mechlib.normalize(np.array(d["Xtr"][s:e])[:, :, [ECG, PPG]])
            Xt = torch.tensor(X)
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), args.bs):
                j = perm[i:i + args.bs]
                opt.zero_grad()
                loss = m(Xt[j].to(DEVICE))
                loss.backward(); opt.step()
                tot += loss.item() * len(j); cnt += len(j)
            del X, Xt
        sch.step()
        hist.append(tot / cnt)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  recon mse {tot/cnt:.5f}  ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    out = ROOT / args.out
    torch.save({"mae": m.state_dict(), "hist": hist,
                "cfg": {"epochs": args.epochs, "n_segments": int(n), "n_patients": int(n_pat)}},
               out)
    (DATA / "mae_pretrain_full.json").write_text(json.dumps(
        {"final_recon_mse": hist[-1], "hist": hist, "epochs": args.epochs,
         "n_segments": int(n), "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"\n[done] {out.name}, recon mse {hist[-1]:.5f}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
