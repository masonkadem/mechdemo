"""xattn_mae.py -- MAE-pretrain the CROSS-ATTENTION trunk, to remove the objective confound.

The calibration notebook compares a self-supervised self-attention encoder (MAE) against a
SUPERVISED cross-attention encoder, so architecture and training objective move together and
neither can be credited. This pretrains cross-attention with the SAME masked-reconstruction
objective the MAE uses, giving the missing cell of the 2x2:

                        self-attention        cross-attention
    self-supervised     models/mae_probe.pt   THIS SCRIPT
    supervised          mae_probe.pt["sup"]   models/xattn_ecgppg.pt

Masking both streams is the load-bearing choice. ECG queries attend to PPG, so masking only the
ECG side leaves PPG fully visible and a query can be reconstructed from its own channel's context
without ever relating the two. Masking both forces the model to use the cross-channel
relationship, which is the thing the architecture exists to represent.

    python xattn_mae.py                 # 30 epochs, ~4 min on a GPU
    python xattn_mae.py --epochs 60     # longer
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mechlib
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS, L, PATCH = 125, 1250, 25


class CrossMAE(nn.Module):
    """Cross-attention encoder (ECG queries PPG) trained by masked reconstruction.

    The encoder half is deliberately parameter-compatible with xattn_model.CrossAttnBP -- same
    names, same shapes -- so the pretrained weights load straight into the supervised class and
    the calibration notebook can reuse it without a bespoke loader.
    """

    def __init__(self, dm=96, heads=4, depth=2, dec_depth=2):
        super().__init__()
        self.n_tok, self.dm = L // PATCH, dm
        self.ecg_emb = nn.Linear(PATCH, dm)
        self.ppg_emb = nn.Linear(PATCH, dm)
        self.pos_e = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        self.pos_p = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        self.mask_q = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        self.mask_kv = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(dm, heads, batch_first=True),
                "norm_q": nn.LayerNorm(dm), "norm_kv": nn.LayerNorm(dm),
                "norm_o": nn.LayerNorm(dm),
                "ff": nn.Sequential(nn.Linear(dm, dm * 2), nn.GELU(), nn.Linear(dm * 2, dm)),
            }) for _ in range(depth)])
        dec = nn.TransformerEncoderLayer(dm, heads, dm * 4, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.dec = nn.TransformerEncoder(dec, dec_depth)
        self.head = nn.Linear(dm, PATCH * 2)

    def _tok(self, x):
        b = x.shape[0]
        e = x[:, :, 0].reshape(b, self.n_tok, PATCH)
        p = x[:, :, 1].reshape(b, self.n_tok, PATCH)
        return e, p

    def encode(self, q, kv):
        for blk in self.blocks:
            a, _ = blk["attn"](blk["norm_q"](q), blk["norm_kv"](kv), blk["norm_kv"](kv),
                               need_weights=False)
            q = q + a
            q = q + blk["ff"](blk["norm_o"](q))
        return q

    def forward(self, x, mask_ratio=0.6):
        b = x.shape[0]
        e, p = self._tok(x)
        target = torch.cat([e, p], -1)
        q = self.ecg_emb(e) + self.pos_e
        kv = self.ppg_emb(p) + self.pos_p

        keep = int(self.n_tok * (1 - mask_ratio))
        idx = torch.rand(b, self.n_tok, device=x.device).argsort(1)
        vis, hid = idx[:, :keep], idx[:, keep:]

        # mask the ECG queries ...
        qv = torch.gather(q, 1, vis[..., None].expand(-1, -1, self.dm))
        # ... and independently mask the PPG keys/values, so neither channel is fully visible
        kidx = torch.rand(b, self.n_tok, device=x.device).argsort(1)[:, :keep]
        kvv = torch.gather(kv, 1, kidx[..., None].expand(-1, -1, self.dm))

        h = self.encode(qv, kvv)
        full = self.mask_q.expand(b, self.n_tok, self.dm).clone().scatter(
            1, vis[..., None].expand(-1, -1, self.dm), h) + self.pos_e
        rec = self.head(self.dec(full))
        return nn.functional.mse_loss(
            torch.gather(rec, 1, hid[..., None].expand(-1, -1, PATCH * 2)),
            torch.gather(target, 1, hid[..., None].expand(-1, -1, PATCH * 2)))

    @torch.no_grad()
    def represent(self, x):
        """Pooled representation, unmasked -- the tensor the calibration head consumes."""
        e, p = self._tok(x)
        return self.encode(self.ecg_emb(e) + self.pos_e, self.ppg_emb(p) + self.pos_p).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-n", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = np.load(DATA / "vitaldb_full_calfree.npz", mmap_mode="r")
    X = mechlib.normalize(np.array(d["Xtr"][:args.pretrain_n])[:, :, [ECG, PPG]])
    Xt = torch.tensor(X)
    print(f"[data] {len(Xt):,} pretraining segments, no labels used", flush=True)

    m = CrossMAE().to(DEVICE)
    print(f"[model] {sum(p.numel() for p in m.parameters()):,} parameters "
          f"({sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('dec.', 'head.', 'mask_'))):,} "
          f"in the encoder that transfers)", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    t0 = time.time()
    hist = []
    for ep in range(1, args.epochs + 1):
        m.train()
        perm = torch.randperm(len(Xt))
        tot = n = 0
        for i in range(0, len(Xt), args.bs):
            j = perm[i:i + args.bs]
            opt.zero_grad()
            loss = m(Xt[j].to(DEVICE))
            loss.backward(); opt.step()
            tot += loss.item() * len(j); n += len(j)
        sch.step()
        hist.append(tot / n)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  recon mse {tot/n:.5f}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "models" / "xattn_mae.pt"
    torch.save({"state_dict": m.state_dict(), "hist": hist,
                "cfg": {"dm": 96, "depth": 2, "epochs": args.epochs,
                        "pretrain_n": args.pretrain_n}}, out)
    (DATA / "xattn_mae.json").write_text(json.dumps(
        {"final_recon_mse": hist[-1], "epochs": args.epochs, "hist": hist,
         "minutes": (time.time() - t0) / 60}, indent=2))
    print(f"\n[done] {out.name}, final recon mse {hist[-1]:.5f}, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
