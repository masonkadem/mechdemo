"""xattn_model.py -- ECG-PPG cross-attention, where arrival time is READABLE, not imposed.

Why this architecture rather than another CNN
---------------------------------------------
The PTT-supervision result showed faithfulness can be induced, but it had to be forced with an
auxiliary loss. Cross-attention makes the mechanism native instead: ECG patches attend to PPG
patches, so the position of the attention mass along the PPG axis IS an arrival-time estimate.
That gives a faithfulness read-out that is architectural rather than imposed --

    attention lag  =  (mean attended PPG position) - (query ECG position)

and it can be compared directly against the PTT measured by signal processing. Agreement means
the model located the pulse; disagreement means it is attending for some other reason. No
intervention, no auxiliary target, no roll audit needed -- although the roll audit is still run
here as an independent check.

Deliberately no convolutional trunk. Patch embedding is a single linear projection over raw
samples, so nothing but attention relates the two channels and the lag read-out cannot be
contaminated by a conv stack that has already mixed them.

    python xattn_model.py --epochs 20
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mechlib
from mechlib import ECG, PPG, _shift_channel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS = 125
L = 1250
DELTAS = (-6, -4, -2, 0)
SLIP_MS = 150.0


class CrossAttnBP(nn.Module):
    """ECG queries, PPG keys/values. One linear patch embedding per channel, no convolutions."""

    def __init__(self, patch=25, dm=96, heads=4, depth=2, n_out=2):
        super().__init__()
        self.patch = patch
        self.n_tok = L // patch
        self.ecg_emb = nn.Linear(patch, dm)
        self.ppg_emb = nn.Linear(patch, dm)
        # learned positional encodings: the lag read-out needs the model to know WHERE a token is
        self.pos_e = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        self.pos_p = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(dm, heads, batch_first=True),
                "norm_q": nn.LayerNorm(dm), "norm_kv": nn.LayerNorm(dm),
                "norm_o": nn.LayerNorm(dm),
                "ff": nn.Sequential(nn.Linear(dm, dm * 2), nn.GELU(), nn.Linear(dm * 2, dm)),
            }) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 64), nn.GELU(),
                                  nn.Linear(64, n_out))

    def tokens(self, x):
        b = x.shape[0]
        e = x[:, :, ECG].reshape(b, self.n_tok, self.patch)
        p = x[:, :, PPG].reshape(b, self.n_tok, self.patch)
        return self.ecg_emb(e) + self.pos_e, self.ppg_emb(p) + self.pos_p

    def forward(self, x, return_attn=False):
        q, kv = self.tokens(x)
        attn_last = None
        for blk in self.blocks:
            a, w = blk["attn"](blk["norm_q"](q), blk["norm_kv"](kv), blk["norm_kv"](kv),
                               need_weights=True, average_attn_weights=True)
            q = q + a
            q = q + blk["ff"](blk["norm_o"](q))
            attn_last = w                       # (B, n_tok_ecg, n_tok_ppg)
        out = self.head(q.mean(1))
        return (out, attn_last) if return_attn else out

    @torch.no_grad()
    def attention_lag_ms(self, x):
        """Per-segment attention lag in ms: how far along the PPG axis each ECG token looks.

        Computed as the attention-weighted mean PPG token index minus the ECG token index,
        averaged over queries, then converted to milliseconds. A model that has located the
        pulse should give a positive lag of roughly one PTT.
        """
        _, w = self.forward(x, return_attn=True)
        b, nq, nk = w.shape
        kpos = torch.arange(nk, device=w.device, dtype=w.dtype)
        qpos = torch.arange(nq, device=w.device, dtype=w.dtype)
        attended = (w * kpos.view(1, 1, -1)).sum(-1)          # (B, nq)
        lag_tokens = (attended - qpos.view(1, -1)).mean(-1)   # (B,)
        return (lag_tokens * self.patch / FS * 1000.0).cpu().numpy()


def audit(pred_fn, X, g, fs=FS):
    """Validated roll audit: negative arm, non-finite PTT dropped, beat-slips discarded."""
    base = mechlib.compute_ptt(X, fs)
    keep = np.isfinite(base)
    nom = np.array([1000.0 * d / fs for d in DELTAS])
    P = np.full((len(X), len(DELTAS)), np.nan)
    for j, dl in enumerate(DELTAS):
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], dl)
        p = mechlib.compute_ptt(Xd, fs)
        dd = (p - base) * 1000.0
        keep &= np.isfinite(p) & (np.abs(dd) <= SLIP_MS)
        P[:, j] = pred_fn(Xd)
    sl = [np.median([np.polyfit(nom, P[i], 1)[0] for i in np.where((g == s) & keep)[0]])
          for s in np.unique(g) if ((g == s) & keep).sum() >= 5]
    sl = np.array(sl)
    return (float(np.median(sl)) if len(sl) else np.nan,
            float(np.mean(sl < 0)) if len(sl) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-n", type=int, default=60000)
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.train_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.train_n]
    gte = d["gte"]
    subs = [s for s in np.unique(gte) if (gte == s).sum() >= 100]
    sel = np.concatenate([np.where(gte == s)[0][:100] for s in subs[:80]])
    Xte, yte, gsel = (mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]]),
                      d["yte"][sel], gte[sel])

    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    Xt = torch.tensor(Xtr)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)

    net = CrossAttnBP().to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[model] cross-attention, {n_par:,} parameters, no convolutions", flush=True)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[j].to(DEVICE)), yt[j].to(DEVICE))
            loss.backward(); opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}  train mse {tot/len(Xt):.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        pred = np.concatenate([net(torch.tensor(Xte[i:i + 512]).to(DEVICE)).cpu().numpy()
                               for i in range(0, len(Xte), 512)]) * sd + mu
    mae = float(np.abs(pred[:, 1] - yte[:, 1]).mean())

    lags = np.concatenate([net.attention_lag_ms(torch.tensor(Xte[i:i + 512]).to(DEVICE))
                           for i in range(0, len(Xte), 512)])
    ptt = mechlib.compute_ptt(Xte, FS) * 1000.0
    ok = np.isfinite(ptt) & np.isfinite(lags)
    from scipy import stats
    r_all = float(stats.spearmanr(lags[ok], ptt[ok]).statistic) if ok.sum() > 100 else np.nan
    rs = []
    for s in np.unique(gsel):
        m = ok & (gsel == s)
        if m.sum() >= 30 and np.std(lags[m]) > 1e-9 and np.std(ptt[m]) > 1e-9:
            rs.append(stats.spearmanr(lags[m], ptt[m]).statistic)
    r_within = float(np.nanmedian(rs)) if rs else np.nan

    def pf(Xr):
        with torch.no_grad():
            return np.concatenate([net(torch.tensor(Xr[i:i + 512]).to(DEVICE)).cpu().numpy()
                                   for i in range(0, len(Xr), 512)])[:, 1] * sd[1] + mu[1]

    slope, frac = audit(pf, Xte, gsel)

    print(f"\n[acc]  DBP MAE {mae:.2f} mmHg")
    print(f"[lag]  attention lag: median {np.median(lags[ok]):+.1f} ms "
          f"(measured PTT median {np.median(ptt[ok]):.1f} ms)")
    print(f"[lag]  correlation with measured PTT: pooled r = {r_all:+.3f}, "
          f"within-subject r = {r_within:+.3f} (n={len(rs)} subjects)")
    print(f"[audit] roll slope {slope:+.4f}, faithful on {frac:.0%} of subjects")
    print("\nThe attention lag is a faithfulness read-out that needs no intervention: if it")
    print("tracks measured PTT, the model located the pulse rather than merely fitting BP.")

    res = {"n_params": n_par, "dbp_mae": mae,
           "attn_lag_median_ms": float(np.median(lags[ok])),
           "ptt_median_ms": float(np.median(ptt[ok])),
           "r_lag_ptt_pooled": r_all, "r_lag_ptt_within": r_within,
           "audit_slope": slope, "frac_faithful": frac}
    (DATA / "xattn_model.json").write_text(json.dumps(res, indent=2, default=float))
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd},
               ROOT / "models" / "xattn_ecgppg.pt")
    print(f"\n[done] data/xattn_model.json")


if __name__ == "__main__":
    main()
