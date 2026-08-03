"""xattn_fine.py -- cross-attention at fine time resolution, so the lag read-out is not quantised.

What the coarse version could not settle
-----------------------------------------
xattn_model.py used patch=25 (200 ms per token, 50 tokens). Measured arrival time is around
250-310 ms, so it spanned about 1.5 tokens and the attention-lag read-out was quantised at the
same scale as the quantity it was trying to measure. The lag came out at +20.5 ms against a
measured 310.5 ms with r = +0.05, which is consistent with "the model ignores arrival time" but
equally consistent with "the read-out cannot resolve it". This version removes that ambiguity.

  * stride-based overlapping windows instead of disjoint patches, so token spacing (the lag
    quantum) is decoupled from token width (the receptive field). stride=4 at 125 Hz gives a
    32 ms quantum, ~8x finer than before, while each token still sees 100 ms of signal.
  * a single linear map per channel, no convolutions, so nothing but attention relates ECG to PPG.
  * an explicit relative-position bias over the query-key lag, which is how the model can express
    "attend d samples ahead" directly rather than having to infer it from absolute encodings.

A NOTE ON WHAT IS BEING MEASURED. compute_ptt returns R-peak -> PPG foot, i.e. PAT = PEP + PTT.
PEP is cardiac and, in these surgical datasets, drug-sensitive: propofol lengthens it, ephedrine
shortens it. So agreement between attention lag and this reference means the model located the
pulse ARRIVAL, not the vascular transit the governing law concerns. Disagreement is the stronger
result either way, since it rules out the composite too.
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
FS, L = 125, 1250
DELTAS = (-6, -4, -2, 0)
SLIP_MS = 150.0


class FineCrossAttn(nn.Module):
    """Overlapping-window tokens: width sets the receptive field, stride sets the lag quantum."""

    def __init__(self, width=12, stride=4, dm=64, heads=4, depth=2, n_out=2, max_lag=48):
        super().__init__()
        self.width, self.stride = width, stride
        self.n_tok = (L - width) // stride + 1
        self.ecg_emb = nn.Linear(width, dm)
        self.ppg_emb = nn.Linear(width, dm)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        # relative-position bias: one learned scalar per query-key offset, so "attend d tokens
        # ahead" is directly expressible rather than something the model must infer
        self.max_lag = max_lag
        self.rel = nn.Parameter(torch.zeros(2 * max_lag + 1))
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "q": nn.Linear(dm, dm), "k": nn.Linear(dm, dm), "v": nn.Linear(dm, dm),
                "o": nn.Linear(dm, dm),
                "n1": nn.LayerNorm(dm), "n2": nn.LayerNorm(dm), "n3": nn.LayerNorm(dm),
                "ff": nn.Sequential(nn.Linear(dm, dm * 2), nn.GELU(), nn.Linear(dm * 2, dm)),
            }) for _ in range(depth)])
        self.heads, self.dm = heads, dm
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 64), nn.GELU(),
                                  nn.Linear(64, n_out))

    def tokens(self, x):
        e = x[:, :, ECG].unfold(1, self.width, self.stride)       # (B, n_tok, width)
        p = x[:, :, PPG].unfold(1, self.width, self.stride)
        return self.ecg_emb(e) + self.pos, self.ppg_emb(p) + self.pos

    def _rel_bias(self, nq, nk, device):
        qi = torch.arange(nq, device=device).view(-1, 1)
        ki = torch.arange(nk, device=device).view(1, -1)
        off = (ki - qi).clamp(-self.max_lag, self.max_lag) + self.max_lag
        return self.rel[off]

    def forward(self, x, return_attn=False):
        q, kv = self.tokens(x)
        w_last = None
        for b in self.blocks:
            qn, kn = b["n1"](q), b["n2"](kv)
            Q, K, V = b["q"](qn), b["k"](kn), b["v"](kn)
            att = (Q @ K.transpose(1, 2)) / (self.dm ** 0.5) + self._rel_bias(
                Q.shape[1], K.shape[1], Q.device)
            w = att.softmax(-1)
            q = q + b["o"](w @ V)
            q = q + b["ff"](b["n3"](q))
            w_last = w
        out = self.head(q.mean(1))
        return (out, w_last) if return_attn else out

    @torch.no_grad()
    def attention_lag_ms(self, x):
        """Attention-weighted mean key offset, in ms. Quantum = stride / fs."""
        _, w = self.forward(x, return_attn=True)
        nq, nk = w.shape[1], w.shape[2]
        kpos = torch.arange(nk, device=w.device, dtype=w.dtype)
        qpos = torch.arange(nq, device=w.device, dtype=w.dtype)
        lag_tok = ((w * kpos.view(1, 1, -1)).sum(-1) - qpos.view(1, -1)).mean(-1)
        return (lag_tok * self.stride / FS * 1000.0).cpu().numpy()


def audit(pred_fn, X, g, fs=FS):
    base = mechlib.compute_ptt(X, fs)
    keep = np.isfinite(base)
    nom = np.array([1000.0 * d / fs for d in DELTAS])
    P = np.full((len(X), len(DELTAS)), np.nan)
    for j, dl in enumerate(DELTAS):
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], dl)
        p = mechlib.compute_ptt(Xd, fs)
        keep &= np.isfinite(p) & (np.abs((p - base) * 1000.0) <= SLIP_MS)
        P[:, j] = pred_fn(Xd)
    sl = [np.median([np.polyfit(nom, P[i], 1)[0] for i in np.where((g == s) & keep)[0]])
          for s in np.unique(g) if ((g == s) & keep).sum() >= 5]
    sl = np.array(sl)
    return (float(np.median(sl)) if len(sl) else np.nan,
            float(np.mean(sl < 0)) if len(sl) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--train-n", type=int, default=40000)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--width", type=int, default=12)
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.train_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.train_n]
    gte = d["gte"]
    subs = [s for s in np.unique(gte) if (gte == s).sum() >= 100]
    sel = np.concatenate([np.where(gte == s)[0][:100] for s in subs[:60]])
    Xte, yte, gsel = (mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]]),
                      d["yte"][sel], gte[sel])

    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    Xt, yt = torch.tensor(Xtr), torch.tensor((ytr - mu) / sd, dtype=torch.float32)

    net = FineCrossAttn(width=args.width, stride=args.stride).to(DEVICE)
    print(f"[model] {sum(p.numel() for p in net.parameters()):,} params, "
          f"{net.n_tok} tokens, width {args.width} ({1000*args.width/FS:.0f} ms), "
          f"stride {args.stride} ({1000*args.stride/FS:.0f} ms lag quantum)", flush=True)

    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 128):
            j = perm[i:i + 128]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(Xt[j].to(DEVICE)), yt[j].to(DEVICE))
            loss.backward(); opt.step()
            tot += float(loss) * len(j)
        sched.step()
        if (ep + 1) % 6 == 0:
            print(f"  epoch {ep+1:2d}  train mse {tot/len(Xt):.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        pred = np.concatenate([net(torch.tensor(Xte[i:i + 256]).to(DEVICE)).cpu().numpy()
                               for i in range(0, len(Xte), 256)]) * sd + mu
    mae = float(np.abs(pred[:, 1] - yte[:, 1]).mean())
    lags = np.concatenate([net.attention_lag_ms(torch.tensor(Xte[i:i + 256]).to(DEVICE))
                           for i in range(0, len(Xte), 256)])
    pat = mechlib.compute_ptt(Xte, FS) * 1000.0
    ok = np.isfinite(pat) & np.isfinite(lags)

    from scipy import stats
    r_all = float(stats.spearmanr(lags[ok], pat[ok]).statistic) if ok.sum() > 100 else np.nan
    rs = [stats.spearmanr(lags[m], pat[m]).statistic
          for s in np.unique(gsel)
          for m in [ok & (gsel == s)]
          if m.sum() >= 30 and np.std(lags[m]) > 1e-9 and np.std(pat[m]) > 1e-9]
    r_within = float(np.nanmedian(rs)) if rs else np.nan

    def pf(Xr):
        with torch.no_grad():
            return np.concatenate([net(torch.tensor(Xr[i:i + 256]).to(DEVICE)).cpu().numpy()
                                   for i in range(0, len(Xr), 256)])[:, 1] * sd[1] + mu[1]

    slope, frac = audit(pf, Xte, gsel)
    print(f"\n[acc]   DBP MAE {mae:.2f} mmHg")
    print(f"[lag]   attention lag median {np.median(lags[ok]):+.1f} ms, sd {np.std(lags[ok]):.1f}")
    print(f"[ref]   measured PAT median {np.median(pat[ok]):.1f} ms  (= PEP + PTT, not PTT)")
    print(f"[lag]   vs PAT: pooled r {r_all:+.3f}, within-subject r {r_within:+.3f} "
          f"({len(rs)} subjects)")
    print(f"[audit] roll slope {slope:+.4f}, faithful {frac:.0%}")
    print(f"\nLag quantum is now {1000*args.stride/FS:.0f} ms against a {np.median(pat[ok]):.0f} ms "
          f"reference, so resolution no longer limits the read-out.")

    res = {"n_params": sum(p.numel() for p in net.parameters()), "n_tokens": net.n_tok,
           "width_ms": 1000 * args.width / FS, "stride_ms": 1000 * args.stride / FS,
           "dbp_mae": mae, "attn_lag_median_ms": float(np.median(lags[ok])),
           "pat_median_ms": float(np.median(pat[ok])),
           "r_lag_pat_pooled": r_all, "r_lag_pat_within": r_within,
           "audit_slope": slope, "frac_faithful": frac}
    (DATA / "xattn_fine.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/xattn_fine.json")


if __name__ == "__main__":
    main()
