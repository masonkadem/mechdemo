"""mae_probe.py -- does self-supervision represent arrival time when supervision does not?

The question
------------
Every model audited so far was trained supervised on BP labels, so the optimiser had no reason to
represent physiology -- it only had to predict a number, and heart rate plus morphology predicts
that number well. A masked autoencoder is trained to RECONSTRUCT the waveform instead, which
cannot be done without representing the timing structure of both channels.

So: pretrain an MAE on raw ECG+PPG with no labels, freeze it, and probe the representation for
arterial arrival time measured from the ABP channel. Three outcomes, all informative:

  MAE probes HIGHER than supervised   the training objective discards recoverable physiology,
                                      which indicts the objective rather than the architecture
  MAE probes the SAME                 the information is equally present either way and simply
                                      is not used, consistent with the probe battery already run
  MAE probes LOWER                    reconstruction does not require arrival time either, and
                                      the signal is weaker than assumed

The probe target is the ABP-derived arrival time (Q-onset to arterial foot), not a PPG estimate,
so the ceiling is the real physiological quantity rather than an optical proxy. Both models are
probed with the same ridge regression on frozen features and scored WITHIN subject, since pooled
scores are dominated by between-subject differences that carry no timing information.

Read the result against what the quantity is worth: perfectly measured arrival time buys
0.23 mmHg [0.13, 0.32] over predicting a subject's own mean. A representation difference here is
a statement about what models encode, not about achievable accuracy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge

import mechlib
import pat_groundtruth as G
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FS, L = 125, 1250
PATCH = 25                      # 200 ms patches -> 50 tokens


class MAE(nn.Module):
    """Masked autoencoder over ECG+PPG patches.

    Patch embedding is linear, and the encoder is a plain transformer, so nothing but attention
    relates the two channels -- the same choice made for the cross-attention model, and for the
    same reason: a convolutional trunk would mix the channels before the representation forms,
    and any timing found afterwards could not be attributed to the representation.
    """

    def __init__(self, dm=128, depth=4, heads=4, dec_depth=2):
        super().__init__()
        self.n_tok = L // PATCH
        self.embed = nn.Linear(PATCH * 2, dm)          # both channels per patch
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, heads, dm * 4, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, depth)
        dec = nn.TransformerEncoderLayer(dm, heads, dm * 4, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.dec = nn.TransformerEncoder(dec, dec_depth)
        self.head = nn.Linear(dm, PATCH * 2)
        self.dm = dm

    def tokens(self, x):
        b = x.shape[0]
        p = x.reshape(b, self.n_tok, PATCH, 2).reshape(b, self.n_tok, PATCH * 2)
        return self.embed(p) + self.pos, p

    def forward(self, x, mask_ratio=0.6):
        z, target = self.tokens(x)
        b, n, _ = z.shape
        keep = int(n * (1 - mask_ratio))
        idx = torch.rand(b, n, device=x.device).argsort(1)
        vis, hid = idx[:, :keep], idx[:, keep:]
        zv = torch.gather(z, 1, vis[..., None].expand(-1, -1, self.dm))
        h = self.enc(zv)
        full = self.mask_tok.expand(b, n, self.dm).clone()
        full = full.scatter(1, vis[..., None].expand(-1, -1, self.dm), h)
        full = full + self.pos
        rec = self.head(self.dec(full))
        loss = nn.functional.mse_loss(
            torch.gather(rec, 1, hid[..., None].expand(-1, -1, PATCH * 2)),
            torch.gather(target, 1, hid[..., None].expand(-1, -1, PATCH * 2)))
        return loss

    @torch.no_grad()
    def represent(self, x):
        """Frozen features: mean-pooled encoder output over all (unmasked) tokens."""
        z, _ = self.tokens(x)
        return self.enc(z).mean(1)


class Supervised(nn.Module):
    """Same trunk, trained on BP labels. The control that isolates the OBJECTIVE."""

    def __init__(self, dm=128, depth=4, heads=4):
        super().__init__()
        self.n_tok = L // PATCH
        self.embed = nn.Linear(PATCH * 2, dm)
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, heads, dm * 4, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, depth)
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 64), nn.GELU(),
                                  nn.Linear(64, 2))
        self.dm = dm

    def _tok(self, x):
        b = x.shape[0]
        p = x.reshape(b, self.n_tok, PATCH, 2).reshape(b, self.n_tok, PATCH * 2)
        return self.embed(p) + self.pos

    def forward(self, x):
        return self.head(self.enc(self._tok(x)).mean(1))

    @torch.no_grad()
    def represent(self, x):
        return self.enc(self._tok(x)).mean(1)


def features(model, X, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model.represent(torch.tensor(X[i:i + bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


def probe_within(F, y, g, min_seg=25, alpha=10.0, seed=0):
    """Ridge on frozen features, fit on TRAIN subjects and scored on HELD-OUT subjects.

    The subject split is not optional. Fitting and scoring on the same rows lets a 128-dimensional
    ridge memorise them: measured on pure Gaussian noise with no signal at all, that version
    returned r = 0.183 -- identical to the best hand-crafted estimator this experiment is meant to
    be compared against, so any result would have been indistinguishable from its own bias.

    Scoring is still WITHIN subject, because a representation can encode who someone is and score
    well pooled while carrying nothing about that person's arrival time changing.
    """
    ok = np.isfinite(y)
    if ok.sum() < 500:
        return np.nan, 0
    subs = np.unique(g)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subs))
    tr_s = set(subs[perm[:len(subs) // 2]].tolist())
    tr = np.array([s in tr_s for s in g]) & ok
    te = (~np.array([s in tr_s for s in g])) & ok
    if tr.sum() < 300 or te.sum() < 300:
        return np.nan, 0
    m = Ridge(alpha=alpha).fit(F[tr], y[tr])
    p = m.predict(F)
    rs = []
    for s in subs:
        k = te & (g == s)
        if k.sum() < min_seg or np.std(y[k]) < 1e-6 or np.std(p[k]) < 1e-9:
            continue
        rs.append(stats.spearmanr(p[k], y[k]).statistic)
    rs = [r for r in rs if np.isfinite(r)]
    return (float(np.median(rs)) if rs else np.nan), len(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-n", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sup-epochs", type=int, default=20)
    ap.add_argument("--probe-subjects", type=int, default=120)
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.pretrain_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.pretrain_n]
    print(f"[data] {len(Xtr)} pretraining segments, no labels used by the MAE", flush=True)

    # ---- probe set: ABP-derived arrival time as the target ---------------------
    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= 50][:args.probe_subjects]
    sel = np.concatenate([np.where(g == s)[0][:50] for s in subs])
    Xraw = d["Xte"][sel]
    Xp = mechlib.normalize(Xraw[:, :, [ECG, PPG]])
    gp = g[sel]
    print(f"[probe] {len(Xp)} segments, {len(subs)} subjects; computing arterial arrival "
          f"time ...", flush=True)
    pat = np.array([G.abp_pat(Xraw[i, :, ECG], Xraw[i, :, 2], FS)
                    for i in range(len(Xraw))]) * 1000.0
    dbp = d["yte"][sel][:, 1]
    print(f"[probe] arrival time measurable on {100*np.isfinite(pat).mean():.0f}%", flush=True)

    res = {}

    # ---- 1. masked autoencoder, no labels -------------------------------------
    torch.manual_seed(0); np.random.seed(0)
    mae = MAE().to(DEVICE)
    opt = torch.optim.AdamW(mae.parameters(), 1.5e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt = torch.tensor(Xtr)
    print(f"\n[mae] {sum(p.numel() for p in mae.parameters()):,} parameters", flush=True)
    for ep in range(args.epochs):
        mae.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            opt.zero_grad()
            loss = mae(Xt[j].to(DEVICE))
            loss.backward(); opt.step()
            tot += float(loss) * len(j)
        sch.step()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}  recon mse {tot/len(Xt):.5f}", flush=True)

    # ---- 2. supervised control, same trunk ------------------------------------
    torch.manual_seed(0); np.random.seed(0)
    sup = Supervised().to(DEVICE)
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    opt2 = torch.optim.AdamW(sup.parameters(), 1e-3, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, args.sup_epochs)
    print(f"\n[sup] {sum(p.numel() for p in sup.parameters()):,} parameters", flush=True)
    for ep in range(args.sup_epochs):
        sup.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 256):
            j = perm[i:i + 256]
            opt2.zero_grad()
            loss = nn.functional.mse_loss(sup(Xt[j].to(DEVICE)), yt[j].to(DEVICE))
            loss.backward(); opt2.step()
            tot += float(loss) * len(j)
        sch2.step()
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}  mse {tot/len(Xt):.4f}", flush=True)

    # ---- 3. probe both frozen representations ---------------------------------
    print(f"\n{'model':14s} {'PAT probe r':>12s} {'DBP probe r':>12s} {'subj':>6s}")
    print("-" * 48)
    for name, model in (("MAE (no labels)", mae), ("supervised", sup)):
        F = features(model, Xp)
        r_pat, n1 = probe_within(F, pat, gp)
        r_dbp, _ = probe_within(F, dbp, gp)
        res[name] = {"r_pat_within": r_pat, "r_dbp_within": r_dbp, "n_subj": n1,
                     "dim": int(F.shape[1])}
        print(f"{name:14s} {r_pat:+12.3f} {r_dbp:+12.3f} {n1:6d}", flush=True)

    # reference: the best hand-crafted estimator against the same ground truth
    res["_reference"] = {"best_ecgppg_estimator_r": 0.183,
                         "ceiling_gain_mmHg": 0.23, "ceiling_ci": [0.13, 0.32]}
    print(f"\nbest hand-crafted ECG-PPG estimator against the same target: r = 0.183")
    print("perfectly measured arrival time is worth 0.23 mmHg [0.13, 0.32], so a difference")
    print("here is about what the representation ENCODES, not about achievable accuracy.")

    (DATA / "mae_probe.json").write_text(json.dumps(res, indent=2, default=float))
    torch.save({"mae": mae.state_dict(), "sup": sup.state_dict()},
               ROOT / "models" / "mae_probe.pt")
    print(f"\n[done] data/mae_probe.json")


if __name__ == "__main__":
    main()
