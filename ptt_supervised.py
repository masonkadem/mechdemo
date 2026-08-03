"""ptt_supervised.py -- can a model be MADE faithful without losing accuracy?

The setup
---------
Everything measured so far is diagnostic: no trained model responds to the arrival-time
intervention (all CIs span zero), and the morphology features the models actually use are not
proxies for transit time (within-subject |r| < 0.07 while retaining partial correlation with BP).

But the same analysis showed the information IS there: measured PTT is predictable from waveform
morphology at R^2 = 0.927. So unfaithfulness is a choice the optimiser makes, not a limit of the
data. This tests whether the choice can be reversed.

A CNN gets two heads on a shared trunk: BP, and an auxiliary head predicting measured PTT. The
auxiliary loss forces transit-time information into the shared representation, so BP must be read
off a code that encodes arrival time. Sweeping the auxiliary weight lambda gives the trade-off
curve:

  lambda = 0     the usual model, for reference
  lambda > 0     BP accuracy versus audit faithfulness

Three outcomes, all informative:
  * faithful at no accuracy cost  -> a constructive fix, and the strongest possible result
  * faithful but less accurate    -> quantifies the price of respecting the governing law
  * accurate but never faithful   -> the shortcut is not merely convenient but preferred, which
                                     strengthens the instrumentation argument

Segments where PTT is unmeasurable (~56%) are MASKED out of the auxiliary loss rather than
imputed. Imputing a constant there would teach the head to predict that constant, which is
exactly the bug that silently flattened the first version of the audit.
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
DELTAS = (-6, -4, -2, 0)          # negative arm, matching the validated audit
SLIP_MS = 150.0
FS = 125


class DualHead(nn.Module):
    """Shared trunk, one head for BP and one for PTT."""

    def __init__(self, n_ch=2, n_out=2):
        super().__init__()
        def blk(i, o, k=7, s=2):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.trunk = nn.Sequential(blk(n_ch, 32), blk(32, 64), blk(64, 96), blk(96, 128),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.bp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))
        self.ptt = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.trunk(x.transpose(1, 2))
        return self.bp(z), self.ptt(z).squeeze(1)


def audit(pred_fn, X, g, fs=FS):
    """The validated audit: negative-arm roll, non-finite PTT dropped, beat-slips discarded.
    Faithful => NEGATIVE slope (a negative shift shortens PTT, and shorter PTT means higher BP)."""
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
    sl = []
    for s in np.unique(g):
        m = (g == s) & keep
        if m.sum() >= 5:
            sl.append(np.median([np.polyfit(nom, P[i], 1)[0] for i in np.where(m)[0]]))
    sl = np.array(sl)
    return (float(np.median(sl)) if len(sl) else np.nan,
            float(np.mean(sl < 0)) if len(sl) else np.nan, len(sl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-n", type=int, default=60000)
    ap.add_argument("--lambdas", default="0,0.1,0.5,2.0")
    args = ap.parse_args()

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.train_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.train_n]
    print("[data] measuring PTT on the training set ...", flush=True)
    ptt_tr = mechlib.compute_ptt(Xtr, FS) * 1000.0
    mask_tr = np.isfinite(ptt_tr)
    print(f"[data] PTT measurable on {100*mask_tr.mean():.0f}% of training segments; the rest "
          f"are MASKED out of the auxiliary loss, not imputed", flush=True)

    gte = d["gte"]
    subs = [s for s in np.unique(gte) if (gte == s).sum() >= 100]
    sel = np.concatenate([np.where(gte == s)[0][:100] for s in subs[:80]])
    Xte = mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]])
    yte, gsel = d["yte"][sel], gte[sel]

    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    pm, ps = float(np.nanmean(ptt_tr[mask_tr])), float(np.nanstd(ptt_tr[mask_tr]) + 1e-9)
    Xt = torch.tensor(Xtr)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    pt = torch.tensor(np.where(mask_tr, (ptt_tr - pm) / ps, 0.0), dtype=torch.float32)
    mt = torch.tensor(mask_tr.astype(np.float32))

    res = {}
    print(f"\n{'lambda':>7s} {'DBP MAE':>9s} {'PTT R2':>8s} {'audit slope':>12s} "
          f"{'frac faithful':>14s}")
    print("-" * 56)
    for lam in [float(x) for x in args.lambdas.split(",")]:
        torch.manual_seed(0); np.random.seed(0)
        net = DualHead().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), 2e-3)
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                j = perm[i:i + 256]
                xb = Xt[j].to(DEVICE)
                opt.zero_grad()
                bp, pp = net(xb)
                loss = nn.functional.mse_loss(bp, yt[j].to(DEVICE))
                if lam > 0:
                    mj = mt[j].to(DEVICE)
                    if mj.sum() > 0:
                        # masked auxiliary loss: only segments with a real PTT contribute
                        aux = (((pp - pt[j].to(DEVICE)) ** 2) * mj).sum() / mj.sum()
                        loss = loss + lam * aux
                loss.backward(); opt.step()

        net.eval()
        with torch.no_grad():
            preds, ptts = [], []
            for i in range(0, len(Xte), 512):
                b, p = net(torch.tensor(Xte[i:i + 512]).to(DEVICE))
                preds.append(b.cpu().numpy()); ptts.append(p.cpu().numpy())
        pred = np.concatenate(preds) * sd + mu
        pptt = np.concatenate(ptts) * ps + pm
        mae = float(np.abs(pred[:, 1] - yte[:, 1]).mean())

        true_ptt = mechlib.compute_ptt(Xte, FS) * 1000.0
        ok = np.isfinite(true_ptt)
        r2 = (1 - np.sum((pptt[ok] - true_ptt[ok]) ** 2)
              / np.sum((true_ptt[ok] - true_ptt[ok].mean()) ** 2)) if ok.sum() > 100 else np.nan

        def pf(Xr):
            with torch.no_grad():
                out = []
                for i in range(0, len(Xr), 512):
                    b, _ = net(torch.tensor(Xr[i:i + 512]).to(DEVICE))
                    out.append(b.cpu().numpy())
            return np.concatenate(out)[:, 1] * sd[1] + mu[1]

        slope, frac, nsub = audit(pf, Xte, gsel)
        res[str(lam)] = {"dbp_mae": mae, "ptt_r2": float(r2), "audit_slope": slope,
                         "frac_faithful": frac, "n_subj": nsub}
        print(f"{lam:7.2f} {mae:9.2f} {r2:8.3f} {slope:+12.4f} {frac:14.0%}", flush=True)
        (DATA / "ptt_supervised.json").write_text(json.dumps(res, indent=2, default=float))

    base = res.get("0.0") or res.get("0")
    if base:
        print(f"\nAgainst the unsupervised baseline (MAE {base['dbp_mae']:.2f}, "
              f"slope {base['audit_slope']:+.4f}):")
        for lam, r in res.items():
            if float(lam) == 0:
                continue
            print(f"  lambda={lam:>4s}  MAE {r['dbp_mae']-base['dbp_mae']:+.2f}  "
                  f"slope {r['audit_slope']-base['audit_slope']:+.4f}  "
                  f"faithful {r['frac_faithful']:.0%}")
        print("\nA negative slope with a faithful fraction above 50% means the auxiliary loss")
        print("made the model respect the arrival-time law. Read the MAE column alongside it:")
        print("that is the price, if any, of the mechanism.")
    print(f"\n[done] data/ptt_supervised.json")


if __name__ == "__main__":
    main()
