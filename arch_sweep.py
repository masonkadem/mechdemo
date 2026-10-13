"""arch_sweep.py -- does architecture change whether a model follows the arrival-time law?

Same data and recipe as the 12k-segment runs in real_audit_controls.py (vitaldb_mini_deep.npz,
Adam 2e-3, StepLR 10/0.5, 30 epochs, best-val epoch kept, seed 0), audited on the same 144
held-out patients. Variants, all width 64 with 200 ms patches:

  depth1 / depth2   the WaveTransformer with 1 or 2 layers (the audited model has 3)
  xattn             ECG and PPG embedded separately; 2 blocks in which PPG tokens attend to ECG
                    tokens (then a feed-forward), so the model is built to relate the two channels
                    in time -- the relation PAT is made of
  dual              dual stream: one bidirectional fusion block, then one shared encoder layer
                    (../BP's BPDualStreamTransformer design, with 200 ms patches)

Writes data/arch_sweep.json.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mechlib
from mechlib import ECG, PPG
from real_audit_controls import DEV, predict, shifted_inputs, subject_slopes

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


class CrossAttn(nn.Module):
    """PPG tokens query ECG tokens. Separate 200 ms patch embeddings per channel, learned
    positions, `depth` blocks of [cross-attention + feed-forward] (post-norm, as in the
    WaveTransformer), mean-pool both streams, linear head."""
    def __init__(self, dm=64, patch=25, heads=4, depth=2, L=1250):
        super().__init__()
        n = L // patch
        self.emb_e = nn.Conv1d(1, dm, patch, patch); self.emb_p = nn.Conv1d(1, dm, patch, patch)
        self.pos_e = nn.Parameter(torch.randn(1, n, dm) * .02)
        self.pos_p = nn.Parameter(torch.randn(1, n, dm) * .02)
        self.att = nn.ModuleList([nn.MultiheadAttention(dm, heads, dropout=.1, batch_first=True)
                                  for _ in range(depth)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(dm, 2 * dm), nn.ReLU(), nn.Dropout(.1),
                                               nn.Linear(2 * dm, dm)) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dm) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(dm) for _ in range(depth)])
        self.head = nn.Linear(2 * dm, 1)

    def forward(self, x):
        e = self.emb_e(x[:, :, 0:1].transpose(1, 2)).transpose(1, 2) + self.pos_e
        p = self.emb_p(x[:, :, 1:2].transpose(1, 2)).transpose(1, 2) + self.pos_p
        for att, ff, n1, n2 in zip(self.att, self.ff, self.n1, self.n2):
            p = n1(p + att(p, e, e, need_weights=False)[0])
            p = n2(p + ff(p))
        return self.head(torch.cat([p.mean(1), e.mean(1)], 1)).squeeze(-1)


class DualStream(nn.Module):
    """Dual stream, as in ../BP/train.py BPDualStreamTransformer but with 200 ms patches:
    ECG and PPG embedded separately, one BIDIRECTIONAL fusion block (each stream attends to the
    other, then its own feed-forward), streams averaged, one shared transformer encoder layer,
    mean-pool, linear head."""
    def __init__(self, dm=64, patch=25, heads=4, L=1250):
        super().__init__()
        n = L // patch
        self.emb = nn.ModuleList([nn.Conv1d(1, dm, patch, patch) for _ in range(2)])
        self.pos = nn.ParameterList([nn.Parameter(torch.randn(1, n, dm) * .02) for _ in range(2)])
        self.att = nn.ModuleList([nn.MultiheadAttention(dm, heads, dropout=.1, batch_first=True)
                                  for _ in range(2)])
        self.n1 = nn.ModuleList([nn.LayerNorm(dm) for _ in range(2)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 2 * dm), nn.ReLU(),
                                               nn.Dropout(.1), nn.Linear(2 * dm, dm)) for _ in range(2)])
        self.enc = nn.TransformerEncoderLayer(dm, heads, 2 * dm, batch_first=True)
        self.head = nn.Linear(dm, 1)

    def forward(self, x):
        s = [self.emb[c](x[:, :, c:c + 1].transpose(1, 2)).transpose(1, 2) + self.pos[c] for c in (0, 1)]
        out = []
        for i in (0, 1):
            a = self.n1[i](s[i] + self.att[i](s[i], s[1 - i], s[1 - i], need_weights=False)[0])
            out.append(a + self.ff[i](a))
        return self.head(self.enc((out[0] + out[1]) / 2).mean(1)).squeeze(-1)


def build(name):
    if name.startswith("depth"):
        return mechlib.WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=int(name[-1]), L=1250)
    if name == "dual":
        return DualStream()
    return CrossAttn()


def train(net, Xtr, ytr, Xva, yva, seed=0, epochs=30, bs=128, lr=2e-3):
    """The walkthrough notebook's recipe (as in real_audit_controls.train), for any model."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    best, state = float("inf"), None
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        vm = float(np.abs(predict(net, Xva) - yva).mean())
        if vm < best:
            best, state = vm, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return net


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="depth1,depth2,xattn,dual")
    names = ap.parse_args().models.split(",")
    out = DATA / "arch_sweep.json"
    res = json.loads(out.read_text()) if out.exists() else {}          # add to earlier results
    deep = mechlib.load_mini(str(DATA / "vitaldb_mini_deep.npz")); fs = int(deep["fs"])
    two = [ECG, PPG]
    Xtr, Xva = mechlib.normalize(deep["Xtr"][:, :, two]), mechlib.normalize(deep["Xva"][:, :, two])
    mini = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))
    Xte, gte = mechlib.normalize(mini["Xte"][:, :, two]), mini["gte"]
    _, rows = shifted_inputs(Xte, fs)
    for target, ti in (("dbp", 1), ("sbp", 0)):
        ytr, yva, yte = deep["ytr"][:, ti], deep["yva"][:, ti], mini["yte"][:, ti]
        base = float(np.abs(yte - ytr.mean()).mean())
        print(f"== {target.upper()}  predict-the-mean MAE {base:.2f}", flush=True)
        for name in names:
            net = train(build(name), Xtr, ytr, Xva, yva)
            n_par = sum(p.numel() for p in net.parameters())
            mae = float(np.abs(predict(net, Xte) - yte).mean())
            r = {m: subject_slopes([predict(net, rw["X"]) for rw in rows[m]], rows[m], gte)
                 for m in ("ppg", "ecg", "both")}
            res[f"{target}_{name}"] = {"params": n_par, "test_mae": mae, "baseline_mae": base,
                                       **{m: {k: v for k, v in r[m].items() if k != "slopes"} for m in r}}
            print(f"  {name:7s} {n_par/1e3:5.0f}k params  MAE {mae:.2f} | expected sign "
                  f"{r['ppg']['frac_negative']:.0%} (p={r['ppg']['p_sign']:.1g}), PPG {r['ppg']['median']:+.4f}"
                  f" ECG {r['ecg']['median']:+.4f} both {r['both']['median']:+.4f}", flush=True)
    out.write_text(json.dumps(res, indent=1))
    print("[done] data/arch_sweep.json")


if __name__ == "__main__":
    main()
