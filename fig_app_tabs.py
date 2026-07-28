"""fig_app_tabs.py -- publication recreation of the mechdemo app's first two tabs as one
2-row x 4-col figure: top row = synthetic sandbox (a-d), bottom row = real VitalDB waveforms
(a-d). Standalone (does not import the Streamlit app); replicates its model + audits.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
NAVY, RED, GREY, GREEN, LIGHT = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a", "#8a9bbf"

BP_MEAN, BP_STD, CONF_NOISE = 120.0, 18.0, 0.35
SHIFT_MS = np.array([-30, -20, -10, 0, 10, 20, 30])
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def ptt_from_bp(bp, p):
    return (140.0 / bp) ** (1.0 / p) * 0.2                  # arrival-time falls as BP rises


def sample(n, seed, p):
    rng = np.random.default_rng(seed)
    bp = rng.uniform(90, 150, n)
    ptt = ptt_from_bp(bp, p) + rng.normal(0, 0.006, n)
    conf = (bp - BP_MEAN) / BP_STD + rng.normal(0, CONF_NOISE, n)
    X = np.stack([ptt, conf], 1)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor((bp - BP_MEAN) / BP_STD, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        path = lambda: nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.physics, self.shortcut = path(), path()
        self.head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.alpha = alpha

    def code(self, X):
        return self.alpha * self.physics(X[:, 0:1]) + (1 - self.alpha) * self.shortcut(X[:, 1:2])

    def forward(self, X):
        return self.head(self.code(X)).squeeze(1)


def roll_curve(net, Xe):
    out = []
    for dm in SHIFT_MS:
        Xs = Xe.clone(); Xs[:, 0] = Xe[:, 0] + dm / 1000.0
        out.append(float((net(Xs).detach().numpy() * BP_STD + BP_MEAN).mean()))
    return np.array(out)


def train_one(alpha, p, epochs=400):
    tr_X, tr_b = sample(3000, 0, p); va_X, va_b = sample(1000, 1, p)
    torch.manual_seed(0); net = Net(alpha); opt = torch.optim.Adam(net.parameters(), 3e-3)
    th, vh = [], []
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(tr_X) - tr_b) ** 2).mean(); loss.backward(); opt.step()
        th.append(loss.item())
        with torch.no_grad():
            vh.append(float(((net(va_X) - va_b) ** 2).mean()))
    net.eval()
    return net, th, vh


def acc_lin_roll(net, Xe, ye):
    acc = r2_score(ye.numpy(), net(Xe).detach().numpy())
    a = net.code(Xe).detach().numpy(); t = Xe[:, 0].numpy(); h = len(t) // 2
    lin = r2_score(t[h:], Ridge().fit(a[:h], t[:h]).predict(a[h:]))
    roll = float(np.polyfit(SHIFT_MS, roll_curve(net, Xe), 1)[0])
    return acc, lin, roll


def main():
    p = 2.0; alpha = 1.0
    net, th, vh = train_one(alpha, p)
    Xe, ye = sample(1500, 7, p)
    curve = roll_curve(net, Xe)
    slope = float(np.polyfit(SHIFT_MS, curve, 1)[0])
    scores = {a: acc_lin_roll(train_one(a, p)[0], *sample(1500, 7, p)) for a in ALPHAS}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))

    # ===== TOP ROW: synthetic =====
    def plabel(ax, s):
        ax.text(-0.22, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    bp = np.random.default_rng(0).uniform(90, 150, 300)
    ax.scatter(bp, ptt_from_bp(bp, p) + np.random.default_rng(1).normal(0, 0.006, 300),
               s=8, alpha=.4, color=NAVY, edgecolor="none")
    ax.set_xlabel("BP (mmHg)"); ax.set_ylabel("PTT (s)"); plabel(ax, "a")

    ax = axes[0, 1]
    ax.plot(th, color=NAVY, lw=1.2, label="train"); ax.plot(vh, color=RED, lw=1.2, ls="--", label="val")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "b")

    ax = axes[0, 2]
    ax.axvline(0, color="#ddd", lw=.8, zorder=0)
    ax.plot(SHIFT_MS, curve, "-o", ms=4, color=NAVY)
    ax.set_xlabel("arrival-time shift (ms)"); ax.set_ylabel("pred. BP (mmHg)"); plabel(ax, "c")

    ax = axes[0, 3]
    rolls = np.array([-scores[a][2] for a in ALPHAS]); use = rolls / max(rolls.max(), 1e-9)
    ax.plot(ALPHAS, [scores[a][0] for a in ALPHAS], "-o", ms=4, color=GREY, label="accuracy")
    ax.plot(ALPHAS, [scores[a][1] for a in ALPHAS], "-s", ms=4, color=LIGHT, label="linear probe")
    ax.plot(ALPHAS, use, "-o", ms=4, color=NAVY, label="roll audit")
    ax.set_xlabel(r"$\alpha$ (arrival-time weight)"); ax.set_ylabel("normalized")
    ax.legend(fontsize=7.5, frameon=False); plabel(ax, "d")

    # ===== BOTTOM ROW: real VitalDB =====
    d = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    yte = d["yte"]; fs = int(d["fs"]); t_axis = np.arange(Xte.shape[1]) / fs
    scalars = mechlib.compute_scalars(Xte, fs, mechlib.ECG, mechlib.PPG)

    ax = axes[1, 0]
    ax.plot(t_axis, Xte[0, :, 0], color=NAVY, lw=1, label="ECG")
    ax.plot(t_axis, Xte[0, :, 1], color=RED, lw=1, label="PPG")
    ax.set_xlabel("time (s)"); ax.set_ylabel("z-scored")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "e")

    ckpt = torch.load(str(DATA / "dbp_transformer.pt"), map_location="cpu", weights_only=False)
    cfg, hist, sd = ckpt["config"], ckpt["history"], ckpt["state_dict"]
    tnet = mechlib.WaveTransformer(**cfg); tnet.load_state_dict(sd); tnet.eval()

    ax = axes[1, 1]
    ax.plot(hist["train_mae"], color=NAVY, lw=1.3, label="train MAE")
    ax.plot(hist["val_mae"], color=RED, lw=1.3, ls="--", label="val MAE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE (mmHg)")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "f")

    # layer features
    @torch.no_grad()
    def layers(X, bs=512):
        outs = None
        for s in range(0, len(X), bs):
            _, acts = tnet(torch.tensor(X[s:s+bs], dtype=torch.float32), return_acts=True)
            pooled = [a.mean(1).numpy() for a in acts]
            outs = pooled if outs is None else [np.concatenate([o, q]) for o, q in zip(outs, pooled)]
        return outs
    stages = layers(Xte)
    xs = ["embed"] + [f"L{i+1}" for i in range(cfg["depth"])]

    ax = axes[1, 2]
    r2_pat = [mechlib.linear_probe(f, scalars["pat"]) for f in stages]
    r2_per = [mechlib.linear_probe(f, scalars["period"]) for f in stages]
    ax.plot(range(len(xs)), r2_pat, "-o", ms=4, color=NAVY, lw=1.3, label="PAT (arrival time)")
    ax.plot(range(len(xs)), r2_per, "-o", ms=4, color=RED, lw=1.3, label="cardiac period")
    ax.axhline(0, color="#bbb", lw=.8); ax.set_xticks(range(len(xs)), xs, fontsize=8, rotation=15)
    ax.set_ylabel("probe $R^2$"); ax.set_xlabel("layer")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "g")

    @torch.no_grad()
    def predict_fn(Xd):
        return tnet(torch.tensor(Xd, dtype=torch.float32)).numpy()
    shift_ms, rcurve, rslope = mechlib.input_shift_audit(predict_fn, Xte, fs)
    ax = axes[1, 3]
    ax.axvline(0, color=GREY, lw=.8, ls=":")
    ax.plot(shift_ms, rcurve, "-o", ms=4, color=NAVY)
    ax.set_xlabel("imposed PPG shift (ms)"); ax.set_ylabel("pred. DBP (mmHg)"); plabel(ax, "h")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.42, wspace=0.42)
    fig.savefig(ROOT / "figures" / "fig_app_tabs.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_app_tabs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_app_tabs.png / .pdf")


if __name__ == "__main__":
    main()
