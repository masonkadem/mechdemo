"""fig_audit.py -- thesis figure: the arrival-time audit, validated in simulation (top) and applied
to transformers trained on VitalDB (bottom).

Top row (ground truth known):   a accuracy vs audit in a toy model, b simulated ECG/PPG,
                                c PAT detector accuracy, d audit recovers the injected slope.
Bottom row (no ground truth):   e VitalDB ECG/PPG, f accuracy vs arrival-time use across training
                                runs, g linear probes, h response by shift type.

Reads data/synth_waveform_audit.json and data/real_audit_controls.json; reruns the toy model
(panel a) and the probes (panel g).
"""
import json
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
import synth_waveform_audit as SW

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


# ----------------------------------------------------------------- toy model (panel a)
# Scalar sandbox: BP is computed from PTT (weight alpha) or from a confound (1 - alpha), and the
# network is given both. Accuracy cannot tell the two routes apart; the shift audit can.
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


def stacked(ax_e, ax_p, ecg, ppg, fs, t0, t1):
    """ECG above, PPG below on a shared time axis (each min-max scaled over the window).
    Filled R peaks are paired with a PPG foot; open ones had no agreeing foot and are dropped.
    A grey line joins each R peak to its own foot, so the pairing is visible even when the
    delay is longer than one beat (VitalDB's PPG offset)."""
    from matplotlib.patches import ConnectionPatch
    sl = slice(int(t0 * fs), int(t1 * fs)); tt = np.arange(len(ecg))[sl] / fs
    sc = lambda v: (v[sl] - v[sl].min()) / (np.ptp(v[sl]) + 1e-9)
    e, q = sc(ecg), sc(ppg)
    ax_e.plot(tt, e, color=NAVY, lw=1.0)
    ax_p.plot(tt, q, color=RED, lw=1.3)
    fid = mechlib.segment_fiducials(ecg, ppg, fs)
    paired = set(fid["r_paired"].tolist())
    for rpk in fid["r_peaks"]:
        if sl.start <= rpk < sl.stop and rpk not in paired:
            ax_e.plot(rpk / fs, e[rpk - sl.start], "o", mfc="white", mec=NAVY, ms=4)
    # No delay value is printed: on VitalDB it includes the instrumental offset, so a number here
    # would read as an (impossible) physiological PAT. The connectors show the pairing instead.
    for rpk, ft in zip(fid["r_paired"], fid["feet"]):
        if not (sl.start <= rpk and ft < sl.stop):
            continue
        ax_e.plot(rpk / fs, e[rpk - sl.start], "o", color=NAVY, ms=4)
        ax_p.plot(ft / fs, q[ft - sl.start], "o", color="k", ms=4)
        ax_p.add_artist(ConnectionPatch((rpk / fs, e[rpk - sl.start]), (ft / fs, q[ft - sl.start]),
                                        "data", "data", axesA=ax_e, axesB=ax_p,
                                        color="#999", lw=.8, ls="--", zorder=0))
    for a_, lab in ((ax_e, "ECG"), (ax_p, "PPG")):
        a_.set_yticks([]); a_.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
    ax_e.set_ylim(-0.08, 1.12); ax_p.set_ylim(-0.1, 1.12)
    ax_e.tick_params(labelbottom=False); ax_e.spines["bottom"].set_visible(False)
    ax_e.tick_params(axis="x", length=0)
    ax_p.set_xlabel("time (s)")


def plabel(ax, s, title):
    ax.text(-0.2, 1.08, s, transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=9.5, loc="left", pad=6)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    swres = json.loads((DATA / "synth_waveform_audit.json").read_text())
    ctl = json.loads((DATA / "real_audit_controls.json").read_text())

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.45, left=0.06, right=0.99, top=0.9, bottom=0.09)
    sub_t = gs[0, 1].subgridspec(2, 1, hspace=0.12)
    sub_b = gs[1, 0].subgridspec(2, 1, hspace=0.12)
    top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(sub_t[0])] + [fig.add_subplot(gs[0, c]) for c in (2, 3)]
    ppg_t = fig.add_subplot(sub_t[1], sharex=top[1])
    bot = [fig.add_subplot(sub_b[0])] + [fig.add_subplot(gs[1, c]) for c in (1, 2, 3)]
    ppg_b = fig.add_subplot(sub_b[1], sharex=bot[0])
    for a_ in top + bot + [ppg_t, ppg_b]:
        a_.spines[["top", "right"]].set_visible(False)

    # ---------------- a: toy model -- accuracy does not see how BP is computed, the audit does
    p = 2.0
    scores = {a: acc_lin_roll(train_one(a, p)[0], *sample(1500, 7, p)) for a in ALPHAS}
    a_ = top[0]
    rolls = np.array([-scores[a][2] for a in ALPHAS]); use = rolls / max(rolls.max(), 1e-9)
    a_.plot(ALPHAS, [scores[a][0] for a in ALPHAS], "-o", ms=4, color=GREY, label="Accuracy ($R^2$)")
    a_.plot(ALPHAS, use, "-o", ms=4, color=NAVY, label="Audit (scaled)")
    a_.set_xlabel(r"Weight on PTT, $\alpha$"); a_.set_ylabel("Score"); a_.set_ylim(-0.05, 1.08)
    a_.legend(fontsize=8, frameon=False, loc="center left")
    plabel(a_, "a", "Accuracy vs. audit (toy model)")

    # VitalDB example for panel e, chosen first so panel b can be drawn at the same heart rate
    d = mechlib.load_mini(str(DATA / "vitaldb_mini.npz")); fs = int(d["fs"])
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    i_ex = 0
    for i_c in range(min(300, len(Xte))):
        fd = mechlib.segment_fiducials(Xte[i_c, :, 0], Xte[i_c, :, 1], fs)
        if np.sum(fd["feet"] < 3 * fs) >= 3:
            i_ex = i_c
            break
    hr_ex = 60.0 / (np.median(np.diff(mechlib.r_peaks(Xte[i_ex, :, 0], fs))) / fs)

    # ---------------- b: simulated ECG and PPG (beat shapes fitted to VitalDB, 400 ms PPG offset)
    seg = SW.make_segment(np.random.default_rng(3), hr=hr_ex, ptt_ms=210)
    stacked(top[1], ppg_t, seg[:, 0], seg[:, 1], SW.FS, 0.0, 3.0)
    ppg_t.set_xlabel("Time (s)")
    plabel(top[1], "b", "Simulated ECG and PPG")

    # ---------------- c: PAT detector vs ground truth
    a_ = top[2]
    Xc, _, pt, _ = SW.make_dataset(400, 1.0, np.random.default_rng(99))
    est = mechlib.compute_ptt(Xc, SW.FS) * 1000.0 - SW.PPG_OFFSET_MS
    ok = np.isfinite(est)
    a_.scatter(pt[ok], est[ok], s=5, alpha=.35, color=NAVY, edgecolors="none")
    lim = [min(pt[ok].min(), est[ok].min()), max(pt[ok].max(), est[ok].max())]
    a_.plot(lim, lim, color="k", lw=.9, ls="--")
    a_.text(.97, .05, f"r = {np.corrcoef(est[ok], pt[ok])[0, 1]:.3f}", transform=a_.transAxes,
            fontsize=8.5, ha="right")
    a_.set_xlabel("Injected PTT (ms)"); a_.set_ylabel("Measured PTT − offset (ms)")
    plabel(a_, "c", "PAT detector accuracy")

    # ---------------- d: audit recovers the injected slope
    a_ = top[3]
    alw = [0.0, 0.25, 0.5, 0.75, 1.0]
    a_.axhline(0, color="k", lw=.6, alpha=.4)
    a_.axhline(-0.22, color=RED, lw=1, ls="--")
    a_.text(.03, -0.215, "Injected", fontsize=8, color=RED, va="bottom", transform=a_.get_yaxis_transform())
    for a0 in alw:
        for sv in swres["alphas"][str(a0)]["slopes"]:
            a_.plot(a0, sv, ".", color=GREY, ms=4, zorder=1)
    a_.plot(alw, [swres["alphas"][str(a0)]["slope"] for a0 in alw], "-o", ms=5, color=NAVY, lw=1.3)
    a_.text(.97, .62, f"r = {swres['slope_vs_alpha_r']:.2f}".replace("-", "−"), transform=a_.transAxes,
            fontsize=8.5, ha="right")
    a_.set_xlabel(r"Weight on PTT, $\alpha$"); a_.set_ylabel("Audit slope (mmHg/ms)")
    plabel(a_, "d", "Audit recovers injected slope")

    # ---------------- e: VitalDB ECG and PPG
    stacked(bot[0], ppg_b, Xte[i_ex, :, 0], Xte[i_ex, :, 1], fs, 0.0, 3.0)
    ppg_b.set_xlabel("Time (s)")
    plabel(bot[0], "e", "VitalDB ECG and PPG")

    # ---------------- f: accuracy vs arrival-time use, every training run, against the data itself
    a_ = bot[1]
    base = ctl["baseline_mae"]
    dat = ctl["data"]["ppg_pat"]
    dlo, dhi = wilson(round(dat["frac_negative"] * dat["n_subjects"]), dat["n_subjects"])
    runs = [k for k in ctl["models"] if k.startswith(("orig", "seed"))]
    a_.axhspan(100 * dlo, 100 * dhi, color="#ddd", lw=0)
    a_.axhline(100 * dat["frac_negative"], color="k", lw=.9, ls="--")
    a_.axhline(50, color="#bbb", lw=.8)
    a_.axvline(base, color=GREY, lw=1, ls=":")
    for k in runs:
        r = ctl["models"][k]["ppg"]; n = r["n_subjects"]
        lo, hi = wilson(round(r["frac_negative"] * n), n)
        m = ctl["models"][k]["test_mae"]
        a_.plot([m, m], [100 * lo, 100 * hi], color=NAVY, lw=1.2, alpha=.6)
        a_.plot(m, 100 * r["frac_negative"], "o", color=NAVY, ms=5)
    maes = [ctl["models"][k]["test_mae"] for k in runs]
    x0 = min(maes) - 0.05
    a_.text(x0 + 0.01, 100 * dhi + 1.5, "Data", fontsize=8, va="bottom")
    a_.text(base - 0.01, 3, "Mean\npredictor", fontsize=8, color=GREY, ha="right", va="bottom")
    a_.set_xlim(x0, base + 0.04); a_.set_ylim(0, 100)
    a_.set_xlabel("Test DBP MAE (mmHg)"); a_.set_ylabel("Patients with expected sign (%)")
    plabel(a_, "f", "Accuracy vs. arrival-time use")

    # ---------------- g: linear probes of hidden layers
    ck = torch.load(str(DATA / "dbp_transformer.pt"), map_location="cpu", weights_only=False)
    tnet = mechlib.WaveTransformer(**ck["config"]); tnet.load_state_dict(ck["state_dict"]); tnet.eval()
    scalars = mechlib.compute_scalars(Xte, fs, mechlib.ECG, mechlib.PPG)
    with torch.no_grad():
        outs = None
        for s in range(0, len(Xte), 512):
            _, acts = tnet(torch.tensor(Xte[s:s + 512], dtype=torch.float32), return_acts=True)
            pooled = [q.mean(1).numpy() for q in acts]
            outs = pooled if outs is None else [np.concatenate([o, q]) for o, q in zip(outs, pooled)]
    xs = ["Embed"] + [f"L{i + 1}" for i in range(ck["config"]["depth"])]
    a_ = bot[2]
    a_.plot(range(len(xs)), [mechlib.linear_probe(f, scalars["period"]) for f in outs], "-o", ms=4,
            color=RED, lw=1.3, label="Cardiac period")
    a_.plot(range(len(xs)), [mechlib.linear_probe(f, scalars["pat"]) for f in outs], "-o", ms=4,
            color=NAVY, lw=1.3, label="PAT")
    a_.axhline(0, color="#bbb", lw=.8); a_.set_xticks(range(len(xs)), xs, fontsize=8)
    a_.set_ylabel("Probe $R^2$"); a_.set_xlabel("Layer"); a_.set_ylim(-0.08, 0.66)
    a_.legend(fontsize=8, frameon=False)
    plabel(a_, "g", "Linear probes of hidden layers")

    # ---------------- h: response by shift type
    a_ = bot[3]
    modes = [("ppg", "PPG\n−48 ms"), ("ppg_small", "PPG\n−16 ms"), ("ecg", "ECG\n+48 ms"),
             ("both", "Both\n−48 ms")]
    xs_ = np.arange(len(modes))
    a_.axhline(0, color="k", lw=.6, alpha=.4)
    for i, k in enumerate(runs):
        a_.plot(xs_, [ctl["models"][k][m]["median"] for m, _ in modes], "-o", ms=4, color=NAVY,
                alpha=.55, lw=1, label="Transformer runs" if i == 0 else None)
    a_.plot(xs_, [ctl["models"]["textbook"][m]["median"] for m, _ in modes], "--s", ms=4,
            color=GREEN, lw=1, label="Textbook control")
    a_.set_xticks(xs_, [lab for _, lab in modes], fontsize=8)
    a_.set_ylabel("Median slope (mmHg/ms)")
    a_.legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 0.12))
    plabel(a_, "h", "Response by shift type")

    for y, txt in ((0.965, "Simulation (ground truth known)"), (0.475, "VitalDB (no ground truth)")):
        fig.text(0.01, y, txt, fontsize=10.5, color="#333")
    fig.savefig(ROOT / "figures" / "fig_audit.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_audit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] figures/fig_audit.png / .pdf")


if __name__ == "__main__":
    main()
