"""fig_app_tabs.py -- publication recreation of the mechdemo app's first two tabs as one
2-row x 4-col figure: top row = synthetic sandbox (a-d), bottom row = real VitalDB waveforms
(a-d). Standalone (does not import the Streamlit app); replicates its model + audits.
"""
import os
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mechlib
import audit_subject as AUD
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


def main():
    p = 2.0; alpha = 1.0
    net, th, vh = train_one(alpha, p)
    Xe, ye = sample(1500, 7, p)
    curve = roll_curve(net, Xe)
    slope = float(np.polyfit(SHIFT_MS, curve, 1)[0])
    scores = {a: acc_lin_roll(train_one(a, p)[0], *sample(1500, 7, p)) for a in ALPHAS}

    # 20-column grid so rows can hold 4 or 5 panels of equal height
    fig = plt.figure(figsize=(17, 10.8))
    gs = fig.add_gridspec(3, 20)
    axes = np.full((3, 5), None, dtype=object)
    for c in range(4):
        axes[0, c] = fig.add_subplot(gs[0, 5 * c:5 * c + 5])
    ppg_ax = {}                                  # e and j: ECG on top, PPG below, shared time axis
    for r_ in (1, 2):
        sub = gs[r_, 0:4].subgridspec(2, 1, hspace=0.12)
        axes[r_, 0] = fig.add_subplot(sub[0])
        ppg_ax[r_] = fig.add_subplot(sub[1], sharex=axes[r_, 0])
        for c in range(1, 5):
            axes[r_, c] = fig.add_subplot(gs[r_, 4 * c:4 * c + 4])
    all_axes = [a for a in axes.ravel() if a is not None] + list(ppg_ax.values())

    # ===== TOP ROW: synthetic =====
    def plabel(ax, s, title=None):
        ax.text(-0.22, 1.06, s, transform=ax.transAxes, fontsize=13, fontweight="bold")
        if title:
            ax.set_title(title, fontsize=8.5, loc="left", pad=6)

    ax = axes[0, 0]
    bp = np.random.default_rng(0).uniform(90, 150, 300)
    ax.scatter(bp, ptt_from_bp(bp, p) + np.random.default_rng(1).normal(0, 0.006, 300),
               s=8, alpha=.4, color=NAVY, edgecolor="none")
    ax.set_xlabel("BP (mmHg)"); ax.set_ylabel("PTT (s)")
    plabel(ax, "a", "Injected law")

    ax = axes[0, 1]
    ax.plot(th, color=NAVY, lw=1.2, label="train"); ax.plot(vh, color=RED, lw=1.2, ls="--", label="val")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "b", "Training")

    ax = axes[0, 2]
    ax.axvline(0, color="#ddd", lw=.8, zorder=0)
    ax.plot(SHIFT_MS, curve, "-o", ms=4, color=NAVY)
    ax.set_xlabel("PTT shift (ms)"); ax.set_ylabel("pred. BP (mmHg)")
    plabel(ax, "c", r"Audit, $\alpha$ = 1")

    ax = axes[0, 3]
    rolls = np.array([-scores[a][2] for a in ALPHAS]); use = rolls / max(rolls.max(), 1e-9)
    ax.plot(ALPHAS, [scores[a][0] for a in ALPHAS], "-o", ms=4, color=GREY, label="accuracy")
    ax.plot(ALPHAS, [scores[a][1] for a in ALPHAS], "-s", ms=4, color=LIGHT, label="linear probe")
    ax.plot(ALPHAS, use, "-o", ms=4, color=NAVY, label="audit (scaled)")
    ax.set_xlabel(r"$\alpha$ (PTT weight)"); ax.set_ylabel("score")
    ax.legend(fontsize=7.5, frameon=False); plabel(ax, "d", r"Accuracy is blind to $\alpha$")


    # ===== SECOND ROW: synthetic WAVEFORMS =====
    # The scalar sandbox above hands PTT to the model as a number, so it never exercises foot
    # detection -- the step that fails on real data. This row generates ECG+PPG with a known
    # PTT-to-BP law and runs the identical pipeline used on VitalDB, so a null in the third row
    # can be attributed to the models rather than to the instrument.
    import synth_waveform_audit as SW
    swres = json.loads((DATA / "synth_waveform_audit.json").read_text())
    rngw = np.random.default_rng(3)

    ax = axes[1, 0]
    seg = SW.make_segment(rngw, hr=68, ptt_ms=210)
    stacked(ax, ppg_ax[1], seg[:, 0], seg[:, 1], SW.FS, 0.0, 3.0)
    plabel(ax, "e", "Synthetic ECG and PPG")

    ax = axes[1, 1]
    _, ysw, pttw, _ = SW.make_dataset(600, 1.0, np.random.default_rng(11))
    ax.scatter(pttw, ysw, s=4, alpha=.35, color=NAVY, edgecolors="none")
    xsw = np.linspace(pttw.min(), pttw.max(), 40)
    ax.plot(xsw, 130.0 - 0.22 * (xsw - 190.0), color=RED, lw=1.5)
    ax.set_xlabel("injected PTT (ms)"); ax.set_ylabel("BP (mmHg)")
    ax.text(.05, .07, "-0.22 mmHg/ms", transform=ax.transAxes, fontsize=8, color=RED)
    plabel(ax, "f", r"Injected law, $\alpha$ = 1")

    ax = axes[1, 2]
    h1 = swres["alphas"]["1.0"]["history"]; ref = swres["alphas"]["1.0"]["ref_rmse"]
    ax.axhline(ref["mean_predictor"], color=GREY, lw=1, ls=":")
    ax.text(len(h1["train"]) - 1, ref["mean_predictor"] - .1, "mean", fontsize=7,
            color=GREY, ha="right", va="top")
    ax.axhline(ref["oracle_ptt"], color=GREEN, lw=1, ls=":")
    ax.text(len(h1["train"]) - 1, ref["oracle_ptt"] - .08, "oracle", fontsize=7,
            color=GREEN, ha="right", va="top")
    ax.plot(h1["train"], color=NAVY, lw=1.3, label="train")
    ax.plot(h1["val"], color=RED, lw=1.3, ls="--", label="val")
    ax.set_ylim(ref["oracle_ptt"] - .6, ref["mean_predictor"] + .4)
    ax.set_xlabel("epoch"); ax.set_ylabel("RMSE (mmHg)")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    plabel(ax, "g", f"{swres.get('arch', 'cnn').capitalize()} training")

    ax = axes[1, 3]
    Xcw, _, ptw, _ = SW.make_dataset(400, 1.0, np.random.default_rng(99))
    estw = mechlib.compute_ptt(Xcw, SW.FS) * 1000.0 - SW.PPG_OFFSET_MS   # remove the known offset
    okw = np.isfinite(estw)
    ax.scatter(ptw[okw], estw[okw], s=5, alpha=.35, color=NAVY, edgecolors="none")
    limw = [min(ptw[okw].min(), estw[okw].min()), max(ptw[okw].max(), estw[okw].max())]
    ax.plot(limw, limw, color="k", lw=.9, ls="--")
    rw = float(np.corrcoef(estw[okw], ptw[okw])[0, 1])
    ax.set_xlabel("injected PTT (ms)"); ax.set_ylabel(f"measured − {SW.PPG_OFFSET_MS:.0f} ms offset")
    ax.text(.97, .05, f"r = {rw:.3f}\n{okw.mean():.0%} measurable", transform=ax.transAxes,
            fontsize=8, ha="right")
    plabel(ax, "h", "Detector vs. ground truth")

    ax = axes[1, 4]
    alw = [0.0, 0.25, 0.5, 0.75, 1.0]
    slw = [swres["alphas"][str(a)]["slope"] for a in alw]
    r2w = [swres["alphas"][str(a)]["test_r2"] for a in alw]
    ax.plot(alw, slw, "-", color=NAVY, lw=1.3)
    for a_, s_, r2_ in zip(alw, slw, r2w):
        poor = r2_ < 0           # a model that fits worse than the mean is flagged, not hidden
        for sv in swres["alphas"][str(a_)].get("slopes", []):
            ax.plot(a_, sv, ".", color=GREY, ms=4, zorder=1)      # individual seeds
        ax.plot(a_, s_, "o", ms=5, color="white" if poor else NAVY, mec=NAVY, mew=1.2)
    ax.axhline(0, color="k", lw=.6, alpha=.4)
    ax.axhline(-0.22, color=RED, lw=1, ls="--")           # the law written into the data (panel f)
    ax.text(.03, -0.215, "injected", fontsize=7.5, color=RED, va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$\alpha$ (PTT weight)"); ax.set_ylabel("audit slope (mmHg/ms)")
    ax.text(.97, .62, f"r = {swres['slope_vs_alpha_r']:+.3f}", transform=ax.transAxes,
            fontsize=9, ha="right")
    plabel(ax, "i", r"Audit recovers the law")

    # ===== THIRD ROW: real VitalDB waveforms, trained transformer =====
    d = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    yte = d["yte"]; fs = int(d["fs"]); t_axis = np.arange(Xte.shape[1]) / fs
    scalars = mechlib.compute_scalars(Xte, fs, mechlib.ECG, mechlib.PPG)

    ax = axes[2, 0]
    i_ex = 0                                  # first segment with >= 3 detected R->foot pairs in the window
    for i_c in range(min(300, len(Xte))):
        fd = mechlib.segment_fiducials(Xte[i_c, :, 0], Xte[i_c, :, 1], fs)
        if np.sum(fd["feet"] < 3 * fs) >= 3:
            i_ex = i_c
            break
    stacked(ax, ppg_ax[2], Xte[i_ex, :, 0], Xte[i_ex, :, 1], fs, 0.0, 3.0)
    plabel(ax, "j", "VitalDB ECG and PPG")

    ckpt = torch.load(str(DATA / "dbp_transformer.pt"), map_location="cpu", weights_only=False)
    cfg, hist, sd = ckpt["config"], ckpt["history"], ckpt["state_dict"]
    tnet = mechlib.WaveTransformer(**cfg); tnet.load_state_dict(sd); tnet.eval()

    ax = axes[2, 1]
    # Mean predictor = always output the training-set mean DBP. Shown so the reader can see how
    # little of the drop in MAE is learning beyond the mean.
    mae_mean = float(np.abs(yte[:, 1] - d["ytr"][:, 1].mean()).mean())
    ax.axhline(mae_mean, color=GREY, lw=1, ls=":")
    ax.text(len(hist["train_mae"]) - 1, mae_mean + 1, f"mean ({mae_mean:.1f})", fontsize=7,
            color=GREY, ha="right", va="bottom")
    ax.plot(hist["train_mae"], color=NAVY, lw=1.3, label="train")
    ax.plot(hist["val_mae"], color=RED, lw=1.3, ls="--", label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("DBP MAE (mmHg)")
    ax.legend(fontsize=8, frameon=False); plabel(ax, "k", "Transformer training")

    # The data behind row 3, so a reader can compare its range to what the simulations assume.
    ax = axes[2, 2]
    ytr_ = d["ytr"]
    groups = [("SBP", 0, ytr_, NAVY, 1), ("SBP", 0, yte, LIGHT, 2),
              ("DBP", 1, ytr_, NAVY, 3.4), ("DBP", 1, yte, LIGHT, 4.4)]
    for lab, col_i, arr, colr, xpos in groups:
        vp = ax.violinplot(arr[:, col_i], positions=[xpos], showextrema=False, widths=0.9)
        for b in vp["bodies"]:
            b.set_facecolor(colr); b.set_alpha(.75); b.set_edgecolor("black"); b.set_linewidth(.7)
        ax.plot(xpos, np.median(arr[:, col_i]), "o", color="white", mec="k", ms=3, zorder=4)
    ax.set_xticks([1, 2, 3.4, 4.4], ["SBP\ntrain", "SBP\ntest", "DBP\ntrain", "DBP\ntest"], fontsize=7)
    ax.set_ylabel("mmHg")
    ax.text(.97, .97, f"{len(np.unique(d['gte']))} test subjects",
            transform=ax.transAxes, fontsize=7, ha="right", va="top", color="#333")
    plabel(ax, "l", "BP distribution")

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

    ax = axes[2, 3]
    r2_pat = [mechlib.linear_probe(f, scalars["pat"]) for f in stages]
    r2_per = [mechlib.linear_probe(f, scalars["period"]) for f in stages]
    ax.plot(range(len(xs)), r2_pat, "-o", ms=4, color=NAVY, lw=1.3, label="PAT")
    ax.plot(range(len(xs)), r2_per, "-o", ms=4, color=RED, lw=1.3, label="cardiac period")
    ax.axhline(0, color="#bbb", lw=.8); ax.set_xticks(range(len(xs)), xs, fontsize=8)
    ax.set_ylabel("probe $R^2$"); ax.set_xlabel("layer")
    ax.legend(fontsize=8, frameon=False); ax.set_ylim(-0.08, 0.66)
    plabel(ax, "m", "Linear probes")

    # Direction-only consistency test. Expected sign under any monotone PAT->BP law is NEGATIVE
    # (later PPG, longer arrival time, lower BP). Magnitude is not compared to a population
    # constant: the constants are subject-specific, and a time-shifted input is off-manifold.
    # Method: non-circular shift, negative arm only, the MEASURED change in PAT on the x-axis,
    # undetectable segments dropped (not imputed), one slope per subject.
    @torch.no_grad()
    def predict_fn(Xd):
        o = tnet(torch.tensor(Xd, dtype=torch.float32)).numpy()
        return np.stack([o, o], 1)

    Xa = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]]); ga = d["gte"]
    pat0 = mechlib.compute_ptt(Xa, fs); okp = np.isfinite(pat0)
    med_pat = float(np.nanmedian(pat0[okp])); b0 = float(d["yte"][okp, 1].mean())

    def control(a_):
        def fn(Xr):
            pp = mechlib.compute_ptt(Xr, fs); pp = np.where(np.isfinite(pp), pp, med_pat)
            o = a_ / pp + b0 - a_ / med_pat
            return np.stack([o, o], 1)
        return fn
    rows = [("textbook", AUD.subject_audit(control(3.0), Xa, ga, fs), GREEN),
            ("transformer", AUD.subject_audit(predict_fn, Xa, ga, fs), NAVY),
            ("inverted", AUD.subject_audit(control(-3.0), Xa, ga, fs), GREY)]
    ax = axes[2, 4]
    rj = np.random.default_rng(5)
    ax.axvspan(-1e3, 0, color="#e6efe8", zorder=0)
    ax.axvline(0, color="k", lw=.8)
    for i, (lab, r, col) in enumerate(rows):
        sl = np.array(r["subj_slopes"])                     # mmHg per ms, same unit as panel i
        ax.scatter(sl, i + rj.uniform(-.17, .17, len(sl)), s=7, alpha=.55, color=col, edgecolors="none")
        ax.plot(np.percentile(sl, [2.5, 97.5]), [i, i], color="k", lw=1.6, solid_capstyle="butt")
        ax.plot(np.median(sl), i, "D", color="white", mec="k", ms=5, zorder=5)
    lim = max(np.abs(np.array(r["subj_slopes"])).max() * 1.05 for _, r, _ in rows)
    ax.set_xlim(-lim, lim); ax.set_ylim(-.6, 2.6)
    ax.set_yticks(range(3), [f"{lab}\n{r['frac_negative']:.0%} neg." for lab, r, _ in rows],
                  fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("per-subject slope (mmHg/ms)")
    ax.text(.03, .97, "expected", transform=ax.transAxes, fontsize=7, color=GREEN, va="top")
    plabel(ax, "n", f"Sign test ({rows[1][1]['n_subjects']} subjects)")

    for ax in all_axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.patch.set_alpha(0)
    fig.subplots_adjust(hspace=0.55, wspace=1.5, bottom=0.06, left=0.085, right=0.985, top=0.95)

    # Row banners: a tinted band behind each row and a rotated tag at the left, so a reader can
    # never mistake a simulation (ground truth known) for a measurement (none).
    from matplotlib.patches import Rectangle
    fig.canvas.draw()
    bands = [("SYNTHETIC", "scalar PTT", "#eaf0f8", NAVY),
             ("SYNTHETIC", "waveforms", "#eaf0f8", NAVY),
             ("REAL", "VitalDB", "#fbeee9", RED)]
    for i, (name, sub, bg, fg) in enumerate(bands):
        last = max(c for c in range(5) if axes[i, c] is not None)
        p0, p3 = axes[i, 1].get_position(), axes[i, last].get_position()   # col 0 is split in rows 2-3
        pad_t, pad_b, x0, x1 = 0.035, 0.05, 0.012, p3.x1 + 0.012
        y0, y1 = p0.y0 - pad_b, p0.y1 + pad_t
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 facecolor=bg, edgecolor="none", zorder=-5))
        fig.add_artist(Rectangle((x0, y0), 0.006, y1 - y0, transform=fig.transFigure,
                                 facecolor=fg, edgecolor="none", zorder=-4))
        fig.text(x0 + 0.016, (y0 + y1) / 2, f"{name}\n{sub}", rotation=90, ha="center",
                 va="center", fontsize=8.5, fontweight="bold", color=fg, linespacing=1.3)

    # Method caveats (direction-only test, PAT includes PEP, dropped segments) live in the caption.
    fig.savefig(ROOT / "figures" / "fig_app_tabs.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "fig_app_tabs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_app_tabs.png / .pdf")


if __name__ == "__main__":
    main()
