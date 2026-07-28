"""ood_benchmark.py -- train the PulseDB benchmark architectures on our VitalDB cache and
show where they break out of distribution, tied to the mechanistic audit.

Models come from the AI4HealthUOL/ppg-ood-generalization repo (Moulaeifard, Charlton &
Strodthoff), vendored under external/ppg-ood. That repo ships architectures and training
code only -- NO pretrained weights -- so every model here is trained locally from scratch.

Three things get logged per model to Weights & Biases:
  1. accuracy on an in-distribution test set and on several OOD conditions,
  2. a linear-probe battery run at every layer of the network (what is decodable, where),
  3. the causal PPG-shift audit + a final causal-direction figure (does it USE arrival time).

The point of pairing 2+3 with 1: a model can hold its ID error while its mechanism is
already wrong, and that shows up as an OOD failure. Probes say a cue is *present*; only
the shift audit says it is *used*.

Usage:
    python ood_benchmark.py --models all --epochs 30
    python ood_benchmark.py --models lenet1d,transformer --epochs 5 --no-wandb
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "external" / "ppg-ood" / "Processing" / "required_codes_files"))

import mechlib
import physics_audit as pa
from mechlib import ECG, PPG, WaveTransformer

MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
TARGETS = ["sbp", "dbp"]


# --------------------------------------------------------------- model zoo
def build_model(name, n_ch=2, n_out=2, L=1250):
    """All the benchmark architectures take (B, C, L); our WaveTransformer takes (B, L, C),
    so it is wrapped below to present one calling convention to the rest of the script."""
    from clinical_ts.xresnet1d import xresnet1d50, xresnet1d101
    from clinical_ts.inception1d import inception1d
    from clinical_ts.lenet1d import lenet1d

    if name == "xresnet1d50":
        return xresnet1d50(input_channels=n_ch, num_classes=n_out, input_size=L)
    if name == "xresnet1d101":
        return xresnet1d101(input_channels=n_ch, num_classes=n_out, input_size=L)
    if name == "inception1d":
        return inception1d(input_channels=n_ch, num_classes=n_out)
    if name == "lenet1d":
        return lenet1d(input_channels=n_ch, num_classes=n_out)
    if name == "transformer":
        return TransformerRegressor(n_ch=n_ch, n_out=n_out, L=L)
    raise ValueError(f"unknown model {name!r}")


ALL_MODELS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]


class TransformerRegressor(nn.Module):
    """WaveTransformer trunk with a 2-output head, exposing per-block activations so the
    probe battery can read the same kind of layer stack it reads from the CNNs."""

    def __init__(self, n_ch=2, n_out=2, L=1250, dm=64, patch=25, heads=4, depth=3):
        super().__init__()
        self.trunk = WaveTransformer(n_ch=n_ch, dm=dm, patch=patch, heads=heads, depth=depth, L=L)
        self.trunk.head = nn.Linear(dm, n_out)

    def forward(self, x):                     # (B, C, L) -> (B, n_out)
        return self.trunk(x.transpose(1, 2))

    @torch.no_grad()
    def layer_acts(self, x):
        _, acts = self.trunk(x.transpose(1, 2), return_acts=True)
        return [a.mean(1) for a in acts]      # mean-pool tokens -> (B, dm) per block


def torch_layers(model, name):
    """Ordered list of submodules whose outputs we probe. For the CNNs this is the stem +
    each residual/inception stage; the transformer handles itself via layer_acts."""
    if name == "transformer":
        return None
    if name.startswith("xresnet"):
        # nn.Sequential: [stem0, stem1, stem2, maxpool, stage0..stageN, head]
        return [m for m in list(model.children())[:-1]]
    if name == "inception1d":
        bb = model.layers[0]
        mods = [bb.im[i] for i in range(len(bb.im))]
        return mods
    if name == "lenet1d":
        return [m for m in model.conv_layers]
    return None


# --------------------------------------------------------------- data / shifts
def subject_split(d, seed=0, frac=(0.6, 0.15, 0.25)):
    """The cached npz splits share subjects across train/test (random-segment split), so a
    model can memorise a subject's baseline and still look accurate. Rebuild the splits so
    they are SUBJECT-DISJOINT -- otherwise 'in distribution' is not a meaningful baseline
    and every OOD gap below would be understated."""
    X = np.concatenate([d["Xtr"], d["Xva"], d["Xte"]])
    y = np.concatenate([d["ytr"], d["yva"], d["yte"]])
    g = np.concatenate([d["gtr"], d["gva"], d["gte"]])
    subs = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_tr = int(frac[0] * len(subs))
    n_va = int(frac[1] * len(subs))
    groups = {"train": subs[:n_tr], "val": subs[n_tr:n_tr + n_va], "test": subs[n_tr + n_va:]}
    out = {}
    for k, s in groups.items():
        m = np.isin(g, s)
        out[k] = dict(X=X[m], y=y[m], g=g[m])
    return out


def make_ood(X, y, g, fs, seed=0, has_ecg=True):
    """OOD conditions built as explicit, physically-meaningful shifts of the held-out test
    set. Each isolates one way deployment data drifts away from PulseDB-style training data.
    Amplitude/baseline shifts are *label-preserving*: true BP does not change when the
    sensor gain does, so any error increase is the model keying on a nuisance feature."""
    rng = np.random.default_rng(seed)
    N = len(X)
    conds = {"id": (X, y)}
    # PPG sits at index 1 alongside ECG, but at index 0 when ECG was dropped (--ppg-only)
    ppg = PPG if has_ecg else 0

    # sensor gain drift: PPG scaled 0.5x. BP unchanged.
    Xa = X.copy()
    Xa[:, :, ppg] *= 0.5
    conds["ppg_gain_0.5x"] = (Xa, y)

    # baseline wander: slow sinusoidal drift added to PPG. BP unchanged.
    t = np.arange(X.shape[1]) / fs
    ph = rng.uniform(0, 2 * np.pi, (N, 1))
    wander = 0.5 * np.sin(2 * np.pi * 0.15 * t[None, :] + ph).astype(np.float32)
    Xw = X.copy()
    Xw[:, :, ppg] += wander * X[:, :, ppg].std(1, keepdims=True)
    conds["baseline_wander"] = (Xw, y)

    # additive sensor noise at 10% of signal std. BP unchanged.
    Xn = X.copy()
    Xn[:, :, ppg] += (0.10 * X[:, :, ppg].std(1, keepdims=True)
                      * rng.standard_normal(X[:, :, ppg].shape).astype(np.float32))
    conds["ppg_noise_10pct"] = (Xn, y)

    # PPG-only deployment: ECG channel zeroed. Any model that truly needs ECG-to-PPG
    # arrival time must collapse here; a morphology-only model will barely move.
    # Undefined when the model was trained PPG-only -- there is no ECG channel to drop.
    if has_ecg:
        Xp = X.copy()
        Xp[:, :, ECG] = 0.0
        conds["ecg_dropout"] = (Xp, y)

    # hypertensive tail: the segments whose true SBP sits in the top quartile. This is
    # label shift, the regime that matters clinically and where regression-to-the-mean hurts.
    hi = y[:, 0] >= np.quantile(y[:, 0], 0.75)
    conds["hypertensive_tail"] = (X[hi], y[hi])
    return conds


# --------------------------------------------------------------- train / eval
def train(model, tr, va, device, epochs=30, bs=256, lr=1e-3, log=None):
    """Plain supervised regression on z-scored targets, early-stopped on val MAE."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, len(tr["X"]) // bs))
    lossf = nn.SmoothL1Loss()

    mu, sd = tr["y"].mean(0), tr["y"].std(0) + 1e-8
    Xtr = torch.tensor(tr["X"].transpose(0, 2, 1))
    ytr = torch.tensor((tr["y"] - mu) / sd)

    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for s in range(0, len(Xtr) - bs + 1, bs):
            idx = perm[s:s + bs]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            try:
                sched.step()
            except Exception:
                pass
            tot += loss.item()
        vp = predict(model, va["X"], device, mu, sd)
        vmae = np.abs(vp - va["y"]).mean(0)
        score = vmae.mean()
        if log:
            log({"epoch": ep, "train_loss": tot / max(1, len(Xtr) // bs),
                 "val_mae_sbp": float(vmae[0]), "val_mae_dbp": float(vmae[1])})
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, (mu, sd)


@torch.no_grad()
def predict(model, X, device, mu, sd, bs=512):
    """(N, L, C) numpy -> (N, 2) predictions in mmHg (undoes target z-scoring)."""
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs].transpose(0, 2, 1), dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) * sd + mu


@torch.no_grad()
def layer_features(model, name, X, device, bs=256, max_dim=512):
    """Activations at each layer, mean-pooled over time -> {layer_name: (N, D)}.
    Wide layers are randomly projected down to `max_dim` so the ridge probe stays cheap
    and comparably conditioned across layers of very different widths."""
    model.eval()
    if name == "transformer":
        feats = {}
        for s in range(0, len(X), bs):
            xb = torch.tensor(X[s:s + bs].transpose(0, 2, 1), dtype=torch.float32, device=device)
            for i, a in enumerate(model.layer_acts(xb)):
                feats.setdefault(f"block{i}", []).append(a.cpu().numpy())
        return {k: np.concatenate(v) for k, v in feats.items()}

    mods = torch_layers(model, name)
    acc, handles, cur = {}, [], {}

    def hook(idx):
        def f(_m, _i, o):
            if isinstance(o, tuple):
                o = o[0]
            cur[idx] = o.mean(-1).cpu().numpy()      # global average pool over time
        return f

    for i, m in enumerate(mods):
        handles.append(m.register_forward_hook(hook(i)))
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs].transpose(0, 2, 1), dtype=torch.float32, device=device)
        cur.clear()
        model(xb)
        for i, v in cur.items():
            acc.setdefault(i, []).append(v)
    for h in handles:
        h.remove()

    out = {}
    rng = np.random.default_rng(0)
    for i, v in sorted(acc.items()):
        F = np.concatenate(v)
        if F.shape[1] > max_dim:
            P = rng.standard_normal((F.shape[1], max_dim)).astype(np.float32) / np.sqrt(max_dim)
            F = F @ P
        out[f"layer{i}"] = F
    return out


def probe_battery(model, name, X, y, scalars, device):
    """Layer-wise linear probes: for every layer, held-out R^2 decoding each physiological
    cue plus the BP targets. Reading down a column shows where a cue becomes linearly
    available; comparing to the causal audit shows whether availability is ever used."""
    feats = layer_features(model, name, X, device)
    targets = {"sbp": y[:, 0], "dbp": y[:, 1]}
    targets.update(scalars)
    rows = {}
    for lname, F in feats.items():
        rows[lname] = {t: mechlib.linear_probe(F, v) for t, v in targets.items()}
    return rows


def causal_audit(model, name, X, fs, device, mu, sd):
    """Roll the PPG channel +/- delta samples -- a real change to pulse arrival time -- and
    measure the response of predicted BP. Faithful physiology = NEGATIVE slope: later PPG
    means longer PTT means lower BP."""
    fn = lambda Xr: predict(model, Xr, device, mu, sd)
    return mechlib.causal_ptt_audit(None, X, fs, device, predict_fn=fn, n_max=min(1000, len(X)))


def physics_audit(model, X, fs, device, mu, sd, has_ecg=True, target=1):
    """The full governing-law battery: perturb each cue in the raw signal and check the
    response sign against physiology (see physics_audit.LAWS)."""
    fn = lambda Xr: predict(model, Xr, device, mu, sd)
    return pa.run_battery(fn, X, fs, target=target, has_ecg=has_ecg, n_max=min(600, len(X)))


# --------------------------------------------------------------- figures
def direction_figure(results, path, gap_cond=None):
    """The summary panel: causal PPG-shift slope (mechanism) against OOD degradation.
    A model in the correct-sign region that also holds its error under shift is the only
    quadrant where the accuracy is coming from the physiology we think it is.
    `gap_cond` picks which OOD condition defines the penalty axis -- the real external set
    when one is present, otherwise the worst synthetic shift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    conds = list(results[names[0]]["ood"])
    if gap_cond is None:
        gap_cond = "ecg_dropout" if "ecg_dropout" in conds else conds[-1]
    slopes = [results[n]["audit"]["dbp"]["dBP_dPTT"] for n in names]
    frac = [results[n]["audit"]["dbp"]["frac_correct_sign"] for n in names]
    gap = [results[n]["ood"][gap_cond]["mae_dbp"] - results[n]["ood"]["id"]["mae_dbp"]
           for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    c = ["tab:red" if s > 0 else "tab:blue" for s in slopes]
    ax[0].barh(names, slopes, color=c)
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("dDBP/dPTT  (mmHg per s)")
    ax[0].set_title("Causal direction: does shifting PPG move BP the physiological way?\n"
                    "blue = correct (negative), red = inverted", fontsize=9)
    for i, f in enumerate(frac):
        ax[0].text(0, i, f"  {f:.0%} segs correct", va="center", fontsize=7)

    ax[1].scatter(slopes, gap, c=c, s=60)
    for n, s, gp in zip(names, slopes, gap):
        ax[1].annotate(n, (s, gp), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[1].axvline(0, color="k", lw=1, ls=":")
    ax[1].axhline(0, color="k", lw=1, ls=":")
    ax[1].set_xlabel("dDBP/dPTT  (mmHg per s)")
    ax[1].set_ylabel(f"OOD penalty: {gap_cond} (mmHg)")
    ax[1].set_title("Mechanism vs OOD fragility", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def physics_figure(name, batt, probe_rows, path):
    """Per-cue: probe R^2 at the last layer (is it DECODABLE) beside the perturbation slope
    (is it USED, and in the direction physics requires). The interesting cell is high R^2 with
    a wrong-signed slope -- information present, mechanism inverted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cues = list(batt)
    last = list(probe_rows)[-1]
    r2 = [max(probe_rows[last].get(c, 0.0), 0.0) for c in cues]
    slope = [batt[c]["slope"] for c in cues]
    ok = [batt[c]["sign_ok"] for c in cues]
    col = ["tab:blue" if o is True else "tab:red" if o is False else "tab:gray" for o in ok]

    fig, ax = plt.subplots(1, 2, figsize=(11, 0.42 * len(cues) + 2.6))
    ax[0].barh(cues, r2, color="tab:purple")
    ax[0].set_xlim(0, 1)
    ax[0].set_xlabel("linear-probe $R^2$ (final layer)")
    ax[0].set_title("Is the cue DECODABLE?", fontsize=9)

    ax[1].barh(cues, slope, color=col)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("dBP / dcue  (perturbation response)")
    ax[1].set_title("Is it USED the way physics says?\n"
                    "blue = sign matches law, red = inverted, grey = no law / control",
                    fontsize=9)
    for i, c in enumerate(cues):
        e = batt[c]["expect"]
        ax[1].text(0, i, "  expect " + ("-" if e < 0 else "+" if e > 0 else "0"),
                   va="center", fontsize=7)
    fig.suptitle(f"{name}: probe vs governing-law perturbation", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def physics_summary_figure(results, path):
    """Cross-model version of the direction figure for PPG-only runs, where PAT does not
    exist: models x cues grid of perturbation-response signs against the governing laws."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    cues = [c for c, v in results[names[0]]["physics"].items() if v["sign_ok"] is not None]
    M = np.array([[1.0 if results[n]["physics"][c]["sign_ok"] else -1.0 for c in cues]
                  for n in names])
    conds = list(results[names[0]]["ood"])
    gapc = conds[-1]

    fig, ax = plt.subplots(1, 2, figsize=(6 + 0.7 * len(cues), 0.5 * len(names) + 3))
    ax[0].imshow(M, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
    ax[0].set_xticks(range(len(cues)), cues, rotation=45, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(names)), names, fontsize=8)
    ax[0].set_title("Governing-law sign check\nblue = matches physics, red = inverted",
                    fontsize=9)
    for i in range(len(names)):
        for j in range(len(cues)):
            ax[0].text(j, i, "OK" if M[i, j] > 0 else "X", ha="center", va="center", fontsize=7)

    n_ok = [int(sum(results[n]["physics"][c]["sign_ok"] for c in cues)) for n in names]
    gap = [results[n]["ood"][gapc]["mae_dbp"] - results[n]["ood"]["id"]["mae_dbp"] for n in names]
    ax[1].scatter(n_ok, gap, s=60, c="tab:blue")
    for n, a, b in zip(names, n_ok, gap):
        ax[1].annotate(n, (a, b), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[1].set_xlabel(f"# governing laws with correct sign (of {len(cues)})")
    ax[1].set_ylabel(f"OOD penalty: {gapc} (mmHg)")
    ax[1].set_title("Mechanism correctness vs OOD fragility", fontsize=9)
    ax[1].axhline(0, color="k", lw=1, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def probe_figure(name, rows, path):
    """Layer (x) by cue (y) heatmap of probe R^2 -- the 'what is decodable, and where' map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(rows)
    cues = list(rows[layers[0]])
    M = np.array([[max(rows[l][c], 0.0) for l in layers] for c in cues])
    fig, ax = plt.subplots(figsize=(1.1 * len(layers) + 3, 0.34 * len(cues) + 2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)), layers, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cues)), cues, fontsize=7)
    ax.set_title(f"{name}: linear-probe R^2 by layer", fontsize=9)
    fig.colorbar(im, ax=ax, label="held-out R^2")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_mini_deep.npz")
    ap.add_argument("--models", default="all", help="comma list or 'all'")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="ppg-ood-audit")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--probe-n", type=int, default=1500, help="segments used for probes/audit")
    ap.add_argument("--ppg-only", action="store_true",
                    help="train on PPG alone so the external PPG-only sets can be scored")
    ap.add_argument("--external", default="",
                    help="comma list name=path of BP-Benchmark folders (PPG-only sets)")
    ap.add_argument("--mimic", default="", help="path to MIMIC-BP root (ECG+PPG OOD set)")
    ap.add_argument("--mimic-patients", type=int, default=0, help="0 = all")
    ap.add_argument("--run-tag", default="", help="suffix for output json/figures to avoid overwrite")
    ap.add_argument("--gbm", action="store_true",
                    help="also train the LightGBM built from audit-passing cues + age/sex")
    ap.add_argument("--ext-cues", action="store_true",
                    help="probe the exhaustive feature battery (fractal dim, APG a-e, HRV, PTT variants)")
    args = ap.parse_args()
    tag = ("_" + args.run_tag) if args.run_tag else ("_ppg" if args.ppg_only else "_ecgppg")

    models = ALL_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    d = mechlib.load_mini(args.data)
    fs = d["fs"]
    # The full-PulseDB npz already carries the official patient-disjoint CalFree splits (and
    # per-segment demographics); the demo cache does not, so re-split it ourselves.
    full = "gtr" in d and len(set(d["gtr"].tolist()).intersection(set(d["gte"].tolist()))) == 0
    if full:
        sp = {"train": dict(X=d["Xtr"], y=d["ytr"], g=d["gtr"]),
              "val": dict(X=d["Xva"], y=d["yva"], g=d["gva"]),
              "test": dict(X=d["Xte"], y=d["yte"], g=d["gte"])}
        demo_keys = [k for k in d if k.startswith(("age_", "sex_", "bmi_"))]
        vitaldb_demo = {k: d[k] for k in demo_keys} if demo_keys else None
        print(f"[data] FULL PulseDB (official CalFree splits){' + demographics' if demo_keys else ''}")
    else:
        sp = subject_split(d, seed=args.seed)
        vitaldb_demo = None
    # ABP is the label source and must never be an input. PPG-only mode drops ECG so the
    # model matches the external BP-Benchmark sets, which carry no ECG channel.
    chans = [PPG] if args.ppg_only else [ECG, PPG]
    for k in sp:
        sp[k]["X"] = mechlib.normalize(sp[k]["X"][:, :, chans])
    n_ch = len(chans)
    L = sp["train"]["X"].shape[1]
    has_ecg = not args.ppg_only
    print(f"[data] subject-disjoint: "
          + " ".join(f"{k}={len(sp[k]['X'])}seg/{len(np.unique(sp[k]['g']))}subj" for k in sp)
          + f"  channels={'PPG' if args.ppg_only else 'ECG+PPG'} L={L}")

    ood = make_ood(sp["test"]["X"], sp["test"]["y"], sp["test"]["g"], fs, seed=args.seed,
                   has_ecg=has_ecg)

    # real external OOD: BP-Benchmark sets (PPG-only, different length -> resampled to L)
    ext = {}
    for spec in [s for s in args.external.split(",") if s.strip()]:
        nm, _, path = spec.partition("=")
        e = pa.load_bpbenchmark(path, name=nm)
        # some sets (UCI2 ~410k seg) are far larger than needed; cap by subject-preserving cut
        if len(e["X"]) > 12000:
            rng = np.random.default_rng(args.seed)
            idx = np.sort(rng.choice(len(e["X"]), 12000, replace=False))
            e = {**e, "X": e["X"][idx], "y": e["y"][idx], "g": e["g"][idx],
                 "demo": ({k: v[idx] for k, v in e["demo"].items()} if e["demo"] else None)}
        Xe = mechlib.normalize(pa.resample_to(e["X"], L))
        if not args.ppg_only:
            print(f"[ext] SKIP {nm}: PPG-only data cannot feed an ECG+PPG model "
                  f"(use --ppg-only)")
            continue
        ext[nm] = dict(X=Xe, y=e["y"], g=e["g"], demo=e.get("demo"))
        print(f"[ext] {nm}: {len(Xe)}seg/{len(np.unique(e['g']))}subj "
              f"SBP {e['y'][:, 0].mean():.1f} DBP {e['y'][:, 1].mean():.1f}")

    # MIMIC-BP: ECG+PPG external OOD (windowed 3750 -> L). PAT audit runs here.
    if args.mimic:
        chans = ("ppg",) if args.ppg_only else ("ecg", "ppg")
        m = pa.load_mimic_bp(args.mimic, channels=chans,
                             max_patients=(args.mimic_patients or None), seed=args.seed)
        Xm, k = pa.window_segments(m["X"], L)              # (N*k, L, C); k windows per record
        ym = np.repeat(m["y"], k, axis=0)                  # record-major, so labels repeat by k
        gm = np.repeat(m["g"], k, axis=0)
        Xm = mechlib.normalize(Xm)
        ext["mimic_bp"] = dict(X=Xm, y=ym, g=gm, demo=None, has_ecg=(not args.ppg_only))
        print(f"[ext] mimic_bp: {len(Xm)}seg/{len(np.unique(gm))}subj "
              f"SBP {ym[:, 0].mean():.1f} DBP {ym[:, 1].mean():.1f}  (ECG+PPG, PAT audit ON)")
    print(f"[ood] conditions: {list(ood)}  external: {list(ext)}")

    # cue scalars for the probe battery, computed once on the ID test subset
    n_probe = min(args.probe_n, len(sp["test"]["X"]))
    Xp, yp = sp["test"]["X"][:n_probe], sp["test"]["y"][:n_probe]
    cache = ROOT / "data" / ("_ood_cues{}{}.npz".format(
        "_ppg" if args.ppg_only else "", "_ext" if args.ext_cues else ""))
    if cache.exists():
        z = np.load(cache)
        scalars = {k: z[k] for k in z.files}
        if len(next(iter(scalars.values()))) != n_probe:
            scalars = None
    else:
        scalars = None
    if scalars is None:
        print("[cues] computing physiological scalars (one-time, ~1-2 min)...")
        if has_ecg:
            scalars = mechlib.compute_scalars(Xp, fs)
        else:
            # No ECG => PAT is undefined; only PPG-intrinsic morphology cues are measurable.
            scalars = mechlib.compute_morphology(Xp, fs, ch=0)
        if args.ext_cues:
            # exhaustive battery: fractal dim, APG a-e ratios, widths, HRV, PTT variants.
            import features_ext as fx
            ppg_ch = PPG if has_ecg else 0
            ecg_ch = ECG if has_ecg else None
            scalars.update(fx.compute_ext(Xp, fs, ppg_ch=ppg_ch, ecg_ch=ecg_ch))
        scalars = {k: v for k, v in scalars.items() if np.isfinite(v).mean() > 0.2}
        np.savez(cache, **scalars)
    print(f"[cues] {len(scalars)} cues: {list(scalars)}")

    # quantify the ID->OOD shift on the cue space so 'out of distribution' is measured, not
    # asserted. KS per cue: 0 identical, 1 disjoint.
    dist_shift = {}
    for nm, e in ext.items():
        try:
            ce = (mechlib.compute_scalars(e["X"][:800], fs) if e.get("has_ecg", False)
                  else mechlib.compute_morphology(e["X"][:800], fs, ch=0))
            dist_shift[nm] = pa.distribution_distance(scalars, ce)
        except Exception as ex:
            print(f"[shift] {nm}: {ex}")
    for nm, ks in dist_shift.items():
        top = sorted(ks.items(), key=lambda t: -t[1])[:4]
        print(f"[shift] {nm}: mean KS {np.mean(list(ks.values())):.2f}  top "
              + ", ".join(f"{k}={v:.2f}" for k, v in top))

    results = {}
    for name in models:
        print(f"\n=== {name} ===")
        run = None
        if not args.no_wandb:
            import wandb
            run = wandb.init(project=args.project, entity=args.entity, name=name,
                             group="ood-audit", reinit=True,
                             config=dict(model=name, epochs=args.epochs, lr=args.lr,
                                         batch_size=args.batch_size, seed=args.seed,
                                         split="subject-disjoint", data=args.data))
        log = (lambda m: run.log(m)) if run else (lambda m: None)

        t0 = time.time()
        model = build_model(name, n_ch=n_ch, L=L)
        n_par = sum(p.numel() for p in model.parameters())
        model, (mu, sd) = train(model, sp["train"], sp["val"], device,
                                epochs=args.epochs, bs=args.batch_size, lr=args.lr, log=log)
        torch.save({"state_dict": model.state_dict(), "model": name, "channels": chans,
                    "mu": mu, "sd": sd, "L": L}, MODEL_DIR / f"{name}{tag}.pt")

        # ---- 1. ID vs OOD accuracy
        oodres = {}
        for cond, (Xc, yc) in ood.items():
            p = predict(model, Xc, device, mu, sd)
            mae = np.abs(p - yc).mean(0)
            oodres[cond] = {"mae_sbp": float(mae[0]), "mae_dbp": float(mae[1])}
            log({f"ood/{cond}/mae_sbp": float(mae[0]), f"ood/{cond}/mae_dbp": float(mae[1])})
        # real external sets: subject-level bootstrap CI, since n_subj can be as low as 40
        extres = {}
        for nm, e in ext.items():
            p = predict(model, e["X"], device, mu, sd)
            bs = pa.bootstrap_mae(p, e["y"], e["g"])
            extres[nm] = {"mae_sbp": bs["mae"][0], "mae_dbp": bs["mae"][1],
                          "dbp_lo": bs["lo"][1], "dbp_hi": bs["hi"][1]}
            oodres[nm] = {"mae_sbp": bs["mae"][0], "mae_dbp": bs["mae"][1]}
            log({f"external/{nm}/mae_sbp": bs["mae"][0], f"external/{nm}/mae_dbp": bs["mae"][1],
                 f"external/{nm}/dbp_ci_lo": bs["lo"][1], f"external/{nm}/dbp_ci_hi": bs["hi"][1]})
            print("  EXT {:<14} DBP {:.2f}  [{:.2f}, {:.2f}] 95% CI".format(
                nm, bs["mae"][1], bs["lo"][1], bs["hi"][1]))

        base = oodres["id"]["mae_dbp"]
        for cond, v in oodres.items():
            v["dbp_penalty"] = v["mae_dbp"] - base
            log({f"ood/{cond}/dbp_penalty": v["dbp_penalty"]})
        print("  ID  MAE  SBP {:.2f} / DBP {:.2f}".format(
            oodres["id"]["mae_sbp"], oodres["id"]["mae_dbp"]))
        for c, v in oodres.items():
            if c != "id":
                print("  OOD {:<18} DBP {:.2f}  ({:+.2f})".format(c, v["mae_dbp"], v["dbp_penalty"]))

        # ---- 2. layer-wise probe battery
        rows = probe_battery(model, name, Xp, yp, scalars, device)
        for lname, r in rows.items():
            for cue, r2 in r.items():
                log({f"probe/{lname}/{cue}": r2})
        pf = probe_figure(name, rows, FIG_DIR / f"probe_{name}{tag}.png")
        if run:
            import wandb
            run.log({"probe_heatmap": wandb.Image(str(pf))})
            tbl = wandb.Table(columns=["layer"] + list(next(iter(rows.values()))))
            for lname, r in rows.items():
                tbl.add_data(lname, *[r[c] for c in next(iter(rows.values()))])
            run.log({"probe_table": tbl})

        # ---- 3. causal PAT audit (needs ECG as the timing reference)
        if has_ecg:
            aud = causal_audit(model, name, Xp, fs, device, mu, sd)
            for t in TARGETS:
                log({f"audit/{t}/dBP_dPTT": aud[t]["dBP_dPTT"],
                     f"audit/{t}/frac_correct_sign": aud[t]["frac_correct_sign"],
                     f"audit/{t}/resp_range_mmHg": aud[t]["resp_range_mmHg"]})
            print("  audit DBP  dBP/dPTT {:+.1f} mmHg/s   correct-sign {:.0%}".format(
                aud["dbp"]["dBP_dPTT"], aud["dbp"]["frac_correct_sign"]))
        else:
            aud = {t: {"dBP_dPTT": float("nan"), "frac_correct_sign": float("nan"),
                       "resp_range_mmHg": float("nan"), "curve": []} for t in TARGETS}
            print("  audit: PAT undefined without ECG -- see physics battery instead")

        # ---- 4. governing-law perturbation battery
        batt = physics_audit(model, Xp, fs, device, mu, sd, has_ecg=has_ecg)
        for cue, v in batt.items():
            log({f"physics/{cue}/slope": v["slope"],
                 f"physics/{cue}/resp_range": v["resp_range"],
                 f"physics/{cue}/sign_ok": (np.nan if v["sign_ok"] is None else int(v["sign_ok"]))})
        laws = [c for c, v in batt.items() if v["sign_ok"] is not None]
        n_ok = sum(batt[c]["sign_ok"] for c in laws)
        print("  physics {}/{} governing-law signs correct: ".format(n_ok, len(laws))
              + " ".join(("+" if batt[c]["sign_ok"] else "-") + c for c in laws))
        log({"physics/n_laws_correct": n_ok, "physics/n_laws": len(laws)})
        pfig = physics_figure(name, batt, rows, FIG_DIR / f"physics_{name}{tag}.png")
        if run:
            import wandb
            run.log({"physics_battery": wandb.Image(str(pfig))})
            t2 = wandb.Table(columns=["cue", "probe_r2_final", "slope", "expect_sign",
                                      "sign_ok", "resp_range_mmHg", "law"])
            lastl = list(rows)[-1]
            for c, v in batt.items():
                t2.add_data(c, rows[lastl].get(c, float("nan")), v["slope"], v["expect"],
                            str(v["sign_ok"]), v["resp_range"], v["law"])
            run.log({"physics_table": t2})

        results[name] = {"params": n_par, "ood": oodres, "probe": rows, "audit": aud,
                         "physics": batt, "external": extres, "train_s": time.time() - t0}
        if run:
            run.summary.update({"params": n_par, "id_mae_dbp": oodres["id"]["mae_dbp"],
                                "audit_dbp_slope": aud["dbp"]["dBP_dPTT"],
                                "audit_dbp_frac_correct": aud["dbp"]["frac_correct_sign"]})
            run.finish()

    # ---- cross-model summary
    if results:
        # PAT slope is NaN in PPG-only mode, so fall back to the rise-time law for the
        # mechanism axis -- it is the strongest PPG-intrinsic cue with a signed prediction.
        has_pat = np.isfinite(next(iter(results.values()))["audit"]["dbp"]["dBP_dPTT"])
        fig = (direction_figure(results, FIG_DIR / f"causal_direction{tag}.png") if has_pat
               else physics_summary_figure(results, FIG_DIR / f"causal_direction{tag}.png"))
        out = ROOT / "data" / f"ood_benchmark{tag}.json"
        out.write_text(json.dumps({"models": results, "dist_shift": dist_shift},
                                  indent=2, default=float))
        print(f"\n[done] {fig}\n[done] {out}")

        if not args.no_wandb:
            import wandb
            run = wandb.init(project=args.project, entity=args.entity,
                             name="summary", group="ood-audit", reinit=True)
            run.log({"causal_direction": wandb.Image(str(fig))})
            conds = list(next(iter(results.values()))["ood"])
            tbl = wandb.Table(columns=["model", "params", "audit_slope_dbp", "frac_correct"]
                              + [f"mae_dbp/{c}" for c in conds])
            for n, r in results.items():
                tbl.add_data(n, r["params"], r["audit"]["dbp"]["dBP_dPTT"],
                             r["audit"]["dbp"]["frac_correct_sign"],
                             *[r["ood"][c]["mae_dbp"] for c in conds])
            run.log({"summary_table": tbl})
            run.finish()

        conds = list(next(iter(results.values()))["ood"])
        gapc = "ecg_dropout" if "ecg_dropout" in conds else conds[-1]
        print("\n{:<14} {:>8} {:>12} {:>8} {:>9} {:>8}".format(
            "model", "ID DBP", gapc[:12], "slope", "correct", "laws"))
        for n, r in results.items():
            laws = [c for c, v in r["physics"].items() if v["sign_ok"] is not None]
            n_ok = sum(r["physics"][c]["sign_ok"] for c in laws)
            print("{:<14} {:>8.2f} {:>12.2f} {:>8.1f} {:>8.0%} {:>7}".format(
                n, r["ood"]["id"]["mae_dbp"], r["ood"][gapc]["mae_dbp"],
                r["audit"]["dbp"]["dBP_dPTT"], r["audit"]["dbp"]["frac_correct_sign"],
                f"{n_ok}/{len(laws)}"))


if __name__ == "__main__":
    main()
