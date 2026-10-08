"""real_audit_controls.py -- is the real transformer's wrong-sign arrival-time response robust?

Three checks on the subject-level shift audit (audit_subject.py), run on the held-out VitalDB
mini test set (144 patients, none of them in the training file):

  1. SEEDS. The original checkpoint (data/dbp_transformer.pt) is one training run. Retrain the same
     recipe (real_transformer_shortcut_walkthrough.ipynb: Adam 2e-3, StepLR 10/0.5, 30 epochs,
     best-val epoch kept) with several seeds and audit each. A finding that flips sign between
     seeds is not a finding.

  2. SPECIFICITY. Three ways to shift the input, at the same nominal sizes:
       ppg  : PPG earlier by d        -> PAT shorter by d   (the audit as used so far)
       ecg  : ECG later by d          -> PAT shorter by d   (same PAT change, other channel)
       both : both earlier by d       -> PAT unchanged      (only the position in the window moves)
     A model that reads relative timing gives ppg == ecg and both ~ 0. A response to `both` means
     the model reacts to where the beats sit in the 10 s window, not to arrival time.
     Plus `ppg_small`: shifts of 8-16 ms only, inside the natural within-patient PAT range, so
     the response is not an artefact of inputs the model never saw.

  3. THE DATA'S OWN SLOPE. Within each patient, regress BP on measured PAT across segments
     (PPG-based, and arterial-line based as a reference). If the data barely carries the
     PAT->BP relation, a model has little reason to learn it.

Slopes are dBP/dPAT in mmHg/ms against the NOMINAL PAT change (the detector measures the
nominal change to within a sample on these inputs; segments where it does not are dropped).
Textbook physiology predicts NEGATIVE.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

import mechlib
from mechlib import ECG, PPG, ABP, _shift_channel

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_SEG = 3
TOL_MS = 20.0                     # measured dPAT must match nominal within this, else beat-slip
SHIFTS = {"ppg": (-6, -4, -2, 0), "ecg": (6, 4, 2, 0), "both": (-6, -4, -2, 0),
          "ppg_small": (-2, -1, 0)}


def train(Xtr, ytr, Xva, yva, seed, epochs=30, bs=128, lr=2e-3):
    """The walkthrough notebook's recipe, verbatim apart from the seed and device."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = mechlib.WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=3, L=Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    hist = {"train_mae": [], "val_mae": []}
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            loss = ((net(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        tm = float(np.abs(predict(net, Xtr) - ytr).mean())
        vm = float(np.abs(predict(net, Xva) - yva).mean())
        hist["train_mae"].append(tm); hist["val_mae"].append(vm)
        if vm < best_val:
            best_val, best_state = vm, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    return net, hist


@torch.no_grad()
def predict(net, X, bs=512):
    net.eval()
    return np.concatenate([net(torch.tensor(X[i:i + bs], device=DEV)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def shifted_inputs(X, fs):
    """For every mode and shift: the shifted batch, the nominal PAT change (ms), and a validity
    mask (baseline and shifted PAT measurable, measured change within TOL_MS of nominal)."""
    base = mechlib.compute_ptt(X, fs)
    out = {}
    for mode, ds in SHIFTS.items():
        rows = []
        for d in ds:
            Xd = X.copy()
            if mode in ("ppg", "ppg_small", "both"):
                Xd[:, :, 1] = _shift_channel(X[:, :, 1], d)
            if mode in ("ecg", "both"):
                Xd[:, :, 0] = _shift_channel(X[:, :, 0], d)
            ms = 1000.0 * d / fs
            nominal = {"ppg": ms, "ppg_small": ms, "ecg": -ms, "both": 0.0}[mode]
            pat = mechlib.compute_ptt(Xd, fs)
            ok = np.isfinite(base) & np.isfinite(pat) & (np.abs((pat - base) * 1000 - nominal) < TOL_MS)
            rows.append({"X": Xd, "x": ms if mode == "both" else nominal, "pat": pat, "ok": ok})
        out[mode] = rows
    return base, out


def subject_slopes(preds, rows, groups):
    """preds: list of per-shift prediction arrays aligned with rows. Per patient, regress the
    patient-mean prediction on the shift; segments must be valid at every shift."""
    ok = np.all([r["ok"] for r in rows], 0)
    x = np.array([r["x"] for r in rows])
    P = np.stack(preds, 1)
    sl = []
    for g in np.unique(groups[ok]):
        m = ok & (groups == g)
        if m.sum() >= MIN_SEG:
            sl.append(float(np.polyfit(x, P[m].mean(0), 1)[0]))
    sl = np.array(sl)
    k = int((sl < 0).sum())
    return {"n_subjects": len(sl), "frac_segments": float(ok.mean()),
            "frac_negative": float(k / len(sl)) if len(sl) else float("nan"),
            "median": float(np.median(sl)) if len(sl) else float("nan"),
            "p_sign": float(binomtest(k, len(sl), 0.5).pvalue) if len(sl) else float("nan"),
            "slopes": sl.tolist()}


def data_slopes(Xs, ys, gs, fs, ti=1):
    """Within-patient BP-vs-PAT slopes from the data itself (no model), PPG and ABP PAT.
    ti selects the label column: 0 = SBP, 1 = DBP."""
    out = {}
    for name, ch in (("ppg_pat", PPG), ("abp_pat", ABP)):
        pat = np.concatenate([mechlib.compute_ptt(X, fs, ecg_pos=ECG, ppg_pos=ch) for X in Xs]) * 1000
        y = np.concatenate(ys)[:, ti]; g = np.concatenate(gs)
        sl, xs, yy = [], [], []
        for s in np.unique(g):
            m = (g == s) & np.isfinite(pat)
            if m.sum() >= 5 and np.ptp(pat[m]) > 5:
                sl.append(float(np.polyfit(pat[m], y[m], 1)[0]))
                xs += list(pat[m] - pat[m].mean()); yy += list(y[m] - y[m].mean())
        sl = np.array(sl); k = int((sl < 0).sum())
        out[name] = {"n_subjects": len(sl), "frac_negative": float(k / len(sl)),
                     "median": float(np.median(sl)), "p_sign": float(binomtest(k, len(sl), 0.5).pvalue),
                     "within_r": float(np.corrcoef(xs, yy)[0, 1]), "slopes": sl.tolist()}
        print(f"[data] {name}: {len(sl)} patients, {k/len(sl):.0%} negative "
              f"(p={out[name]['p_sign']:.2g}), median {np.median(sl):+.3f} mmHg/ms, "
              f"within-patient r {out[name]['within_r']:+.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--target", choices=["dbp", "sbp"], default="dbp")
    args = ap.parse_args()
    ti = {"sbp": 0, "dbp": 1}[args.target]
    # DBP keeps the original file names; SBP gets its own (no original SBP checkpoint exists)
    suffix = "" if args.target == "dbp" else f"_{args.target}"

    deep = mechlib.load_mini(str(DATA / "vitaldb_mini_deep.npz")); fs = int(deep["fs"])
    two = [ECG, PPG]
    Xtr, ytr = mechlib.normalize(deep["Xtr"][:, :, two]), deep["ytr"][:, ti]
    Xva, yva = mechlib.normalize(deep["Xva"][:, :, two]), deep["yva"][:, ti]
    mini = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))
    Xte, yte, gte = mechlib.normalize(mini["Xte"][:, :, two]), mini["yte"][:, ti], mini["gte"]
    base_mae = float(np.abs(yte - ytr.mean()).mean())
    print(f"[data] train {len(Xtr)} (deep file), audit {len(Xte)} segments / {len(np.unique(gte))} "
          f"patients (mini file); predict-the-mean MAE {base_mae:.2f}", flush=True)

    res = {"target": args.target, "baseline_mae": base_mae, "models": {}}
    res["data"] = data_slopes([deep[k] for k in ("Xtr", "Xva", "Xte")],
                              [deep[k] for k in ("ytr", "yva", "yte")],
                              [deep[k] for k in ("gtr", "gva", "gte")], fs, ti)

    base_pat, rows = shifted_inputs(Xte, fs)
    med, b0 = float(np.nanmedian(base_pat)), float(yte[np.isfinite(base_pat)].mean())

    def audit(name, pred_rows, mae=None):
        r = {"test_mae": mae}
        for mode in SHIFTS:
            r[mode] = subject_slopes(pred_rows(mode), rows[mode], gte)
        res["models"][name] = r
        print(f"{name:12s} MAE {mae if mae is not None else float('nan'):5.2f} | " + " | ".join(
            f"{m} {r[m]['frac_negative']:.0%} neg, med {r[m]['median']:+.4f}" for m in SHIFTS), flush=True)

    # positive / inverted controls: BP = a/PAT, computed from the detector's own PAT
    for name, a in (("textbook", 3.0), ("inverted", -3.0)):
        def ctl(mode, a=a):
            return [a / np.where(np.isfinite(rw["pat"]), rw["pat"], med) + b0 - a / med for rw in rows[mode]]
        audit(name, ctl)

    nets = {}
    if args.target == "dbp":
        ck = torch.load(str(DATA / "dbp_transformer.pt"), map_location=DEV, weights_only=False)
        net0 = mechlib.WaveTransformer(**ck["config"]).to(DEV); net0.load_state_dict(ck["state_dict"])
        nets["original"] = net0
    for s in range(args.seeds):
        net, hist = train(Xtr, ytr, Xva, yva, seed=s)
        torch.save({"state_dict": net.state_dict(), "history": hist, "config": net.cfg},
                   DATA / f"{args.target}_transformer_seed{s}.pt")
        nets[f"seed{s}"] = net
    for name, net in nets.items():
        mae = float(np.abs(predict(net, Xte) - yte).mean())
        audit(name, lambda mode, net=net: [predict(net, rw["X"]) for rw in rows[mode]], mae)

    out = DATA / f"real_audit_controls{suffix}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"[done] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
