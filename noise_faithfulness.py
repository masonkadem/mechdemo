"""noise_faithfulness.py -- does training with realistic PPG-artifact augmentation improve the
model's MECHANISM (not just accuracy)?

Trains two matched XResNet1d50 (ECG+PPG) models on VitalDB:
  clean : no augmentation
  aug   : each batch's PPG randomly corrupted with realistic wearable artifacts
          (motion spikes, baseline wander, amplitude drift, additive noise)

Then compares on three axes:
  1. accuracy  : ID (VitalDB test) + OOD (MIMIC-BP)
  2. mechanism : roll-audit dDBP/dPTT slope + fraction-correct-sign
  3. shortcut  : cardiac-period vs PAT linear-probe decodability

Hypothesis: augmentation forces reliance on robust morphology, REDUCING the cardiac-period
shortcut and IMPROVING the roll-audit -- a causal intervention on faithfulness.

    python noise_faithfulness.py --epochs 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
import ood_benchmark as ob
import physics_audit as pa
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def augment(Xb, fs, rng):
    """Apply realistic PPG artifacts to a (B, L, C) batch's PPG channel (index 1). Each sample
    gets a random subset at random strength -- mimics wearable motion/contact noise."""
    X = Xb.copy(); L = X.shape[1]; t = np.arange(L) / fs
    for i in range(len(X)):
        p = X[i, :, PPG]
        s = p.std() + 1e-6
        if rng.random() < 0.6:                              # baseline wander
            f = rng.uniform(0.05, 0.4); ph = rng.uniform(0, 2 * np.pi)
            p = p + rng.uniform(0.2, 0.8) * s * np.sin(2 * np.pi * f * t + ph)
        if rng.random() < 0.6:                              # additive noise
            p = p + rng.uniform(0.03, 0.15) * s * rng.standard_normal(L)
        if rng.random() < 0.4:                              # amplitude drift
            p = p * (1 + rng.uniform(-0.3, 0.3) * np.sin(2 * np.pi * rng.uniform(0.02, 0.1) * t))
        if rng.random() < 0.3:                              # motion spikes
            k = rng.integers(1, 4)
            for _ in range(k):
                c = rng.integers(0, L); w = rng.integers(3, 15)
                p[max(0, c - w):c + w] += rng.uniform(-2, 2) * s
        X[i, :, PPG] = p
    return X.astype(np.float32)


def train(model, sp, fs, device, epochs, bs=256, lr=1e-3, aug=False, seed=0):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * max(1, len(sp["train"]["X"]) // bs))
    lossf = torch.nn.SmoothL1Loss()
    mu, sd = sp["train"]["y"].mean(0), sp["train"]["y"].std(0) + 1e-8
    Xtr = sp["train"]["X"]; ytr = (sp["train"]["y"] - mu) / sd
    rng = np.random.default_rng(seed)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr) - bs + 1, bs):
            idx = perm[s:s + bs]
            xb = augment(Xtr[idx], fs, rng) if aug else Xtr[idx]
            xb = torch.tensor(xb.transpose(0, 2, 1), dtype=torch.float32, device=device)
            yb = torch.tensor(ytr[idx], dtype=torch.float32, device=device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            try:
                sched.step()
            except Exception:
                pass
        vp = ob.predict(model, sp["val"]["X"], device, mu, sd)
        vmae = np.abs(vp - sp["val"]["y"]).mean()
        if vmae < best:
            best = vmae; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, (mu, sd)


def audit_model(model, mu, sd, X, fs, scalars, period):
    fn = lambda Xr: ob.predict(model, Xr, DEVICE, mu, sd)
    aud = mechlib.causal_ptt_audit(None, X, fs, DEVICE, predict_fn=fn, n_max=min(1000, len(X)))
    feats = ob.layer_features(model, "xresnet1d50", X, DEVICE)
    pat_r2 = max(max(mechlib.linear_probe(f, scalars["pat"]) for f in feats.values()), 0)
    per_r2 = max(max(mechlib.linear_probe(f, period) for f in feats.values()), 0)
    return {"slope": aud["dbp"]["dBP_dPTT"], "frac_correct": aud["dbp"]["frac_correct_sign"],
            "pat_probe": pat_r2, "period_probe": per_r2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/vitaldb_full_calfree.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train-n", type=int, default=60000)
    ap.add_argument("--mimic", default="C:/Users/mason/OneDrive - McMaster University/2026/BP")
    args = ap.parse_args()

    d = mechlib.load_mini(args.data); fs = d["fs"]
    tn = args.train_n
    sp = {"train": dict(X=mechlib.normalize(d["Xtr"][:tn][:, :, [ECG, PPG]]), y=d["ytr"][:tn]),
          "val": dict(X=mechlib.normalize(d["Xva"][:8000][:, :, [ECG, PPG]]), y=d["yva"][:8000]),
          "test": dict(X=mechlib.normalize(d["Xte"][:8000][:, :, [ECG, PPG]]), y=d["yte"][:8000])}
    print(f"[nf] train {len(sp['train']['X'])} seg", flush=True)

    # audit data + cues (shared)
    Xa = sp["test"]["X"][:1000]
    scalars = mechlib.compute_scalars(Xa, fs); period = scalars["period"]

    # MIMIC OOD
    m = pa.load_mimic_bp(args.mimic, channels=("ecg", "ppg"), max_patients=200)
    Xm, k = pa.window_segments(m["X"], 1250); ym = np.repeat(m["y"], k, 0); gm = np.repeat(m["g"], k, 0)
    ii = np.sort(np.random.default_rng(0).choice(len(Xm), 4000, False))
    Xm = mechlib.normalize(Xm[ii]); ym, gm = ym[ii], gm[ii]

    results = {}
    for tag, aug in [("clean", False), ("aug", True)]:
        print(f"\n[nf] training {tag} (aug={aug}) ...", flush=True)
        model = ob.build_model("xresnet1d50", n_ch=2, L=1250)
        model, (mu, sd) = train(model, sp, fs, DEVICE, args.epochs, aug=aug)
        idmae = np.abs(ob.predict(model, sp["test"]["X"], DEVICE, mu, sd) - sp["test"]["y"]).mean(0)
        mmae = np.abs(ob.predict(model, Xm, DEVICE, mu, sd) - ym).mean(0)
        au = audit_model(model, mu, sd, Xa, fs, scalars, period)
        results[tag] = {"id_dbp": float(idmae[1]), "mimic_dbp": float(mmae[1]), **au}
        print(f"[nf] {tag}: ID DBP {idmae[1]:.2f}  MIMIC {mmae[1]:.2f}  "
              f"roll {au['slope']:+.1f} ({au['frac_correct']:.0%})  "
              f"probe PAT {au['pat_probe']:.2f} period {au['period_probe']:.2f}", flush=True)

    (ROOT / "data" / "noise_faithfulness.json").write_text(json.dumps(results, indent=2, default=float))
    c, a = results["clean"], results["aug"]
    print("\n[nf] === clean -> aug ===")
    print(f"  ID DBP        {c['id_dbp']:.2f} -> {a['id_dbp']:.2f}")
    print(f"  MIMIC DBP     {c['mimic_dbp']:.2f} -> {a['mimic_dbp']:.2f}")
    print(f"  roll slope    {c['slope']:+.1f} -> {a['slope']:+.1f}  (more negative = more faithful)")
    print(f"  period probe  {c['period_probe']:.2f} -> {a['period_probe']:.2f}  (lower = less shortcut)")
    print("[done] data/noise_faithfulness.json")


if __name__ == "__main__":
    main()
