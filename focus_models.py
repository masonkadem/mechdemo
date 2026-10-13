"""focus_models.py -- the four models the story uses, trained properly, with learning curves.

{1-layer, 3-layer WaveTransformer} x {12k, 140k training segments} x {SBP, DBP}, seed 0.
Both sizes come from the same file (vitaldb_train_pulsedb.npz: every 3rd segment of the official
PulseDB VitalDB training set, 1,164 patients), so the only difference is the amount of data:
  12k  : a random 12,000-segment sample, the original recipe (30 epochs, LR halved every 10)
  140k : all of it, --big-epochs epochs at LR 5e-4, halved at the same fractions of training
Every epoch logs train and validation MAE, so a collapsed or undertrained run is visible.
Each model is audited on the 144 held-out patients (PPG-earlier, ECG-later, both shifted) and on
150 patients it trained on (PulseDB CalBased test segments).
Results are written after every model to data/focus_models.json; checkpoints to
data/focus_{target}_{depth}L_{data}.pt.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import mechlib
from mechlib import ECG, PPG
from real_audit_controls import DEV, predict, shifted_inputs, subject_slopes
from quick_scaling import load_calbased, within_r

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
# Waveform files are read from outside OneDrive: OneDrive replaced two of the .npz files in data/
# with copies that fail their CRC check (vitaldb_full_calfree, vitaldb_mini_deep).
LOCAL = Path.home() / "mechdemo_local"


def train(net, Xtr, ytr, Xva, yva, epochs, step, tag, bs=128, lr=2e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.5)
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    hist = {"train_mae": [], "val_mae": []}
    best, state = float("inf"), None
    for ep in range(epochs):
        t0 = time.time()
        net.train(); perm = torch.randperm(len(Xt), device=DEV); err = []
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            out = net(Xt[b])
            loss = ((out - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            err.append(float((out.detach() - yt[b]).abs().mean()))
        sched.step()
        vm = float(np.abs(predict(net, Xva) - yva).mean())
        hist["train_mae"].append(float(np.mean(err))); hist["val_mae"].append(vm)
        if not np.isfinite(vm):                        # diverged: stop, keep the record
            print(f"  [{tag}] diverged at epoch {ep + 1}", flush=True)
            if state is None:
                state = {k: v.clone() for k, v in net.state_dict().items()}
            break
        if vm < best:
            best, state = vm, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  [{tag}] epoch {ep + 1}/{epochs}  train {hist['train_mae'][-1]:.2f}  val {vm:.2f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    net.load_state_dict(state); net.eval()
    return net, hist


def audit(net, rows, groups, modes):
    return {m: {k: v for k, v in subject_slopes([predict(net, rw["X"]) for rw in rows[m]],
                                                rows[m], groups).items() if k != "slopes"}
            for m in modes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-epochs", type=int, default=12)
    ap.add_argument("--only", default="", help="retrain just this key, e.g. sbp_3L_140k")
    ap.add_argument("--lr", type=float, default=2e-3, help="only used with --only")
    ap.add_argument("--seeds", default="0", help="comma-separated; seed 0 keeps the plain key")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    fs = 125; two = [ECG, PPG]
    mini = mechlib.load_mini(str(LOCAL / "vitaldb_mini.npz"))
    Xcf, gcf = mechlib.normalize(mini["Xte"][:, :, two]), mini["gte"]
    _, rows_cf = shifted_inputs(Xcf, fs)
    out = DATA / "focus_models.json"
    res = json.loads(out.read_text()) if out.exists() else {}

    big = mechlib.load_mini(str(LOCAL / "vitaldb_train_pulsedb.npz"))
    Xall, Xva = mechlib.normalize(big["Xtr"]), mechlib.normalize(big["Xva"])
    sub = np.sort(np.random.default_rng(0).choice(len(Xall), 12000, replace=False))
    print(f"[data] 140k: {len(Xall)} segments / {len(np.unique(big['gtr']))} patients; 12k sample covers "
          f"{len(np.unique(big['gtr'][sub]))} of them", flush=True)
    Xcb, ycb, gcb = load_calbased()
    _, rows_cb = shifted_inputs(Xcb, fs)
    for data_name in ("12k", "140k"):
        if data_name == "12k":
            Xtr, ytr_all = Xall[sub], big["ytr"][sub]
            epochs, step = 30, 10
        else:
            Xtr, ytr_all = Xall, big["ytr"]
            epochs = args.big_epochs; step = max(epochs // 3, 1)
        d = {"ytr": ytr_all, "yva": big["yva"]}
        for depth in (1, 3):
            for (target, ti), seed in [(t, s) for s in seeds for t in (("sbp", 0), ("dbp", 1))]:
                key = f"{target}_{depth}L_{data_name}" + (f"_s{seed}" if seed else "")
                # 140k models use 5e-4: at 2e-3 the 3-layer SBP model never left the mean and a
                # 1-layer SBP seed diverged. 12k models keep the original recipe's 2e-3.
                lr = args.lr if args.only else (5e-4 if data_name == "140k" else 2e-3)
                if args.only and key not in args.only.split(","):
                    continue
                if key in res and not args.only:
                    print(f"[skip] {key} already done", flush=True); continue
                print(f"[train] {key}: {len(Xtr)} segments, {epochs} epochs", flush=True)
                net = mechlib.WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=depth, L=1250)
                if key in res:                          # keep the failed run on record
                    old = res.pop(key)
                    res[f"{key}_lr{old.get('lr', 2e-3):g}"] = old
                net, hist = train(net, Xtr, d["ytr"][:, ti], Xva, d["yva"][:, ti], epochs, step, key,
                                  lr=lr, seed=seed)
                torch.save({"state_dict": net.state_dict(), "history": hist, "config": net.cfg},
                           DATA / f"focus_{key}.pt")
                mu = float(d["ytr"][:, ti].mean()); yte = mini["yte"][:, ti]
                r = {"target": target, "depth": depth, "data": data_name, "epochs": epochs, "lr": lr,
                     "seed": seed,
                     "history": hist, "val_mean_mae": float(np.abs(d["yva"][:, ti] - mu).mean()),
                     "calfree": {"mae": float(np.abs(predict(net, Xcf) - yte).mean()),
                                 "mean_mae": float(np.abs(yte - mu).mean()),
                                 **audit(net, rows_cf, gcf, ("ppg", "ecg", "both"))}}
                if True:                                # both sizes trained on the CalBased patients
                    p = predict(net, Xcb)
                    r["calbased"] = {"mae": float(np.abs(p - ycb[:, ti]).mean()),
                                     "mean_mae": float(np.abs(ycb[:, ti] - mu).mean()),
                                     "within_r": within_r(p, ycb[:, ti], gcb),
                                     **audit(net, rows_cb, gcb, ("ppg", "both"))}
                res[key] = r
                out.write_text(json.dumps(res, indent=1))
                c = r["calfree"]
                print(f"[result] {key}: MAE {c['mae']:.2f} vs mean {c['mean_mae']:.2f} | textbook sign "
                      f"{c['ppg']['frac_negative']:.0%} (p={c['ppg']['p_sign']:.1g}) | PPG {c['ppg']['median']:+.4f}"
                      f" ECG {c['ecg']['median']:+.4f} both {c['both']['median']:+.4f}"
                      + (f" | seen patients {r['calbased']['ppg']['frac_negative']:.0%}"
                         f" (within r {r['calbased']['within_r']:+.2f})" if "calbased" in r else ""),
                      flush=True)
    print("[done] data/focus_models.json")


if __name__ == "__main__":
    main()
