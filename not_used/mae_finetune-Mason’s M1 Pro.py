"""mae_finetune.py -- does a more faithful representation predict blood pressure better?

The probe result says the MAE encodes arterial arrival time at r = 0.43 where a supervised model
with the same trunk reaches 0.20. That is a statement about what is represented, not about
accuracy. This asks the second question.

Four arms, all sharing one encoder architecture so the comparison isolates the training recipe:

  scratch          random init, trained on BP labels          -- the usual baseline
  linear probe     MAE frozen, only a small head trained      -- is the representation LINEARLY useful?
  fine-tune        MAE init, everything trained on BP labels  -- is it a better starting point?
  fine-tune (lo)   same, encoder at a tenth of the head's lr  -- the usual recipe, which preserves
                                                                pretrained structure

Three seeds each, because a single seed already produced one result in this project that did not
survive replication (PTT supervision, 66% -> 76%, which became 68 -> 59 -> 61 across seeds).

Set expectations against the ceiling: perfectly measured arrival time is worth 0.23 mmHg
[0.13, 0.32] over predicting a subject's own mean. If the MAE's representational advantage
converts fully into accuracy, the gain should be a fraction of a millimetre. A null here is the
informative outcome -- it would mean a representation can be more faithful without being more
accurate, which is this project's argument in its cleanest form.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import mechlib
import eval_protocols as ep
from mae_probe import MAE, Supervised, PATCH, L, FS
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_from_mae(mae_state, freeze):
    """A supervised model whose encoder is the MAE's, optionally frozen.

    Supervised and MAE share embed/pos/enc parameter names by construction, so the pretrained
    encoder transfers with a strict load of that subset -- and a strict load is what catches a
    silent shape mismatch that would otherwise leave the encoder randomly initialised while the
    run still completes.
    """
    m = Supervised().to(DEVICE)
    sub = {k: v for k, v in mae_state.items()
           if k.startswith(("embed.", "pos", "enc."))}
    missing, unexpected = m.load_state_dict(sub, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    assert all(k.startswith("head.") for k in missing), f"encoder not loaded: {missing[:4]}"
    if freeze:
        for n, p in m.named_parameters():
            if not n.startswith("head."):
                p.requires_grad = False
    return m


def train(model, Xt, yt, epochs, lr, enc_lr=None, bs=256, seed=0):
    torch.manual_seed(seed)
    if enc_lr is None:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    else:
        head = [p for n, p in model.named_parameters() if n.startswith("head.")]
        enc = [p for n, p in model.named_parameters()
               if not n.startswith("head.") and p.requires_grad]
        groups = [{"params": head, "lr": lr}, {"params": enc, "lr": enc_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep_i in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[j].to(DEVICE)), yt[j].to(DEVICE))
            loss.backward(); opt.step()
        sch.step()
    return model


@torch.no_grad()
def predict(model, X, mu, sd, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.tensor(X[i:i + bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out) * sd + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=80000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    ck = torch.load(ROOT / "models" / "mae_probe.pt", map_location=DEVICE, weights_only=False)
    mae_state = ck["mae"]

    d = mechlib.load_mini(str(DATA / "vitaldb_full_calfree.npz"))
    Xtr = mechlib.normalize(d["Xtr"][:args.train_n][:, :, [ECG, PPG]])
    ytr = d["ytr"][:args.train_n]
    mu, sd = ytr.mean(0), ytr.std(0) + 1e-9
    Xt = torch.tensor(Xtr)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)

    g = d["gte"]
    subs = [s for s in np.unique(g) if (g == s).sum() >= 60]
    sel = np.concatenate([np.where(g == s)[0][:60] for s in subs])
    Xte, yte, gte = (mechlib.normalize(d["Xte"][sel][:, :, [ECG, PPG]]),
                     d["yte"][sel], g[sel])
    print(f"[data] train {len(Xt)}, test {len(Xte)} over {len(subs)} subjects", flush=True)

    arms = {
        "scratch": dict(init=None, freeze=False, lr=1e-3, enc_lr=None),
        "linear probe (MAE frozen)": dict(init="mae", freeze=True, lr=3e-3, enc_lr=None),
        "fine-tune (MAE)": dict(init="mae", freeze=False, lr=1e-3, enc_lr=None),
        "fine-tune (MAE, low enc lr)": dict(init="mae", freeze=False, lr=1e-3, enc_lr=1e-4),
    }

    res = {}
    print(f"\n{'arm':30s} {'seed':>4s} {'ID DBP':>8s} {'k=5':>7s} {'k=20':>7s}")
    print("-" * 60)
    for name, cfg in arms.items():
        rows = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            m = (build_from_mae(mae_state, cfg["freeze"]) if cfg["init"] == "mae"
                 else Supervised().to(DEVICE))
            m = train(m, Xt, yt, args.epochs, cfg["lr"], cfg["enc_lr"], seed=seed)
            p = predict(m, Xte, mu, sd)[:, 1]
            idm = float(np.abs(p - yte[:, 1]).mean())
            cur = ep.anchor_curve(p, yte[:, 1], gte, min_seg=40)
            rows.append({"seed": seed, "id": idm, "k5": cur[5], "k20": cur[20]})
            print(f"{name:30s} {seed:4d} {idm:8.2f} {cur[5]:7.2f} {cur[20]:7.2f}", flush=True)
            res[name] = {"runs": rows}
            (DATA / "mae_finetune.json").write_text(json.dumps(res, indent=2, default=float))
        a = np.array([[r["id"], r["k5"], r["k20"]] for r in rows])
        res[name] = {"runs": rows, "id_mean": float(a[:, 0].mean()),
                     "id_sd": float(a[:, 0].std()), "k20_mean": float(a[:, 2].mean()),
                     "k20_sd": float(a[:, 2].std())}
        print(f"{'':30s} {'mean':>4s} {a[:,0].mean():8.2f} {a[:,1].mean():7.2f} "
              f"{a[:,2].mean():7.2f}   (sd {a[:,0].std():.2f})", flush=True)

    print(f"\n{'arm':30s} {'ID DBP':>14s} {'k=20':>14s}")
    base = res["scratch"]
    for name, r in res.items():
        if "id_mean" not in r:
            continue
        print(f"{name:30s} {r['id_mean']:8.2f} +/-{r['id_sd']:.2f} "
              f"{r['k20_mean']:8.2f} +/-{r['k20_sd']:.2f}")
    print(f"\nCeiling for context: perfectly measured arrival time is worth 0.23 mmHg "
          f"[0.13, 0.32].")
    print("A difference here smaller than the seed spread is not a difference.")
    (DATA / "mae_finetune.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/mae_finetune.json")


if __name__ == "__main__":
    main()
