"""quick_scaling.py -- one bigger model per target, three questions, about 30 minutes.

Trains one DBP and one SBP transformer (seed 0, --epochs, same recipe otherwise) on
data/vitaldb_train_pulsedb.npz (140k segments, 1,164 patients; pulsedb_train_subset.py) and asks:

  1. CalFree   -- the 144 held-out VitalDB patients, same audit as real_audit_controls.py:
                  does more training data make a model follow the arrival-time law more?
  2. CalBased  -- new segments from patients the model trained on (PulseDB's CalBased test
                  file); the model knows each patient's baseline. Does it follow the law, and
                  track within-patient change, when the baseline is known?
  3. MIMIC-BP  -- out of distribution: MAE, bias, within-patient tracking.

Compares with the 12k-segment runs in data/real_audit_controls*.json and data/mimic_ood_runs.json.
Writes data/quick_scaling.json.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import mechlib
import physics_audit as pa
from mechlib import ECG, PPG
from real_audit_controls import DEV, predict, shifted_inputs, subject_slopes, train
import arch_sweep as AS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CALBASED = ROOT.parent / "BP" / "dbdata" / "VitalDB_CalBased_Test_Subset.mat"


def within_r(p, y, g):
    """Correlation of prediction with label after removing each patient's mean from both."""
    pc, yc = p.astype(float).copy(), y.astype(float).copy()
    for s in np.unique(g):
        m = g == s
        pc[m] -= pc[m].mean(); yc[m] -= yc[m].mean()
    return float(np.corrcoef(pc, yc)[0, 1])


def load_calbased(n_patients=150, per_patient=20, seed=0):
    """Segments of patients that were in the TRAINING split (not the 10% held out for validation),
    reproducing pulsedb_train_subset.py's patient split."""
    with h5py.File(CALBASED, "r") as h:
        s = h["Subset"]
        subj = np.array([''.join(chr(c) for c in h[r][:].flatten()) for r in s["Subject"][0]])
        ids = np.unique(subj)
        assert len(ids) == 1293, len(ids)            # CalBased = the 1,293 training patients
        val_ids = ids[np.random.default_rng(0).choice(len(ids), int(round(0.1 * len(ids))), replace=False)]
        rng = np.random.default_rng(seed)
        pats = rng.choice(np.setdiff1d(ids, val_ids), n_patients, replace=False)
        idx = np.sort(np.concatenate([rng.permutation(np.where(subj == p)[0])[:per_patient] for p in pats]))
        sig = s["Signals"]
        X = np.empty((len(idx), 1250, 2), np.float32)
        for a in range(0, sig.shape[2], 8166):
            loc = idx[(idx >= a) & (idx < a + 8166)]
            if len(loc):
                X[np.searchsorted(idx, loc)] = np.transpose(sig[:, :2, a:a + 8166][:, :, loc - a], (2, 0, 1))
        y = np.stack([s["SBP"][0, idx], s["DBP"][0, idx]], 1).astype(np.float32)
    g = np.unique(subj[idx], return_inverse=True)[1]
    return mechlib.normalize(X), y, g


def audit(net, rows, groups):
    pr = {m: [predict(net, rw["X"]) for rw in rows[m]] for m in ("ppg", "both")}
    return {m: subject_slopes(pr[m], rows[m], groups) for m in pr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3, help="transformer layers (3 = the audited model)")
    ap.add_argument("--targets", default="dbp,sbp")
    args = ap.parse_args()
    tag = "" if args.depth == 3 else f"_depth{args.depth}"

    big = mechlib.load_mini(str(DATA / "vitaldb_train_pulsedb.npz")); fs = 125
    Xtr, Xva = mechlib.normalize(big["Xtr"]), mechlib.normalize(big["Xva"])
    mini = mechlib.load_mini(str(DATA / "vitaldb_mini.npz"))
    Xcf, gcf = mechlib.normalize(mini["Xte"][:, :, [ECG, PPG]]), mini["gte"]
    Xcb, ycb, gcb = load_calbased()
    print(f"[data] train {len(Xtr)}, CalFree audit {len(Xcf)} segs / {len(np.unique(gcf))} pts, "
          f"CalBased {len(Xcb)} segs / {len(np.unique(gcb))} seen pts", flush=True)
    _, rows_cf = shifted_inputs(Xcf, fs)
    _, rows_cb = shifted_inputs(Xcb, fs)
    m = pa.load_mimic_bp(str(ROOT.parent / "BP"), channels=("ecg", "ppg"))
    W, k = pa.window_segments(m["X"], 1250); W = mechlib.normalize(W)

    old = {"dbp": json.loads((DATA / "real_audit_controls.json").read_text()),
           "sbp": json.loads((DATA / "real_audit_controls_sbp.json").read_text())}
    res = {}
    for target, ti in [t for t in (("dbp", 1), ("sbp", 0)) if t[0] in args.targets.split(",")]:
        if args.depth == 3:
            net, hist = train(Xtr, big["ytr"][:, ti], Xva, big["yva"][:, ti], seed=0, epochs=args.epochs)
            best_val = float(min(hist["val_mae"]))
        else:
            net = AS.train(AS.build(f"depth{args.depth}"), Xtr, big["ytr"][:, ti], Xva, big["yva"][:, ti],
                           epochs=args.epochs)
            best_val = float(np.abs(predict(net, Xva) - big["yva"][:, ti]).mean())
        torch.save({"state_dict": net.state_dict(), "config": net.cfg},
                   DATA / f"{target}_big{tag}_transformer.pt")
        mu = float(big["ytr"][:, ti].mean())
        r = {"epochs_run": args.epochs, "depth": args.depth, "best_val_mae": best_val}
        # 1. CalFree
        ycf = mini["yte"][:, ti]
        r["calfree"] = {"mae": float(np.abs(predict(net, Xcf) - ycf).mean()),
                        "mean_mae": float(np.abs(ycf - mu).mean()), **audit(net, rows_cf, gcf)}
        # 2. CalBased
        p = predict(net, Xcb)
        r["calbased"] = {"mae": float(np.abs(p - ycb[:, ti]).mean()),
                         "mean_mae": float(np.abs(ycb[:, ti] - mu).mean()),
                         "within_r": within_r(p, ycb[:, ti], gcb), **audit(net, rows_cb, gcb)}
        # 3. MIMIC
        pm = predict(net, W).reshape(-1, k).mean(1)
        r["mimic"] = {"mae": float(np.abs(pm - m["y"][:, ti]).mean()),
                      "mean_mae": float(np.abs(m["y"][:, ti] - mu).mean()),
                      "bias": float((pm - m["y"][:, ti]).mean()),
                      "within_r": within_r(pm, m["y"][:, ti], m["g"])}
        res[target] = r
        oldneg = [v["ppg"]["frac_negative"] for kk, v in old[target]["models"].items()
                  if kk not in ("textbook", "inverted")]
        print(f"\n== {target.upper()} (big model, depth {args.depth}, {args.epochs} epochs)", flush=True)
        for nm in ("calfree", "calbased"):
            q = r[nm]
            print(f"  {nm:8s} MAE {q['mae']:.2f} vs mean {q['mean_mae']:.2f} | expected sign "
                  f"{q['ppg']['frac_negative']:.0%} of {q['ppg']['n_subjects']} pts "
                  f"(p={q['ppg']['p_sign']:.1g}), median {q['ppg']['median']:+.4f}; joint-shift "
                  f"median {q['both']['median']:+.4f}"
                  + (f" | within-patient r {q['within_r']:+.2f}" if "within_r" in q else ""), flush=True)
        q = r["mimic"]
        print(f"  mimic    MAE {q['mae']:.2f} vs mean {q['mean_mae']:.2f}, bias {q['bias']:+.1f}, "
              f"within-patient r {q['within_r']:+.3f}")
        print(f"  12k models, CalFree expected sign: {min(oldneg):.0%}-{max(oldneg):.0%}", flush=True)
    (DATA / f"quick_scaling{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[done] data/quick_scaling{tag}.json")


if __name__ == "__main__":
    main()
