"""mimic_models.py -- the VitalDB result, replicated on MIMIC-BP (a second hospital).

1-layer transformers (the focus-model recipe) trained on MIMIC-BP alone, SBP and DBP, 3 seeds,
on 12k windows (30 epochs, LR 2e-3) and on all training windows (12 epochs, LR 5e-4). MIMIC's
30 s records are cut into 10 s windows; patients are split train / validation (150) / test (150)
with no overlap. Each model is audited on 150 held-out MIMIC patients (20 windows each) and,
for transfer, on the 144 held-out VitalDB patients used throughout. The data's own law on MIMIC
is in data/pat_bp_quality_mimic.json (SBP 80%, DBP 72% of patients).
Writes data/mimic_models.json.
"""
import json
from pathlib import Path

import numpy as np

import mechlib
import physics_audit as pa
from mechlib import ECG, PPG
from focus_models import audit, train
from real_audit_controls import predict, shifted_inputs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOCAL = Path.home() / "mechdemo_local"
FS = 125


def main():
    m = pa.load_mimic_bp(str(ROOT.parent / "BP"), channels=("ecg", "ppg"))
    W, k = pa.window_segments(m["X"], 1250)
    W = mechlib.normalize(W)
    y = np.repeat(m["y"], k, axis=0).astype(np.float32); g = np.repeat(m["g"], k)
    pats = np.random.default_rng(0).permutation(np.unique(g))
    te_p, va_p = pats[:150], pats[150:300]
    tr = ~np.isin(g, np.concatenate([te_p, va_p])); va = np.isin(g, va_p)
    rng = np.random.default_rng(1)
    te_idx = np.sort(np.concatenate([rng.permutation(np.where(g == p)[0])[:20] for p in te_p]))
    Xtr_all, ytr_all = W[tr], y[tr]
    Xva, yva = W[va], y[va]
    Xte, yte, gte = W[te_idx], y[te_idx], g[te_idx]
    sub = np.sort(np.random.default_rng(0).choice(len(Xtr_all), 12000, replace=False))
    print(f"[mimic] train {tr.sum()} windows / {len(np.unique(g[tr]))} patients, val {va.sum()}, "
          f"audit {len(Xte)} windows / {len(te_p)} patients", flush=True)
    _, rows_m = shifted_inputs(Xte, FS)

    mini = mechlib.load_mini(str(LOCAL / "vitaldb_mini.npz"))
    Xv, gv = mechlib.normalize(mini["Xte"][:, :, [ECG, PPG]]), mini["gte"]
    _, rows_v = shifted_inputs(Xv, FS)

    out = DATA / "mimic_models.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for size, (idx, epochs, step, lr) in {"12k": (sub, 30, 10, 2e-3),
                                          "full": (slice(None), 12, 4, 5e-4)}.items():
        for target, ti in (("sbp", 0), ("dbp", 1)):
            for seed in (0, 1, 2):
                key = f"{target}_1L_{size}_s{seed}"
                if key in res:
                    continue
                Xtr, ytr = Xtr_all[idx], ytr_all[idx, ti]
                net = mechlib.WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=1, L=1250)
                net, hist = train(net, Xtr, ytr, Xva, yva[:, ti], epochs, step, key, lr=lr, seed=seed)
                mu = float(ytr.mean())
                r = {"target": target, "size": size, "seed": seed, "n_train": int(len(Xtr)),
                     "history": hist,
                     "mimic": {"mae": float(np.abs(predict(net, Xte) - yte[:, ti]).mean()),
                               "mean_mae": float(np.abs(yte[:, ti] - mu).mean()),
                               **audit(net, rows_m, gte, ("ppg", "both"))},
                     "vitaldb": {"mae": float(np.abs(predict(net, Xv) - mini["yte"][:, ti]).mean()),
                                 "mean_mae": float(np.abs(mini["yte"][:, ti] - mu).mean()),
                                 **audit(net, rows_v, gv, ("ppg", "both"))}}
                res[key] = r
                out.write_text(json.dumps(res, indent=1))
                a, b = r["mimic"], r["vitaldb"]
                print(f"[result] {key}: MIMIC MAE {a['mae']:.2f} vs mean {a['mean_mae']:.2f}, textbook "
                      f"{a['ppg']['frac_negative']:.0%} | on VitalDB MAE {b['mae']:.2f} vs {b['mean_mae']:.2f}, "
                      f"textbook {b['ppg']['frac_negative']:.0%}", flush=True)
    print("[done] data/mimic_models.json")


if __name__ == "__main__":
    main()
