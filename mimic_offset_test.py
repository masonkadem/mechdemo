"""mimic_offset_test.py -- is the VitalDB -> MIMIC-BP failure partly a device-timing problem?

VitalDB's PPG channel lags its arterial line by ~476 ms, MIMIC-BP's by ~279 ms
(pat_bp_quality.py). A model trained on VitalDB therefore sees MIMIC's PPG ~200 ms earlier than
anything in training; a model that uses arrival time the textbook way reads that as a shorter
PAT and predicts higher pressure. Test: run the VitalDB-trained 1-layer focus models (12k and
140k segments, SBP and DBP, 3 seeds; focus_models.py) on all of MIMIC-BP with the PPG
  as recorded       (0 ms)
  device-matched    shifted later by the difference in PPG-minus-arterial-line delay, so only
                    the instrument offset is removed and physiology is left alone
  total-matched     shifted later by the difference in ECG->PPG delay (also removes a real
                    physiological difference; shown for comparison)
and report MAE, bias, and within-patient tracking (correlation after removing each patient's
mean). Prediction: matching the device offset shrinks the bias, most for law-following models.
Writes data/mimic_offset_test.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
import physics_audit as pa
from mechlib import _shift_channel
from quick_scaling import within_r
from real_audit_controls import DEV, predict

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOCAL = Path.home() / "mechdemo_local"
FS = 125


def main():
    tv = json.loads((DATA / "pat_bp_quality_vitaldb.json").read_text())["timing_ms"]
    tm = json.loads((DATA / "pat_bp_quality_mimic.json").read_text())["timing_ms"]
    device = tv["ppg_minus_abp_median"] - tm["ppg_minus_abp_median"]
    total = tv["ppg_pat_median"] - tm["ppg_pat_median"]
    conds = {"as recorded": 0.0, "device-matched": device, "total-matched": total}
    print("[shift] " + ", ".join(f"{k} {v:+.0f} ms" for k, v in conds.items()), flush=True)

    m = pa.load_mimic_bp(str(ROOT.parent / "BP"), channels=("ecg", "ppg"))
    X30, y, g = m["X"].astype(np.float32), m["y"], m["g"]
    print(f"[mimic] {len(y)} segments, {len(np.unique(g))} patients", flush=True)
    big = np.load(LOCAL / "vitaldb_train_pulsedb.npz")
    mu = {"sbp": float(big["ytr"][:, 0].mean()), "dbp": float(big["ytr"][:, 1].mean())}
    focus = json.loads((DATA / "focus_models.json").read_text())

    keys = [f"{t}_1L_{d}" + s for t in ("sbp", "dbp") for d in ("12k", "140k") for s in ("", "_s1", "_s2")]
    nets = {}
    for k in keys:
        ck = torch.load(str(DATA / f"focus_{k}.pt"), map_location=DEV, weights_only=False)
        net = mechlib.WaveTransformer(**ck["config"]).to(DEV); net.load_state_dict(ck["state_dict"]); net.eval()
        nets[k] = net

    res = {"shift_ms": conds, "models": {k: {"audit_textbook": focus[k]["calfree"]["ppg"]["frac_negative"]}
                                         for k in keys}}
    for cname, ms in conds.items():
        Xc = X30.copy()
        Xc[:, :, 1] = _shift_channel(X30[:, :, 1], int(round(ms * FS / 1000)))
        W, k_ = pa.window_segments(Xc, 1250); W = mechlib.normalize(W)
        del Xc
        for k in keys:
            t, ti = k[:3], {"sbp": 0, "dbp": 1}[k[:3]]
            p = predict(nets[k], W).reshape(-1, k_).mean(1)
            e = p - y[:, ti]
            res["models"][k][cname] = {"mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                                       "within_r": within_r(p, y[:, ti], g),
                                       "mean_mae": float(np.abs(y[:, ti] - mu[t]).mean())}
        del W
        print(f"[done] {cname}", flush=True)

    print(f"\n{'model':16s} {'audit':>5s} | " + " | ".join(f"{c:^26s}" for c in conds))
    print(f"{'':16s} {'':5s} | " + " | ".join(f"{'MAE':>6s} {'bias':>6s} {'within r':>9s}" for _ in conds))
    for k in keys:
        r = res["models"][k]
        print(f"{k:16s} {r['audit_textbook']:5.0%} | " + " | ".join(
            f"{r[c]['mae']:6.2f} {r[c]['bias']:+6.1f} {r[c]['within_r']:+9.3f}" for c in conds))
    for t in ("sbp", "dbp"):
        print(f"{t.upper()} predict-the-VitalDB-mean MAE on MIMIC: {res['models'][f'{t}_1L_12k']['as recorded']['mean_mae']:.2f}")
    (DATA / "mimic_offset_test.json").write_text(json.dumps(res, indent=1))
    print("[done] data/mimic_offset_test.json")


if __name__ == "__main__":
    main()
