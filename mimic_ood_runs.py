"""mimic_ood_runs.py -- do models that follow the arrival-time law transfer better out of distribution?

Every audited VitalDB transformer (data/{dbp,sbp}_transformer*.pt) is run on MIMIC-BP (ICU, a
different hospital, monitor and population; ../BP). MIMIC segments are 30 s; each is cut into
three 10 s windows, every window is z-scored per channel exactly as in training, and the three
predictions are averaged against the segment's label.

Per run: in-distribution MAE (VitalDB test), out-of-distribution MAE, the predict-the-VitalDB-mean
baseline on MIMIC, and the run's share of patients with the expected sign from
real_audit_controls*.json. Writes data/mimic_ood_runs.json.
"""
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
import physics_audit as pa

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(net, X, bs=1024):
    return np.concatenate([net(torch.tensor(X[i:i + bs], device=DEV)).cpu().numpy()
                           for i in range(0, len(X), bs)])


def main():
    m = pa.load_mimic_bp(str(ROOT.parent / "BP"), channels=("ecg", "ppg"))
    W, k = pa.window_segments(m["X"], 1250)                 # (N*3, 1250, 2), record-major
    W = mechlib.normalize(W)
    y = m["y"]
    print(f"[mimic] {len(y)} segments, {len(np.unique(m['g']))} patients, {k} windows each", flush=True)

    deep = mechlib.load_mini(str(DATA / "vitaldb_mini_deep.npz"))
    out = {"n_segments": int(len(y)), "n_patients": int(len(np.unique(m["g"]))), "runs": {}}
    sets = [("dbp", 1, "real_audit_controls.json", ""), ("sbp", 0, "real_audit_controls_sbp.json", ""),
            ("dbp", 1, "real_audit_controls_full.json", "_full"),
            ("sbp", 0, "real_audit_controls_sbp_full.json", "_full")]
    for target, ti, ctl_file, tag in sets:
        if not (DATA / ctl_file).exists():
            continue
        ctl = json.loads((DATA / ctl_file).read_text())
        mean_pred = float(np.abs(y[:, ti] - deep["ytr"][:, ti].mean()).mean())
        out[f"{target}_mean_predictor_mae"] = mean_pred
        print(f"[{target}] predict-the-VitalDB-mean MAE on MIMIC {mean_pred:.2f}", flush=True)
        for name in ctl["models"]:
            if name in ("textbook", "inverted"):
                continue
            ck_name = "dbp_transformer.pt" if name == "original" else f"{target}{tag}_transformer_{name}.pt"
            ck = torch.load(str(DATA / ck_name), map_location=DEV, weights_only=False)
            net = mechlib.WaveTransformer(**ck["config"]).to(DEV); net.load_state_dict(ck["state_dict"])
            net.eval()
            p = predict(net, W).reshape(-1, k).mean(1)
            e = p - y[:, ti]
            # within-patient tracking: remove each patient's mean prediction and mean label, so a
            # population offset (which calibration fixes) cannot count for or against the model
            pc, yc = p.copy(), y[:, ti].astype(float).copy()
            for s in np.unique(m["g"]):
                ms = m["g"] == s
                pc[ms] -= pc[ms].mean(); yc[ms] -= yc[ms].mean()
            r = {"target": target + tag,
                 "id_mae": ctl["models"][name]["test_mae"],
                 "ood_mae": float(np.abs(e).mean()),
                 "ood_bias": float(e.mean()),
                 "ood_mae_debiased": float(np.abs(e - e.mean()).mean()),
                 "ood_within_r": float(np.corrcoef(pc, yc)[0, 1]),
                 "ood_within_mae": float(np.abs(pc - yc).mean()),
                 "expected_sign": ctl["models"][name]["ppg"]["frac_negative"]}
            out["runs"][f"{target}{tag}_{name}"] = r
            print(f"  {name:9s} ID {r['id_mae']:.2f}  OOD {r['ood_mae']:.2f} (bias {r['ood_bias']:+.1f},"
                  f" debiased {r['ood_mae_debiased']:.2f})  within-patient r {r['ood_within_r']:+.3f}"
                  f" MAE {r['ood_within_mae']:.2f}  expected sign {r['expected_sign']:.0%}", flush=True)
        out[f"{target}_within_mae_flat"] = float(np.mean([np.abs(y[m["g"] == s, ti] - y[m["g"] == s, ti].mean()).mean()
                                                         * (m["g"] == s).sum() for s in np.unique(m["g"])]) * len(np.unique(m["g"])) / len(y))
        print(f"  within-patient MAE of a flat (per-patient mean) prediction: "
              f"{out[f'{target}_within_mae_flat']:.2f}")
        rr = [v for v in out["runs"].values() if v["target"] == target + tag]
        f = [v["expected_sign"] for v in rr]
        for key in ("ood_mae", "ood_mae_debiased", "ood_within_r", "id_mae"):
            print(f"  r(expected sign, {key}) = {np.corrcoef(f, [v[key] for v in rr])[0, 1]:+.2f}")
        pen = [v["ood_mae"] - v["id_mae"] for v in rr]
        print(f"  r(expected sign, OOD penalty) = {np.corrcoef(f, pen)[0, 1]:+.2f}", flush=True)
    (DATA / "mimic_ood_runs.json").write_text(json.dumps(out, indent=1))
    print("[done] data/mimic_ood_runs.json")


if __name__ == "__main__":
    main()
