"""ppg_audit.py -- roll-audit for PPG-ONLY models.

The PAT roll-audit needs ECG as a timing reference. Without ECG, we instead perturb the
INTRA-BEAT morphological quantities the PPG-only model actually relies on (identified from its
LightGBM feature importance: diastolic width, reflection/augmentation timing, rise time), each
with a signed governing law. This is the PPG-only analogue of the PAT roll-audit.

Perturbations (from physics_audit, validated to move their target cue):
  decay  -- stretch/compress the diastolic runoff (relates to compliance/SVR)
  aix    -- shift the reflected/augmentation wave (reflection timing; stiffer = earlier = higher BP)
  rise   -- warp the systolic upstroke duration (faster = stiffer = higher BP)

Reports each model's response slope + sign, parallel to the ECG+PPG mechanism table.
"""
import json
from pathlib import Path

import numpy as np
import torch

import mechlib
import physics_audit as pa
import ood_benchmark as ob
from mechlib import ECG, PPG

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["lenet1d", "inception1d", "xresnet1d50", "xresnet1d101", "transformer"]
CUES = ["decay", "aix", "rise"]                            # PPG-intrinsic, no ECG needed


def main():
    d = mechlib.load_mini("data/vitaldb_full_calfree.npz"); fs = d["fs"]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["gte"]), 1000, replace=False))
    X = mechlib.normalize(d["Xte"][sel][:, :, [PPG]])       # PPG-only, channel 0

    results = {}
    for mk in MODELS:
        ck = torch.load(ROOT / "models" / f"{mk}_ppg.pt", map_location=DEVICE, weights_only=False)
        m = ob.build_model(mk, n_ch=1, L=1250); m.load_state_dict(ck["state_dict"])
        m.to(DEVICE).eval()
        mu, sd = ck["mu"], ck["sd"]
        fn = lambda Xr: ob.predict(m, Xr, DEVICE, mu, sd)
        batt = pa.run_battery(fn, X, fs, cues=CUES, target=1, has_ecg=False, n_max=800)
        results[mk] = {c: {"slope": batt[c]["slope"], "expect": batt[c]["expect"],
                           "sign_ok": batt[c]["sign_ok"], "resp_range": batt[c]["resp_range"]}
                       for c in CUES}
        row = "  ".join(f"{c} {batt[c]['slope']:+.1f}({'ok' if batt[c]['sign_ok'] else 'x' if batt[c]['sign_ok'] is False else '-'})"
                        for c in CUES)
        print(f"[ppg-audit] {mk:14s} {row}")

    (ROOT / "data" / "ppg_audit.json").write_text(json.dumps(results, indent=2, default=float))
    print("[done] data/ppg_audit.json")


if __name__ == "__main__":
    main()
