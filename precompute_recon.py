"""Mechanism-attribution battery for a real ECG+PPG blood-pressure model.

Train an ABP-waveform-reconstruction model (ECG+PPG -> BP, with an auxiliary head that
rebuilds the arterial pressure wave), then run the causal donor-swap audit across a battery
of physiologically-defined candidate cues:

    PAT (arrival time) | PPG rise-time | augmentation index | APG stiffness | HR | amplitude

For each cue we report (a) how DECODABLE it is from the features (linear probe R^2) and
(b) whether it is CAUSALLY USED (donor-swap frac in the physiological direction; 0.5 = chance).
The same audit run on the same model discriminates between cues -> it is self-validating.
Candidate shape cues are validated against the ground-truth ABP-derived versions.

Robustness: repeated over several seeds; we report mean +/- std of the per-cue frac.
Writes data/capstone.json + data/capstone.npz (consumed by app_faithfulness.py).
"""
import json, os, numpy as np, torch, torch.nn as nn
import mechlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM, EPOCHS, SEEDS = 1.0, 30, [0, 1, 2]
ECG, PPG, ABP = mechlib.ECG, mechlib.PPG, mechlib.ABP
CUE_CACHE = "data/_cue_cache_deep.npz"          # v2: mini_deep + tangent-foot PAT + extra morphology

d = mechlib.load_mini("data/vitaldb_mini_deep.npz"); fs = int(d["fs"])
Xtr = mechlib.normalize(d["Xtr"][:, :, [ECG, PPG]]); ytr = d["ytr"]
Xte = mechlib.normalize(d["Xte"][:, :, [ECG, PPG]]); yte = d["yte"]; gte = d["gte"]
L = Xtr.shape[1]


def stdz_wave(w):
    w = w.astype(np.float32).copy(); w -= w.mean(1, keepdims=True)
    w /= w.std(1, keepdims=True) + 1e-8; return w


Atr, Ate = stdz_wave(d["Xtr"][:, :, ABP]), stdz_wave(d["Xte"][:, :, ABP])


SCALAR_KEYS = ["pat", "rise", "aix", "apg", "kurt", "notch", "decay", "peak", "hr", "period", "amp"]
ABP_KEYS = mechlib.MORPH_KEYS   # shape cues we can also read off the ground-truth ABP wave


def load_or_compute_cues():
    """Cue extraction is the slow part (savgol + peak-finding over every segment). Cache it
    so a container restart never re-pays it."""
    if os.path.exists(CUE_CACHE):
        z = np.load(CUE_CACHE)
        return ({k: z[f"s_{k}"] for k in SCALAR_KEYS}, {k: z[f"a_{k}"] for k in ABP_KEYS})
    print("computing cues (PPG) + ground-truth morphology (ABP) [one-time cache] ...", flush=True)
    sc = mechlib.compute_scalars(d["Xte"][:, :, [ECG, PPG]], fs, 0, 1)
    ab = mechlib.compute_morphology(d["Xte"], fs, ABP)
    np.savez_compressed(CUE_CACHE, **{f"s_{k}": sc[k] for k in SCALAR_KEYS},
                        **{f"a_{k}": ab[k] for k in ABP_KEYS})
    return sc, ab


scalars_te, morph_abp = load_or_compute_cues()

# pick a clean, legible example segment for the Tab-3 cue-sanity overlay: well-separated beats
# (moderate count -> readable) with clearly visible dicrotic notches, and a physiological PAT.
_fids = [mechlib.segment_fiducials(Xte[i, :, 0], Xte[i, :, 1], fs) for i in range(min(600, len(Xte)))]
_cand = [i for i, f in enumerate(_fids)
         if 7 <= len(f["feet"]) <= 12                          # ~45-75 bpm over the 10 s window
         and len(f["notches"]) >= max(len(f["feet"]) - 2, 3)   # notch found on ~every beat
         and len(f["pat_ms"]) and 180 <= np.median(f["pat_ms"]) <= 340]
EX = max(_cand, key=lambda i: len(_fids[i]["notches"])) if _cand else \
    int(np.argmax([len(f["notches"]) for f in _fids]))
exf = _fids[EX]
print(f"example segment for sanity overlay: idx {EX}  "
      f"({len(exf['feet'])} feet, {len(exf['notches'])} notches, PAT {np.median(exf['pat_ms']):.0f} ms)",
      flush=True)


class ReconCNN(nn.Module):
    """Shared conv encoder -> pooled features -> BP head; a decoder rebuilds the ABP wave."""
    def __init__(self, w=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, w, 7, 2, 3), nn.ReLU(), nn.Conv1d(w, w*2, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(w*2, w*2, 7, 2, 3), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(w*2, 2)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(w*2, w*2, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w*2, w, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w, 1, 7, 2, 3, output_padding=1))

    def encode(self, x): return self.enc(x.transpose(1, 2))
    def features(self, x): return self.pool(self.encode(x)).flatten(1)
    def forward(self, x): return self.head(self.features(x))
    def reconstruct(self, x):
        y = self.dec(self.encode(x))
        return nn.functional.interpolate(y, size=L, mode="linear", align_corners=False).squeeze(1)


def train(seed, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ReconCNN().to(device); opt = torch.optim.Adam(net.parameters(), 2e-3)
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    At = torch.tensor(Atr, device=device)
    bp_var = torch.tensor(ytr.var(0), dtype=torch.float32, device=device)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = perm[s:s+bs]
            z = net.encode(Xt[b]); f = net.pool(z).flatten(1)
            abp = nn.functional.interpolate(net.dec(z), size=L, mode="linear",
                                            align_corners=False).squeeze(1)
            loss = (((net.head(f) - yt[b]) ** 2) / bp_var).mean() + LAM * ((abp - At[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


@torch.no_grad()
def features_and_corr(net):
    F, corrs = [], []
    for s in range(0, len(Xte), 512):
        x = torch.tensor(Xte[s:s+512], device=device)
        F.append(net.features(x).cpu().numpy())
        ah = net.reconstruct(x).cpu().numpy(); at = Ate[s:s+512]
        corrs += [np.corrcoef(ah[i], at[i])[0, 1] for i in range(len(ah))]
    return np.concatenate(F), float(np.nanmean(corrs))


print(f"training reconstruction model x{len(SEEDS)} seeds (lambda={LAM}) ...", flush=True)
per_seed, recon_corrs, maes, overlay = [], [], [], None
for sd in SEEDS:
    net = train(sd)
    feats, rc = features_and_corr(net)
    head = lambda Fm: net.head(torch.tensor(Fm, dtype=torch.float32, device=device)).detach().cpu().numpy()
    prof = mechlib.mechanism_profile(feats, head, scalars_te, target=1)
    pred = mechlib.predict(net, Xte, device); mae = mechlib.calibrated_mae(pred, yte, gte, K=3)
    per_seed.append(prof); recon_corrs.append(rc); maes.append(float(mae[1]))
    top = sorted(prof.items(), key=lambda kv: -kv[1]["probe_r2"])[0]
    print(f"  seed {sd}  recon {rc:.2f}  MAEcal_DBP {mae[1]:.1f}  "
          f"PAT frac {prof['PAT (arrival time)']['frac_correct']:.2f}  "
          f"top-decodable {top[0].split(' (')[0]} (R2 {top[1]['probe_r2']:.2f}, used {top[1]['frac_correct']:.2f})",
          flush=True)
    if sd == SEEDS[-1]:
        with torch.no_grad():
            rc_ex = net.reconstruct(torch.tensor(Xte[EX:EX + 1], device=device)).cpu().numpy()[0]
        overlay = (Ate[EX], rc_ex)

# aggregate across seeds
names = list(per_seed[0].keys())
cues = {}
for n in names:
    fr = np.array([p[n]["frac_correct"] for p in per_seed])
    pr = np.array([p[n]["probe_r2"] for p in per_seed])
    dp = np.array([p[n]["dependence"] for p in per_seed])
    cues[n] = {"frac_mean": float(fr.mean()), "frac_std": float(fr.std()),
               "probe_mean": float(pr.mean()), "dep_mean": float(dp.mean()), "dep_std": float(dp.std()),
               "expect_sign": per_seed[0][n]["expect_sign"]}

cue_val = {}
for cue in ABP_KEYS:                              # validate every PPG shape cue vs the ABP wave
    p, a = scalars_te[cue], morph_abp[cue]; mm = np.isfinite(p) & np.isfinite(a)
    cue_val[cue] = float(np.corrcoef(p[mm], a[mm])[0, 1]) if mm.sum() > 10 else float("nan")

res = {"lambda": LAM, "n_seeds": len(SEEDS), "recon_corr": float(np.mean(recon_corrs)),
       "mae_cal_dbp": float(np.mean(maes)), "cues": cues, "cue_validation": cue_val}
json.dump(res, open("data/capstone.json", "w"), indent=2)
np.savez_compressed("data/capstone.npz",
    t=np.arange(L) / fs, abp_true=overlay[0], abp_recon=overlay[1],
    ex_ecg=Xte[EX, :, 0], ex_ppg=Xte[EX, :, 1], ex_fs=fs,
    ex_r_peaks=exf["r_peaks"], ex_feet=exf["feet"], ex_sys_peaks=exf["sys_peaks"],
    ex_notches=exf["notches"], ex_pat_ms=exf["pat_ms"])
print("cue validation (PPG vs ABP):", {k: round(v, 2) for k, v in cue_val.items()}, flush=True)
print("wrote data/capstone.json + .npz", flush=True); print("DONE", flush=True)
