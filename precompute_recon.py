"""Real-data faithfulness dial: sweep how much a model routes BP through the reconstructed
ABP pressure waveform, and watch the audits — the real-data analogue of the synthetic tab.

    BP = alpha * head_recon(features_of(ABP_hat))  +  (1-alpha) * head_shortcut(features)
    loss = MSE(BP)/Var(BP)  +  lambda * MSE(ABP_hat, ABP_true)   # decoder always trained

alpha = 0  -> pure direct shortcut ;  alpha = 1 -> BP flows only through the rebuilt wave.
As alpha rises we expect (mirroring the synthetic sandbox):
    accuracy            ~ flat   (both pathways can fit BP)
    morphology probe    ~ flat   (the wave is decodable regardless, decoder always on)
    donor-swap          tracks alpha  (only the causal audit sees the faithful routing)

Also computes, at alpha = 1: the mechanism-faithfulness profile across candidate cues
(PAT / rise-time / augmentation index / APG stiffness / HR / amplitude), and validates the
PPG-derived shape cues against the ground-truth ABP-derived ones. Writes data/capstone.*.
"""
import json, numpy as np, torch, torch.nn as nn
import mechlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM, EPOCHS = 1.0, 60
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ECG, PPG, ABP = mechlib.ECG, mechlib.PPG, mechlib.ABP

d = mechlib.load_mini("data/vitaldb_mini.npz"); fs = int(d["fs"])
Xtr = mechlib.normalize(d["Xtr"][:, :, [ECG, PPG]]); ytr = d["ytr"]
Xte = mechlib.normalize(d["Xte"][:, :, [ECG, PPG]]); yte = d["yte"]; gte = d["gte"]
L = Xtr.shape[1]


def center_by_group(v, g):
    """Remove each subject's mean (per-subject calibration): the space where the model is
    actually accurate. A single amplitude-normalized window cannot fix absolute BP, so all
    R^2 must be measured on within-subject variation."""
    out = np.asarray(v, dtype=np.float64).copy()
    for gid in np.unique(g):
        idx = g == gid
        out[idx] -= out[idx].mean(0)
    return out


def stdz_wave(w):
    w = w.astype(np.float32).copy(); w -= w.mean(1, keepdims=True)
    w /= w.std(1, keepdims=True) + 1e-8; return w


Atr, Ate = stdz_wave(d["Xtr"][:, :, ABP]), stdz_wave(d["Xte"][:, :, ABP])
y_mu, y_sd = ytr.mean(0), ytr.std(0) + 1e-6

print("computing cues (PPG) + ground-truth morphology (ABP) ...", flush=True)
scalars_te = mechlib.compute_scalars(d["Xte"][:, :, [ECG, PPG]], fs, 0, 1)
morph_abp = mechlib.compute_morphology(d["Xte"], fs, ABP)          # ground-truth shape cues
ptt_tr = mechlib.compute_ptt(d["Xtr"][:, :, [ECG, PPG]], fs, 0, 1)


class AlphaReconCNN(nn.Module):
    """Encoder -> {pooled features, reconstructed ABP}. BP is a convex blend of a shortcut
    head (on pooled features) and a recon head (on features of the rebuilt ABP wave)."""
    def __init__(self, w=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, w, 7, 2, 3), nn.ReLU(), nn.Conv1d(w, w*2, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(w*2, w*2, 7, 2, 3), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(w*2, w*2, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w*2, w, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w, 1, 7, 2, 3, output_padding=1))
        self.renc = nn.Sequential(
            nn.Conv1d(1, w, 7, 2, 3), nn.ReLU(), nn.Conv1d(w, w, 7, 2, 3), nn.ReLU())
        self.shead = nn.Linear(w*2, 2)          # direct shortcut pathway
        self.rhead = nn.Linear(w, 2)            # routed-through-reconstruction pathway

    def parts(self, x):
        z = self.enc(x.transpose(1, 2))
        p = self.pool(z).flatten(1)
        abp = nn.functional.interpolate(self.dec(z), size=L, mode="linear",
                                        align_corners=False).squeeze(1)
        r = self.pool(self.renc(abp.unsqueeze(1))).flatten(1)
        return p, r, abp

    def forward(self, x, alpha):
        p, r, _ = self.parts(x)
        return alpha * self.rhead(r) + (1 - alpha) * self.shead(p)


def train(alpha, seed=0, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    net = AlphaReconCNN().to(device); opt = torch.optim.Adam(net.parameters(), 2e-3)
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    At = torch.tensor(Atr, device=device)
    bp_var = torch.tensor(ytr.var(0), dtype=torch.float32, device=device)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = perm[s:s+bs]
            p, r, abp = net.parts(Xt[b])
            bp = alpha * net.rhead(r) + (1 - alpha) * net.shead(p)
            loss = (((bp - yt[b]) ** 2) / bp_var).mean() + LAM * ((abp - At[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


@torch.no_grad()
def collect(net):
    """Per-test pooled feats p, recon-pathway BP rp, shortcut BP sp, recon-feats r, recon corr."""
    P, R, RP, SP, corrs = [], [], [], [], []
    for s in range(0, len(Xte), 512):
        x = torch.tensor(Xte[s:s+512], device=device)
        p, r, abp = net.parts(x)
        P.append(p.cpu().numpy()); R.append(r.cpu().numpy())
        RP.append(net.rhead(r).cpu().numpy()); SP.append(net.shead(p).cpu().numpy())
        ah = abp.cpu().numpy(); at = Ate[s:s+512]
        corrs += [np.corrcoef(ah[i], at[i])[0, 1] for i in range(len(ah))]
    return (np.concatenate(P), np.concatenate(R), np.concatenate(RP), np.concatenate(SP),
            float(np.nanmean(corrs)))


def r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum() / (((true - true.mean(0)) ** 2).sum() + 1e-9))


def corr2(pred, true):
    """Squared Pearson correlation, clipped at 0 (anti-correlation = not faithful). Robust to
    the additive shortcut-on-base variance that makes raw R^2 brittle when the signal is weak."""
    a = np.asarray(pred, float) - np.mean(pred); b = np.asarray(true, float) - np.mean(true)
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    c = float((a * b).sum() / denom)
    return c * c if c > 0 else 0.0


def sweep_metrics(alpha, P, RP, SP, morph_probe_target, n_pairs=1500, seed=0):
    # everything on within-subject (per-subject-centered) DBP -- where the model is accurate
    yc = center_by_group(yte[:, 1], gte)
    RPc = center_by_group(RP[:, 1], gte); SPc = center_by_group(SP[:, 1], gte)
    bp = alpha * RPc + (1 - alpha) * SPc
    acc = corr2(bp, yc)                                            # within-subject DBP (corr^2)
    mae_dbp = float(np.abs((alpha * RP[:, 1] + (1 - alpha) * SP[:, 1]) - yte[:, 1]).mean())
    probe = mechlib.linear_probe(P, morph_probe_target)           # morphology decodable from feats
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(P), n_pairs); donor = rng.integers(0, len(P), n_pairs)
    bp_sw = alpha * RPc[donor] + (1 - alpha) * SPc[base]           # swap recon pathway to donor
    swap = corr2(bp_sw, yc[donor])                                 # follows donor's within-subj DBP?
    return {"acc": acc, "mae_dbp": mae_dbp, "probe_morph": max(probe, 0.0), "swap": swap}


def run():
  print(f"training alpha sweep {ALPHAS} (lambda={LAM}) ...", flush=True)
  morph_target = scalars_te["rise"]                                # PPG morphology cue (decodable)
  sweep = {"acc": [], "mae_dbp": [], "probe_morph": [], "swap": []}
  overlay = None; prof = None; recon1 = None
  for al in ALPHAS:
    net = train(al); P, R, RP, SP, rc = collect(net)
    m = sweep_metrics(al, P, RP, SP, morph_target)
    for k in sweep:
        sweep[k].append(m[k])
    print(f"  alpha {al:.2f}   acc R2 {m['acc']:.2f}   morph-probe R2 {m['probe_morph']:.2f}   "
          f"donor-swap R2 {m['swap']:.2f}   MAE_DBP {m['mae_dbp']:.1f}   recon {rc:.2f}", flush=True)
    if al == 1.0:
        recon1 = rc
        head = lambda F: net.rhead(torch.tensor(F, dtype=torch.float32, device=device)).detach().cpu().numpy()
        prof = mechlib.mechanism_profile(R, head, scalars_te, target=1)
        with torch.no_grad():
            abp1 = net.parts(torch.tensor(Xte[:1], device=device))[2].cpu().numpy()[0]
        overlay = (Ate[0], abp1)

  cue_val = {}                                                    # PPG cue vs ground-truth ABP cue
  for cue in ["rise", "aix", "apg"]:
    p, a = scalars_te[cue], morph_abp[cue]; mm = np.isfinite(p) & np.isfinite(a)
    cue_val[cue] = float(np.corrcoef(p[mm], a[mm])[0, 1]) if mm.sum() > 10 else float("nan")
  print("cue validation (PPG vs ground-truth ABP):", {k: round(v, 2) for k, v in cue_val.items()}, flush=True)

  mtr = np.isfinite(ptt_tr); coef = np.polyfit(ptt_tr[mtr], ytr[mtr, 1], 1)   # analytic PAT->DBP
  dbp_mean, sbp_mean = ytr[:, 1].mean(), ytr[:, 0].mean()
  def analytic_fn(Xr):
    p = mechlib.compute_ptt(Xr, fs, 0, 1)
    return np.stack([np.full(len(Xr), sbp_mean),
                     np.where(np.isfinite(p), coef[0] * p + coef[1], dbp_mean)], 1).astype(np.float32)
  au = mechlib.causal_ptt_audit(None, d["Xte"][:, :, [ECG, PPG]], fs, device, ppg_pos=1, predict_fn=analytic_fn)
  analytic = {"in_dbp_slope": au["dbp"]["dBP_dPTT"], "fit_slope_mmHg_per_ms": float(coef[0] * 1e-3)}
  print(f"analytic PAT->DBP fit slope {coef[0]*1e-3:+.3f} mmHg/ms  audit slope {au['dbp']['dBP_dPTT']:+.2f}", flush=True)

  res = {"lambda": LAM, "alphas": ALPHAS, "sweep": sweep, "recon_corr": recon1,
         "profile": prof, "cue_validation": cue_val, "analytic": analytic}
  json.dump(res, open("data/capstone.json", "w"), indent=2)
  np.savez_compressed("data/capstone.npz",
      alphas=np.array(ALPHAS), acc=np.array(sweep["acc"]),
      probe_morph=np.array(sweep["probe_morph"]), swap=np.array(sweep["swap"]),
      t=np.arange(L) / fs, abp_true=overlay[0], abp_recon=overlay[1])
  print("wrote data/capstone.json + .npz", flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    run()
