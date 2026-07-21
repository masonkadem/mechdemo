"""Capstone v2: does reconstructing the ABP *waveform* as an intermediate make a
model FAITHFUL to pulse transit time (PTT)?

Same CNN backbone (ECG + PPG -> BP). The only difference is the auxiliary objective:

    vanilla :  loss = MSE(BP)/Var(BP)
    aux     :  loss = MSE(BP)/Var(BP)  +  lambda * MSE(ABP_hat, ABP_true)   # rebuild the wave

The aux model is forced to rebuild the arterial pressure waveform (morphology) from
ECG + PPG through a shared feature map. We then measure FOUR different things people
conflate with faithfulness:

    1. accuracy            -- calibrated BP MAE
    2. decodability        -- linear-probe R^2 for PTT in the pooled features
    3. reconstruction      -- per-segment correlation of ABP_hat with true ABP (morphology)
    4. causal faithfulness -- donor-swap dDBP/dPTT + input-space PTT-shift audit

Faithful physiology = NEGATIVE dBP/dPTT (longer transit time -> lower BP). An analytic
detect-PTT-then-map model is included as a faithful-by-construction positive control.
Writes data/capstone.json + data/capstone.npz (consumed by app_faithfulness.py).
"""
import json, numpy as np, torch, torch.nn as nn
import mechlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM = 1.0                     # reconstruction weight
EPOCHS = 60
ECG, PPG, ABP = mechlib.ECG, mechlib.PPG, mechlib.ABP

d = mechlib.load_mini("data/vitaldb_mini.npz")
fs = d["fs"]
Xtr = mechlib.normalize(d["Xtr"][:, :, [ECG, PPG]]); ytr = d["ytr"]
Xte = mechlib.normalize(d["Xte"][:, :, [ECG, PPG]]); yte = d["yte"]; gte = d["gte"]


def stdz_wave(w):
    """Per-segment standardized ABP shape (morphology target; absolute mmHg removed)."""
    w = w.astype(np.float32).copy()
    w -= w.mean(1, keepdims=True); w /= w.std(1, keepdims=True) + 1e-8
    return w


Atr = stdz_wave(d["Xtr"][:, :, ABP])          # true ABP waveform, standardized
Ate = stdz_wave(d["Xte"][:, :, ABP])

print("computing measured PTT + candidate cues (test) ...", flush=True)
ptt_te = mechlib.compute_ptt(d["Xte"][:, :, [ECG, PPG]], fs, ecg_pos=0, ppg_pos=1)
ptt_tr = mechlib.compute_ptt(d["Xtr"][:, :, [ECG, PPG]], fs, ecg_pos=0, ppg_pos=1)
scalars_te = mechlib.compute_scalars(d["Xte"][:, :, [ECG, PPG]], fs, ecg_pos=0, ppg_pos=1)


class ReconCNN(nn.Module):
    """Shared conv encoder -> pooled features -> BP head (audit-compatible).
    A decoder branch rebuilds the ABP waveform from the same temporal feature map."""
    def __init__(self, w=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, w, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(w, w*2, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(w*2, w*2, 7, 2, 3), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(w*2, 2)                       # BP (mmHg)
        self.dec = nn.Sequential(                            # ABP waveform reconstruction
            nn.ConvTranspose1d(w*2, w*2, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w*2, w, 7, 2, 3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(w, 1, 7, 2, 3, output_padding=1))

    def encode(self, x): return self.enc(x.transpose(1, 2))   # (B, C, T')
    def features(self, x): return self.pool(self.encode(x)).flatten(1)
    def forward(self, x): return self.head(self.features(x))
    def reconstruct(self, x, L):
        y = self.dec(self.encode(x))
        return nn.functional.interpolate(y, size=L, mode="linear", align_corners=False).squeeze(1)
    def forward_aux(self, x, L):
        z = self.encode(x)
        return self.head(self.pool(z).flatten(1)), \
            nn.functional.interpolate(self.dec(z), size=L, mode="linear", align_corners=False).squeeze(1)


def train(lam, seed=0, epochs=EPOCHS, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    net = ReconCNN().to(device); opt = torch.optim.Adam(net.parameters(), 2e-3)
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    At = torch.tensor(Atr, device=device)
    bp_var = torch.tensor(ytr.var(0), dtype=torch.float32, device=device)
    L = Xtr.shape[1]
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = perm[s:s+bs]
            bp_hat, abp_hat = net.forward_aux(Xt[b], L)
            loss = (((bp_hat - yt[b])**2) / bp_var).mean()
            if lam > 0:
                loss = loss + lam * ((abp_hat - At[b])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def recon_corr(net):
    """Mean per-segment correlation of reconstructed vs true ABP morphology (test)."""
    L = Xte.shape[1]; corrs = []
    with torch.no_grad():
        for s in range(0, len(Xte), 512):
            xb = torch.tensor(Xte[s:s+512], device=device)
            pr = net.reconstruct(xb, L).cpu().numpy()
            tr = Ate[s:s+512]
            for i in range(len(pr)):
                c = np.corrcoef(pr[i], tr[i])[0, 1]
                if np.isfinite(c): corrs.append(c)
    return float(np.mean(corrs))


def features_of(net):
    return np.concatenate([net.features(torch.tensor(Xte[s:s+512], device=device)).detach().cpu().numpy()
                           for s in range(0, len(Xte), 512)])


def per_segment_recon_vs_causal(net, n=600, deltas=(-6, -4, -2, 0, 2, 4, 6), seed=1):
    """For a sample of test segments: reconstruction quality (corr with true ABP) and the
    per-segment causal DBP response to an imposed PTT shift. Used for the morphology-vs-
    faithfulness scatter (are the best-reconstructed segments the PTT-responsive ones?)."""
    L = Xte.shape[1]; rng = np.random.default_rng(seed)
    sel = rng.choice(len(Xte), min(n, len(Xte)), replace=False); sel.sort()
    Xs = Xte[sel]; dt = np.array(deltas) / fs
    with torch.no_grad():
        pr = net.reconstruct(torch.tensor(Xs, device=device), L).cpu().numpy()
    corr = np.array([np.corrcoef(pr[i], Ate[sel][i])[0, 1] for i in range(len(sel))])
    preds = np.zeros((len(Xs), len(deltas)), np.float32)
    for j, dd in enumerate(deltas):
        Xd = Xs.copy(); Xd[:, :, 1] = np.roll(Xs[:, :, 1], int(dd), axis=1)
        preds[:, j] = mechlib.predict(net, Xd, device)[:, 1]
    slope = np.array([np.polyfit(dt, preds[i], 1)[0] for i in range(len(Xs))])
    m = np.isfinite(corr) & np.isfinite(slope)
    return corr[m], slope[m]


def evaluate(net, tag):
    pred = mechlib.predict(net, Xte, device)
    cal = mechlib.calibrated_mae(pred, yte, gte, K=3)
    feats = features_of(net)
    probe = mechlib.linear_probe(feats, ptt_te * 1000)
    rc = recon_corr(net)
    au = mechlib.causal_ptt_audit(net, Xte, fs, device, ppg_pos=1)
    head = lambda F: net.head(torch.tensor(F, dtype=torch.float32, device=device)).detach().cpu().numpy()
    ds = mechlib.donor_swap(feats, head, ptt_te, target=1)
    prof = mechlib.mechanism_profile(feats, head, scalars_te, target=1)
    prof_str = "  ".join(f"{k.split()[0]} {v['frac_correct']:.2f}" for k, v in prof.items())
    print(f"{tag:>10}  MAE(cal) DBP {cal[1]:.1f}  probe PTT R2 {probe:.2f}  recon corr {rc:.2f}  "
          f"[donor-swap PTT frac {ds['donor_swap_frac_correct']:.2f}]  profile[{prof_str}]", flush=True)
    return {"mae_cal_sbp": float(cal[0]), "mae_cal_dbp": float(cal[1]),
            "probe_ptt_r2": probe, "recon_corr": rc,
            "shift_ms": au["shift_ms"],
            "curve_dbp": au["dbp"]["curve"], "curve_sbp": au["sbp"]["curve"],
            "in_dbp_slope": au["dbp"]["dBP_dPTT"], "in_dbp_frac": au["dbp"]["frac_correct_sign"],
            "ds_dbp_slope": ds["donor_swap_slope"], "ds_dbp_frac": ds["donor_swap_frac_correct"],
            "profile": prof}


def readoff_model(seed=0, epochs=EPOCHS, bs=128):
    """Faithful-by-design: reconstruct the ABP waveform in mmHg, then read SBP/DBP off its
    peak/trough. The pressure wave is an interpretable, auditable bottleneck."""
    torch.manual_seed(seed); np.random.seed(seed)
    net = ReconCNN().to(device); opt = torch.optim.Adam(net.parameters(), 2e-3)
    Xt = torch.tensor(Xtr, device=device)
    Mt = torch.tensor(d["Xtr"][:, :, ABP].astype(np.float32), device=device)   # mmHg ABP
    L = Xtr.shape[1]
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = perm[s:s+bs]
            loss = ((net.reconstruct(Xt[b], L) - Mt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        rec = np.concatenate([net.reconstruct(torch.tensor(Xte[s:s+512], device=device), L).cpu().numpy()
                              for s in range(0, len(Xte), 512)])
    sbp_hat, dbp_hat = rec.max(1), rec.min(1)
    mae = [float(np.abs(sbp_hat - yte[:, 0]).mean()), float(np.abs(dbp_hat - yte[:, 1]).mean())]
    print(f"   readoff  reconstruct-then-read-off MAE  SBP {mae[0]:.1f}  DBP {mae[1]:.1f} mmHg", flush=True)
    return {"mae_sbp": mae[0], "mae_dbp": mae[1]}, rec[0], d["Xte"][0, :, ABP].astype(np.float32)


def analytic_positive_control():
    """Faithful BY CONSTRUCTION: detect PTT, map DBP = m*PTT + c (m fit on train, < 0).
    If the audit is a valid instrument it should read this as faithful (negative slope)."""
    m = np.isfinite(ptt_tr)
    coef = np.polyfit(ptt_tr[m], ytr[m, 1], 1)
    dbp_mean, sbp_mean = ytr[:, 1].mean(), ytr[:, 0].mean()
    def predict_fn(Xr):
        p = mechlib.compute_ptt(Xr, fs, ecg_pos=0, ppg_pos=1)
        dbp = np.where(np.isfinite(p), coef[0] * p + coef[1], dbp_mean)
        return np.stack([np.full(len(Xr), sbp_mean), dbp], 1).astype(np.float32)
    au = mechlib.causal_ptt_audit(None, Xte, fs, device, ppg_pos=1, predict_fn=predict_fn)
    print(f"  analytic  input-shift dDBP/dPTT {au['dbp']['dBP_dPTT']:+.2f} "
          f"frac {au['dbp']['frac_correct_sign']:.2f}  (PTT->DBP {coef[0]*1e-3:+.3f} mmHg/ms)", flush=True)
    return {"in_dbp_slope": au["dbp"]["dBP_dPTT"], "in_dbp_frac": au["dbp"]["frac_correct_sign"],
            "curve_dbp": au["dbp"]["curve"], "shift_ms": au["shift_ms"]}


def example_reconstructions(net, k=1):
    """A few (true, reconstructed) ABP segments for the app's morphology overlay."""
    L = Xte.shape[1]
    with torch.no_grad():
        pr = net.reconstruct(torch.tensor(Xte[:k], device=device), L).cpu().numpy()
    return Ate[:k], pr


print(f"training vanilla (lam=0) and aux (ABP-recon, lam={LAM}) ...", flush=True)
van = train(0.0); aux = train(LAM)
readoff, rec_mmHg, abp_mmHg = readoff_model()
res = {"lambda": LAM, "primary_target": "DBP", "aux_kind": "abp_waveform",
       "vanilla": evaluate(van, "vanilla"),
       "aux": evaluate(aux, "aux (ABP)"),
       "analytic": analytic_positive_control(),
       "readoff": readoff}
json.dump(res, open("data/capstone.json", "w"), indent=2)

t_true, t_rec = example_reconstructions(aux, k=1)
seg_corr, seg_slope = per_segment_recon_vs_causal(aux)
np.savez_compressed("data/capstone.npz",
    shift_ms=np.array(res["vanilla"]["shift_ms"]),
    van_dbp=np.array(res["vanilla"]["curve_dbp"]), aux_dbp=np.array(res["aux"]["curve_dbp"]),
    van_sbp=np.array(res["vanilla"]["curve_sbp"]), aux_sbp=np.array(res["aux"]["curve_sbp"]),
    analytic_dbp=np.array(res["analytic"]["curve_dbp"]),
    t=np.arange(Xte.shape[1]) / fs, abp_true=t_true[0], abp_recon=t_rec[0],
    seg_recon_corr=seg_corr, seg_causal_slope=seg_slope,
    readoff_t=np.arange(Xte.shape[1]) / fs, readoff_abp_true=abp_mmHg, readoff_abp_rec=rec_mmHg)
print("wrote data/capstone.json + .npz", flush=True); print("DONE", flush=True)
