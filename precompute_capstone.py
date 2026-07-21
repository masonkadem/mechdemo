"""Capstone: does a reconstruction / PTT-consistency objective make a model FAITHFUL?

Same architecture, same data, same init. Only difference: an auxiliary head that must
reconstruct the measured pulse transit time (PTT) from the shared features.

    vanilla :  loss = MSE(BP)/Var(BP)
    aux     :  loss = MSE(BP)/Var(BP)  +  lambda * MSE(PTT_hat, PTT_measured)

We then run the identical causal PTT-shift audit on both. Physiological faithfulness =
NEGATIVE dBP/dPTT (longer transit time -> lower BP). Saves data/capstone.json + .npz.
"""
import json, numpy as np, torch, torch.nn as nn
import mechlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM = 3.0
d = mechlib.load_mini("data/vitaldb_mini.npz")
ECG, PPG = mechlib.ECG, mechlib.PPG
Xtr = mechlib.normalize(d["Xtr"][:, :, [ECG, PPG]]); ytr = d["ytr"]
Xte = mechlib.normalize(d["Xte"][:, :, [ECG, PPG]]); yte = d["yte"]; gte = d["gte"]
fs = d["fs"]

print("computing measured PTT (train/test)...", flush=True)
ptt_tr = mechlib.compute_ptt(Xtr, fs, ecg_pos=0, ppg_pos=1)
ptt_te = mechlib.compute_ptt(Xte, fs, ecg_pos=0, ppg_pos=1)
p_mu, p_sd = np.nanmean(ptt_tr), np.nanstd(ptt_tr) + 1e-6
mask_tr = np.isfinite(ptt_tr)
ptt_tr_s = np.where(mask_tr, (ptt_tr - p_mu) / p_sd, 0.0)   # 0 (not NaN) where masked: NaN*0=NaN


class AuxCNN(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, w, 7, 2, 3), nn.ReLU(), nn.Conv1d(w, w*2, 7, 2, 3), nn.ReLU(),
            nn.Conv1d(w*2, w*2, 7, 2, 3), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(w*2, 2)          # BP (mmHg) -- audit-compatible
        self.ptt_head = nn.Linear(w*2, 1)       # reconstruct transit time (standardized)

    def features(self, x): return self.body(x.transpose(1, 2))
    def forward(self, x): return self.head(self.features(x))
    def forward_aux(self, x):
        f = self.features(x); return self.head(f), self.ptt_head(f).squeeze(1)


def train(lam, seed=0, epochs=60, bs=128):
    torch.manual_seed(seed); np.random.seed(seed)
    net = AuxCNN().to(device); opt = torch.optim.Adam(net.parameters(), 2e-3)
    Xt = torch.tensor(Xtr, device=device); yt = torch.tensor(ytr, device=device)
    pt = torch.tensor(ptt_tr_s, dtype=torch.float32, device=device)
    mk = torch.tensor(mask_tr, dtype=torch.float32, device=device)
    bp_var = torch.tensor(ytr.var(0), dtype=torch.float32, device=device)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            b = perm[s:s+bs]
            bp_hat, ptt_hat = net.forward_aux(Xt[b])
            loss = (((bp_hat - yt[b])**2) / bp_var).mean()
            if lam > 0 and mk[b].sum() > 0:
                loss = loss + lam * (((ptt_hat - pt[b])**2) * mk[b]).sum() / mk[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def evaluate(net, tag):
    pred = mechlib.predict(net, Xte, device)
    raw = np.abs(pred - yte).mean(0); cal = mechlib.calibrated_mae(pred, yte, gte, K=3)
    feats = np.concatenate([net.features(torch.tensor(Xte[s:s+512], device=device)).detach().cpu().numpy()
                            for s in range(0, len(Xte), 512)])
    probe = mechlib.linear_probe(feats, ptt_te * 1000)
    au = mechlib.causal_ptt_audit(net, Xte, fs, device, ppg_pos=1)          # input-shift
    head = lambda F: net.head(torch.tensor(F, dtype=torch.float32, device=device)).detach().cpu().numpy()
    ds = mechlib.donor_swap(feats, head, ptt_te, target=1)                  # DBP subspace donor-swap
    print(f"{tag:>10}  MAE(cal) SBP {cal[0]:.1f} DBP {cal[1]:.1f}  probe PTT R2 {probe:.2f}  "
          f"[input-shift DBP dBP/dPTT {au['dbp']['dBP_dPTT']:+.2f}]  "
          f"[donor-swap DBP slope {ds['donor_swap_slope']:+.2f} frac_ok {ds['donor_swap_frac_correct']:.2f}]",
          flush=True)
    return {"mae_raw_sbp": float(raw[0]), "mae_raw_dbp": float(raw[1]),
            "mae_cal_sbp": float(cal[0]), "mae_cal_dbp": float(cal[1]),
            "probe_ptt_r2": probe,
            "shift_ms": au["shift_ms"],
            "in_dbp_slope": au["dbp"]["dBP_dPTT"], "in_dbp_frac": au["dbp"]["frac_correct_sign"],
            "in_sbp_slope": au["sbp"]["dBP_dPTT"], "in_sbp_frac": au["sbp"]["frac_correct_sign"],
            "curve_dbp": au["dbp"]["curve"], "curve_sbp": au["sbp"]["curve"],
            "ds_dbp_slope": ds["donor_swap_slope"], "ds_dbp_frac": ds["donor_swap_frac_correct"]}


def analytic_positive_control():
    """POSITIVE CONTROL: a model that is faithful BY CONSTRUCTION. Detect PTT with the
    algorithm and map it monotonically to DBP: DBP = m*PTT + c (m fit on train, < 0).
    The audit rolls PPG -> the detector recomputes a longer PTT -> DBP drops. If the audit
    is a valid instrument, it must read this model as FAITHFUL (negative slope)."""
    m = np.isfinite(ptt_tr)
    coef = np.polyfit(ptt_tr[m], ytr[m, 1], 1)              # DBP ~ PTT (seconds)
    dbp_mean = ytr[:, 1].mean()
    def predict_fn(Xr):
        p = compute_ptt_local(Xr)
        dbp = np.where(np.isfinite(p), coef[0] * p + coef[1], dbp_mean)
        return np.stack([np.full(len(Xr), ytr[:, 0].mean()), dbp], 1).astype(np.float32)
    au = mechlib.causal_ptt_audit(None, Xte, fs, device, ppg_pos=1, predict_fn=predict_fn)
    print(f"  analytic  [donor-swap n/a]  input-shift DBP dBP/dPTT {au['dbp']['dBP_dPTT']:+.2f} "
          f"frac_correct {au['dbp']['frac_correct_sign']:.2f}  (PTT->DBP slope {coef[0]*1e-3:+.3f} mmHg/ms)",
          flush=True)
    return {"in_dbp_slope": au["dbp"]["dBP_dPTT"], "in_dbp_frac": au["dbp"]["frac_correct_sign"],
            "curve_dbp": au["dbp"]["curve"], "shift_ms": au["shift_ms"]}

def compute_ptt_local(Xr):
    return mechlib.compute_ptt(Xr, fs, ecg_pos=0, ppg_pos=1)

print(f"training vanilla (lam=0) and aux (lam={LAM})...", flush=True)
res = {"lambda": LAM, "primary_target": "DBP",
       "vanilla": evaluate(train(0.0), "vanilla"),
       "aux": evaluate(train(LAM), "aux (PTT)"),
       "analytic": analytic_positive_control()}
json.dump(res, open("data/capstone.json", "w"), indent=2)
np.savez_compressed("data/capstone.npz",
    shift_ms=np.array(res["vanilla"]["shift_ms"]),
    van_dbp=np.array(res["vanilla"]["curve_dbp"]), aux_dbp=np.array(res["aux"]["curve_dbp"]),
    van_sbp=np.array(res["vanilla"]["curve_sbp"]), aux_sbp=np.array(res["aux"]["curve_sbp"]),
    analytic_dbp=np.array(res["analytic"]["curve_dbp"]))
print("wrote data/capstone.json + .npz", flush=True); print("DONE", flush=True)
