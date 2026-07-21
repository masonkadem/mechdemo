"""mechlib -- small self-contained toolkit for the mechanistic BP demo.

No heavy dependencies on any training pipeline: just numpy / scipy / torch /
sklearn. Provides data loading, PTT detection, the causal PTT-shift audit, a
linear probe, and per-subject offset calibration.

Verified channel identities in the cached data:  ECG = 0, PPG = 1, ABP = 2.
"""
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ECG, PPG, ABP = 0, 1, 2


# ----------------------------------------------------------------- data
def load_mini(path="data/vitaldb_mini.npz"):
    """Return dict with Xtr/ytr/gtr, Xva/yva/gva, Xte/yte/gte (float32) and fs.
    Signals are (N, 1250, 3) = [ECG, PPG, ABP]; labels are (N, 2) = [SBP, DBP]."""
    d = np.load(path)
    out = {k: (d[k].astype(np.float32) if d[k].dtype == np.float16 else d[k]) for k in d.files}
    out["fs"] = int(d["fs"])
    return out


def normalize(X):
    """Per-segment, per-channel: remove baseline, scale to unit std."""
    X = X.astype(np.float32).copy()
    X -= X.mean(1, keepdims=True)
    X /= X.std(1, keepdims=True) + 1e-8
    return X


# ----------------------------------------------------------------- PTT
def _z(x): return (x - x.mean()) / (x.std() + 1e-8)


def segment_ptt(ecg, ppg, fs):
    """Median ECG-R-peak -> following PPG-foot delay (seconds) for one segment.
    Returns nan if too few clean beats are found."""
    ez, pz = _z(ecg), _z(ppg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    if len(r) < 3:
        return np.nan
    ptts = []
    for rp in r:
        lo = rp + max(int(0.05 * fs), 1)
        hi = min(rp + int(0.50 * fs), len(pz))
        if lo >= hi:
            continue
        foot = lo + int(np.argmin(pz[lo:hi]))
        ptt = (foot - rp) / fs
        if 0.05 < ptt < 0.5:
            ptts.append(ptt)
    return float(np.median(ptts)) if len(ptts) >= 2 else np.nan


def compute_ptt(X, fs, ecg_pos=ECG, ppg_pos=PPG):
    """Per-segment PTT (s) for a batch (N, L, C). nan where undetectable."""
    return np.array([segment_ptt(X[i, :, ecg_pos], X[i, :, ppg_pos], fs) for i in range(len(X))])


# ----------------------------------------------------------------- model helpers
@torch.no_grad()
def predict(model, X, device, bs=512):
    """Batched forward of a torch model over numpy (N, L, C) -> (N, out)."""
    model.eval(); out = []
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs], dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------- audits
def causal_ptt_audit(model, X, fs, device, ppg_pos=PPG, deltas=(-6, -4, -2, 0, 2, 4, 6),
                     n_max=1500, seed=0, predict_fn=None):
    """INPUT-space causal test: shift the PPG channel by +/- delta samples (later PPG =
    longer PTT) and measure how predicted BP responds. Faithful physiology = NEGATIVE
    slope (longer PTT -> lower BP). Returns per-target slope stats + mean response curves.
    DBP is the theoretically PTT-coupled target (pulse propagates during diastole).

    predict_fn : optional callable (X_rolled) -> (M, 2); lets you audit a non-torch
    'model' such as an analytic detect-PTT-then-map estimator. Defaults to the torch model."""
    if predict_fn is None:
        predict_fn = lambda Xr: predict(model, Xr, device)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), min(n_max, len(X)), replace=False); sel.sort()
    Xs = X[sel]; dt = np.array(deltas) / fs
    preds = np.zeros((len(Xs), len(deltas), 2), np.float32)
    for j, d in enumerate(deltas):
        Xd = Xs.copy()
        Xd[:, :, ppg_pos] = np.roll(Xs[:, :, ppg_pos], int(d), axis=1)
        preds[:, j] = predict_fn(Xd)
    out = {"shift_ms": (dt * 1000).tolist()}
    for k, name in [(0, "sbp"), (1, "dbp")]:
        slopes = np.array([np.polyfit(dt, preds[i, :, k], 1)[0] for i in range(len(Xs))])
        out[name] = {
            "dBP_dPTT": float(np.median(slopes)),
            "frac_correct_sign": float(np.mean(slopes < 0)),
            "resp_range_mmHg": float((preds[:, :, k].max(1) - preds[:, :, k].min(1)).mean()),
            "curve": preds[:, :, k].mean(0).tolist(),
        }
    return out


def probe_direction(feats, target):
    """Unit vector in RAW feature space along which `target` is most linearly decodable
    (the linear-probe direction, mapped back through the standardizer)."""
    m = np.isfinite(target); f, t = feats[m], target[m]
    mu, sd = f.mean(0), f.std(0) + 1e-8
    w = Ridge(alpha=1.0).fit((f - mu) / sd, t).coef_
    u = w / sd
    return u / (np.linalg.norm(u) + 1e-8)


def donor_swap(feats, head, ptt, target=1, n_pairs=1500, seed=0):
    """ACTIVATION-space causal test (interchange intervention): replace ONLY the PTT
    subspace of a base activation with a donor's, and check whether the output moves as
    the donor's PTT implies. Faithful = output DROPS when the donor's PTT is LONGER
    (slope of dBP vs dPTT is negative). This is the real-data analogue of the synthetic
    donor-swap, restricted to the decodable PTT direction.
      feats : (N, D) activations;  head : callable (M, D) -> (M, 2) BP in mmHg
      ptt   : (N,) measured PTT in seconds;  target : 0=SBP, 1=DBP"""
    m = np.isfinite(ptt); feats, ptt = feats[m], ptt[m]
    u = probe_direction(feats, ptt * 1000.0)                 # PTT direction (ms)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(feats), n_pairs)
    donor = rng.integers(0, len(feats), n_pairs)
    proj = (feats[donor] - feats[base]) @ u                   # donor-minus-base PTT component
    f_patched = feats[base] + np.outer(proj, u)              # swap ONLY the PTT subspace
    dBP = head(f_patched)[:, target] - head(feats[base])[:, target]
    dPTT = ptt[donor] - ptt[base]                            # seconds
    slope = float(np.polyfit(dPTT, dBP, 1)[0])              # mmHg per second
    frac_correct = float(np.mean((dPTT > 0) == (dBP < 0)))  # longer PTT -> lower BP
    return {"donor_swap_slope": slope, "donor_swap_frac_correct": frac_correct,
            "dPTT_ms": (dPTT * 1000), "dBP": dBP}


def linear_probe(feats, target):
    """Held-out R^2 decoding `target` from `feats` (disjoint fit/score halves)."""
    m = np.isfinite(target); feats, target = feats[m], target[m]
    h = len(target) // 2
    sc = StandardScaler().fit(feats[:h])
    fit = Ridge(alpha=1.0).fit(sc.transform(feats[:h]), target[:h])
    return float(r2_score(target[h:], fit.predict(sc.transform(feats[h:]))))


# ----------------------------------------------------------------- calibration
def calibrated_mae(pred, y, grp, K=3):
    """Per-subject offset calibration: K anchor segments fix each subject's baseline,
    score the rest. A constant offset does not change the audit slope."""
    errs = []
    for g in np.unique(grp):
        idx = np.where(grp == g)[0]
        if len(idx) <= K:
            continue
        off = (y[idx[:K]] - pred[idx[:K]]).mean(0)
        errs.append(np.abs((pred[idx[K:]] + off) - y[idx[K:]]))
    return np.concatenate(errs).mean(0) if errs else np.array([np.nan, np.nan])
