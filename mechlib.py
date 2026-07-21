"""mechlib -- small self-contained toolkit for the mechanistic BP demo.

No heavy dependencies on any training pipeline: just numpy / scipy / torch /
sklearn. Provides data loading, PTT detection, the causal PTT-shift audit, a
linear probe, and per-subject offset calibration.

Verified channel identities in the cached data:  ECG = 0, PPG = 1, ABP = 2.
"""
import numpy as np
import torch
from scipy.signal import find_peaks, savgol_filter
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


# ----------------------------------------------------------------- mechanism profile
def _ppg_fiducials(ppg_z, r_peaks, fs):
    """Per-beat PPG (foot, systolic-peak) index pairs following each ECG R-peak."""
    feet, peaks = [], []
    for rp in r_peaks:
        lo = rp + max(int(0.05 * fs), 1); hi = min(rp + int(0.5 * fs), len(ppg_z))
        if lo >= hi:
            continue
        foot = lo + int(np.argmin(ppg_z[lo:hi]))
        hi2 = min(foot + int(0.4 * fs), len(ppg_z))
        if foot + 1 >= hi2:
            continue
        peak = foot + int(np.argmax(ppg_z[foot:hi2]))
        feet.append(foot); peaks.append(peak)
    return feet, peaks


def _pulse_feet(wz, fs):
    f, _ = find_peaks(-wz, distance=max(int(0.4 * fs), 1), prominence=0.3)
    return f


def wave_morphology(wave, fs):
    """Shape cues from ONE pulsatile wave (PPG or ABP), median over beats (nan if too few):
      rise -- systolic rise time (s): foot -> systolic peak
      aix  -- reflection / augmentation index: (secondary-peak height) / (pulse height)
      apg  -- second-derivative (acceleration) b/a ratio: an arterial-stiffness index."""
    wz = _z(wave); feet = _pulse_feet(wz, fs)
    out = {"rise": np.nan, "aix": np.nan, "apg": np.nan}
    if len(feet) < 3:
        return out
    d2 = np.gradient(np.gradient(savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)))
    rises, aixs, apgs = [], [], []
    for k in range(len(feet) - 1):
        s, e = feet[k], feet[k + 1]
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        beat = wz[s:e]; pk = int(np.argmax(beat))
        if pk < 2 or pk > len(beat) - 2:
            continue
        pp = beat[pk] - beat[0]
        if pp < 1e-3:
            continue
        rises.append(pk / fs)
        tail = beat[pk + int(0.05 * fs):]
        if len(tail) > 3:
            sp, _ = find_peaks(tail)
            if len(sp):
                aixs.append((tail[sp].max() - beat[0]) / pp)
        d2b = d2[s:e]; ap, _ = find_peaks(d2b, prominence=0.02 * (d2b.max() - d2b.min() + 1e-9))
        if len(ap):
            ai = ap[0]; hi = min(ai + int(0.25 * fs), len(d2b))
            if ai + 1 < hi and abs(d2b[ai]) > 1e-9:
                apgs.append(d2b[ai:hi].min() / d2b[ai])
    return {"rise": float(np.median(rises)) if len(rises) >= 2 else np.nan,
            "aix": float(np.median(aixs)) if len(aixs) >= 2 else np.nan,
            "apg": float(np.median(apgs)) if len(apgs) >= 2 else np.nan}


def compute_morphology(X, fs, ch):
    """Per-segment {rise, aix, apg} from channel `ch` of a batch (N, L, C). Use on the ABP
    channel to get ground-truth shape cues for validating the PPG-derived ones."""
    keys = ["rise", "aix", "apg"]; acc = {k: [] for k in keys}
    for i in range(len(X)):
        m = wave_morphology(X[i, :, ch], fs)
        for k in keys:
            acc[k].append(m[k])
    return {k: np.array(v) for k, v in acc.items()}


def segment_scalars(ecg, ppg, fs):
    """Candidate BP cues from one ECG+PPG segment (nan where undetectable):
      pat            -- ECG-R -> PPG-foot delay (s); arrival-time law (PAT, PEP-confounded)
      rise/aix/apg   -- PPG wave-shape / arterial-stiffness morphology cues
      hr             -- heart rate (bpm); exploratory, ambiguous BP sign
      amp            -- PPG pulse amplitude (au); negative control (removed by per-segment norm)."""
    ez, pz = _z(ecg), _z(ppg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    out = {"pat": np.nan, "hr": np.nan, "amp": np.nan}
    out.update(wave_morphology(ppg, fs))
    if len(r) < 3:
        return out
    pats = []
    for rp in r:
        lo = rp + max(int(0.05 * fs), 1); hi = min(rp + int(0.5 * fs), len(pz))
        if lo >= hi:
            continue
        ptt = (lo + int(np.argmin(pz[lo:hi])) - rp) / fs
        if 0.05 < ptt < 0.5:
            pats.append(ptt)
    if len(pats) >= 2:
        out["pat"] = float(np.median(pats))
    rr = np.diff(r) / fs; rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) >= 1:
        out["hr"] = float(60.0 / np.median(rr))
    feet, peaks = _ppg_fiducials(pz, r, fs)
    amps = [pz[pk] - pz[ft] for ft, pk in zip(feet, peaks)]
    if len(amps) >= 2:
        out["amp"] = float(np.median(amps))
    return out


def compute_scalars(X, fs, ecg_pos=ECG, ppg_pos=PPG):
    """Per-segment cue dict {pat, rise, aix, apg, hr, amp} for a batch (N, L, C)."""
    keys = ["pat", "rise", "aix", "apg", "hr", "amp"]; acc = {k: [] for k in keys}
    for i in range(len(X)):
        s = segment_scalars(X[i, :, ecg_pos], X[i, :, ppg_pos], fs)
        for k in keys:
            acc[k].append(s.get(k, np.nan))
    return {k: np.array(v) for k, v in acc.items()}


def subspace_swap(feats, head, mech, target=1, expect_sign=-1, n_pairs=1500, seed=0):
    """Generalized activation-space donor-swap for ANY scalar cue `mech`. Patch only the
    linearly-decodable direction of `mech` from a donor into a base activation and measure
    how the BP output moves. `expect_sign` is the physiological sign of dBP/dmech
    (-1 = longer cue -> lower BP, like PAT/rise; 0 = no expected sign, e.g. HR/control).
    Returns slope (mmHg per cue-unit) and frac_correct (share of pairs in the expected
    direction; 0.5 = chance)."""
    m = np.isfinite(mech); feats, mech = feats[m], mech[m]
    if len(feats) < 10 or np.std(mech) < 1e-9:
        return {"slope": float("nan"), "frac_correct": float("nan")}
    u = probe_direction(feats, mech)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(feats), n_pairs); donor = rng.integers(0, len(feats), n_pairs)
    proj = (feats[donor] - feats[base]) @ u
    dBP = head(feats[base] + np.outer(proj, u))[:, target] - head(feats[base])[:, target]
    dM = mech[donor] - mech[base]
    slope = float(np.polyfit(dM, dBP, 1)[0])
    ref = expect_sign if expect_sign != 0 else 1
    frac = float(np.mean(np.sign(dBP) == ref * np.sign(dM)))
    return {"slope": slope, "frac_correct": frac}


def mechanism_profile(feats, head, scalars, target=1):
    """Run subspace_swap across the candidate cues -> 'faithful to WHAT'. `scalars` is the
    dict from compute_scalars. Expected signs encode the textbook physiology per cue."""
    specs = [("PAT (arrival time)", "pat", -1), ("PPG rise-time (morphology)", "rise", -1),
             ("augmentation index (morphology)", "aix", +1), ("APG stiffness (morphology)", "apg", +1),
             ("heart rate", "hr", 0), ("PPG amplitude (control)", "amp", 0)]
    return {name: {**subspace_swap(feats, head, scalars[key], target, sgn), "expect_sign": sgn}
            for name, key, sgn in specs if key in scalars}


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
