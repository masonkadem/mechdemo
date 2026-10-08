"""mechlib -- small self-contained toolkit for the mechanistic BP demo.

No heavy dependencies on any training pipeline: just numpy / scipy / torch /
sklearn. Provides data loading, PTT detection, the causal PTT-shift audit, a
linear probe, and per-subject offset calibration.

Verified channel identities in the cached data:  ECG = 0, PPG = 1, ABP = 2.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, cheby2, find_peaks, savgol_filter, sosfiltfilt
from scipy.stats import kurtosis as _kurtosis
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ECG, PPG, ABP = 0, 1, 2


# ----------------------------------------------------------------- model
class WaveTransformer(nn.Module):
    """Patchify -> TransformerEncoder -> mean-pool -> linear head, for a single scalar
    regression target (e.g. DBP). Shared by the real-data walkthrough notebook and the
    Streamlit demo so the architecture used to train a checkpoint can never drift from the
    architecture used to load it -- reconstruct with `WaveTransformer(**checkpoint["config"])`.
    """
    def __init__(self, n_ch=2, dm=64, patch=25, heads=4, depth=3, L=1250):
        super().__init__()
        self.cfg = dict(n_ch=n_ch, dm=dm, patch=patch, heads=heads, depth=depth, L=L)
        n_tok = L // patch
        self.embed = nn.Conv1d(n_ch, dm, patch, patch)                # (B,C,L) -> (B,dm,n_tok)
        self.pos = nn.Parameter(torch.randn(1, n_tok, dm) * 0.02)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dm, heads, dm * 2, batch_first=True), depth)
        self.head = nn.Linear(dm, 1)

    def forward(self, x, return_acts=False):
        t = self.embed(x.transpose(1, 2)).transpose(1, 2) + self.pos    # (B, n_tok, dm) patch embed
        acts = [t]
        for layer in self.tr.layers:
            t = layer(t)
            acts.append(t)                                             # one entry per transformer block
        y = self.head(t.mean(1)).squeeze(-1)
        return (y, acts) if return_acts else y


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

# Search span for the R-peak -> PPG-foot delay. Wide on purpose: VitalDB's PPG channel carries a
# fixed instrumental offset of several hundred ms (Ruffolo et al. 2025 report ~350 ms), so the
# stored delay can approach or exceed one RR interval. Nothing is assumed about where in this
# span the foot lies; the pairing below picks the delay that is the same for every beat.
PAT_RANGE = (0.05, 1.5)   # s
PAIR_TOL = 0.04           # s: a beat agrees with the segment delay if within this tolerance


def r_peaks(ecg, fs):
    """ECG R-peaks, Pan-Tompkins style (Pan & Tompkins 1985): 5-15 Hz zero-phase bandpass ->
    derivative -> square -> 150 ms moving integration -> peaks >= 0.3 s apart. The passband
    suppresses T waves, which a plain peak-pick on the raw ECG mistakes for R peaks. Each peak is
    placed at the largest |bandpassed ECG| within 60 ms, so inverted leads also work."""
    bp = sosfiltfilt(butter(2, [5, 15], btype="band", fs=fs, output="sos"), ecg)
    w = max(int(0.15 * fs), 1)
    mwi = np.convolve(np.gradient(bp) ** 2, np.ones(w) / w, mode="same")
    pk, _ = find_peaks(mwi, distance=max(int(0.3 * fs), 1), height=0.25 * np.percentile(mwi, 99))
    h = int(0.06 * fs)
    return np.array([max(p - h, 0) + int(np.argmax(np.abs(bp[max(p - h, 0):p + h]))) for p in pk], int)


def ppg_feet(ppg, fs):
    """PPG pulse feet found on the PPG alone (no ECG): 4th-order Chebyshev-II 0.5-10 Hz zero-phase
    filter (Liang et al. 2018), beats at the steepest upslope, foot by the intersecting tangent --
    the tangent at the steepest point meets the horizontal through the preceding minimum.
    Returns float indices."""
    p = sosfiltfilt(cheby2(4, 20, [0.5, 10], btype="band", fs=fs, output="sos"), ppg)
    d = np.gradient(p)
    up, _ = find_peaks(d, distance=max(int(0.3 * fs), 1), height=0.3 * np.percentile(d, 99))
    feet = []
    for k in up:
        a = max(k - int(0.3 * fs), 0)
        tr = a + int(np.argmin(p[a:k + 1]))
        feet.append(k - (p[k] - p[tr]) / d[k])
    return np.array(feet, float)


def pair_beats(r, feet, fs):
    """Pair PPG feet with R-peaks. Every R-peak up to 1.5 s before a foot is a candidate source.
    The segment delay is the one most beats agree on (within PAIR_TOL); among near-ties, the one
    with the smallest spread wins. That tie-break is what resolves a delay longer than one beat:
    the true pairing stays constant while the wrong one drifts with heart-rate variability (the
    beat-matching idea of Ruffolo et al. 2025). Returns (foot_idx, r_idx, delay_s), aligned."""
    cands = [[(f - q) / fs for q in r if PAT_RANGE[0] < (f - q) / fs < PAT_RANGE[1]] for f in feet]
    best, best_key = None, None
    for c in sorted({round(v, 3) for cs in cands for v in cs}):
        m = [min(cs, key=lambda v: abs(v - c)) for cs in cands if cs]
        m = [v for v in m if abs(v - c) < PAIR_TOL]
        if len(m) >= 2:
            key = (len(m), -np.std(m))
            if best_key is None or key > best_key:
                best, best_key = c, key
    if best is None:
        return np.array([], int), np.array([], int), np.array([])
    # re-centre on the agreed delay, then keep the beats that match it
    mm = [min(cs, key=lambda v: abs(v - best)) for cs in cands if cs]
    best = float(np.median([v for v in mm if abs(v - best) < PAIR_TOL]))
    fi, ri, dl = [], [], []
    for f, cs in zip(feet, cands):
        if not cs:
            continue
        v = min(cs, key=lambda v: abs(v - best))
        if abs(v - best) < PAIR_TOL:
            fi.append(int(round(f))); ri.append(int(round(f - v * fs))); dl.append(v)
    return np.array(fi, int), np.array(ri, int), np.array(dl)


def segment_ptt(ecg, ppg, fs):
    """Median ECG-R-peak -> PPG-foot delay (seconds) for one segment; nan if fewer than 3 beats
    agree on a delay. Under a channel offset this is offset + PAT: use it for within-subject
    changes, never as an absolute transit time."""
    _, _, dl = pair_beats(r_peaks(ecg, fs), ppg_feet(ppg, fs), fs)
    return float(np.median(dl)) if len(dl) >= 3 else np.nan


def compute_ptt(X, fs, ecg_pos=ECG, ppg_pos=PPG):
    """Per-segment pulse ARRIVAL time (s) for a batch (N, L, C). nan where undetectable.

    NAMING, because it matters for every claim built on this function. What is computed is
    R-peak -> PPG foot, i.e.

        PAT = PEP + PTT   (+ any instrumental ECG-PPG offset)

    where PEP (pre-ejection period) is cardiac -- electrical activation to aortic valve opening --
    and only PTT carries arterial stiffness. The name `compute_ptt` is kept for compatibility with
    the existing call sites, but the quantity is PAT and should be described as such.

    This distinction is sharper in surgical datasets than elsewhere: PEP is drug-sensitive.
    Propofol depresses contractility and lengthens it; ephedrine and epinephrine shorten it. So in
    VitalDB and MIMIC a substantial part of the measured interval varies with anaesthetic and
    vasopressor state rather than with vascular tone, and an audit that perturbs this quantity is
    perturbing a cardiac-plus-vascular composite, not the Moens-Korteweg variable.
    """
    return np.array([segment_ptt(X[i, :, ecg_pos], X[i, :, ppg_pos], fs) for i in range(len(X))])


def segment_fiducials(ecg, ppg, fs):
    """Per-beat fiducial indices for ONE ECG+PPG segment -- the points the PAT / morphology cues
    are built from, for a sanity-check overlay. Returns a dict of index arrays:
      r_peaks   : all ECG R-peaks
      r_paired  : the R-peak each foot was paired with (aligned with feet)
      feet      : intersecting-tangent PPG feet that agree with the segment delay (int)
      sys_peaks : PPG systolic peak after each foot
      notches   : dicrotic-notch index on the downslope (only where detected)
      pat_ms    : per-beat delay (R-peak -> foot) in ms, aligned with feet."""
    r = r_peaks(ecg, fs)
    fi, ri, dl = pair_beats(r, ppg_feet(ppg, fs), fs)
    pz = _z(ppg)
    keep, sys_peaks, notches = [], [], []
    for j, f in enumerate(fi):
        hp = min(f + int(0.4 * fs), len(pz))                     # systolic peak within 400 ms
        if f < 0 or f + 2 >= hp:
            continue
        pk = f + int(np.argmax(pz[f:hp]))
        keep.append(j); sys_peaks.append(pk)
        down = pz[pk:min(pk + int(0.4 * fs), len(pz))]           # dicrotic notch on downslope
        ni_rel, _ = find_peaks(-down, prominence=0.05)
        if len(ni_rel):
            notches.append(pk + int(ni_rel[0]))
    keep = np.array(keep, int)
    return {"r_peaks": r, "r_paired": ri[keep], "feet": fi[keep],
            "sys_peaks": np.array(sys_peaks, int), "notches": np.array(notches, int),
            "pat_ms": dl[keep] * 1000.0}


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
def _shift_channel(x, d, edge=8):
    """Shift a 1-D-per-row channel (N, L) by d samples WITHOUT the circular wrap that np.roll
    introduces. The vacated end is filled by edge-value replication (hold the boundary sample),
    which avoids injecting the far-end of the waveform as a spurious pulse. `edge` samples at
    each end are additionally held flat so a reviewer cannot attribute the response to a
    wrap-around discontinuity."""
    N, L = x.shape
    out = np.empty_like(x)
    d = int(d)
    if d == 0:
        out[:] = x
    elif d > 0:                                   # shift later (right): pad front with x[:,0]
        out[:, d:] = x[:, :L - d]
        out[:, :d] = x[:, :1]
    else:                                         # shift earlier (left): pad tail with x[:,-1]
        out[:, :L + d] = x[:, -d:]
        out[:, L + d:] = x[:, -1:]
    return out


def causal_ptt_audit(model, X, fs, device, ppg_pos=PPG, deltas=(-6, -4, -2, 0, 2, 4, 6),
                     n_max=1500, seed=0, predict_fn=None, null_channels=None):
    """INPUT-space causal test: shift the PPG channel by +/- delta samples (later PPG =
    longer PTT) and measure how predicted BP responds. Faithful physiology = NEGATIVE
    slope (longer PTT -> lower BP). Returns per-target slope stats + mean response curves.
    DBP is the theoretically PTT-coupled target (pulse propagates during diastole).

    Hardened for publication:
      * non-circular shift (`_shift_channel`) so no wrap-around artifact drives the response;
      * `null_channels`: also shift a channel that carries NO arrival-time information as a
        negative control -- a faithful response to the PPG shift should be specific, so the
        null slope should sit near zero. Returns null slope stats under out['null'].

    predict_fn : optional callable (X_rolled) -> (M, 2); lets you audit a non-torch
    'model' such as an analytic detect-PTT-then-map estimator. Defaults to the torch model."""
    if predict_fn is None:
        predict_fn = lambda Xr: predict(model, Xr, device)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), min(n_max, len(X)), replace=False); sel.sort()
    Xs = X[sel]; dt = np.array(deltas) / fs

    def _sweep(shift_pos):
        preds = np.zeros((len(Xs), len(deltas), 2), np.float32)
        for j, d in enumerate(deltas):
            Xd = Xs.copy()
            Xd[:, :, shift_pos] = _shift_channel(Xs[:, :, shift_pos], d)
            preds[:, j] = predict_fn(Xd)
        return preds

    preds = _sweep(ppg_pos)
    out = {"shift_ms": (dt * 1000).tolist()}
    for k, name in [(0, "sbp"), (1, "dbp")]:
        slopes = np.array([np.polyfit(dt, preds[i, :, k], 1)[0] for i in range(len(Xs))])
        out[name] = {
            "dBP_dPTT": float(np.median(slopes)),
            "frac_correct_sign": float(np.mean(slopes < 0)),
            "resp_range_mmHg": float((preds[:, :, k].max(1) - preds[:, :, k].min(1)).mean()),
            "curve": preds[:, :, k].mean(0).tolist(),
        }
    # negative control: shifting a non-arrival-time channel should barely move BP
    if null_channels is not None:
        for nc in ([null_channels] if isinstance(null_channels, int) else null_channels):
            npreds = _sweep(nc)
            nsl = np.array([np.polyfit(dt, npreds[i, :, 1], 1)[0] for i in range(len(Xs))])
            out.setdefault("null", {})[f"ch{nc}"] = {
                "dBP_dPTT": float(np.median(nsl)),
                "resp_range_mmHg": float((npreds[:, :, 1].max(1) - npreds[:, :, 1].min(1)).mean()),
            }
    return out


def input_shift_audit(predict_fn, X, fs, ppg_pos=PPG, deltas=(-8, -6, -4, -2, 0, 2, 4, 6, 8)):
    """INPUT-space causal test for a single scalar-output model: np.roll the PPG channel by
    +/- delta samples -- a real change to arrival time, not an activation edit -- and watch how
    the output responds. Faithful PAT/PTT-use = NEGATIVE slope (later PPG = longer arrival time
    = lower predicted BP). `predict_fn`: (N, L, C) ndarray -> (N,) ndarray of scalar predictions.
    Returns (shift_ms, mean_response_curve, slope_mmHg_per_ms)."""
    dt_ms = np.array(deltas) / fs * 1000
    curve = []
    for delta in deltas:
        Xd = X.copy()
        Xd[:, :, ppg_pos] = np.roll(X[:, :, ppg_pos], int(delta), axis=1)
        curve.append(float(np.mean(predict_fn(Xd))))
    slope = float(np.polyfit(dt_ms, curve, 1)[0])
    return dt_ms, np.array(curve), slope


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


MORPH_KEYS = ["rise", "aix", "apg", "kurt", "notch", "decay", "peak"]


def wave_morphology(wave, fs):
    """Shape cues from ONE pulsatile wave (PPG or ABP), median over beats (nan if too few):
      rise  -- systolic rise time (s): foot -> systolic peak
      aix   -- reflection / augmentation index: (secondary-peak height) / (pulse height)
      apg   -- second-derivative (acceleration) b/a ratio: an arterial-stiffness index
      kurt  -- kurtosis (peakedness) of the pulse shape; a waveform / quality descriptor
      notch -- dicrotic-notch relative height: (notch - foot) / (systolic peak - foot)
      decay -- diastolic decay slope (1/s): normalized fall from the notch to the next foot
      peak  -- systolic peak height above the segment baseline (z-units)."""
    wz = _z(wave); feet = _pulse_feet(wz, fs)
    out = {k: np.nan for k in MORPH_KEYS}
    if len(feet) < 3:
        return out
    d2 = np.gradient(np.gradient(savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)))
    acc = {k: [] for k in MORPH_KEYS}
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
        acc["rise"].append(pk / fs)
        acc["peak"].append(float(beat[pk]))
        acc["kurt"].append(float(_kurtosis(beat)))
        tail = beat[pk + int(0.05 * fs):]
        if len(tail) > 3:
            sp, _ = find_peaks(tail)
            if len(sp):
                acc["aix"].append((tail[sp].max() - beat[0]) / pp)
        d2b = d2[s:e]; ap, _ = find_peaks(d2b, prominence=0.02 * (d2b.max() - d2b.min() + 1e-9))
        if len(ap):
            ai = ap[0]; hi = min(ai + int(0.25 * fs), len(d2b))
            if ai + 1 < hi and abs(d2b[ai]) > 1e-9:
                acc["apg"].append(d2b[ai:hi].min() / d2b[ai])
        # dicrotic notch: first local minimum on the systolic downslope after the peak
        down = beat[pk:]
        ni_rel, _ = find_peaks(-down, prominence=0.02 * pp)
        if len(ni_rel):
            ni = pk + int(ni_rel[0])
            acc["notch"].append((beat[ni] - beat[0]) / pp)
            dia = beat[ni:]                              # notch -> next foot: diastolic runoff
            if len(dia) > 3:
                t = np.arange(len(dia)) / fs
                acc["decay"].append(float(np.polyfit(t, dia, 1)[0]))
    return {k: (float(np.median(acc[k])) if len(acc[k]) >= 2 else np.nan) for k in MORPH_KEYS}


def compute_morphology(X, fs, ch):
    """Per-segment morphology dict (MORPH_KEYS) from channel `ch` of a batch (N, L, C). Use on
    the ABP channel to get ground-truth shape cues for validating the PPG-derived ones."""
    keys = MORPH_KEYS; acc = {k: [] for k in keys}
    for i in range(len(X)):
        m = wave_morphology(X[i, :, ch], fs)
        for k in keys:
            acc[k].append(m[k])
    return {k: np.array(v) for k, v in acc.items()}


def segment_scalars(ecg, ppg, fs):
    """Candidate BP cues from one ECG+PPG segment (nan where undetectable):
      pat                  -- ECG-R -> PPG-foot delay (s), intersecting-tangent foot; arrival-time
                              law (PAT, PEP-confounded)
      rise/aix/apg         -- PPG wave-shape / arterial-stiffness morphology cues
      kurt/notch/decay/peak-- PPG pulse kurtosis, dicrotic-notch height, diastolic decay, peak height
      hr                   -- heart rate (bpm); exploratory, ambiguous BP sign
      amp                  -- PPG pulse amplitude (au); negative control (removed by per-segment norm)."""
    pz = _z(ppg)
    r = r_peaks(ecg, fs)
    out = {"pat": np.nan, "hr": np.nan, "amp": np.nan, "period": np.nan}
    out.update(wave_morphology(ppg, fs))
    pf = _pulse_feet(pz, fs)                                  # cardiac period from PPG feet
    if len(pf) >= 3:
        out["period"] = float(np.median(np.diff(pf)) / fs)
    if len(r) < 3:
        return out
    out["pat"] = segment_ptt(ecg, ppg, fs)                   # paired R-peak -> tangent foot
    rr = np.diff(r) / fs; rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) >= 1:
        out["hr"] = float(60.0 / np.median(rr))
    feet, peaks = _ppg_fiducials(pz, r, fs)
    amps = [pz[pk] - pz[ft] for ft, pk in zip(feet, peaks)]
    if len(amps) >= 2:
        out["amp"] = float(np.median(amps))
    return out


def compute_scalars(X, fs, ecg_pos=ECG, ppg_pos=PPG):
    """Per-segment cue dict {pat, rise, aix, apg, kurt, notch, decay, peak, hr, period, amp}
    for a batch (N, L, C)."""
    keys = ["pat", "rise", "aix", "apg", "kurt", "notch", "decay", "peak", "hr", "period", "amp"]
    acc = {k: [] for k in keys}
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
        return {"slope": float("nan"), "frac_correct": float("nan"), "dependence": float("nan")}
    u = probe_direction(feats, mech)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, len(feats), n_pairs); donor = rng.integers(0, len(feats), n_pairs)
    proj = (feats[donor] - feats[base]) @ u
    dBP = head(feats[base] + np.outer(proj, u))[:, target] - head(feats[base])[:, target]
    dM = mech[donor] - mech[base]
    slope = float(np.polyfit(dM, dBP, 1)[0])
    fr_raw = float(np.mean(np.sign(dBP) == np.sign(dM)))          # positive-ref association
    ref = expect_sign if expect_sign != 0 else 1
    frac = float(np.mean(np.sign(dBP) == ref * np.sign(dM)))      # physiological-direction frac
    dependence = max(fr_raw, 1 - fr_raw)                          # sign-agnostic: does output USE it?
    return {"slope": slope, "frac_correct": frac, "dependence": dependence}


def mechanism_profile(feats, head, scalars, target=1):
    """Run subspace_swap across the candidate cues -> 'faithful to WHAT'. `scalars` is the
    dict from compute_scalars. Expected signs encode the textbook physiology per cue."""
    specs = [("PAT (arrival time)", "pat", -1), ("PPG rise-time (morphology)", "rise", -1),
             ("augmentation index (morphology)", "aix", +1), ("APG stiffness (morphology)", "apg", +1),
             ("dicrotic notch (morphology)", "notch", 0), ("diastolic decay (morphology)", "decay", 0),
             ("PPG kurtosis (morphology)", "kurt", 0), ("PPG peak height (morphology)", "peak", 0),
             ("cardiac period (f2f)", "period", 0), ("heart rate", "hr", 0),
             ("PPG amplitude (control)", "amp", 0)]

    def entry(key, sgn):
        s = subspace_swap(feats, head, scalars[key], target, sgn)
        s["expect_sign"] = sgn
        s["probe_r2"] = max(linear_probe(feats, scalars[key]), 0.0)   # how DECODABLE the cue is
        return s
    return {name: entry(key, sgn) for name, key, sgn in specs if key in scalars}


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
