"""physics_audit.py -- signal-space perturbations whose effect on BP is fixed by physiology,
plus the external BP-Benchmark loaders.

The idea: a linear probe says a cue is DECODABLE; it never says the model USES it. So for
each cue we know a governing law for, we edit the raw waveform to move that cue in a known
direction and check whether the prediction moves the way physics says it must. Probe R^2 high
+ response sign wrong = the model has the information and is using it backwards, which is the
signature that survives in-distribution accuracy and breaks out of distribution.

Sign conventions below are all "effect on BP of INCREASING the named quantity".
"""
import numpy as np
from scipy.signal import find_peaks

from mechlib import ECG, PPG, _z


# --------------------------------------------------------------- governing laws
# expect: sign of dBP/d(cue). 0 = no confident textbook prediction (exploratory).
# law:    the physical relation being tested.
LAWS = {
    "pat": dict(expect=-1, law="Moens-Korteweg / Bramwell-Hill: PWV ~ sqrt(Eh/2rho R), and "
                                "stiffer vessel at higher BP => faster wave => SHORTER arrival "
                                "time. Longer PAT must predict LOWER BP."),
    "rise": dict(expect=-1, law="Systolic upstroke time falls as ejection meets a stiffer, "
                                "higher-pressure arterial tree. Longer rise => LOWER BP."),
    "aix": dict(expect=+1, law="Augmentation index: faster PWV returns the reflected wave into "
                               "systole, raising late systolic pressure. Higher AIx => HIGHER BP."),
    "apg": dict(expect=+1, law="APG b/a stiffness index rises with arterial stiffening, which "
                               "tracks higher pressure. Higher => HIGHER BP."),
    "hr": dict(expect=+1, law="Rate-pressure coupling under sympathetic drive (BP = CO x SVR, "
                              "CO = HR x SV). Weak/ambiguous: SV falls as HR rises."),
    "period": dict(expect=-1, law="Cardiac period is 60/HR, so it carries the inverse of the "
                                  "HR relation."),
    "notch": dict(expect=0, law="Dicrotic notch height moves with reflection timing and SVR; "
                                "no clean monotone BP law."),
    "decay": dict(expect=0, law="Diastolic decay ~ RC windkessel time constant; depends on both "
                                "compliance and resistance, so the BP sign is not fixed."),
    "kurt": dict(expect=0, law="Pulse kurtosis is a shape summary with no direct pressure law."),
    "peak": dict(expect=0, law="Peak height after per-segment normalization is near-arbitrary."),
    "amp": dict(expect=0, law="NEGATIVE CONTROL: per-segment amplitude normalization removes "
                              "scale, so a faithful model must be INSENSITIVE to gain."),
}


# --------------------------------------------------------------- perturbations
# Each returns a copy of X with one physiological quantity moved in a known direction.
# `delta` is signed and in the cue's own units where meaningful.

def perturb_pat(X, fs, delta_s, ppg_pos=PPG):
    """Shift PPG later by delta_s seconds => arrival time (PAT) INCREASES by delta_s.
    Requires ECG present as the timing reference."""
    Xo = X.copy()
    Xo[:, :, ppg_pos] = np.roll(X[:, :, ppg_pos], int(round(delta_s * fs)), axis=1)
    return Xo


def perturb_rise(X, fs, delta, ppg_pos=PPG):
    """Warp each pulse's systolic upstroke to be longer (delta>0) or shorter (delta<0) while
    holding period and amplitude. Implemented as a piecewise-linear time warp anchored on
    pulse feet, so the beat grid and pulse height are preserved and only upstroke DURATION moves."""
    Xo = X.copy()
    L = X.shape[1]
    for i in range(len(X)):
        w = X[i, :, ppg_pos]
        wz = _z(w)
        feet = find_peaks(-wz, distance=max(int(0.4 * fs), 1), prominence=0.3)[0]
        if len(feet) < 2:
            continue
        out = w.copy()
        for a, b in zip(feet[:-1], feet[1:]):
            seg = w[a:b]
            n = len(seg)
            if n < 8:
                continue
            pk = int(np.argmax(seg))
            if pk < 2 or pk > n - 3:
                continue
            pk2 = int(np.clip(pk * (1.0 + delta), 2, n - 3))   # move the peak's position in time
            src = np.concatenate([np.linspace(0, pk, pk2, endpoint=False),
                                  np.linspace(pk, n - 1, n - pk2)])
            out[a:b] = np.interp(src, np.arange(n), seg)
        Xo[i, :, ppg_pos] = out
    return Xo


def perturb_aix(X, fs, delta, ppg_pos=PPG):
    """Add/remove a reflected wave: superpose a delayed, scaled copy of each pulse in late
    systole. delta>0 raises the augmentation index.

    The reflection is injected as a bump on the DIASTOLIC side of each beat (peak + 0.25 s)
    rather than a whole-signal roll: rolling the full waveform also raises the primary peak,
    which is what the measured AIx divides by, and that made the net AIx move the wrong way.
    Validated to give median dAIx > 0 for delta > 0."""
    Xo = X.copy()
    lag = int(0.25 * fs)
    for i in range(len(X)):
        w = X[i, :, ppg_pos]
        wz = _z(w)
        feet = find_peaks(-wz, distance=max(int(0.4 * fs), 1), prominence=0.3)[0]
        if len(feet) < 2:
            continue
        out = w.copy()
        for a, b in zip(feet[:-1], feet[1:]):
            seg = w[a:b]
            n = len(seg)
            if n < 8:
                continue
            pk = int(np.argmax(seg))
            hgt = seg[pk] - seg.min()
            j = pk + lag                                   # reflected-wave arrival, in diastole
            if j >= n - 2:
                continue
            width = max(int(0.08 * fs), 2)                 # gaussian bump, ~80 ms wide
            t = np.arange(n)
            out[a:b] = seg + delta * hgt * np.exp(-0.5 * ((t - j) / width) ** 2)
        Xo[i, :, ppg_pos] = out
    return Xo


def perturb_hr(X, fs, delta, ppg_pos=PPG, ecg_pos=ECG, both=True):
    """Resample the whole segment in time by (1+delta) and crop/pad back to length, changing
    beat rate without changing pulse shape. delta<0 => faster heart rate."""
    Xo = X.copy()
    L = X.shape[1]
    src = np.clip(np.arange(L) * (1.0 + delta), 0, L - 1)
    chans = [ppg_pos] + ([ecg_pos] if both else [])
    for c in chans:
        for i in range(len(X)):
            Xo[i, :, c] = np.interp(src, np.arange(L), X[i, :, c])
    return Xo


def perturb_amp(X, fs, delta, ppg_pos=PPG):
    """NEGATIVE CONTROL: pure gain change on PPG. True BP is unchanged, so a faithful model
    must not respond."""
    Xo = X.copy()
    Xo[:, :, ppg_pos] = X[:, :, ppg_pos] * (1.0 + delta)
    return Xo


def perturb_decay(X, fs, delta, ppg_pos=PPG):
    """Multiply the diastolic tail of each beat by an exponential, lengthening (delta>0) or
    shortening the windkessel-like decay."""
    Xo = X.copy()
    for i in range(len(X)):
        w = X[i, :, ppg_pos]
        wz = _z(w)
        feet = find_peaks(-wz, distance=max(int(0.4 * fs), 1), prominence=0.3)[0]
        if len(feet) < 2:
            continue
        out = w.copy()
        for a, b in zip(feet[:-1], feet[1:]):
            seg = w[a:b].copy()
            n = len(seg)
            if n < 8:
                continue
            pk = int(np.argmax(seg))
            tail = seg[pk:]
            base = tail[-1]
            t = np.arange(len(tail)) / max(fs, 1)
            out[a + pk:b] = base + (tail - base) * np.exp(-delta * t / 0.3)
        Xo[i, :, ppg_pos] = out
    return Xo


PERTURBATIONS = {
    "pat": (perturb_pat, [-0.048, -0.024, 0.0, 0.024, 0.048], "s"),
    "rise": (perturb_rise, [-0.30, -0.15, 0.0, 0.15, 0.30], "frac"),
    "aix": (perturb_aix, [-0.30, -0.15, 0.0, 0.15, 0.30], "frac"),
    "hr": (perturb_hr, [-0.15, -0.075, 0.0, 0.075, 0.15], "frac (neg=faster)"),
    "decay": (perturb_decay, [-1.0, -0.5, 0.0, 0.5, 1.0], "1/s"),
    "amp": (perturb_amp, [-0.4, -0.2, 0.0, 0.2, 0.4], "frac"),
}

# `period` and `hr` are the same manipulation with opposite sign; `hr` covers both.
PPG_ONLY = [k for k in PERTURBATIONS if k != "pat"]   # runnable without an ECG channel


def run_battery(predict_fn, X, fs, cues=None, target=1, has_ecg=True, n_max=800, seed=0):
    """For each cue: sweep the perturbation, fit dBP/dcue, and compare its sign to the
    governing law. `predict_fn`: (N, L, C) -> (N, 2) in mmHg. target 0=SBP, 1=DBP.

    Returns {cue: {slope, expect, sign_ok, curve, deltas, units, law, resp_range}}.
    Note `hr` warps the time axis, so its response mixes rate with any length sensitivity;
    it is reported but treated as exploratory (expect=+1 is weak)."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), min(n_max, len(X)), replace=False)
    sel.sort()
    Xs = X[sel]
    cues = cues or (list(PERTURBATIONS) if has_ecg else PPG_ONLY)
    # PPG is channel 1 next to ECG, but channel 0 when the model is PPG-only
    ppg_pos = PPG if has_ecg else 0

    out = {}
    for cue in cues:
        fn, deltas, units = PERTURBATIONS[cue]
        curve = []
        for dlt in deltas:
            kw = {"ppg_pos": ppg_pos}
            if cue == "hr":
                kw["both"] = has_ecg          # no ECG channel to co-warp in PPG-only mode
                kw["ecg_pos"] = ECG
            Xd = Xs if dlt == 0 else fn(Xs, fs, dlt, **kw)
            curve.append(float(predict_fn(Xd)[:, target].mean()))
        curve = np.array(curve)
        slope = float(np.polyfit(deltas, curve, 1)[0])
        exp = LAWS[cue]["expect"]
        out[cue] = {
            "slope": slope,
            "expect": exp,
            # sign_ok is None for cues with no confident law; for `amp` (negative control)
            # "correct" means a NEAR-ZERO response, judged on the response range instead.
            "sign_ok": (None if exp == 0 else bool(np.sign(slope) == exp)),
            "curve": curve.tolist(),
            "deltas": list(deltas),
            "units": units,
            "resp_range": float(curve.max() - curve.min()),
            "law": LAWS[cue]["law"],
        }
    return out


# --------------------------------------------------------------- external datasets
def load_bpbenchmark(folder, name=None):
    """Load a BP-Benchmark external set (BCG / UCI / Sensors / PPG-BP) from its 5-fold
    signal_fold_*.mat files, following Preprocess_External_dataset.py conventions.
    These sets are PPG-ONLY -- there is no ECG channel, so PAT is undefined on them.
    Returns dict(X=(N, L, 1), y=(N, 2) [SBP, DBP], g=subject ids, name)."""
    import scipy.io as sio
    from pathlib import Path

    folder = Path(folder)
    S, SP, DP, PAT_, AGE, SEX = [], [], [], [], [], []

    def _find(m, opts):
        for k in m:
            if k.lower() in opts or any(k.lower().startswith(o) for o in opts):
                return k
        return None

    for f in sorted(folder.glob("signal_fold_*.mat")):
        m = sio.loadmat(f)
        n = len(m["signal"])
        S.append(np.asarray(m["signal"], dtype=np.float32))
        SP.append(np.asarray(m["SP"], dtype=np.float32)[:, 0])
        DP.append(np.asarray(m["DP"], dtype=np.float32)[:, 0])
        PAT_.append(np.array([str(x[0]).strip() for x in m["patient"]]))
        ak = _find(m, {"age", "age(year)"})
        sk = _find(m, {"gender", "sex", "sex(m/f)"})
        AGE.append(np.asarray(m[ak], float)[:, 0] if ak else np.full(n, np.nan))
        # sex -> 1 male, 0 female, nan unknown
        if sk:
            SEX.append(np.array([1.0 if str(x[0]).strip().lower().startswith("m") else
                                 0.0 if str(x[0]).strip().lower().startswith("f") else np.nan
                                 for x in m[sk]]))
        else:
            SEX.append(np.full(n, np.nan))
    if not S:
        raise FileNotFoundError(f"no signal_fold_*.mat in {folder}")
    X = np.concatenate(S)[:, :, None]
    y = np.stack([np.concatenate(SP), np.concatenate(DP)], 1)
    g = np.concatenate(PAT_)
    _, g = np.unique(g, return_inverse=True)
    age, sex = np.concatenate(AGE), np.concatenate(SEX)
    demo = None if not np.isfinite(age).any() else {"age": age, "sex": sex}
    return dict(X=X, y=y, g=g, demo=demo, name=name or folder.stem)


def load_mimic_bp(root, channels=("ecg", "ppg"), max_patients=None, seed=0):
    """MIMIC-BP (Gonzalez et al. 2023) external OOD set: per-patient .npy waveforms with
    continuous-ABP-derived SBP/DBP labels. Unlike the BP-Benchmark sets this has ECG *and*
    PPG at 125 Hz in VitalDB's channel convention (ECG=0, PPG=1, ABP=2), so the full PAT
    roll-audit runs on it. Layout (see BP/read_data.py):
        {root}/{abp,ecg,ppg,resp}/p######_{wav}.npy  each (30, 3750)
        {root}/labels/p######_labels.npy             (30, 2) = [SBP, DBP]
    Returns dict(X=(N, 3750, C), y=(N, 2), g=patient index, name).
    Different hospital than VitalDB (BIDMC ICU vs SNUH surgical) => genuine distribution shift."""
    from pathlib import Path
    root = Path(root)
    pats = sorted(p.stem[:-len("_labels")] for p in (root / "labels").glob("p*_labels.npy"))
    if max_patients:
        rng = np.random.default_rng(seed)
        pats = list(rng.choice(pats, min(max_patients, len(pats)), replace=False))
    Xs, ys, gs = [], [], []
    for gi, pid in enumerate(pats):
        try:
            lab = np.load(root / "labels" / f"{pid}_labels.npy").astype(np.float32)  # (30,2)
            chans = [np.load(root / c / f"{pid}_{c}.npy").astype(np.float32) for c in channels]
        except FileNotFoundError:
            continue
        W = np.stack(chans, -1)                            # (30, 3750, C)
        ok = np.isfinite(W).all((1, 2)) & np.isfinite(lab).all(1)
        ok &= (lab[:, 0] > 60) & (lab[:, 0] < 220) & (lab[:, 1] > 30) & (lab[:, 1] < 140)
        if ok.sum() == 0:
            continue
        Xs.append(W[ok]); ys.append(lab[ok]); gs.append(np.full(ok.sum(), gi))
    if not Xs:
        raise FileNotFoundError(f"no usable MIMIC-BP patients under {root}")
    return dict(X=np.concatenate(Xs), y=np.concatenate(ys), g=np.concatenate(gs), name="mimic_bp")


def window_segments(X, L_out):
    """Cut each (L, C) record into consecutive non-overlapping windows of length L_out and
    stack -> (N*k, L_out, C). Preferred over resampling for MIMIC-BP (3750 @125Hz = 30 s ->
    three 10 s / 1250-sample windows), because it keeps the native frequency content the
    model was trained on instead of stretching the time axis."""
    N, L, C = X.shape
    k = L // L_out
    if k < 1:
        return resample_to(X, L_out).astype(np.float32), 1
    W = X[:, :k * L_out].reshape(N, k, L_out, C).reshape(N * k, L_out, C)
    return W.astype(np.float32), k       # k = windows per input record (row-major, record-major)


def resample_to(X, L_out):
    """Linear-resample (N, L, C) to length L_out. Used to bring external sets onto the
    input length the model was trained at (BCG 625, PPG-BP 262 -> common L)."""
    N, L, C = X.shape
    if L == L_out:
        return X.astype(np.float32)
    src = np.linspace(0, L - 1, L_out)
    out = np.empty((N, L_out, C), np.float32)
    for c in range(C):
        for i in range(N):
            out[i, :, c] = np.interp(src, np.arange(L), X[i, :, c])
    return out


def distribution_distance(cues_id, cues_ood):
    """Quantify the ID->OOD shift per cue with a KS statistic (0 = identical, 1 = disjoint),
    so the 'this really is out of distribution' claim is measured, not asserted. `cues_*` are
    the scalar dicts from compute_scalars/compute_morphology."""
    from scipy.stats import ks_2samp
    out = {}
    for k in cues_id:
        if k not in cues_ood:
            continue
        a = np.asarray(cues_id[k], float); b = np.asarray(cues_ood[k], float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) > 20 and len(b) > 20:
            out[k] = float(ks_2samp(a, b).statistic)
    return out


def bootstrap_mae(pred, y, g, n_boot=1000, seed=0):
    """Subject-level bootstrap CI on MAE. The external sets have as few as 40 subjects, so a
    bare MAE overstates precision -- resample SUBJECTS, not segments."""
    rng = np.random.default_rng(seed)
    subs = np.unique(g)
    idx_by = {s: np.where(g == s)[0] for s in subs}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(subs, len(subs), replace=True)
        ii = np.concatenate([idx_by[s] for s in pick])
        stats.append(np.abs(pred[ii] - y[ii]).mean(0))
    stats = np.array(stats)
    return {"mae": np.abs(pred - y).mean(0).tolist(),
            "lo": np.percentile(stats, 2.5, axis=0).tolist(),
            "hi": np.percentile(stats, 97.5, axis=0).tolist()}
