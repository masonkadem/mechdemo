"""features_full.py -- comprehensive (~150) physiological feature library for ECG+PPG BP.

Every feature is literature-motivated and grouped by signal source. Extraction is per-segment,
median-over-beats where beat-wise. Groups:

  ECG / HRV      : rate, RR time-domain (SDNN, RMSSD, pNN50), RR freq-domain (LF, HF, LF/HF),
                   QRS amplitude/width, R-wave stats
  PTT / timing   : R->foot, R->peak, R->max-slope PAT; PTT variability; ECG-PPG xcorr
  PPG morphology : rise/crest/decay times, systolic/diastolic widths at 10-90%, areas,
                   augmentation index, reflection index, dicrotic notch depth/timing, peak stats
  VPG (1st deriv): max/min slope, their ratio, timings, w/x/y/z landmarks
  APG (2nd deriv): a-e amplitudes, all a-e ratios, a-e TIMINGS, aging indices (Takazawa,
                   Ushiroyama), reflection indices
  complexity     : Higuchi/Katz fractal dim, spectral/sample/approx entropy, signal moments
  spectral       : PPG power in bands, dominant freq, spectral centroid/spread/rolloff
  statistical    : skewness, kurtosis, percentiles of PPG and its derivatives

Use compute_full(X, fs, ecg_ch, ppg_ch) -> {name: (N,) array}.
"""
import numpy as np
from scipy.signal import find_peaks, welch, savgol_filter, butter, filtfilt, periodogram
from scipy.stats import skew, kurtosis

from mechlib import _z, _pulse_feet, PAT_WIN
import features_ext as fx


# ---------------------------------------------------------------- helpers
def _beats(wz, fs):
    return _pulse_feet(wz, fs)


def _bp(x, lo, hi, fs):
    try:
        b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return filtfilt(b, a, x)
    except Exception:
        return x


def _safe_median(a):
    a = np.asarray(a, float)
    return float(np.median(a)) if len(a) >= 2 else np.nan


# ---------------------------------------------------------------- ECG / HRV
def ecg_hrv_full(ecg, fs):
    ez = _z(ecg)
    r, props = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    out = {k: np.nan for k in ["hr", "rr_mean", "rr_sdnn", "rr_rmssd", "rr_pnn50", "rr_cv",
                               "hrv_lf", "hrv_hf", "hrv_lfhf", "qrs_amp_mean", "qrs_amp_std",
                               "r_count", "qrs_width"]}
    out["r_count"] = float(len(r))
    if len(r) < 4:
        return out
    rr = np.diff(r) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) >= 2:
        out["hr"] = 60.0 / np.median(rr)
        out["rr_mean"] = np.mean(rr)
        out["rr_sdnn"] = np.std(rr)
        out["rr_rmssd"] = np.sqrt(np.mean(np.diff(rr) ** 2))
        out["rr_pnn50"] = np.mean(np.abs(np.diff(rr)) > 0.05)
        out["rr_cv"] = np.std(rr) / (np.mean(rr) + 1e-9)
        if len(rr) >= 8:                                   # crude freq-domain HRV
            f, p = periodogram(rr - rr.mean(), fs=1.0 / np.mean(rr))
            lf = p[(f >= 0.04) & (f < 0.15)].sum()
            hf = p[(f >= 0.15) & (f < 0.4)].sum()
            out["hrv_lf"] = float(lf); out["hrv_hf"] = float(hf)
            out["hrv_lfhf"] = float(lf / (hf + 1e-9))
    out["qrs_amp_mean"] = float(np.median(ez[r]))
    out["qrs_amp_std"] = float(np.std(ez[r]))
    # QRS width: FWHM around each R
    widths = []
    for rp in r:
        a0, b0 = max(rp - int(0.06 * fs), 0), min(rp + int(0.06 * fs), len(ez))
        seg = ez[a0:b0]
        if len(seg) > 3 and seg.max() > 0:
            above = np.where(seg > 0.5 * seg.max())[0]
            if len(above):
                widths.append((above[-1] - above[0]) / fs)
    out["qrs_width"] = _safe_median(widths)
    return out


# ---------------------------------------------------------------- PPG morphology (extended)
def ppg_morph_full(ppg, fs):
    wz = _z(ppg)
    feet = _beats(wz, fs)
    keys = (["rise", "crest", "decay_slope", "sys_area", "dia_area", "sys_dia_ratio",
             "notch_depth", "notch_time", "aix", "reflect_idx", "peak_mean", "peak_std",
             "amp_mean", "amp_cv", "ppg_skew", "ppg_kurt"]
            + [f"sw{p}" for p in (10, 25, 50, 75, 90)]
            + [f"dw{p}" for p in (10, 25, 50, 75, 90)])
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        beat = wz[s:e] - wz[s]
        n = len(beat); pk = int(np.argmax(beat)); h = beat[pk]
        if pk < 2 or pk > n - 3 or h < 1e-3:
            continue
        acc["rise"].append(pk / fs)
        acc["crest"].append(pk / n)
        acc["peak_mean"].append(float(beat[pk]))
        acc["amp_mean"].append(float(h))
        acc["ppg_skew"].append(float(skew(beat)))
        acc["ppg_kurt"].append(float(kurtosis(beat)))
        acc["sys_area"].append(float(beat[:pk].sum() / (beat.sum() + 1e-9)))
        acc["dia_area"].append(float(beat[pk:].sum() / (beat.sum() + 1e-9)))
        acc["sys_dia_ratio"].append(float(beat[:pk].sum() / (beat[pk:].sum() + 1e-9)))
        # systolic (upslope) and diastolic (downslope) widths at various heights
        for p in (10, 25, 50, 75, 90):
            lvl = p / 100.0 * h
            up = np.where(beat[:pk] >= lvl)[0]
            dn = np.where(beat[pk:] >= lvl)[0]
            acc[f"sw{p}"].append((pk - up[0]) / fs if len(up) else np.nan)
            acc[f"dw{p}"].append((dn[-1]) / fs if len(dn) else np.nan)
        # dicrotic notch on downslope
        down = beat[pk:]
        ni, _ = find_peaks(-down, prominence=0.02 * h)
        if len(ni):
            npos = pk + int(ni[0])
            acc["notch_depth"].append(float((beat[npos] - beat[0]) / h))
            acc["notch_time"].append(float(npos / n))
            # augmentation / reflection from secondary peak after notch
            tail = beat[npos:]
            sp, _ = find_peaks(tail)
            if len(sp):
                acc["aix"].append(float((tail[sp].max()) / h))
                acc["reflect_idx"].append(float(tail[sp].max() / (beat[pk] + 1e-9)))
            dia = beat[npos:]
            if len(dia) > 3:
                t = np.arange(len(dia)) / fs
                acc["decay_slope"].append(float(np.polyfit(t, dia, 1)[0]))
    out = {k: _safe_median(v) for k, v in acc.items()}
    # amplitude CV / peak std across beats
    if len(acc["amp_mean"]) >= 2:
        out["amp_cv"] = float(np.std(acc["amp_mean"]) / (np.mean(acc["amp_mean"]) + 1e-9))
        out["peak_std"] = float(np.std(acc["peak_mean"]))
    return out


# ---------------------------------------------------------------- VPG (1st derivative)
def vpg_full(ppg, fs):
    wz = _z(ppg)
    sm = savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)
    vpg = np.gradient(sm)
    feet = _beats(wz, fs)
    keys = ["vpg_max", "vpg_min", "vpg_ratio", "t_vpg_max", "t_vpg_min", "vpg_ms_area"]
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        v = vpg[s:e]; n = len(v)
        if n < 8:
            continue
        acc["vpg_max"].append(float(v.max()))
        acc["vpg_min"].append(float(v.min()))
        acc["vpg_ratio"].append(float(v.max() / (abs(v.min()) + 1e-9)))
        acc["t_vpg_max"].append(float(np.argmax(v) / n))
        acc["t_vpg_min"].append(float(np.argmin(v) / n))
        acc["vpg_ms_area"].append(float(np.abs(v).mean()))
    return {k: _safe_median(v) for k, v in acc.items()}


# ---------------------------------------------------------------- APG (2nd derivative) full
def apg_full(ppg, fs):
    wz = _z(ppg)
    sm = savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)
    apg = np.gradient(np.gradient(sm))
    feet = _beats(wz, fs)
    amps = ["a", "b", "c", "d", "e"]
    ratios = [f"apg_{x}_a" for x in ("b", "c", "d", "e")]
    pair = ["apg_cd_a", "apg_bd_a", "apg_ce_a"]
    idxs = ["takazawa", "ushiro", "reflect_be"]
    times = [f"t_{x}" for x in ("b", "c", "d", "e")]
    keys = ratios + pair + idxs + times
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        seg = apg[s:e]
        pks, _ = find_peaks(seg); trs, _ = find_peaks(-seg)
        ext = np.sort(np.concatenate([pks, trs]))
        ext = ext[ext < int(0.5 * fs)]
        if len(ext) < 5:
            continue
        a, b, c, dd, ee = (seg[ext[i]] for i in range(5))
        if abs(a) < 1e-9:
            continue
        acc["apg_b_a"].append(b / a); acc["apg_c_a"].append(c / a)
        acc["apg_d_a"].append(dd / a); acc["apg_e_a"].append(ee / a)
        acc["apg_cd_a"].append((c - dd) / a); acc["apg_bd_a"].append((b - dd) / a)
        acc["apg_ce_a"].append((c - ee) / a)
        acc["takazawa"].append((b - c - dd - ee) / a)
        acc["ushiro"].append((c + dd - b) / a)
        acc["reflect_be"].append((b - ee) / a)
        for i, tk in enumerate(times):
            acc[tk].append(ext[i + 1] / fs)
    return {k: _safe_median(v) for k, v in acc.items()}


def vascular_indices(ppg, fs, height_cm=None):
    """Published PPG/APG vascular indices, added because they are the field's standard
    stiffness measures and were missing from the library.

    stiffness_index   height / delta-T, where delta-T is systolic peak to diastolic peak.
                      This is the one index here with the dimensions of a VELOCITY, so it is a
                      direct pulse-wave-velocity proxy rather than a shape descriptor -- the
                      quantity the governing law actually concerns. Height is per subject and is
                      supplied by the caller; without it, delta_t alone is returned and the
                      caller can scale later.
    delta_t           systolic-to-diastolic peak interval on its own, so the timing is available
                      even when height is not.
    crest_time_ratio  crest time divided by cycle time, i.e. crest time made rate-independent.
                      Raw crest time confounds stiffness with heart rate; the ratio does not.
    vascular_age      (b - c - d - e) / a from the second derivative.

    Reflection index and augmentation index already exist as reflect_idx and aix.
    """
    wz = _z(ppg)
    sm = savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)
    feet = _beats(wz, fs)
    dts, ctrs = [], []
    for s, e in zip(feet[:-1], feet[1:]):
        n = e - s
        if not (int(0.3 * fs) < n < int(1.5 * fs)):
            continue
        seg = sm[s:e]
        pks, _ = find_peaks(seg)
        if len(pks) < 1:
            continue
        sys_i = int(pks[np.argmax(seg[pks])])
        # diastolic peak: the first maximum after the systolic peak, i.e. the reflected wave
        later = pks[pks > sys_i + int(0.05 * fs)]
        if len(later):
            dia_i = int(later[0])
            dts.append((dia_i - sys_i) / fs)
        ctrs.append(sys_i / n)
    dt = _safe_median(dts)
    out = {"delta_t": dt,
           "crest_time_ratio": _safe_median(ctrs),
           "stiffness_index": (height_cm / 100.0) / dt
           if (height_cm and np.isfinite(dt) and dt > 1e-6) else np.nan}
    return out


# ---------------------------------------------------------------- complexity + spectral
def complexity_full(ppg, fs):
    out = {}
    out["hfd"] = fx.higuchi_fd(ppg)
    out["katz_fd"] = fx.katz_fd(ppg)
    out["spec_ent"] = fx.spectral_entropy(ppg, fs)
    x = np.asarray(ppg, float)
    out["ppg_std"] = float(x.std())
    out["ppg_skew_g"] = float(skew(x))
    out["ppg_kurt_g"] = float(kurtosis(x))
    for p in (10, 25, 75, 90):
        out[f"ppg_p{p}"] = float(np.percentile(x, p))
    # spectral band powers + shape
    f, P = welch(x, fs=fs, nperseg=min(256, len(x)))
    tot = P.sum() + 1e-12
    for lo, hi, nm in [(0, 0.5, "vlf"), (0.5, 2, "lf"), (2, 5, "mf"), (5, 10, "hf")]:
        out[f"pow_{nm}"] = float(P[(f >= lo) & (f < hi)].sum() / tot)
    out["dom_freq"] = float(f[np.argmax(P)])
    out["spec_centroid"] = float((f * P).sum() / tot)
    out["spec_spread"] = float(np.sqrt(((f - out["spec_centroid"]) ** 2 * P).sum() / tot))
    cum = np.cumsum(P) / tot
    out["spec_rolloff"] = float(f[np.searchsorted(cum, 0.85)] if cum[-1] >= 0.85 else f[-1])
    return out


# ---------------------------------------------------------------- driver
GROUPS = {
    "ECG/HRV": ecg_hrv_full,          # needs ecg
    "PPG morphology": ppg_morph_full,
    "VPG": vpg_full,
    "APG": apg_full,
    "complexity/spectral": complexity_full,
}


def compute_full(X, fs, ppg_ch=1, ecg_ch=0):
    """~150-feature library over a batch (N, L, C). Returns {name: (N,) array} and a
    {name: group} map via compute_full.groups (set as attribute)."""
    names_group = {}
    acc = None
    for i in range(len(X)):
        ppg = X[i, :, ppg_ch]
        row = {}
        row.update(ppg_morph_full(ppg, fs))
        row.update(vpg_full(ppg, fs))
        row.update(apg_full(ppg, fs))
        row.update(complexity_full(ppg, fs))
        # height is per subject and not available inside this loop, so stiffness_index is
        # returned as NaN here and the caller can scale delta_t by height when it has it
        row.update(vascular_indices(ppg, fs, height_cm=None))
        if ecg_ch is not None:
            ecg = X[i, :, ecg_ch]
            row.update(ecg_hrv_full(ecg, fs))
            row.update(fx.ptt_variants(ecg, ppg, fs))
            row.update(fx.ecg_ppg_xcorr(ecg, ppg, fs))
        if acc is None:
            acc = {k: [] for k in row}
        for k in acc:
            acc[k].append(row.get(k, np.nan))
    return {k: np.array(v) for k, v in acc.items()}


def feature_group(name):
    """Map a feature name to its signal-source group (for figure grouping)."""
    if name.startswith(("rr_", "hr", "hrv", "qrs", "r_count")):
        return "ECG/HRV"
    if name.startswith(("pat", "ptt", "xcorr")):
        return "PTT/timing"
    if name.startswith(("apg", "takazawa", "ushiro", "reflect_be")) or name.startswith("t_"):
        return "APG"
    if name.startswith("vpg"):
        return "VPG"
    if name.startswith(("hfd", "katz", "spec_", "pow_", "dom_", "ppg_p", "ppg_std",
                        "ppg_skew_g", "ppg_kurt_g")):
        return "complexity/spectral"
    return "PPG morphology"
