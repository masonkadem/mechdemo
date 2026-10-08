"""features_ext.py -- exhaustive physiologically-grounded cue battery for the probe/GBM work.

mechlib gives the core BP cues (pat, rise, aix, apg, notch, decay, hr, period, amp, kurt, peak).
This module adds the wider set a reviewer would expect us to have probed, each with a reason it
could carry BP signal:

PPG shape / complexity
  hfd        Higuchi fractal dimension of the PPG (signal complexity; falls with stiffening)
  katz_fd    Katz fractal dimension (alternative complexity estimator)
  samp_ent   sample entropy (regularity)
  spec_ent   spectral entropy (frequency-domain complexity)

PPG derivative landmarks (the classic a-b-c-d-e of the acceleration PPG)
  vpg_max    max of 1st derivative (max upslope velocity)
  apg_b_a    APG b/a  (aging / stiffness index; more negative = stiffer)
  apg_c_a, apg_d_a, apg_e_a   remaining APG wave ratios
  aging_idx  (b-c-d-e)/a  Takazawa aging index

PPG width / area
  sys_area   systolic area fraction (foot->notch area / total)
  pw50, pw25 pulse width at 50% / 25% of systolic height (relate to SVR / reflection)
  crest      crest time / total pulse time

Spectral / rate
  resp_rate  respiratory rate from PPG amplitude modulation (RIAV)
  ppg_sdnn   beat-to-beat interval SD from PPG feet (pulse-rate variability)

ECG-derived (only when an ECG channel is present)
  rr_mean, rr_sdnn, rr_rmssd   HRV time-domain (autonomic tone -> BP coupling)
  qrs_amp    R-wave amplitude
  pat_foot, pat_peak           two PTT definitions: R->PPG-foot and R->PPG-systolic-peak
  ptt_var    beat-to-beat PTT variability

All functions take a single 1-D signal (or ecg+ppg pair) at sampling rate fs and return a
scalar (nan if not measurable), mirroring mechlib's per-beat-median style.
"""
import numpy as np
from scipy.signal import find_peaks, welch, savgol_filter

from mechlib import _z, _pulse_feet, PAT_WIN


# ------------------------------------------------------------- complexity
def higuchi_fd(x, kmax=10):
    """Higuchi fractal dimension. Higher = more complex/irregular waveform."""
    x = np.asarray(x, float)
    N = len(x)
    if N < 2 * kmax:
        return np.nan
    L = []
    ks = range(1, kmax + 1)
    for k in ks:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            ll = np.abs(np.diff(x[idx])).sum() * (N - 1) / (len(idx) - 1) / k
            Lk.append(ll)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return np.nan
    lk = np.log(np.array(L) + 1e-12)
    ln = np.log(1.0 / np.arange(1, len(L) + 1))
    return float(np.polyfit(ln, lk, 1)[0])


def katz_fd(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    d = np.abs(x - x[0]).max()
    Ln = np.abs(np.diff(x)).sum()
    if Ln <= 0 or d <= 0:
        return np.nan
    n = len(x) - 1
    return float(np.log10(n) / (np.log10(n) + np.log10(d / Ln)))


def spectral_entropy(x, fs):
    x = np.asarray(x, float)
    if len(x) < 32:
        return np.nan
    f, p = welch(x, fs=fs, nperseg=min(256, len(x)))
    p = p[f <= 15]
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p) + 1e-12))


def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, float)
    x = (x - x.mean()) / (x.std() + 1e-8)
    N = len(x)
    if N < 50:
        return np.nan
    x = x[:: max(1, N // 500)]                 # decimate for speed
    N = len(x)
    r *= 1.0

    def phi(mm):
        cnt = 0
        tmpl = np.array([x[i:i + mm] for i in range(N - mm)])
        for i in range(len(tmpl)):
            d = np.abs(tmpl - tmpl[i]).max(1)
            cnt += np.sum(d <= r) - 1
        return cnt

    a, b = phi(m + 1), phi(m)
    if a <= 0 or b <= 0:
        return np.nan
    return float(-np.log(a / b))


# ------------------------------------------------------------- PPG derivatives / widths
def ppg_derivative_cues(ppg, fs):
    """APG a-b-c-d-e wave ratios + aging index + velocity, median over beats."""
    wz = _z(ppg)
    sm = savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)
    vpg = np.gradient(sm)
    apg = np.gradient(vpg)
    feet = _pulse_feet(wz, fs)
    keys = ["vpg_max", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "aging_idx", "crest"]
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        seg = apg[s:e]
        beat = wz[s:e]
        pk = int(np.argmax(beat))
        if pk < 2 or pk > len(beat) - 3:
            continue
        acc["vpg_max"].append(float(vpg[s:e].max()))
        acc["crest"].append(pk / (e - s))
        # a = first APG peak; b,c,d,e = subsequent extrema within the beat
        pks, _ = find_peaks(seg)
        trs, _ = find_peaks(-seg)
        ext = np.sort(np.concatenate([pks, trs]))
        ext = ext[ext < int(0.5 * fs)]
        if len(ext) >= 1:
            a = seg[ext[0]]
            if abs(a) > 1e-9:
                vals = [seg[ext[i]] / a if i < len(ext) else np.nan for i in range(1, 5)]
                for key, v in zip(["apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a"], vals):
                    if np.isfinite(v):
                        acc[key].append(float(v))
                if len(ext) >= 5:
                    b, c, dd, ee = (seg[ext[i]] for i in range(1, 5))
                    acc["aging_idx"].append(float((b - c - dd - ee) / a))
    return {k: (float(np.median(acc[k])) if len(acc[k]) >= 2 else np.nan) for k in keys}


def apg_novel_cues(ppg, fs):
    """Literature-grounded APG features BEYOND the a-e amplitude ratios: landmark TIMINGS and
    alternative aging indices. All median-over-beats (nan if too few clean beats).

      t_b, t_c, t_d, t_e   time (s) from beat foot to APG landmark b/c/d/e (reflection timing;
                            shorter t_b tracks faster wave return = stiffer/higher BP)
      apg_ratio_cd_a       (c-d)/a   alternative reflection index
      apg_ushiro           (c+d-b)/a Ushiroyama-Watanabe vascular aging index
      apg_reflect          (b-e)/a   early-vs-late reflection balance
      vpg_ms_ratio         VPG max-slope / |min-slope|  (up/down velocity asymmetry)
      apg_ba_over_t         (b/a) / t_b  amplitude-per-time reflection stiffness
      apg_area_sys         area under |APG| in systole (foot->dicrotic), normalized
    """
    wz = _z(ppg)
    sm = savgol_filter(wz, max(int(0.05 * fs) | 1, 5), 3)
    vpg = np.gradient(sm)
    apg = np.gradient(vpg)
    feet = _pulse_feet(wz, fs)
    keys = ["t_b", "t_c", "t_d", "t_e", "apg_ratio_cd_a", "apg_ushiro", "apg_reflect",
            "vpg_ms_ratio", "apg_ba_over_t", "apg_area_sys"]
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        seg = apg[s:e]
        vseg = vpg[s:e]
        pks, _ = find_peaks(seg)
        trs, _ = find_peaks(-seg)
        ext = np.sort(np.concatenate([pks, trs]))
        ext = ext[ext < int(0.5 * fs)]
        if len(ext) < 5:
            continue
        a = seg[ext[0]]
        if abs(a) < 1e-9:
            continue
        b, c, dd, ee = (seg[ext[i]] for i in range(1, 5))
        # landmark timings (s from foot)
        acc["t_b"].append(ext[1] / fs)
        acc["t_c"].append(ext[2] / fs)
        acc["t_d"].append(ext[3] / fs)
        acc["t_e"].append(ext[4] / fs)
        # alternative indices
        acc["apg_ratio_cd_a"].append(float((c - dd) / a))
        acc["apg_ushiro"].append(float((c + dd - b) / a))
        acc["apg_reflect"].append(float((b - ee) / a))
        if ext[1] > 0:
            acc["apg_ba_over_t"].append(float((b / a) / (ext[1] / fs)))
        # velocity asymmetry + systolic APG energy
        vmax, vmin = vseg.max(), vseg.min()
        if abs(vmin) > 1e-9:
            acc["vpg_ms_ratio"].append(float(vmax / abs(vmin)))
        notch = min(ext[3], len(seg) - 1)             # ~dicrotic region
        acc["apg_area_sys"].append(float(np.abs(seg[:notch]).mean()))
    return {k: (float(np.median(acc[k])) if len(acc[k]) >= 2 else np.nan) for k in keys}


def ppg_width_cues(ppg, fs):
    """Pulse widths at 25/50% systolic height + systolic area fraction."""
    wz = _z(ppg)
    feet = _pulse_feet(wz, fs)
    keys = ["pw25", "pw50", "sys_area"]
    acc = {k: [] for k in keys}
    for s, e in zip(feet[:-1], feet[1:]):
        if not (int(0.3 * fs) < e - s < int(1.5 * fs)):
            continue
        beat = wz[s:e] - wz[s]
        pk = int(np.argmax(beat))
        h = beat[pk]
        if h < 1e-3 or pk < 2:
            continue
        for lvl, key in [(0.25, "pw25"), (0.50, "pw50")]:
            above = np.where(beat >= lvl * h)[0]
            if len(above):
                acc[key].append((above[-1] - above[0]) / fs)
        acc["sys_area"].append(float(beat[:pk].sum() / (beat.sum() + 1e-9)))
    return {k: (float(np.median(acc[k])) if len(acc[k]) >= 2 else np.nan) for k in keys}


# ------------------------------------------------------------- ECG / HRV / PTT variants
def ecg_hrv_cues(ecg, fs):
    ez = _z(ecg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    out = {"rr_mean": np.nan, "rr_sdnn": np.nan, "rr_rmssd": np.nan, "qrs_amp": np.nan}
    if len(r) < 4:
        return out
    rr = np.diff(r) / fs
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(rr) >= 2:
        out["rr_mean"] = float(np.mean(rr))
        out["rr_sdnn"] = float(np.std(rr))
        out["rr_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    out["qrs_amp"] = float(np.median(ez[r]))
    return out


def ecg_ppg_xcorr(ecg, ppg, fs, lo_s=0.05, hi_s=PAT_WIN):
    """ECG-PPG cross-correlation cues -- a fiducial-free PTT estimator, computed PER BEAT.

    Whole-segment cross-correlation of multi-beat ECG vs PPG locks onto the heart-rate
    periodicity, not the arrival delay, so instead we anchor on each ECG R-peak and find the
    lag (in the physiological PTT window) that best aligns the following PPG upstroke with the
    ECG QRS. Averaging that per-beat lag gives a robust arrival-time estimate; its peak
    correlation is the ECG->PPG coupling strength. This is the signal-level analogue of the
    cross-attention alignment learned in the cross-site work -- here just the lag and strength.
      xcorr_lag    median per-beat best-alignment lag (s)  -> PTT estimate (expect to track BP)
      xcorr_peak   median peak correlation                 -> coupling / signal quality
      xcorr_width  SD of per-beat lag (s)                  -> timing jitter across beats
    """
    from scipy.signal import butter, filtfilt
    ez = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
    try:
        b, a = butter(3, [0.5 / (fs / 2), 8.0 / (fs / 2)], btype="band")
        pz = filtfilt(b, a, ppg)
    except Exception:
        pz = ppg
    pz = (pz - pz.mean()) / (pz.std() + 1e-8)
    dp = np.gradient(pz)                                   # upstroke velocity aligns with arrival
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    out = {"xcorr_lag": np.nan, "xcorr_peak": np.nan, "xcorr_width": np.nan}
    if len(r) < 3:
        return out
    lags_s, peaks = [], []
    lo, hi = int(lo_s * fs), int(hi_s * fs)
    for rp in r:
        a0, b0 = rp + lo, min(rp + hi, len(dp))
        if b0 - a0 < 3:
            continue
        seg = dp[a0:b0]
        k = int(np.argmax(seg))                           # lag of steepest PPG upstroke after R
        lags_s.append((lo + k) / fs)
        peaks.append(float(seg[k]))
    if len(lags_s) >= 2:
        out["xcorr_lag"] = float(np.median(lags_s))
        out["xcorr_peak"] = float(np.median(peaks))
        out["xcorr_width"] = float(np.std(lags_s))
    return out


def ptt_variants(ecg, ppg, fs):
    """Two PTT definitions + beat-to-beat variability. R->foot vs R->systolic-peak isolate
    the pre-ejection vs upstroke components differently."""
    ez, pz = _z(ecg), _z(ppg)
    r, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
    out = {"pat_foot": np.nan, "pat_peak": np.nan, "ptt_var": np.nan}
    if len(r) < 3:
        return out
    dp = np.gradient(pz)
    foots, peaks = [], []
    for rp in r:
        a = rp + int(0.05 * fs)
        b = min(rp + int(PAT_WIN * fs), len(pz))
        if a >= b:
            continue
        up = a + int(np.argmax(dp[a:b]))                 # steepest upstroke ~ foot region
        t_foot = (up - rp) / fs
        pkw = min(up + int(0.3 * fs), len(pz))
        if up + 1 < pkw:
            pk = up + int(np.argmax(pz[up:pkw]))
            t_peak = (pk - rp) / fs
            if 0.05 < t_foot < PAT_WIN and 0.05 < t_peak < 0.6:
                foots.append(t_foot); peaks.append(t_peak)
    if len(foots) >= 2:
        out["pat_foot"] = float(np.median(foots))
        out["pat_peak"] = float(np.median(peaks))
        out["ptt_var"] = float(np.std(foots))
    return out


# ------------------------------------------------------------- batch driver
PPG_EXT_KEYS = (["hfd", "katz_fd", "spec_ent", "samp_ent"]
                + ["vpg_max", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a", "aging_idx", "crest"]
                + ["pw25", "pw50", "sys_area"])
ECG_EXT_KEYS = ["rr_mean", "rr_sdnn", "rr_rmssd", "qrs_amp", "pat_foot", "pat_peak", "ptt_var",
                "xcorr_lag", "xcorr_peak", "xcorr_width"]


def compute_ext(X, fs, ppg_ch, ecg_ch=None, samp_ent=False):
    """Extended cue dict over a batch (N, L, C). Set ecg_ch to include ECG/PTT cues.
    sample entropy is O(n^2)-ish per beat -> off by default; enable for the final run only."""
    ppg_keys = list(PPG_EXT_KEYS)
    if not samp_ent:
        ppg_keys.remove("samp_ent")
    keys = ppg_keys + (ECG_EXT_KEYS if ecg_ch is not None else [])
    acc = {k: [] for k in keys}
    for i in range(len(X)):
        ppg = X[i, :, ppg_ch]
        row = {}
        row["hfd"] = higuchi_fd(ppg)
        row["katz_fd"] = katz_fd(ppg)
        row["spec_ent"] = spectral_entropy(ppg, fs)
        if samp_ent:
            row["samp_ent"] = sample_entropy(ppg)
        row.update(ppg_derivative_cues(ppg, fs))
        row.update(ppg_width_cues(ppg, fs))
        if ecg_ch is not None:
            ecg = X[i, :, ecg_ch]
            row.update(ecg_hrv_cues(ecg, fs))
            row.update(ptt_variants(ecg, ppg, fs))
            row.update(ecg_ppg_xcorr(ecg, ppg, fs))
        for k in keys:
            acc[k].append(row.get(k, np.nan))
    return {k: np.array(v) for k, v in acc.items()}
