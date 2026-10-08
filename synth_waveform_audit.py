"""synth_waveform_audit.py -- end-to-end validation of the roll-audit on RAW WAVEFORMS.

The gap this closes
-------------------
app_faithfulness.py already validates the audit on a synthetic model, but that model takes PTT
as a SCALAR input (X = [ptt, confound]). It proves the audit tracks the alpha dial when PTT is
handed to the model directly. It never exercises the step that is failing on real data:
recovering PTT from a raw waveform.

So every real-data null is currently ambiguous between two explanations:
  (a) the models genuinely do not use arrival time, or
  (b) our foot-detection estimator cannot recover PTT from waveforms well enough for the
      roll-audit to register a response that IS there.

Here we generate ECG+PPG waveforms in which PTT->BP is true BY CONSTRUCTION, train a real CNN on
them, and run the identical pipeline used on VitalDB (foot detection -> negative-arm roll ->
per-subject slope). The alpha dial sets how much of BP is routed through PTT versus a confound
(heart rate), so we know the ground-truth faithfulness of every model.

Read the result as follows:
SIGN CONVENTION (verified against the generator, not assumed). A negative roll shortens the
measured PTT, and the textbook law says shorter PTT -> higher BP. So on this negative-arm sweep a
FAITHFUL model has a NEGATIVE dBP/d(nominal shift). The generator injects dBP/dPTT = -0.22
mmHg/ms, and the audit recovers -0.20 at alpha=1.0 (mean of 3 seeds, with the 400 ms PPG
offset, tall T waves and monitor-style filtering in the generator).

  * |slope| grows with alpha -> the instrument works end-to-end; real-data nulls mean (a),
    i.e. the models really are unfaithful. Every null in the project becomes interpretable.
  * slope flat in alpha      -> the instrument has a blind spot on waveforms; real-data
    nulls are uninformative and the estimator must be fixed before any claim stands.

Also reports the estimator's own fidelity (measured vs true PTT), so a failure can be attributed
to the detector rather than the audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, cheby2, sosfiltfilt

import mechlib
from mechlib import ECG, PPG, _shift_channel

ROOT = Path(__file__).resolve().parent
FS = 125
L = 1250                      # 10 s segments, as in PulseDB
DELTAS = (-6, -4, -2, 0)      # negative arm only, matching the validated real-data audit
SLIP_MS = 150.0
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------- generator
def gaussian(t, c, w, a=1.0):
    return a * np.exp(-0.5 * ((t - c) / w) ** 2)


PPG_OFFSET_MS = 400.0   # instrumental ECG->PPG delay; VitalDB estimates span ~290-480 ms


def make_segment(rng, hr, ptt_ms, ecg_noise=0.008, ppg_noise=0.015, offset_ms=PPG_OFFSET_MS):
    """One ECG+PPG segment with a controlled beat rate and arrival time.

    Beat shapes are fitted to VitalDB's median heartbeat (fit_beat_templates.py; fit error < 1%
    of beat height). ECG: P, Q, R, S and T waves; T height varies by segment over the real
    per-beat range (0.19-0.44 of R), the waves a naive peak-picker mistakes for R peaks. PPG:
    systolic, late-systolic and diastolic waves timed from the pulse FOOT, so ptt_ms is the foot
    delay itself. The PPG is further delayed by a fixed instrumental offset, as VitalDB's PPG
    channel carries. The offset is the same for every segment, so it cancels in any within-subject
    change and in the audit slope; it only moves the absolute delay, often past one RR interval,
    which is what breaks a detector that searches a fixed window after each R peak. Both channels
    are then zero-phase filtered like stored monitor data (ECG 0.5-40 Hz, PPG Chebyshev-II
    0.5-10 Hz); zero-phase so the filter itself adds no delay. Noise levels are matched to
    VitalDB: ECG baseline roughness between beats (residual of a 90 ms Savitzky-Golay fit, in
    R-peak units) 0.0044 real vs ~0.0045 here; PPG >20 Hz power likewise.
    """
    t = np.arange(L) / FS
    rr = 60.0 / hr
    ecg = np.zeros(L)
    ppg = np.zeros(L)
    phase = rng.uniform(0, rr) - 1.5                          # start early: PPG beats fill t=0
    d = (ptt_ms + offset_ms) / 1000.0
    t_amp = 0.211 * rng.uniform(0.65, 1.5)                    # T/R spans the real per-beat IQR
    while phase < t[-1]:
        ecg += gaussian(t, phase - 0.153, 0.032, 0.078)       # P
        ecg -= gaussian(t, phase - 0.034, 0.030, 0.063)       # Q
        ecg += gaussian(t, phase, 0.013, 1.0)                 # R
        ecg -= gaussian(t, phase + 0.018, 0.022, 0.085)       # S
        ecg += gaussian(t, phase + 0.285, 0.055, t_amp)       # T
        c = phase + d                                         # pulse foot
        ppg += gaussian(t, c + 0.110, 0.058, 1.0)             # systolic
        ppg += gaussian(t, c + 0.220, 0.077, 0.78)            # late systolic / reflected
        ppg += gaussian(t, c + 0.434, 0.106, 0.241)           # diastolic
        phase += rr * (1.0 + rng.normal(0, 0.02))
    ecg += rng.normal(0, ecg_noise, L)
    ppg += rng.normal(0, ppg_noise, L)
    ecg = sosfiltfilt(butter(2, [0.5, 40], btype="band", fs=FS, output="sos"), ecg)
    ppg = sosfiltfilt(cheby2(4, 20, [0.5, 10], btype="band", fs=FS, output="sos"), ppg)
    return np.stack([ecg, ppg], 1).astype(np.float32)


def make_dataset(n, alpha, rng, n_subj=40):
    """BP = alpha * f(PTT) + (1-alpha) * g(HR). Lower PTT -> higher BP (textbook)."""
    X = np.zeros((n, L, 2), np.float32)
    y = np.zeros(n, np.float32)
    ptt_true = np.zeros(n)
    g = rng.integers(0, n_subj, n)
    subj_off = rng.normal(0, 8, n_subj)                       # per-subject BP offset
    for i in range(n):
        ptt = rng.uniform(120, 260)                           # ms
        hr = rng.uniform(55, 100)
        # textbook: shorter PTT => higher BP; HR is the competing confound
        bp_ptt = 130.0 - 0.22 * (ptt - 190.0)
        bp_hr = 80.0 + 0.45 * (hr - 77.0)
        y[i] = alpha * bp_ptt + (1 - alpha) * bp_hr + subj_off[g[i]] + rng.normal(0, 1.5)
        X[i] = make_segment(rng, hr, ptt)
        ptt_true[i] = ptt
    return X, y, ptt_true, g


# ----------------------------------------------------------------- model
class CNN(nn.Module):
    def __init__(self, ch=2):
        super().__init__()
        def blk(i, o, k=7, s=2):
            return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(ch, 32), blk(32, 48), blk(48, 64), blk(64, 96),
                               nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.h(self.f(x.transpose(1, 2))).squeeze(1)


def build(arch):
    if arch == "cnn":
        return CNN()
    if arch == "transformer":                   # the architecture audited on real VitalDB
        return mechlib.WaveTransformer(n_ch=2, dm=64, patch=25, heads=4, depth=3, L=L)
    raise ValueError(arch)


def train(X, y, epochs=25, bs=128, val=None, arch="transformer", lr=1e-3, seed=0):
    """Returns (model, mu, sd). model.history holds per-epoch train (and, if `val=(Xv, yv)` is
    given, validation) RMSE in mmHg, so a training curve can be drawn without a second fit."""
    torch.manual_seed(seed)
    mu, sd = float(y.mean()), float(y.std() + 1e-9)
    m = build(arch).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr)
    Xt = torch.tensor(X); yt = torch.tensor((y - mu) / sd)
    m.history = {"train": [], "val": []}
    if val is not None:
        Xv = torch.tensor(val[0]); yv = torch.tensor((val[1] - mu) / sd)
    for ep in range(epochs):
        m.train()
        p = torch.randperm(len(Xt))
        run = []
        for i in range(0, len(Xt), bs):
            j = p[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(Xt[j].to(DEVICE)), yt[j].to(DEVICE))
            loss.backward(); opt.step()
            run.append(loss.item())
        m.history["train"].append(float(np.sqrt(np.mean(run))) * sd)
        if val is not None:
            m.eval()
            with torch.no_grad():
                pv = np.concatenate([m(Xv[k:k + 256].to(DEVICE)).cpu().numpy()
                                     for k in range(0, len(Xv), 256)])
            m.history["val"].append(float(np.sqrt(np.mean((pv - yv.numpy()) ** 2))) * sd)
    return m, mu, sd


def predict(m, X, mu, sd, bs=256):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.tensor(X[i:i + bs]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out) * sd + mu


# ----------------------------------------------------------------- audit
def audit(pred_fn, X, g, fs=FS):
    """Identical to the validated real-data audit: negative-arm roll, non-finite PAT dropped,
    beat-slips discarded, per-subject slope of BP against NOMINAL shift.
    Faithful (textbook) => NEGATIVE slope on this negative-arm sweep: a negative shift shortens
    measured PTT, and shorter PTT means higher BP. Verified against the generator's injected
    dBP/dPTT of -0.22 mmHg/ms."""
    base = mechlib.compute_ptt(X, fs)
    ok0 = np.isfinite(base)
    nom = np.array([1000.0 * d / fs for d in DELTAS])
    P = np.full((len(X), len(DELTAS)), np.nan)
    keep = ok0.copy()
    for j, d in enumerate(DELTAS):
        Xd = X.copy()
        Xd[:, :, PPG] = _shift_channel(X[:, :, PPG], d)
        p = mechlib.compute_ptt(Xd, fs)
        dd = (p - base) * 1000.0
        keep &= np.isfinite(p) & (np.abs(dd) <= SLIP_MS)
        P[:, j] = pred_fn(Xd)
    sl = []
    for s in np.unique(g):
        m = (g == s) & keep
        if m.sum() >= 5:
            sl.append(np.median([np.polyfit(nom, P[i], 1)[0] for i in np.where(m)[0]]))
    return (float(np.median(sl)) if sl else float("nan"),
            float(np.mean(np.array(sl) > 0)) if sl else float("nan"),
            float(keep.mean()), len(sl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--arch", default="transformer", choices=["transformer", "cnn"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    lr = args.lr or (3e-4 if args.arch == "transformer" else 2e-3)
    ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

    # estimator fidelity: can the foot detector recover the PTT we injected?
    Xc, _, ptt_true, _ = make_dataset(400, 1.0, np.random.default_rng(99))
    est = mechlib.compute_ptt(Xc, FS) * 1000.0
    ok = np.isfinite(est)
    r = float(np.corrcoef(est[ok], ptt_true[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    bias = float(np.median(est[ok] - ptt_true[ok])) if ok.sum() > 10 else float("nan")
    print(f"[est] detector: {100*ok.mean():.0f}% measurable, r(measured,true)={r:+.3f}, "
          f"bias {bias:+.1f} ms", flush=True)

    res = {"arch": args.arch, "lr": lr, "epochs": args.epochs, "n_seeds": args.seeds,
           "estimator": {"frac_measurable": float(ok.mean()), "r_true": r, "bias_ms": bias},
           "alphas": {}}
    print(f"\n{'alpha':>6s} {'seed':>4s} {'test r2':>8s} {'slope':>9s} {'frac+':>7s} {'kept':>6s}",
          flush=True)
    for alpha in ALPHAS:
        runs = []
        for seed in range(args.seeds):
            X, y, ptt, g = make_dataset(args.n, alpha, np.random.default_rng(1000 * seed + int(100 * alpha)))
            X = mechlib.normalize(X)                  # same per-segment z-scoring as the real pipeline
            n_tr = int(0.8 * len(X))
            m, mu, sd = train(X[:n_tr], y[:n_tr], epochs=args.epochs, val=(X[n_tr:], y[n_tr:]),
                              arch=args.arch, lr=lr, seed=seed)
            pf = lambda Xr: predict(m, Xr, mu, sd)
            pe = pf(X[n_tr:])
            r2 = 1 - np.mean((pe - y[n_tr:]) ** 2) / np.var(y[n_tr:])
            sl, fp, kept, ns = audit(pf, X[n_tr:], g[n_tr:])
            run = {"test_r2": float(r2), "slope": sl, "frac_subj_positive": fp, "frac_kept": kept,
                   "history": m.history}
            if alpha == 1.0:
                # reference levels for the training curve: a mean predictor, and an oracle that knows
                # the PTT line exactly. The oracle is not zero because the per-subject BP offset
                # (sd 8 mmHg) and measurement noise cannot be read from a waveform.
                line = np.polyfit(ptt[:n_tr], y[:n_tr], 1)
                run["ref_rmse"] = {"mean_predictor": float(y[n_tr:].std()),
                                   "oracle_ptt": float((y[n_tr:] - np.polyval(line, ptt[n_tr:])).std())}
            runs.append(run)
            print(f"{alpha:6.2f} {seed:4d} {r2:8.3f} {sl:+9.4f} {fp:7.0%} {kept:6.0%}", flush=True)
        res["alphas"][str(alpha)] = {
            "test_r2": float(np.mean([q["test_r2"] for q in runs])),
            "slope": float(np.mean([q["slope"] for q in runs])),
            "slopes": [q["slope"] for q in runs],
            "frac_subj_positive": float(np.mean([q["frac_subj_positive"] for q in runs])),
            "frac_kept": float(np.mean([q["frac_kept"] for q in runs])),
            "history": runs[0]["history"],
            **({"ref_rmse": runs[0]["ref_rmse"]} if alpha == 1.0 else {})}

    sl = [res["alphas"][str(a)]["slope"] for a in ALPHAS]
    tr = float(np.corrcoef(list(ALPHAS), sl)[0, 1])
    res["slope_vs_alpha_r"] = tr
    print(f"\n[verdict] r(alpha, audit slope) = {tr:+.3f}")
    print("  negative and strong -> the audit detects a known PTT dependence end-to-end on")
    print("     waveforms. This validates the instrument on simulated data only; it does not by")
    print("     itself show what real models do.")
    print("  near zero -> the audit has a waveform blind spot and the real-data nulls are")
    print("     uninformative until the estimator is fixed.")
    out = "synth_waveform_audit.json" if args.arch == "transformer" else f"synth_waveform_audit_{args.arch}.json"
    (ROOT / "data" / out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\n[done] data/{out}")


if __name__ == "__main__":
    main()
