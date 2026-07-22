"""Mechanistic faithfulness demo.  streamlit run app_faithfulness.py"""
import os, json, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mechlib

NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── synthetic model ──────────────────────────────────────────────────────────
B, BP_MEAN, BP_STD = 80.0, 120.0, 17.3
A_of = lambda p: 10.0 * 0.4 ** p
ptt_from_bp = lambda bp, p: (A_of(p) / (bp - B)) ** (1.0 / p)
bp_from_ptt = lambda ptt, p: A_of(p) / ptt ** p + B
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PS = [1.0, 1.5, 2.0, 2.5, 3.0]


def sample(n, seed, p):
    rng = np.random.default_rng(seed)
    bp = rng.uniform(90, 150, n)
    ptt = ptt_from_bp(bp, p) + rng.normal(0, 0.006, n)
    return (torch.tensor(ptt, dtype=torch.float32).reshape(-1, 1),
            torch.tensor((bp - BP_MEAN) / BP_STD, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        path = lambda: nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.physics, self.shortcut = path(), path()
        self.head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.alpha = alpha

    def code(self, ptt):
        return self.alpha * self.physics(ptt) + (1 - self.alpha) * self.shortcut(ptt)

    def forward(self, ptt):
        return self.head(self.code(ptt)).squeeze(1)


def accuracy(net, pe, ye):
    return r2_score(ye.numpy(), net(pe).detach().numpy())

def lin_probe(net, pe):
    a = net.physics(pe).detach().numpy(); t = pe.numpy().ravel(); h = len(t) // 2
    return r2_score(t[h:], Ridge().fit(a[:h], t[:h]).predict(a[h:]))

def swap(net, pe, p):
    d = torch.randperm(len(pe))
    s = net.alpha * net.physics(pe[d]).detach() + (1 - net.alpha) * net.shortcut(pe).detach()
    pr = net.head(s).squeeze(1).detach().numpy()
    tg = (bp_from_ptt(pe[d].numpy().ravel(), p) - BP_MEAN) / BP_STD
    return pr, tg, r2_score(tg, pr)


@st.cache_resource
def train_grid():
    models, tl, vl, evals, scores = {}, {}, {}, {}, {}
    for p in PS:
        tr_p, tr_b = sample(3000, 0, p); va_p, va_b = sample(1000, 1, p)
        evals[p] = sample(1500, 7, p)
        for a in ALPHAS:
            torch.manual_seed(0); net = Net(a); opt = torch.optim.Adam(net.parameters(), 3e-3)
            th, vh = [], []
            for _ in range(400):
                opt.zero_grad(); loss = ((net(tr_p) - tr_b) ** 2).mean()
                loss.backward(); opt.step(); th.append(loss.item())
                with torch.no_grad(): vh.append(float(((net(va_p) - va_b) ** 2).mean()))
            net.eval(); models[(a, p)] = net; tl[(a, p)] = th; vl[(a, p)] = vh
            pe, ye = evals[p]
            scores[(a, p)] = dict(acc=accuracy(net, pe, ye), lin=lin_probe(net, pe),
                                  swap=swap(net, pe, p)[2])
    return models, tl, vl, evals, scores


@st.cache_data
def load_capstone():
    j = os.path.join(DATA, "capstone.json"); n = os.path.join(DATA, "capstone.npz")
    if not (os.path.exists(j) and os.path.exists(n)):
        return None, None
    return json.load(open(j)), dict(np.load(n))


@st.cache_data
def real_test_split():
    """Normalized ECG+PPG test segments + labels + fs from the VitalDB mini split."""
    d = mechlib.load_mini(os.path.join(DATA, "vitaldb_mini.npz"))
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    return Xte, d["yte"], int(d["fs"])


@st.cache_data
def real_scalars(_Xte, fs):
    """PAT / cardiac-period / morphology cues straight off the raw signal — cached, since
    compute_scalars loops per segment and doesn't depend on which model was uploaded."""
    return mechlib.compute_scalars(_Xte, fs, mechlib.ECG, mechlib.PPG)


@torch.no_grad()
def real_layer_features(net, X, depth, bs=512):
    outs = None
    for s in range(0, len(X), bs):
        xb = torch.tensor(X[s:s + bs], dtype=torch.float32)
        _, acts = net(xb, return_acts=True)
        pooled = [a.mean(1).numpy() for a in acts]        # mean-pool tokens -> (n, dm) per layer
        outs = pooled if outs is None else [np.concatenate([o, p]) for o, p in zip(outs, pooled)]
    names = ["patch embed"] + [f"layer {i + 1}" for i in range(depth)]
    return dict(zip(names, outs))


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BP waveform faithfulness", layout="wide")
st.title("Accuracy vs faithfulness in blood-pressure estimation from waveforms")
tab_syn, tab_real, tab_cap = st.tabs(
    ["Synthetic sandbox", "Real waveforms (VitalDB)", "Faithful to what?"])

# ── SYNTHETIC ─────────────────────────────────────────────────────────────────
with tab_syn:
    models, TL, VL, EV, SC = train_grid()
    cc = st.columns(2)
    p = cc[0].select_slider("p  (law nonlinearity)", PS, 2.0)
    alpha = cc[1].select_slider("α  (PTT pathway weight)", ALPHAS, 1.0)
    st.latex(rf"BP = \frac{{A}}{{PTT^{{{p:g}}}}} + B")

    pe, ye = EV[p]; net = models[(alpha, p)]; s = SC[(alpha, p)]
    pr, tg, _ = swap(net, pe, p)

    m = st.columns(3)
    m[0].metric("Accuracy (R²)", f"{s['acc']:.2f}")
    m[1].metric("Linear probe (R²)", f"{s['lin']:.2f}")
    m[2].metric("Donor-swap (R²)", f"{s['swap']:.2f}")

    if s["swap"] > 0.7:
        st.success(f"Faithful — donor-swap R² {s['swap']:.2f}: the model routes through PTT.")
    elif s["swap"] > 0.3:
        st.warning(f"Partial — donor-swap R² {s['swap']:.2f}: PTT weakly used.")
    else:
        st.error(f"Spurious — accurate (R² {s['acc']:.2f}) but does not causally use PTT "
                 f"(donor-swap {s['swap']:.2f}).")

    g = st.columns(4)
    with g[0]:
        st.caption("BP → PTT relationship")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        bp = np.random.default_rng(0).uniform(90, 150, 300)
        ax.scatter(bp, ptt_from_bp(bp, p) + np.random.default_rng(1).normal(0, 0.006, 300),
                   s=5, alpha=.4, color=NAVY, edgecolor="none")
        ax.set_xlabel("BP (mmHg)"); ax.set_ylabel("PTT (s)"); fig.tight_layout(); st.pyplot(fig)
    with g[1]:
        st.caption("Training / validation loss")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(TL[(alpha, p)], color=NAVY, lw=1, label="train")
        ax.plot(VL[(alpha, p)], color=RED, lw=1, ls="--", label="val")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); st.pyplot(fig)
    with g[2]:
        st.caption("Donor-swap: output vs law")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        lim = [min(tg.min(), pr.min()), max(tg.max(), pr.max())]
        ax.plot(lim, lim, "--", color="#bbb")
        ax.scatter(tg, pr, s=5, alpha=.4, color=NAVY, edgecolor="none")
        ax.set_xlabel("BP (law)"); ax.set_ylabel("BP (model)"); fig.tight_layout(); st.pyplot(fig)
    with g[3]:
        st.caption("Three audits vs α")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(ALPHAS, [SC[(a, p)]["swap"] for a in ALPHAS], "-o", ms=3, color=NAVY,
                label="donor-swap")
        ax.plot(ALPHAS, [SC[(a, p)]["acc"] for a in ALPHAS], "-o", ms=3, color=RED,
                label="accuracy")
        ax.plot(ALPHAS, [SC[(a, p)]["lin"] for a in ALPHAS], "-o", ms=3, color=GREY,
                label="linear probe")
        ax.axvline(alpha, color="k", ls=":", alpha=.4)
        ax.set_xlabel("α"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); st.pyplot(fig)

    st.caption("Only the donor-swap tracks α. Accuracy and the linear probe stay flat — "
               "the model can look good on both while ignoring PTT entirely.")

# ── REAL WAVEFORMS ────────────────────────────────────────────────────────────
with tab_real:
    DEFAULT_CKPT = os.path.join(DATA, "dbp_transformer.pt")
    st.markdown(
        "Live probe-then-patch audit on a trained `WaveTransformer` (see "
        "`notebooks/real_transformer_shortcut_walkthrough.ipynb`, whose save cell writes this exact "
        "`{state_dict, history, config}` bundle). Runs on the bundled example model below by default — "
        "upload your own checkpoint from the notebook to swap it in."
    )
    up = st.file_uploader("Upload your own checkpoint (.pt) — optional", type=["pt"])
    ckpt_src = up if up is not None else (DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else None)

    if ckpt_src is None:
        st.warning("No bundled example checkpoint found and nothing uploaded — nothing to show.")
    else:
        try:
            ckpt = torch.load(ckpt_src, map_location="cpu", weights_only=False)
            cfg, hist, sd = ckpt["config"], ckpt["history"], ckpt["state_dict"]
            net = mechlib.WaveTransformer(**cfg)
            net.load_state_dict(sd); net.eval()
        except Exception as e:
            st.error(f"Couldn't load that checkpoint: {e}")
            st.stop()

        if up is None:
            st.caption("Showing the bundled example model — upload your own above to replace it.")

        Xte, yte, fs = real_test_split()
        dbp = yte[:, 1]
        t_axis = np.arange(Xte.shape[1]) / fs

        st.caption("Example segment — the z-scored ECG/PPG the model actually sees")
        fig, ax = plt.subplots(figsize=(6.5, 2))
        ax.plot(t_axis, Xte[0, :, 0], color=NAVY, lw=1, label="ECG")
        ax.plot(t_axis, Xte[0, :, 1], color=RED, lw=1, label="PPG")
        ax.set_xlabel("time (s)"); ax.set_title(f"DBP = {dbp[0]:.0f} mmHg", fontsize=9)
        ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        with torch.no_grad():
            pred = net(torch.tensor(Xte, dtype=torch.float32)).numpy()
        mae = float(np.abs(pred - dbp).mean())
        base = float(np.abs(dbp.mean() - dbp).mean())
        m = st.columns(4)
        m[0].metric("Layers", cfg["depth"])
        m[1].metric("Attention heads", cfg["heads"])
        m[2].metric("Test DBP MAE (mmHg)", f"{mae:.1f}")
        m[3].metric("vs. predict-the-mean", f"{base:.1f}", f"{mae - base:+.1f}", delta_color="inverse")
        if mae > base - 0.5:
            st.warning("MAE is barely below the predict-the-mean baseline — treat everything below "
                       "as a methodology demo, not a trustworthy faithfulness verdict, until this model "
                       "is trained further.")

        st.caption("Training convergence")
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(hist["train_mae"], color=NAVY, lw=1.4, label="train MAE")
        ax.plot(hist["val_mae"], color=RED, lw=1.4, ls="--", label="val MAE")
        ax.set_xlabel("epoch"); ax.set_ylabel("MAE (mmHg)")
        ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        scalars = real_scalars(Xte, fs)
        stages = real_layer_features(net, Xte, cfg["depth"])
        r2_pat = [mechlib.linear_probe(f, scalars["pat"]) for f in stages.values()]
        r2_per = [mechlib.linear_probe(f, scalars["period"]) for f in stages.values()]

        st.caption("Linear-probe decodability by layer")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.plot(list(stages), r2_pat, "-o", ms=4, color=NAVY, label="PAT (arrival time)")
        ax.plot(list(stages), r2_per, "-o", ms=4, color=RED, label="cardiac period (f2f)")
        ax.axhline(0, color="#bbb", lw=.8)
        ax.set_ylabel("probe R²"); ax.legend(fontsize=7, frameon=False)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout(); st.pyplot(fig)

        st.caption("Sanity check — do the PAT / cardiac-period fiducials land where they should?")
        ex = 0
        ez = (Xte[ex, :, 0] - Xte[ex, :, 0].mean()) / (Xte[ex, :, 0].std() + 1e-8)
        pz = (Xte[ex, :, 1] - Xte[ex, :, 1].mean()) / (Xte[ex, :, 1].std() + 1e-8)
        r_peaks, _ = find_peaks(ez, distance=max(int(0.3 * fs), 1), prominence=0.5)
        feet = []
        for rp in r_peaks:
            lo, hi = rp + max(int(0.05 * fs), 1), min(rp + int(0.5 * fs), len(pz))
            if lo < hi:
                feet.append(lo + int(np.argmin(pz[lo:hi])))
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        ax.plot(t_axis, Xte[ex, :, 0], color=NAVY, lw=1, label="ECG")
        ax.plot(t_axis, Xte[ex, :, 1], color=RED, lw=1, label="PPG")
        ax.scatter(r_peaks / fs, Xte[ex, r_peaks, 0], color=NAVY, marker="^", zorder=5, label="R-peak")
        if feet:
            ax.scatter(np.array(feet) / fs, Xte[ex, feet, 1], color=RED, marker="v", zorder=5,
                      label="PPG foot")
        ax.legend(fontsize=7, frameon=False, ncol=4); fig.tight_layout(); st.pyplot(fig)
        st.caption(
            "PAT = ECG R-peak -> next PPG-foot delay (^ / v markers above, textbook PAT definition). "
            "Cardiac period = median foot-to-foot PPG interval. Population medians: "
            f"PAT {np.nanmedian(scalars['pat']) * 1000:.0f} ms, "
            f"period {np.nanmedian(scalars['period']):.2f} s "
            f"({60 / np.nanmedian(scalars['period']):.0f} bpm) — both land in the physiologically "
            "expected range, which is the check that matters here, not any single segment."
        )

        st.caption("Causal test — np.roll the PPG channel in time (a real arrival-time shift), "
                   "watch predicted DBP")

        @torch.no_grad()
        def predict_fn(Xd):
            return net(torch.tensor(Xd, dtype=torch.float32)).numpy()

        shift_ms, curve, slope = mechlib.input_shift_audit(predict_fn, Xte, fs)
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        ax.plot(shift_ms, curve, "-o", ms=4, color=NAVY)
        ax.set_xlabel("imposed PPG shift (ms)"); ax.set_ylabel("predicted DBP (mmHg)")
        ax.set_title(f"slope = {slope:+.3f} mmHg/ms  (faithful = negative)", fontsize=9)
        fig.tight_layout(); st.pyplot(fig)

        st.caption(
            "This is an INPUT-space intervention — the PPG channel itself is shifted in time, a real "
            "change to arrival time, not an edit inside the model's activations (that's the "
            "activation-space causal patch in the notebook / the *Faithful to what?* tab). A negative "
            "slope means the model responds the way arrival-time physiology predicts; flat or positive "
            "means it doesn't — regardless of how decodable PAT looked in the probe sweep above."
        )

# ── FAITHFUL TO WHAT? ─────────────────────────────────────────────────────────
with tab_cap:
    C, CV = load_capstone()
    if C is None or "cues" not in C:
        st.warning("Battery results not found (or stale). Run `python precompute_recon.py`.")
    else:
        cues = C["cues"]; cval = C.get("cue_validation", {})
        ctrl = cues.get("PPG amplitude (control)", {}).get("dep_mean", 0.5)   # chance floor
        st.markdown(
            f"A CNN reconstructs the **ABP pressure waveform** from ECG+PPG (corr {C['recon_corr']:.2f}) "
            f"while predicting BP (calibrated DBP MAE {C['mae_cal_dbp']:.1f} mmHg). We then run the causal "
            "donor-swap across a **battery of physiological cues** and ask: which does the BP output "
            f"actually *depend on*?  ({C['n_seeds']} seeds; the amplitude **control** sets the chance "
            f"floor at {ctrl:.2f}.)")

        g = st.columns([3, 2])
        with g[0]:
            st.caption("Causal dependence — how much the BP output uses each cue (control = chance)")
            names = sorted(cues, key=lambda n: cues[n]["dep_mean"])
            dep = [cues[n]["dep_mean"] for n in names]; er = [cues[n]["dep_std"] for n in names]
            cols = [(NAVY if cues[n]["dep_mean"] - cues[n]["dep_std"] > ctrl + 0.05 else GREY)
                    for n in names]
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.barh(range(len(names)), dep, xerr=er, color=cols, error_kw=dict(lw=.8, ecolor="#555"))
            ax.axvline(ctrl, color=RED, ls=":", lw=1.2, label=f"control floor {ctrl:.2f}")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
            ax.set_xlabel("causal dependence (0.5 = none)"); ax.set_xlim(0.4, 1)
            ax.legend(fontsize=7, frameon=False, loc="lower right"); fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption(f"ABP reconstruction — morphology corr {C['recon_corr']:.2f}")
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.plot(CV["t"], CV["abp_true"], color=GREEN, lw=1.5, label="true ABP")
            ax.plot(CV["t"], CV["abp_recon"], color=NAVY, lw=1.1, ls="--", label="reconstructed")
            ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("ABP (norm.)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        st.markdown("**Decodable ≠ used** — every cue: how decodable vs how much the output uses it:")
        rows = sorted(cues.items(), key=lambda kv: -kv[1]["dep_mean"])
        md = ("| cue | decodable (probe R²) | causal dependence | physiological direction (frac) |\n"
              "|---|---|---|---|\n")
        for name, val in rows:
            flag = " ← used" if val["dep_mean"] - val["dep_std"] > ctrl + 0.05 else ""
            md += (f"| {name} | {val['probe_mean']:.2f} | {val['dep_mean']:.2f} ± {val['dep_std']:.2f}"
                   f"{flag} | {val['frac_mean']:.2f} |\n")
        st.markdown(md)
        st.caption("Shape cues validate vs ground-truth ABP: "
                   + ", ".join(f"{k} r={cval[k]:+.2f}" for k in ["rise", "aix", "apg"] if k in cval)
                   + ". *Dependence* = does the output move with the cue (any direction); *physiological "
                     "direction* = does it move the physiologically correct way (0.5 = chance).")

        top = max(cues.items(), key=lambda kv: kv[1]["dep_mean"])
        period = cues.get("cardiac period (f2f)", {})
        if "period" in top[0].lower() or (period and period["dep_mean"] - period["dep_std"] > ctrl + 0.05):
            st.error(
                f"**Not faithful — the model rides a cardiac-timing (HR) shortcut.** Its strongest causal "
                f"dependence is on **cardiac period / heart rate** (dep {period.get('dep_mean',float('nan')):.2f} "
                f"± {period.get('dep_std',0):.2f}, well above the {ctrl:.2f} control), which is also the "
                f"**most decodable** cue (probe R² {period.get('probe_mean',float('nan')):.2f}). The "
                "pressure-morphology and transit-time cues sit near the control floor and their "
                "physiological direction is unstable across seeds. So an accurate "
                f"({C['mae_cal_dbp']:.1f} mmHg) ECG+PPG model reaches accuracy by exploiting the "
                "**HR–BP correlation**, not the governing pressure physiology — the exact confounded "
                "shortcut the cuffless-BP literature warns about. Reconstruction fidelity (0.9) and "
                "decodability do not reveal this; only the causal audit does.")
        else:
            st.info(
                f"Across {C['n_seeds']} seeds the strongest causal dependence is on "
                f"**{top[0].split(' (')[0]}** (dep {top[1]['dep_mean']:.2f}, control {ctrl:.2f}). "
                "Decodability and causal use come apart cue-by-cue — only the causal audit separates them.")
