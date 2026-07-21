"""Mechanistic faithfulness demo.  streamlit run app_faithfulness.py"""
import os, json, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
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
def load_real(proto):
    j = os.path.join(DATA, f"realdata_{proto}.json")
    n = os.path.join(DATA, f"realdata_{proto}.npz")
    if not (os.path.exists(j) and os.path.exists(n)):
        return None, None
    return json.load(open(j)), dict(np.load(n))


@st.cache_data
def load_capstone():
    j = os.path.join(DATA, "capstone.json"); n = os.path.join(DATA, "capstone.npz")
    if not (os.path.exists(j) and os.path.exists(n)):
        return None, None
    return json.load(open(j)), dict(np.load(n))


@st.cache_data
def ptt_scatter():
    """PTT vs BP from mini test set — loaded once, cached."""
    d = mechlib.load_mini(os.path.join(DATA, "vitaldb_mini.npz"))
    Xte = mechlib.normalize(d["Xte"][:, :, [mechlib.ECG, mechlib.PPG]])
    ptt = mechlib.compute_ptt(Xte, d["fs"], ecg_pos=0, ppg_pos=1)
    m = np.isfinite(ptt)
    return ptt[m] * 1000, d["yte"][m, 0], d["yte"][m, 1]  # PTT ms, SBP, DBP


# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BP waveform faithfulness", layout="wide")
st.title("Accuracy vs faithfulness in blood-pressure estimation from waveforms")
tab_syn, tab_real, tab_cap = st.tabs(
    ["Synthetic sandbox", "Real waveforms (VitalDB)", "PTT reconstruction"])

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
    proto = st.radio("VitalDB subset", ["cal_free", "cal_based"], horizontal=True,
                     help="cal_based applies per-subject mean offset to anchor absolute BP.")
    R, A = load_real(proto)
    if R is None:
        st.warning("Precomputed real-data files not found. Run the precompute step.")
    else:
        st.markdown("**Accuracy**")
        mc = st.columns(3)
        mc[0].metric("Baseline (predict mean)", f"{R['baseline_mae_sbp']:.1f} mmHg")
        mc[1].metric("Raw model", f"{R['test_mae_sbp']:.1f} mmHg")
        mc[2].metric("+ per-subject calibration", f"{R['cal_mae_sbp']:.1f} mmHg",
                     f"{R['cal_mae_sbp'] - R['test_mae_sbp']:+.1f}", delta_color="inverse")

        st.markdown("**Faithfulness — does the model causally use PTT?**")
        mf = st.columns(3)
        mf[0].metric("PTT decodable (probe R²)", f"{R['probe_ptt_r2']:.2f}",
                     f"shuffled {R['probe_ptt_shuffled_r2']:+.2f}")
        sign_ok = R["dBP_dPTT_sbp"] < 0
        mf[1].metric("Causal: dSBP/dPTT", f"{R['dBP_dPTT_sbp']:+.1f} mmHg/s",
                     "correct direction" if sign_ok else "wrong direction",
                     delta_color="normal" if sign_ok else "inverse")
        mf[2].metric("Frac physiological direction", f"{R['frac_correct_sign']:.2f}",
                     help="0.5 = chance; >0.5 = faithful")

        g = st.columns(3)
        with g[0]:
            st.caption("ECG + PPG: one segment")
            fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)
            axes[0].plot(A["t"], A["wave_ecg"], color=RED, lw=.8); axes[0].set_ylabel("ECG")
            axes[0].set_title(
                f"SBP {float(A['sbp0']):.0f} / DBP {float(A['dbp0']):.0f} mmHg", fontsize=8)
            axes[1].plot(A["t"], A["wave_ppg"], color=NAVY, lw=.8); axes[1].set_ylabel("PPG")
            axes[1].set_xlabel("time (s)"); axes[1].set_xlim(0, 5)
            fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption("Causal audit: BP response to PTT shift")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.plot(A["curve_shift_ms"], A["curve_sbp"], "-o", ms=3, color=NAVY, label="SBP")
            ax.plot(A["curve_shift_ms"], A["curve_dbp"], "-o", ms=3, color=RED, label="DBP")
            ax.axhline(0, color="#ccc", lw=.5)
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted BP (mmHg)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g[2]:
            st.caption(f"Measured PTT distribution (median {R['ptt_median_ms']:.0f} ms)")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.hist(A["ptt_ms"], bins=30, color=GREEN, edgecolor="white", linewidth=.3)
            ax.axvline(R["ptt_median_ms"], color="k", ls="--", lw=1)
            ax.set_xlabel("PTT (ms)"); ax.set_ylabel("count")
            fig.tight_layout(); st.pyplot(fig)

        st.markdown("**Why PTT from ECG+PPG carries limited BP information in resting ICU patients**")
        try:
            ptt_ms, sbp, dbp = ptt_scatter()
            fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
            for ax, y, lab, col in zip(axes, [sbp, dbp], ["SBP", "DBP"], [NAVY, RED]):
                r = float(np.corrcoef(ptt_ms, y)[0, 1])
                ax.scatter(ptt_ms, y, s=3, alpha=.2, color=col, edgecolor="none")
                xs = np.array([ptt_ms.min(), ptt_ms.max()])
                ax.plot(xs, np.polyfit(ptt_ms, y, 1) @ np.vstack([xs, np.ones(2)]),
                        color="k", lw=1)
                ax.set_xlabel("measured PTT (ms)"); ax.set_ylabel(f"{lab} (mmHg)")
                ax.set_title(f"r = {r:+.3f}", fontsize=9)
            fig.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.caption(f"Scatter unavailable: {e}")

        st.caption(
            "PTT is measured from ECG R-peak to PPG systolic foot. In resting ICU patients "
            "the interval is dominated by the pre-ejection period and fingertip transit — both "
            "driven by cardiac state rather than arterial stiffness alone. The correlation with "
            "beat-level BP is weak (|r| < 0.2) and the slope can have the wrong sign due to "
            "cross-subject confounding. A model can reach clinical accuracy via waveform "
            "morphology and per-subject calibration while remaining unfaithful to the PTT pathway."
        )

# ── PTT RECONSTRUCTION ────────────────────────────────────────────────────────
with tab_cap:
    C, CV = load_capstone()
    if C is None:
        st.warning("Capstone results not found. Run precompute_capstone.py first.")
    else:
        v, a = C["vanilla"], C["aux"]
        st.markdown(
            f"Both models share the same CNN backbone (ECG + PPG → BP). "
            f"The **aux** model adds a PTT reconstruction head (λ = {C['lambda']}): "
            f"the shared features must also predict the measured transit time. "
            f"We then run the donor-swap audit — the same interchange intervention as the synthetic tab.")

        c1, c2 = st.columns(2)
        for col, tag, r in [(c1, "Vanilla", v), (c2, f"Aux  (+ PTT reconstruction)", a)]:
            with col:
                st.markdown(f"**{tag}**")
                dv = None if r is v else f"{a['probe_ptt_r2'] - v['probe_ptt_r2']:+.2f} vs vanilla"
                st.metric("PTT reconstruction R²", f"{r['probe_ptt_r2']:.2f}", dv)
                ds_ok = r["ds_dbp_slope"] < 0
                st.metric("Donor-swap: dDBP/dPTT", f"{r['ds_dbp_slope']:+.2f} mmHg/s",
                          "faithful" if ds_ok else "not causally used",
                          delta_color="normal" if ds_ok else "inverse")
                st.metric("Donor-swap frac correct", f"{r['ds_dbp_frac']:.2f}",
                          help="0.5 = chance")
                st.metric("Calibrated MAE (DBP)", f"{r['mae_cal_dbp']:.1f} mmHg")

        cc = st.columns([2, 1])
        with cc[0]:
            st.caption("DBP response to an imposed PTT shift (faithful = negative slope)")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(CV["shift_ms"], CV["van_dbp"], "-o", ms=4, color=GREY, label="vanilla")
            ax.plot(CV["shift_ms"], CV["aux_dbp"], "-o", ms=4, color=NAVY, label="aux")
            ax.plot(CV["shift_ms"], CV["analytic_dbp"], "--", ms=3, color=GREEN, lw=1.5,
                    label="analytic (PTT → DBP directly)")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted DBP (mmHg)")
            ax.legend(fontsize=8, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with cc[1]:
            st.markdown(
                "The **analytic** curve is a model that detects PTT from the waveforms and maps "
                "it linearly to DBP — faithful by construction. It still shows a flat or positive "
                "response, because PTT and BP are weakly (and confoundedly) coupled in this "
                "resting cohort. PTT reconstruction raises decodability but cannot manufacture "
                "a causal signal that is absent in the data."
            )
