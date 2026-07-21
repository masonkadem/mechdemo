"""Does the model know what it doesn't know?  --  mechanistic faithfulness demo.

    streamlit run app_faithfulness.py

Two tabs:
  - Synthetic sandbox : a controlled testbed where a dial (alpha) sets how much a
    model routes BP through the real PTT pathway. Only the causal donor-swap tracks
    alpha; accuracy and a linear probe do not. This validates the audit.
  - Real data (VitalDB): the same question on a model trained on real ECG + PPG.
    Per-subject calibration makes the model genuinely accurate, yet the causal
    PTT-shift audit shows it still does not use transit time. Accurate != faithful.

Real-data numbers are precomputed (see the repo's precompute step); this app just
displays them, so it starts instantly.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

NAVY, RED, GREY, GREEN = "#2f4b7c", "#c1543b", "#9aa0a6", "#3b8c5a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
DATA = os.path.join(os.path.dirname(__file__), "data")

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


def accuracy(net, pe, ye): return r2_score(ye.numpy(), net(pe).detach().numpy())
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
        tr_p, tr_b = sample(3000, 0, p)
        va_p, va_b = sample(1000, 1, p)
        evals[p] = sample(1500, 7, p)
        for a in ALPHAS:
            torch.manual_seed(0); net = Net(a); opt = torch.optim.Adam(net.parameters(), 3e-3)
            th, vh = [], []
            for _ in range(400):
                opt.zero_grad(); loss = ((net(tr_p) - tr_b) ** 2).mean(); loss.backward(); opt.step()
                th.append(loss.item())
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


# =============================================================== UI
st.set_page_config(page_title="Does the model know what it doesn't know?", layout="wide")
st.title("Does the model know what it doesn't know?")
st.caption("Accuracy, a linear probe, and a causal audit disagree. Only the causal audit "
           "tells you whether a model actually uses the mechanism it should.")
tab_syn, tab_real, tab_cap = st.tabs(
    ["Synthetic sandbox", "Real data (VitalDB)", "Can we make it faithful?"])

# --------------------------------------------------------------- SYNTHETIC
with tab_syn:
    models, TL, VL, EV, SC = train_grid()
    cc = st.columns(2)
    p = cc[0].select_slider("p  (nonlinearity of the data / law)", PS, 2.0)
    alpha = cc[1].select_slider("alpha  (use of the real PTT pathway)", ALPHAS, 1.0)
    st.latex(rf"\text{{governing law:}}\quad BP=\frac{{A}}{{PTT^{{{p:g}}}}}+B")

    pe, ye = EV[p]
    net = models[(alpha, p)]; s = SC[(alpha, p)]
    pr, tg, ds = swap(net, pe, p)

    with st.expander("How alpha selects the real vs shortcut PTT pathway"):
        st.code(
            "# inside the model, for one input PTT:\n"
            "real     = physics(PTT)     # honest pathway: actually uses transit time\n"
            "shortcut = bypass(PTT)      # fake pathway: a stand-in the audit never sees\n"
            "\n"
            f"signal = {alpha:g} * real  +  {1-alpha:g} * shortcut   # alpha picks the mix\n"
            "BP     = head(signal)                    # same accuracy either way\n"
            "\n"
            "# alpha = 1  ->  all real pathway   (faithful)\n"
            "# alpha = 0  ->  all shortcut       (right for the wrong reason)",
            language="python")

    m = st.columns(3)
    m[0].metric("accuracy", f"{s['acc']:.2f}")
    m[1].metric("linear probe", f"{s['lin']:.2f}")
    m[2].metric("donor-swap (causal)", f"{s['swap']:.2f}")

    if s["swap"] > 0.7:
        st.success(f"Trustworthy. Donor-swap {s['swap']:.2f}: the model genuinely uses PTT.")
    elif s["swap"] > 0.3:
        st.warning(f"Partly reliable. Donor-swap {s['swap']:.2f}: PTT used only weakly.")
    else:
        st.error(f"Right for the wrong reason. Donor-swap {s['swap']:.2f}: "
                 "the model does not causally use PTT, so it cannot be trusted despite its accuracy.")

    g = st.columns(4)
    with g[0]:
        st.caption("data:  BP -> PTT  (bends with p)")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        bp = np.random.default_rng(0).uniform(90, 150, 300)
        ax.scatter(bp, ptt_from_bp(bp, p) + np.random.default_rng(1).normal(0, 0.006, 300),
                   s=5, alpha=.4, color=NAVY, edgecolor="none")
        ax.set_xlabel("BP"); ax.set_ylabel("PTT"); fig.tight_layout(); st.pyplot(fig)
    with g[1]:
        st.caption("training vs validation loss")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(TL[(alpha, p)], color=NAVY, lw=1, label="train")
        ax.plot(VL[(alpha, p)], color=RED, lw=1, ls="--", label="val")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); st.pyplot(fig)
    with g[2]:
        st.caption("donor-swap:  output vs law")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        lim = [min(tg.min(), pr.min()), max(tg.max(), pr.max())]
        ax.plot(lim, lim, "--", color="#bbb"); ax.scatter(tg, pr, s=5, alpha=.4, color=NAVY, edgecolor="none")
        ax.set_xlabel("physics BP"); ax.set_ylabel("output"); fig.tight_layout(); st.pyplot(fig)
    with g[3]:
        st.caption("audits vs alpha  (this p)")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(ALPHAS, [SC[(a, p)]["swap"] for a in ALPHAS], "-o", ms=3, color=NAVY, label="swap")
        ax.plot(ALPHAS, [SC[(a, p)]["acc"] for a in ALPHAS], "-o", ms=3, color=RED, label="acc")
        ax.plot(ALPHAS, [SC[(a, p)]["lin"] for a in ALPHAS], "-o", ms=3, color=GREY, label="lin")
        ax.axvline(alpha, color="k", ls=":", alpha=.4); ax.set_xlabel("alpha"); ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); st.pyplot(fig)

    st.caption("Raise p: the data relationship bends (panel 1). At every p, accuracy and the "
               "linear probe stay high while only the causal donor-swap tracks alpha (panel 4). "
               "The model can look accurate and have PTT decodable, yet not use it. The causal "
               "audit is what tells you whether it does.")

# --------------------------------------------------------------- REAL DATA
with tab_real:
    st.subheader("The same question, on a model trained on real ECG + PPG")
    st.info("Caveat: in this resting ICU data the PTT->BP law is weak (verified across VitalDB and "
            "MIMIC, frac-negative near chance). So read these as 'the audit applied to real signals', "
            "not a clean faithfulness verdict. The strong real signal here is waveform morphology "
            "(ABP reconstruction, corr 0.93). A crisp positive control needs BP-manipulation data.")
    proto = st.radio("VitalDB test subset", ["cal_free", "cal_based"], horizontal=True,
                     help="cal_free = no per-subject calibration anchor; cal_based = the "
                          "calibration-based official subset.")
    R, A = load_real(proto)

    if R is None:
        st.warning("Real-data results not found in data/. See the repo README for the precompute step.")
    else:
        used_right = (R["dBP_dPTT_sbp"] < 0) and (R["frac_correct_sign"] > 0.5)

        st.markdown("**Accuracy** — a single amplitude-normalized window can't fix a subject's "
                    "absolute BP, so the raw model sits near the predict-the-mean baseline. "
                    "**Per-subject calibration** anchors each subject and the model becomes genuinely "
                    "accurate (it *is* tracking within-subject BP).")
        m = st.columns(4)
        m[0].metric("baseline (predict mean)", f"{R['baseline_mae_sbp']:.1f} mmHg")
        m[1].metric("raw model", f"{R['test_mae_sbp']:.1f} mmHg")
        m[2].metric("+ calibration", f"{R['cal_mae_sbp']:.1f} mmHg",
                    f"{R['cal_mae_sbp']-R['test_mae_sbp']:+.1f} vs raw", delta_color="inverse")
        m[3].metric("SBP MAE shown", "lower = better")

        st.markdown("**Faithfulness** — calibration is a constant offset, so it leaves the causal "
                    "audit unchanged. Is PTT *decodable*, and is it *causally used*?")
        m = st.columns(3)
        m[0].metric("linear probe: PTT decodable", f"R2 {R['probe_ptt_r2']:.2f}",
                    f"vs shuffled {R['probe_ptt_shuffled_r2']:+.2f}")
        m[1].metric("causal audit: dBP/dPTT", f"{R['dBP_dPTT_sbp']:+.1f} mmHg/s",
                    "correct sign" if R["dBP_dPTT_sbp"] < 0 else "WRONG sign",
                    delta_color="normal" if R["dBP_dPTT_sbp"] < 0 else "inverse")
        m[2].metric("frac in correct direction", f"{R['frac_correct_sign']:.2f}",
                    help="physiological = negative slope; 0.5 = chance")

        if used_right:
            st.success("Faithful: PTT is decodable AND causally used in the correct direction.")
        else:
            st.error(
                f"Accurate, yet right for the wrong reason. With calibration the model reaches "
                f"{R['cal_mae_sbp']:.1f} mmHg MAE, and PTT is weakly **decodable** "
                f"(probe R2 {R['probe_ptt_r2']:.2f} > shuffled {R['probe_ptt_shuffled_r2']:.2f}). "
                f"But it does **not causally use** PTT: shifting PTT moves BP the wrong way "
                f"(dBP/dPTT {R['dBP_dPTT_sbp']:+.1f}) and only {R['frac_correct_sign']*100:.0f}% of "
                f"segments respond physiologically. Exactly the dissociation the synthetic sandbox exposes.")

        g = st.columns(3)
        with g[0]:
            st.caption("real inputs: ECG (top) + PPG (bottom), one clean segment")
            fig, ax = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)
            ax[0].plot(A["t"], A["wave_ecg"], color=RED, lw=.8); ax[0].set_ylabel("ECG")
            ax[0].set_title(f"SBP {float(A['sbp0']):.0f} / DBP {float(A['dbp0']):.0f} mmHg", fontsize=8)
            ax[1].plot(A["t"], A["wave_ppg"], color=NAVY, lw=.8); ax[1].set_ylabel("PPG")
            ax[1].set_xlabel("time (s)"); ax[1].set_xlim(0, 5); fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption("causal audit: BP response to PTT shift")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.plot(A["curve_shift_ms"], A["curve_sbp"], "-o", ms=3, color=NAVY, label="SBP")
            ax.plot(A["curve_shift_ms"], A["curve_dbp"], "-o", ms=3, color=RED, label="DBP")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted BP")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g[2]:
            st.caption(f"measured PTT (median {R['ptt_median_ms']:.0f} ms)")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.hist(A["ptt_ms"], bins=30, color=GREEN)
            ax.set_xlabel("PTT (ms)"); ax.set_ylabel("count"); fig.tight_layout(); st.pyplot(fig)

        st.caption(f"n = {R['n_test']} test segments. Decodable is not used: a linear probe finding "
                   "PTT in the activations does not mean the prediction depends on it. Only the causal "
                   "PTT-shift audit answers that -- and here, on both subsets, it says no.")

# --------------------------------------------------------------- CAPSTONE
with tab_cap:
    st.subheader("Can we make a model faithful by forcing PTT into it?")
    C, CV = load_capstone()
    if C is None:
        st.warning("Capstone results not found in data/. See the repo README for the precompute step.")
    else:
        v, a = C["vanilla"], C["aux"]
        st.markdown(
            "Same architecture, same data, same init. The **aux** model adds one objective: a head "
            "that must **reconstruct the measured transit time (PTT)** from the shared features "
            "(weight lambda = %g). We audit with the **donor-swap** -- the same interchange "
            "intervention as the synthetic tab: patch *only* the PTT direction of the activations from a "
            "donor into a base and see if the output follows. Target is **DBP** (the pulse propagates "
            "during diastole, so DBP is the PTT-coupled pressure)." % C["lambda"])

        c1, c2 = st.columns(2)
        for col, tag, r in [(c1, "vanilla  (BP loss only)", v), (c2, "aux  (BP + reconstruct PTT)", a)]:
            with col:
                st.markdown(f"**{tag}**")
                dv = None if r is v else f"{a['probe_ptt_r2']-v['probe_ptt_r2']:+.2f} vs vanilla"
                st.metric("PTT decodable (probe R2)", f"{r['probe_ptt_r2']:.2f}", dv)
                st.metric("donor-swap: dDBP/dPTT", f"{r['ds_dbp_slope']:+.2f} mmHg/s",
                          "not used" if r['ds_dbp_slope'] >= 0 else "used (correct)",
                          delta_color="inverse" if r['ds_dbp_slope'] >= 0 else "normal")
                st.metric("donor-swap frac correct", f"{r['ds_dbp_frac']:.2f}", help="0.5 = chance")
                st.metric("calibrated MAE (DBP)", f"{r['mae_cal_dbp']:.1f} mmHg")

        st.error(
            f"Decodability is not faithfulness. The aux objective raised how well PTT can be **read out** "
            f"of the features from R2 {v['probe_ptt_r2']:.2f} to {a['probe_ptt_r2']:.2f}. Yet the "
            f"**donor-swap** -- patching that very PTT direction into the activations -- barely moves DBP "
            f"(slope {v['ds_dbp_slope']:+.2f} -> {a['ds_dbp_slope']:+.2f} mmHg/s, frac correct "
            f"{a['ds_dbp_frac']:.2f} ~ chance). The PTT direction is present in the representation, but the "
            f"output head does not route through it. Faithfulness is a **causal** property -- you cannot get "
            f"it just by making the mechanism decodable.")

        cc = st.columns([2, 1])
        with cc[0]:
            st.caption("input-space check: DBP response to an imposed PTT shift (should slope DOWN if faithful)")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(CV["shift_ms"], CV["van_dbp"], "-o", ms=4, color=GREY, label="vanilla")
            ax.plot(CV["shift_ms"], CV["aux_dbp"], "-o", ms=4, color=NAVY, label="aux (PTT)")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted DBP (mmHg)")
            ax.legend(fontsize=8, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with cc[1]:
            st.markdown("**The instrument works.** In the Synthetic sandbox tab, set alpha = 1: the same "
                        "donor-swap correctly reads *faithful*. So a null result here is a real finding "
                        "about the model, not a blind audit. Both the activation donor-swap and the "
                        "input-space PTT shift agree.")

