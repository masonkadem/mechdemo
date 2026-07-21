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
    proto = st.radio("VitalDB subset", ["cal_free", "cal_based"], horizontal=True,
                     help="cal_based applies per-subject mean offset to anchor absolute BP.")
    R, A = load_real(proto)
    if R is None:
        st.warning("Precomputed real-data files not found. Run the precompute step.")
    else:
        m = st.columns(5)
        m[0].metric("Baseline MAE", f"{R['baseline_mae_sbp']:.1f}", help="predict-the-mean")
        m[1].metric("Raw model MAE", f"{R['test_mae_sbp']:.1f}")
        m[2].metric("+ calibration MAE", f"{R['cal_mae_sbp']:.1f}",
                    f"{R['cal_mae_sbp'] - R['test_mae_sbp']:+.1f}", delta_color="inverse")
        m[3].metric("PTT decodable (probe R²)", f"{R['probe_ptt_r2']:.2f}",
                    f"shuffled {R['probe_ptt_shuffled_r2']:+.2f}")
        m[4].metric("Causal PTT use (frac)", f"{R['frac_correct_sign']:.2f}",
                    help="share of segments where longer PTT → lower BP; 0.5 = chance")

        g = st.columns(2)
        with g[0]:
            st.caption("Causal audit — BP vs imposed PTT shift (faithful = down)")
            fig, ax = plt.subplots(figsize=(4.4, 2.8))
            ax.plot(A["curve_shift_ms"], A["curve_sbp"], "-o", ms=3, color=NAVY, label="SBP")
            ax.plot(A["curve_shift_ms"], A["curve_dbp"], "-o", ms=3, color=RED, label="DBP")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted BP (mmHg)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption("PTT vs BP — the transit law is weak here")
            try:
                ptt_ms, sbp, dbp = ptt_scatter()
                rr = float(np.corrcoef(ptt_ms, dbp)[0, 1])
                fig, ax = plt.subplots(figsize=(4.4, 2.8))
                ax.scatter(ptt_ms, dbp, s=3, alpha=.2, color=RED, edgecolor="none")
                xs = np.array([ptt_ms.min(), ptt_ms.max()])
                ax.plot(xs, np.polyfit(ptt_ms, dbp, 1) @ np.vstack([xs, np.ones(2)]),
                        color="k", lw=1)
                ax.set_xlabel("measured PTT (ms)"); ax.set_ylabel("DBP (mmHg)")
                ax.set_title(f"r = {rr:+.3f}", fontsize=9)
                fig.tight_layout(); st.pyplot(fig)
            except Exception as e:
                st.caption(f"unavailable: {e}")

        st.caption(
            f"Calibration reaches {R['cal_mae_sbp']:.1f} mmHg SBP MAE — genuinely accurate — and "
            f"PTT is weakly decodable (R² {R['probe_ptt_r2']:.2f}). Yet the causal audit shows the "
            f"model does not use it: only {R['frac_correct_sign']*100:.0f}% of segments respond in "
            "the physiological direction. The interval detected from ECG→PPG is really pulse "
            "*arrival* time (PAT = pre-ejection period + transit); PEP moves independently of BP, "
            "so in resting ICU data the PTT→BP law is weak and can invert (right panel). "
            "Accurate ≠ faithful — and the mechanism may simply not be in this modality."
        )

# ── FAITHFUL TO WHAT? ─────────────────────────────────────────────────────────
with tab_cap:
    C, CV = load_capstone()
    if C is None or "sweep" not in C:
        st.warning("α-sweep results not found (or stale). Run `python precompute_recon.py`.")
    else:
        sw = C["sweep"]; al = C["alphas"]
        st.markdown(
            "Same dial as the synthetic tab — now on the **real** ECG+PPG model. "
            f"**α sets how much BP is routed through the reconstructed ABP pressure wave** "
            f"(λ = {C['lambda']:g}); the rest comes from a direct shortcut head:")
        st.latex(r"BP=\alpha\,\text{head}_{\text{recon}}\!\big(\text{ABP}_{\text{rebuilt}}\big)"
                 r"+(1-\alpha)\,\text{head}_{\text{shortcut}}(\text{features})")

        # headline: the sweep — accuracy & morphology-probe flat, donor-swap tracks alpha
        g1 = st.columns([3, 2])
        with g1[0]:
            st.caption("Three audits vs α  (accuracy & probe flat; only the donor-swap tracks faithfulness)")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(al, sw["swap"], "-o", ms=4, color=NAVY, label="donor-swap (causal)")
            ax.plot(al, sw["acc"], "-o", ms=4, color=RED, label="accuracy (R²)")
            ax.plot(al, sw["probe_morph"], "-o", ms=4, color=GREY, label="morphology probe (R²)")
            ax.set_xlabel("α  (reliance on reconstructed waveform)"); ax.set_ylabel("score")
            ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8, frameon=False)
            fig.tight_layout(); st.pyplot(fig)
        with g1[1]:
            st.caption(f"ABP reconstruction at α=1 — morphology corr {C['recon_corr']:.2f}")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(CV["t"], CV["abp_true"], color=GREEN, lw=1.4, label="true ABP")
            ax.plot(CV["t"], CV["abp_recon"], color=NAVY, lw=1.1, ls="--", label="reconstructed")
            ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("ABP (norm.)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        def row(name, arr): return f"| {name} | " + " | ".join(f"{x:.2f}" for x in arr) + " |"
        st.markdown(
            "| metric \\ α | " + " | ".join(f"{x:g}" for x in al) + " |\n"
            "|" + "---|" * (len(al) + 1) + "\n"
            + row("accuracy (R²)", sw["acc"]) + "\n"
            + row("morphology probe (R²)", sw["probe_morph"]) + "\n"
            + row("**donor-swap (causal)**", sw["swap"]))
        st.caption(
            "As reliance on the reconstruction rises, accuracy barely moves and morphology stays "
            "equally decodable — but the **causal donor-swap climbs from ~0 to "
            f"{max(sw['swap']):.2f}**. Faithfulness is the one property that tracks how much the "
            "model actually routes BP through the pressure wave. Exactly the synthetic-tab signature, "
            "reproduced on real signals — with the reconstruction as the faithfulness lever.")

        # mechanism profile at alpha=1 — faithful to WHICH cue?
        st.markdown("**At α = 1, faithful to _which_ cue?**  (donor-swap per candidate mechanism)")
        prof = C.get("profile", {}); cval = C.get("cue_validation", {})
        p2 = st.columns([3, 2])
        with p2[0]:
            if prof:
                names = list(prof.keys()); fracs = [prof[n]["frac_correct"] for n in names]
                cols = [GREEN if (prof[n]["expect_sign"] != 0 and prof[n]["frac_correct"] > 0.5)
                        else GREY for n in names]
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.barh(range(len(names)), fracs, color=cols)
                ax.axvline(0.5, color="k", ls=":", lw=1)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
                ax.set_xlabel("frac in expected physiological direction"); ax.set_xlim(0, 1)
                ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig)
        with p2[1]:
            st.caption("Shape cues are real — PPG-derived vs ground-truth ABP-derived:")
            st.markdown("\n".join(
                f"- {k}: r = {cval[k]:+.2f}" for k in ["rise", "aix", "apg"] if k in cval))
            st.caption("Green bars pass chance (0.5): the model causally uses those cues.")

        # data-driven verdict
        morph_fracs = [prof[n]["frac_correct"] for n in prof
                       if prof[n]["expect_sign"] != 0 and "morphology" in n]
        pat_f = prof.get("PAT (arrival time)", {}).get("frac_correct", float("nan"))
        best_morph = max(morph_fracs) if morph_fracs else float("nan")
        if np.isfinite(best_morph) and best_morph > 0.5 and pat_f < 0.5:
            st.success(
                f"**Faithful to morphology, not to arrival time.** At α=1 the model fails the PAT "
                f"audit (frac {pat_f:.2f} < 0.5) but passes a wave-shape / stiffness cue "
                f"(frac {best_morph:.2f} > 0.5). Routing BP through the rebuilt pressure wave makes the "
                "model use the mechanism ECG+PPG actually carries — and the audit says faithful *to "
                "what*, not just yes/no. Reconstruction is a usable lever for building faithful models.")
        else:
            st.info(
                f"At α=1 the causal audit reads PAT frac {pat_f:.2f} and best morphology cue "
                f"{best_morph:.2f}. The α-sweep still shows the donor-swap is the only property that "
                "tracks reconstruction reliance — faithfulness is causal, and separable from accuracy "
                "and decodability.")
