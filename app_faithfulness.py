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

        g = st.columns(3)
        with g[0]:
            st.caption("ECG + PPG — one segment")
            fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.6), sharex=True)
            axes[0].plot(A["t"], A["wave_ecg"], color=RED, lw=.8); axes[0].set_ylabel("ECG")
            axes[0].set_title(
                f"SBP {float(A['sbp0']):.0f} / DBP {float(A['dbp0']):.0f} mmHg", fontsize=8)
            axes[1].plot(A["t"], A["wave_ppg"], color=NAVY, lw=.8); axes[1].set_ylabel("PPG")
            axes[1].set_xlabel("time (s)"); axes[1].set_xlim(0, 5)
            fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption("Causal audit — BP vs imposed PTT shift (faithful = down)")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.plot(A["curve_shift_ms"], A["curve_sbp"], "-o", ms=3, color=NAVY, label="SBP")
            ax.plot(A["curve_shift_ms"], A["curve_dbp"], "-o", ms=3, color=RED, label="DBP")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted BP (mmHg)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g[2]:
            st.caption("PTT vs BP — the transit law is weak here")
            try:
                ptt_ms, sbp, dbp = ptt_scatter()
                rr = float(np.corrcoef(ptt_ms, dbp)[0, 1])
                fig, ax = plt.subplots(figsize=(3.4, 2.6))
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
    if C is None or "recon_corr" not in C.get("vanilla", {}):
        st.warning("Reconstruction results not found (or stale). Run `python precompute_recon.py`.")
    else:
        v, a = C["vanilla"], C["aux"]
        st.markdown(
            f"Same CNN backbone (ECG + PPG → BP). The **aux** model adds one objective — "
            f"reconstruct the **ABP pressure waveform** from the shared features (λ = {C['lambda']:g}). "
            "We then measure four things people conflate with faithfulness.")

        def cell(x): return f"{x:.2f}"
        st.markdown(
            "| model | DBP MAE (cal) ↓ | PTT decodable (probe R²) | ABP morphology (recon corr) | "
            "causal PTT use (donor-swap frac) |\n"
            "|---|---|---|---|---|\n"
            f"| vanilla | {v['mae_cal_dbp']:.1f} | {cell(v['probe_ptt_r2'])} | — | {cell(v['ds_dbp_frac'])} |\n"
            f"| **+ ABP reconstruction** | {a['mae_cal_dbp']:.1f} | **{cell(a['probe_ptt_r2'])}** "
            f"| **{cell(a['recon_corr'])}** | {cell(a['ds_dbp_frac'])} |")
        st.caption(
            f"Reconstruction makes PTT **{a['probe_ptt_r2']/max(v['probe_ptt_r2'],1e-6):.1f}× more "
            f"decodable** ({v['probe_ptt_r2']:.2f} → {a['probe_ptt_r2']:.2f}) and rebuilds the pressure "
            f"wave at corr {a['recon_corr']:.2f} — yet causal PTT use stays at chance "
            f"({v['ds_dbp_frac']:.2f} → {a['ds_dbp_frac']:.2f}). Accuracy, decodability, and "
            "reconstruction fidelity are each **independent** of mechanistic faithfulness.")

        g1 = st.columns(2)
        with g1[0]:
            st.caption(f"ABP reconstruction (aux) — morphology corr {a['recon_corr']:.2f}")
            fig, ax = plt.subplots(figsize=(4.6, 2.6))
            ax.plot(CV["t"], CV["abp_true"], color=GREEN, lw=1.4, label="true ABP")
            ax.plot(CV["t"], CV["abp_recon"], color=NAVY, lw=1.1, ls="--", label="reconstructed")
            ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("ABP (norm.)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g1[1]:
            st.caption("Mechanism profile — faithful to WHICH cue? (aux model)")
            prof = a.get("profile", {})
            if prof:
                names = list(prof.keys()); fracs = [prof[n]["frac_correct"] for n in names]
                cols = [GREEN if (prof[n]["expect_sign"] != 0 and prof[n]["frac_correct"] > 0.5)
                        else GREY for n in names]
                fig, ax = plt.subplots(figsize=(4.6, 2.6))
                ax.barh(range(len(names)), fracs, color=cols)
                ax.axvline(0.5, color="k", ls=":", lw=1)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
                ax.set_xlabel("frac in expected direction"); ax.set_xlim(0, 1)
                ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig)
            else:
                st.caption("profile not in data")

        g2 = st.columns(2)
        with g2[0]:
            st.caption("Morphology ⊥ faithfulness — per test segment (aux)")
            if "seg_recon_corr" in CV:
                sc, sl = CV["seg_recon_corr"], CV["seg_causal_slope"]
                rr = float(np.corrcoef(sc, sl)[0, 1])
                fig, ax = plt.subplots(figsize=(4.6, 2.6))
                ax.scatter(sc, sl, s=5, alpha=.3, color=NAVY, edgecolor="none")
                ax.axhline(0, color="#ccc", lw=.6)
                ax.set_xlabel("reconstruction quality (corr)")
                ax.set_ylabel("causal PTT response")
                ax.set_title(f"r = {rr:+.2f}  (no link)", fontsize=9)
                fig.tight_layout(); st.pyplot(fig)
            else:
                st.caption("per-segment data not in file")
        with g2[1]:
            st.caption("DBP vs imposed PTT shift (faithful = negative slope)")
            fig, ax = plt.subplots(figsize=(4.6, 2.6))
            ax.plot(CV["shift_ms"], CV["van_dbp"], "-o", ms=3, color=GREY, label="vanilla")
            ax.plot(CV["shift_ms"], CV["aux_dbp"], "-o", ms=3, color=NAVY, label="aux (ABP)")
            ax.plot(CV["shift_ms"], CV["analytic_dbp"], "--", color=GREEN, lw=1.4,
                    label="analytic (faithful by constr.)")
            ax.set_xlabel("imposed PTT shift (ms)"); ax.set_ylabel("predicted DBP (mmHg)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        st.markdown("**The constructive turn — a faithful-by-design model**")
        ro = C.get("readoff")
        rc = st.columns([1, 2])
        with rc[0]:
            if ro:
                st.metric("read-off SBP MAE", f"{ro['mae_sbp']:.1f} mmHg", help="raw, uncalibrated")
                st.metric("read-off DBP MAE", f"{ro['mae_dbp']:.1f} mmHg", help="raw, uncalibrated")
        with rc[1]:
            if "readoff_abp_rec" in CV:
                st.caption("predict the ABP wave in mmHg, then read SBP/DBP off peak & trough — "
                           "the mechanism is transparent and auditable by construction")
                fig, ax = plt.subplots(figsize=(6, 2.4))
                ax.plot(CV["readoff_t"], CV["readoff_abp_true"], color=GREEN, lw=1.4, label="true ABP")
                ax.plot(CV["readoff_t"], CV["readoff_abp_rec"], color=NAVY, lw=1.1, ls="--",
                        label="reconstructed")
                ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("mmHg")
                ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        prof = a.get("profile", {})
        morph = prof.get("PPG rise-time (morphology)", {}).get("frac_correct", float("nan"))
        pat_f = a["ds_dbp_frac"]
        if np.isfinite(morph) and morph > 0.5 and pat_f < 0.5:
            st.success(
                f"**Faithful to morphology, not to PAT.** The reconstruction model fails the "
                f"arrival-time audit (frac {pat_f:.2f}) but passes the wave-shape / stiffness cue "
                f"(frac {morph:.2f} > 0.5). The audit doesn't just say faithful/unfaithful — it says "
                "faithful to *what*. Forcing the model to rebuild the pressure wave routes it through "
                "the mechanism ECG+PPG actually carries (morphology / arterial stiffness), not the "
                "one it doesn't (transit time). That reframes a 'failed PTT audit' as a correct "
                "mechanistic verdict — and points to reconstruction as the way to build faithful models.")
        else:
            st.info(
                f"**Reconstruction ≠ causal use.** Even at recon corr {a['recon_corr']:.2f}, the model "
                f"does not causally route DBP through PTT (frac {pat_f:.2f}); the analytic control — "
                "faithful to PAT by construction — also fails (green curve), because the arrival-time "
                "law is sign-inverted in this resting cohort. Faithfulness needs the mechanism present "
                "in the data AND causally used; reconstruction alone gives neither for free.")
