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
    if C is None or "cues" not in C:
        st.warning("Battery results not found (or stale). Run `python precompute_recon.py`.")
    else:
        cues = C["cues"]; cval = C.get("cue_validation", {})
        st.markdown(
            f"A CNN reconstructs the **ABP pressure waveform** from ECG+PPG (corr {C['recon_corr']:.2f}) "
            f"while predicting BP (calibrated DBP MAE {C['mae_cal_dbp']:.1f} mmHg). We then run the causal "
            "donor-swap across a **battery of physiological cues** and ask, for each: is it *decodable*, "
            f"and is it *causally used*?  ({C['n_seeds']} seeds; error bars = ±std.)")

        # headline: mechanism battery — which cue does the model causally use?
        g = st.columns([3, 2])
        with g[0]:
            st.caption("Mechanism battery — frac of segments responding in the physiological direction")
            names = list(cues.keys())
            fr = [cues[n]["frac_mean"] for n in names]; er = [cues[n]["frac_std"] for n in names]
            cols = [GREEN if (cues[n]["expect_sign"] != 0 and cues[n]["frac_mean"] > 0.5)
                    else (RED if cues[n]["expect_sign"] != 0 else GREY) for n in names]
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            ax.barh(range(len(names)), fr, xerr=er, color=cols, error_kw=dict(lw=.8, ecolor="#555"))
            ax.axvline(0.5, color="k", ls=":", lw=1)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
            ax.set_xlabel("causally used  (frac in expected direction; 0.5 = chance)")
            ax.set_xlim(0, 1); ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption(f"ABP reconstruction — morphology corr {C['recon_corr']:.2f}")
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.plot(CV["t"], CV["abp_true"], color=GREEN, lw=1.5, label="true ABP")
            ax.plot(CV["t"], CV["abp_recon"], color=NAVY, lw=1.1, ls="--", label="reconstructed")
            ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("ABP (norm.)")
            ax.legend(fontsize=7, frameon=False); fig.tight_layout(); st.pyplot(fig)

        st.markdown("**Decodable ≠ used** — every cue, ranked by how decodable it is:")
        rows = sorted(cues.items(), key=lambda kv: -kv[1]["probe_mean"])
        md = "| cue | decodable (probe R²) | causally used (frac ± std) |\n|---|---|---|\n"
        for name, val in rows:
            used = f"{val['frac_mean']:.2f} ± {val['frac_std']:.2f}"
            flag = " ✅" if (val["expect_sign"] != 0 and val["frac_mean"] > 0.5) else ""
            md += f"| {name} | {val['probe_mean']:.2f} | {used}{flag} |\n"
        st.markdown(md)
        st.caption("Shape cues are physiologically real — PPG-derived vs ground-truth ABP-derived: "
                   + ", ".join(f"{k} r={cval[k]:+.2f}" for k in ["rise", "aix", "apg"] if k in cval)
                   + ". The audit discriminates *on the same model* — so a null on PAT is a real "
                     "verdict, not a blind instrument.")

        pat = cues.get("PAT (arrival time)", {})
        morph = [(n, v) for n, v in cues.items() if v["expect_sign"] != 0 and "morphology" in n
                 and v["frac_mean"] > 0.5]
        pat_f = pat.get("frac_mean", float("nan"))
        if morph and np.isfinite(pat_f) and pat_f < 0.5:
            best = max(morph, key=lambda nv: nv[1]["frac_mean"])
            st.success(
                f"**These models are pulse-wave-analysis estimators, not transit-time estimators.** "
                f"Across {C['n_seeds']} seeds the causal audit finds arrival time (PAT) is *not* used "
                f"(frac {pat_f:.2f} < 0.5) while a wave-shape / stiffness cue **is** "
                f"({best[0].split(' (')[0]}, frac {best[1]['frac_mean']:.2f} > 0.5). The mechanism the "
                "network relies on is the pressure-wave morphology ECG+PPG actually carries — the audit "
                "tells us faithful *to what*, not merely yes/no.")
        else:
            st.info(
                f"Across {C['n_seeds']} seeds: PAT frac {pat_f:.2f}. Decodability and causal use come "
                "apart cue-by-cue — the causal audit is the only thing that separates them.")
