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
# Two input channels: an ARRIVAL-TIME channel (ptt, the physics signal) and a SEPARATE
# CONFOUND channel that also correlates with BP (the real-data analogue of an HR shortcut).
# The physics branch reads ptt; the shortcut branch reads the confound. α mixes them.
B, BP_MEAN, BP_STD = 80.0, 120.0, 17.3
A_of = lambda p: 10.0 * 0.4 ** p
ptt_from_bp = lambda bp, p: (A_of(p) / (bp - B)) ** (1.0 / p)
bp_from_ptt = lambda ptt, p: A_of(p) / ptt ** p + B
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PS = [1.0, 1.5, 2.0, 2.5, 3.0]
CONF_NOISE = 0.35
SHIFT_MS = np.array([-30, -20, -10, 0, 10, 20, 30])       # imposed arrival-time perturbations


def sample(n, seed, p):
    rng = np.random.default_rng(seed)
    bp = rng.uniform(90, 150, n)
    ptt = ptt_from_bp(bp, p) + rng.normal(0, 0.006, n)                # arrival-time channel
    conf = (bp - BP_MEAN) / BP_STD + rng.normal(0, CONF_NOISE, n)     # separate confound channel
    X = np.stack([ptt, conf], 1)
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor((bp - BP_MEAN) / BP_STD, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        path = lambda: nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.physics, self.shortcut = path(), path()     # physics <- ptt, shortcut <- confound
        self.head = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
        self.alpha = alpha

    def code(self, X):
        return self.alpha * self.physics(X[:, 0:1]) + (1 - self.alpha) * self.shortcut(X[:, 1:2])

    def forward(self, X):
        return self.head(self.code(X)).squeeze(1)


def accuracy(net, Xe, ye):
    return r2_score(ye.numpy(), net(Xe).detach().numpy())

def lin_probe(net, Xe):
    """Is arrival-time (ptt) linearly decodable from the internal code? (Decodable ≠ used.)"""
    a = net.code(Xe).detach().numpy(); t = Xe[:, 0].numpy(); h = len(t) // 2
    return r2_score(t[h:], Ridge().fit(a[:h], t[:h]).predict(a[h:]))

def roll_curve(net, Xe):
    """Perturbation audit: shift ONLY the arrival-time channel by ±ms and read mean predicted BP."""
    out = []
    for dm in SHIFT_MS:
        Xs = Xe.clone(); Xs[:, 0] = Xe[:, 0] + dm / 1000.0
        out.append(float((net(Xs).detach().numpy() * BP_STD + BP_MEAN).mean()))
    return np.array(out)

def roll_slope(net, Xe):
    return float(np.polyfit(SHIFT_MS, roll_curve(net, Xe), 1)[0])    # mmHg per ms (faithful < 0)


@st.cache_resource
def train_grid():
    models, tl, vl, evals, scores = {}, {}, {}, {}, {}
    for p in PS:
        tr_X, tr_b = sample(3000, 0, p); va_X, va_b = sample(1000, 1, p)
        evals[p] = sample(1500, 7, p)
        for a in ALPHAS:
            torch.manual_seed(0); net = Net(a); opt = torch.optim.Adam(net.parameters(), 3e-3)
            th, vh = [], []
            for _ in range(400):
                opt.zero_grad(); loss = ((net(tr_X) - tr_b) ** 2).mean()
                loss.backward(); opt.step(); th.append(loss.item())
                with torch.no_grad(): vh.append(float(((net(va_X) - va_b) ** 2).mean()))
            net.eval(); models[(a, p)] = net; tl[(a, p)] = th; vl[(a, p)] = vh
            Xe, ye = evals[p]
            scores[(a, p)] = dict(acc=accuracy(net, Xe, ye), lin=lin_probe(net, Xe),
                                  roll=roll_slope(net, Xe))
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

    Xe, ye = EV[p]; net = models[(alpha, p)]; s = SC[(alpha, p)]
    curve = roll_curve(net, Xe); slope = s["roll"]

    m = st.columns(3)
    m[0].metric("Accuracy (R²)", f"{s['acc']:.2f}")
    m[1].metric("Linear probe (R²)", f"{s['lin']:.2f}")
    m[2].metric("Roll audit (mmHg/ms)", f"{slope:+.2f}")

    if slope < -0.20:
        st.success(f"Faithful — roll audit {slope:+.2f} mmHg/ms: longer arrival time lowers BP, "
                   "the model uses arrival-time physics.")
    elif slope < -0.08:
        st.warning(f"Partial — roll audit {slope:+.2f} mmHg/ms: arrival time weakly used.")
    else:
        st.error(f"Spurious — accurate (R² {s['acc']:.2f}) but the roll audit is flat "
                 f"({slope:+.2f} mmHg/ms): the model rides the confound, not arrival time.")

    LIGHT = "#8a9bbf"
    g = st.columns(4)
    with g[0]:
        st.caption("BP → arrival-time law")
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
        st.caption("Roll audit — perturb arrival time")
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.axvline(0, color="#ddd", lw=.8, zorder=0)
        ax.plot(SHIFT_MS, curve, "-o", ms=3.5, color=NAVY)
        ax.set_xlabel("arrival-time shift (ms)"); ax.set_ylabel("pred. BP (mmHg)")
        ax.set_title(f"slope {slope:+.2f} (faithful<0)", fontsize=8)
        fig.tight_layout(); st.pyplot(fig)
    with g[3]:
        st.caption("Audits vs α")
        rolls = np.array([-SC[(a, p)]["roll"] for a in ALPHAS])
        use = rolls / max(rolls.max(), 1e-9)                  # arrival-time use, normalized to [0,1]
        fig, ax = plt.subplots(figsize=(3, 2.5))
        ax.plot(ALPHAS, [SC[(a, p)]["acc"] for a in ALPHAS], "-o", ms=3, color=GREY, label="accuracy")
        ax.plot(ALPHAS, [SC[(a, p)]["lin"] for a in ALPHAS], "-s", ms=3, color=LIGHT, label="linear probe")
        ax.plot(ALPHAS, use, "-o", ms=3.5, color=NAVY, label="roll audit")
        ax.axvline(alpha, color="k", ls=":", alpha=.4)
        ax.set_xlabel("α"); ax.legend(fontsize=6.5, frameon=False)
        fig.tight_layout(); st.pyplot(fig)

    st.caption("Only the roll audit tracks α. Accuracy and the linear probe stay high at every α — "
               "the model looks good on both while ignoring arrival time and riding the confound.")

# ── REAL WAVEFORMS ────────────────────────────────────────────────────────────
with tab_real:
    DEFAULT_CKPT = os.path.join(DATA, "dbp_transformer.pt")
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
        FS = (3.0, 3.0)                                     # small square figures

        m = st.columns(2)
        m[0].metric("Layers", cfg["depth"])
        m[1].metric("Attention heads", cfg["heads"])

        scalars = real_scalars(Xte, fs)
        stages = real_layer_features(net, Xte, cfg["depth"])
        xs = list(stages)

        row1 = st.columns(2)
        with row1[0]:
            st.caption("Example segment — z-scored ECG/PPG the model sees")
            fig, ax = plt.subplots(figsize=FS)
            ax.plot(t_axis, Xte[0, :, 0], color=NAVY, lw=1, label="ECG")
            ax.plot(t_axis, Xte[0, :, 1], color=RED, lw=1, label="PPG")
            ax.set_xlabel("time (s)"); ax.set_title(f"DBP = {dbp[0]:.0f} mmHg", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with row1[1]:
            st.caption("Training convergence")
            fig, ax = plt.subplots(figsize=FS)
            ax.plot(hist["train_mae"], color=NAVY, lw=1.3, label="train MAE")
            ax.plot(hist["val_mae"], color=RED, lw=1.3, ls="--", label="val MAE")
            ax.set_xlabel("epoch"); ax.set_ylabel("MAE (mmHg)")
            ax.legend(fontsize=6.5, frameon=False); fig.tight_layout(); st.pyplot(fig)

        row2 = st.columns(2)
        with row2[0]:
            st.caption("Linear-probe decodability by layer")
            r2_pat = [mechlib.linear_probe(f, scalars["pat"]) for f in stages.values()]
            r2_per = [mechlib.linear_probe(f, scalars["period"]) for f in stages.values()]
            fig, ax = plt.subplots(figsize=FS)
            ax.plot(xs, r2_pat, "-o", ms=3.5, color=NAVY, lw=1.3, label="PAT (arrival time)")
            ax.plot(xs, r2_per, "-o", ms=3.5, color=RED, lw=1.3, label="cardiac period (f2f)")
            ax.axhline(0, color="#bbb", lw=.8)
            ax.set_ylabel("probe R²"); ax.legend(fontsize=6.5, frameon=False)
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
            fig.tight_layout(); st.pyplot(fig)
        with row2[1]:
            st.caption("Causal test — shift PPG in time, watch predicted DBP")

            @torch.no_grad()
            def predict_fn(Xd):
                return net(torch.tensor(Xd, dtype=torch.float32)).numpy()

            shift_ms, curve, slope = mechlib.input_shift_audit(predict_fn, Xte, fs)
            fig, ax = plt.subplots(figsize=FS)
            ax.axvline(0, color=GREY, lw=.8, ls=":")
            ax.plot(shift_ms, curve, "-o", ms=3.5, color=NAVY)
            ax.set_xlabel("imposed PPG shift (ms)"); ax.set_ylabel("pred. DBP (mmHg)")
            ax.set_title(f"slope {slope:+.3f} mmHg/ms (faithful<0)", fontsize=8)
            fig.tight_layout(); st.pyplot(fig)

        st.caption(
            f"Decodable ≠ used: **cardiac period** is the most linearly decodable cue, yet the causal "
            f"shift test is flat (slope {slope:+.3f} mmHg/ms) — the model isn't using arrival-time "
            f"physics (PAT). Population PAT median {np.nanmedian(scalars['pat']) * 1000:.0f} ms, "
            f"cardiac period {np.nanmedian(scalars['period']):.2f} s "
            f"({60 / np.nanmedian(scalars['period']):.0f} bpm).")

# ── FAITHFUL TO WHAT? ─────────────────────────────────────────────────────────
with tab_cap:
    C, CV = load_capstone()
    if C is None or "cues" not in C:
        st.warning("Battery results not found (or stale). Run `python precompute_recon.py`.")
    else:
        cues = C["cues"]; cval = C.get("cue_validation", {})
        ctrl = cues.get("PPG amplitude (control)", {}).get("dep_mean", 0.5)   # chance floor
        roll = C.get("roll_slope_mean", float("nan"))
        st.markdown(
            "**We train the model to reconstruct the ABP pressure waveform** from ECG+PPG, then ask what it "
            "actually uses: **probe** it (is a cue linearly decodable?) and **roll-audit** it (shift the PPG "
            "in time — does predicted BP causally respond the way arrival-time physics demands?).")

        m = st.columns(3)
        m[0].metric("Reconstruction corr", f"{C['recon_corr']:.2f}")
        m[1].metric("Calibrated DBP MAE", f"{C['mae_cal_dbp']:.1f} mmHg")
        m[2].metric("Roll audit (mmHg/ms)", f"{roll:+.3f}")

        g = st.columns(3)
        with g[0]:
            st.caption("1 · Reconstruct the ABP wave")
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.plot(CV["t"], CV["abp_true"], color=GREEN, lw=1.5, label="true ABP")
            ax.plot(CV["t"], CV["abp_recon"], color=NAVY, lw=1.1, ls="--", label="reconstructed")
            ax.set_xlim(0, 5); ax.set_xlabel("time (s)"); ax.set_ylabel("ABP (norm.)")
            ax.set_title(f"corr {C['recon_corr']:.2f}", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False); fig.tight_layout(); st.pyplot(fig)
        with g[1]:
            st.caption("2 · Probe — what's decodable")
            names = sorted(cues, key=lambda n: cues[n]["probe_mean"])
            pr = [cues[n]["probe_mean"] for n in names]
            top_dec = names[-1]
            cols = [(NAVY if n == top_dec else GREY) for n in names]
            fig, ax = plt.subplots(figsize=(3.4, 2.6))
            ax.barh(range(len(names)), pr, color=cols)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([n.split(" (")[0] for n in names], fontsize=6.5)
            ax.set_xlabel("linear-probe R²"); fig.tight_layout(); st.pyplot(fig)
        with g[2]:
            st.caption("3 · Roll audit — shift PPG in time")
            if "roll_ms" in CV:
                fig, ax = plt.subplots(figsize=(3.4, 2.6))
                ax.axvline(0, color="#ddd", lw=.8, zorder=0)
                ax.plot(CV["roll_ms"], CV["roll_curve"], "-o", ms=3.5, color=NAVY)
                ax.set_xlabel("PPG shift (ms)"); ax.set_ylabel("pred. DBP (mmHg)")
                ax.set_title(f"slope {roll:+.3f} (faithful<0)", fontsize=8)
                fig.tight_layout(); st.pyplot(fig)
            else:
                st.info("Re-run `python precompute_recon.py` to add the roll audit.")

        top = max(cues.items(), key=lambda kv: kv[1]["probe_mean"])
        if not (roll < -0.05):     # flat / wrong-signed roll audit -> not using arrival time
            st.error(
                f"**Reconstruction ≠ faithfulness.** The model rebuilds the ABP wave (corr "
                f"{C['recon_corr']:.2f}) and **{top[0].split(' (')[0]}** is the most decodable cue (probe R² "
                f"{top[1]['probe_mean']:.2f}) — yet the roll audit is flat (slope {roll:+.3f} mmHg/ms): "
                "shifting arrival time barely moves predicted BP. So it rides the cardiac-timing (HR) "
                "confound, not arrival-time physics. Only the causal roll audit reveals this — reconstruction "
                "fidelity and decodability do not.")
        else:
            st.success(
                f"**Faithful** — roll audit {roll:+.3f} mmHg/ms (negative): predicted BP falls as arrival time "
                "grows, so the model uses arrival-time physics, not just the confound.")

        with st.expander("Full cue battery — decodable vs causally used (donor-swap dependence)"):
            rows = sorted(cues.items(), key=lambda kv: -kv[1]["dep_mean"])
            md = ("| cue | decodable (probe R²) | causal dependence | physiological direction (frac) |\n"
                  "|---|---|---|---|\n")
            for name, val in rows:
                flag = " ← used" if val["dep_mean"] - val["dep_std"] > ctrl + 0.05 else ""
                md += (f"| {name} | {val['probe_mean']:.2f} | {val['dep_mean']:.2f} ± {val['dep_std']:.2f}"
                       f"{flag} | {val['frac_mean']:.2f} |\n")
            st.markdown(md)
            st.caption("Shape cues validate vs ground-truth ABP: "
                       + ", ".join(f"{k} r={cval[k]:+.2f}"
                                   for k in ["rise", "aix", "apg", "kurt", "notch", "decay", "peak"]
                                   if k in cval and np.isfinite(cval[k]))
                       + f". Control floor {ctrl:.2f}. *Dependence* = does the output move with the cue "
                         "(any direction); *physiological direction* = the physiologically correct way "
                         "(0.5 = chance).")
