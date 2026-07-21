"""Streamlit dashboard: does the model know what it does not know?

    streamlit run audit_lab/app_faithfulness.py

Governing law:  BP = A / PTT^p + B.  p is the NONLINEARITY of the data (visible
in the BP vs PTT plot). Physiology: BP is the cause, PTT the derived noisy marker.
alpha sets how much the model routes BP through the real PTT pathway.

Finding baked in: as the data gets more nonlinear (larger p), the causal
donor-swap still tracks alpha, and the network keeps its PTT code simple, so a
linear probe stays fine. Decodable is not the same as used; the causal audit is
what catches a model that is right for the wrong reason.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

NAVY, RED, GREY = "#2f4b7c", "#c1543b", "#9aa0a6"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

B, BP_MEAN, BP_STD = 80.0, 120.0, 17.3
A_of = lambda p: 10.0 * 0.4 ** p                           # keeps PTT near [0.15, 0.4] across p
ptt_from_bp = lambda bp, p: (A_of(p) / (bp - B)) ** (1.0 / p)   # BP -> PTT (the cause)
bp_from_ptt = lambda ptt, p: A_of(p) / ptt ** p + B            # inverse the model learns
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


# =============================================================== UI
st.set_page_config(page_title="Does the model know what it doesn't know?", layout="wide")
models, TL, VL, EV, SC = train_grid()

st.title("Does the model know what it doesn't know?")
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
