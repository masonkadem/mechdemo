"""fig_shift_explainer.py -- how the audit's shift works, for slides and study.

Left: one heartbeat. The ECG stays put; the PPG is slid 48 ms earlier, so the pulse arrives
sooner after the R peak and PAT is 48 ms shorter. Top right: the same 48 ms shift on the real
array, (B, 1250, 2): the PPG samples move 6 places left, the last value fills the end, the
ECG is untouched (mechlib._shift_channel). Bottom right: what a
model following the textbook law should do (shorter PAT -> higher BP, -0.22 mmHg/ms).

Illustrative signals (Gaussian bumps), no data. The VitalDB PPG channel offset is left out.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fs = 125
t = np.arange(0, 1.2, 1 / fs)
bump = lambda c, w, h: h * np.exp(-0.5 * ((t - c) / w) ** 2)
R, PAT, SHIFT = 0.15, 0.30, 0.048                 # R peak at 0.15 s, PAT 300 ms, shift 48 ms
ecg = bump(R, 0.012, 1) + bump(R + 0.25, 0.04, 0.3)
pulse = lambda foot: bump(foot + 0.13, 0.06, 1) + bump(foot + 0.40, 0.08, 0.35)
ppg = pulse(R + PAT)
ppg_s = pulse(R + PAT - SHIFT)
BLUE, ORANGE, RED = "#2f6db5", "#e07b1a", "#c0392b"

fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1], hspace=0.55, wspace=0.28)


def pat_arrow(ax, x1, x2, y, label, color):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    ax.text((x1 + x2) / 2, y + 0.08, label, ha="center", va="bottom", color=color, fontsize=10)


ax = fig.add_subplot(gs[0, 0])
ax.plot(t, ecg, "k", lw=1.2)
ax.axvline(R, color="grey", ls=":", lw=1)
ax.text(R + 0.01, 0.95, "R peak (heartbeat)", fontsize=9, va="top")
ax.set_title("ECG: stays where it is", loc="left", fontsize=10)

ax = fig.add_subplot(gs[1, 0], sharex=ax)
ax.plot(t, ppg, color=BLUE, lw=1.6)
ax.axvline(R, color="grey", ls=":", lw=1)
ax.axvline(R + PAT, color=BLUE, ls=":", lw=1)
pat_arrow(ax, R, R + PAT, 0.25, "PAT", BLUE)
ax.set_ylim(-0.1, 1.15)
ax.set_title("PPG, original", loc="left", fontsize=10)

ax = fig.add_subplot(gs[2, 0], sharex=ax)
ax.plot(t, ppg, color=BLUE, lw=1, alpha=0.35)
ax.plot(t, ppg_s, color=ORANGE, lw=1.6)
ax.axvline(R, color="grey", ls=":", lw=1)
ax.axvline(R + PAT - SHIFT, color=ORANGE, ls=":", lw=1)
pat_arrow(ax, R, R + PAT - SHIFT, 0.25, "PAT − 48 ms", ORANGE)
ax.annotate("", xy=(R + PAT + 0.13 - SHIFT, 1.05), xytext=(R + PAT + 0.13, 1.05),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
ax.text(R + PAT + 0.15, 1.05, "moved 48 ms earlier (left)", color=ORANGE, fontsize=9, va="center")
ax.set_ylim(-0.1, 1.25)
ax.set_title("PPG, shifted: pulse arrives sooner, so PAT is shorter", loc="left", fontsize=10)
ax.set_xlabel("Time (s)")
for a in fig.axes:
    a.set_yticks([]); a.spines[["top", "right", "left"]].set_visible(False)

# top right: the same 48 ms shift on the real array, (B, 1250, 2)
ax = fig.add_subplot(gs[0:2, 1]); ax.axis("off"); ax.set_xlim(-1.6, 9.6); ax.set_ylim(-1.0, 5.0)
ax.set_title("What the code does: shift d = −6 samples (6 × 8 ms = 48 ms)", loc="left", fontsize=10)
ax.text(-1.6, 4.6, "X: (B, 1250, 2).  Channel 0 = ECG: unchanged.  Channel 1 = PPG: shifted.",
        fontsize=8, va="center")


def block(x0, x1, y, label, idx, fc="white", note=None):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, 0.7, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec="black", lw=1))
    ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="center", fontsize=9)
    ax.text((x0 + x1) / 2, y + 0.8, idx, ha="center", va="bottom", fontsize=7, color="grey")
    if note:
        ax.text((x0 + x1) / 2, y - 0.08, note, ha="center", va="top", fontsize=7.5, color="grey")


GREY_FC, FILL = "#eeeeee", "#fde7d3"
ax.text(-0.2, 3.35, "PPG in", ha="right", va="center", fontsize=9)
block(0.0, 2.3, 3.0, "x₀ … x₅", "index 0 … 5", fc=GREY_FC, note="dropped")
block(2.4, 9.6, 3.0, "x₆  x₇  …  x₁₂₄₉", "index 6 … 1249", note="1244 samples")
ax.text(-0.2, 1.05, "PPG out", ha="right", va="center", fontsize=9)
block(0.0, 7.2, 0.7, "x₆  x₇  …  x₁₂₄₉", "index 0 … 1243")
block(7.3, 9.6, 0.7, "x₁₂₄₉ × 6", "1244 … 1249", fc=FILL, note="last value repeated")
for xa, xb in ((2.4, 0.0), (9.6, 7.2)):                       # the kept block slides 6 places left
    ax.annotate("", xy=(xb, 1.62), xytext=(xa, 2.62),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2, shrinkA=0, shrinkB=0))
code = ("out[:, 0:1244, 1] = X[:, 6:1250, 1]    # (B, 1244)\n"
        "out[:, 1244:, 1]  = X[:, 1249:, 1]     # (B, 1) copied into (B, 6)\n"
        "out[:, :, 0]      = X[:, :, 0]         # ECG untouched")
ax.text(-1.6, -0.95, code, family="monospace", fontsize=7.5, va="bottom")

# bottom right: what a textbook model should do
ax = fig.add_subplot(gs[2, 1])
dpat = np.array([-48, -32, -16, 0])
ax.plot(dpat, -0.22 * dpat, "o-", color=RED)
ax.set(xlabel="Change in PAT (ms)", ylabel="Change in\npredicted BP (mmHg)", xticks=dpat)
ax.set_title("Textbook model: shorter PAT → higher BP", loc="left", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

FIG = Path(__file__).resolve().parent / "figures"
fig.savefig(FIG / "fig_shift_explainer.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_shift_explainer.pdf", bbox_inches="tight")
print("wrote", FIG / "fig_shift_explainer.png")
