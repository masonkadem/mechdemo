"""fig_arch.py -- the audited transformer as a residual stream, with tensor shapes.

Vertical residual stream: the patch embedding (trapezoid) writes 50 tokens of width 64 into the
stream; each layer's attention and MLP read from it and add back to it; the read-out (triangle)
averages the tokens and maps them to one BP value.

Shapes are those of every audited checkpoint (mechlib.WaveTransformer defaults,
real_audit_controls.py): B segments per batch (128 when training), 10 s of ECG + PPG at 125 Hz
= 1250 x 2, 25-sample (200 ms) patches -> 50 tokens, width 64, 4 heads, MLP hidden 128, 3 layers.

Coordinates are in inches (figure size = axis limits, equal aspect).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"

INK, DIM = "#222222", "#666666"
FS_NAME, FS_DIM = 10, 8
XS = 1.0                                   # x of the residual stream
XB = 2.45                                  # x of the branches


def dims(ax, x, y, s, **kw):
    ax.text(x, y, s, color=DIM, **{"fontsize": FS_DIM, "ha": "left", "va": "center", **kw})


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=9,
                                shrinkA=0, shrinkB=0), zorder=2)


def line(ax, xs, ys, lw=1.0, z=2):
    ax.plot(xs, ys, color=INK, lw=lw, zorder=z, solid_capstyle="butt")


def poly(ax, pts):
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=INK, lw=1.2, zorder=3))


def add(ax, y, r=0.13):
    ax.add_patch(Circle((XS, y), r, fc="white", ec=INK, lw=1.2, zorder=4))
    line(ax, [XS - 0.6 * r, XS + 0.6 * r], [y, y], z=5)
    line(ax, [XS, XS], [y - 0.6 * r, y + 0.6 * r], z=5)


def branch(ax, y_tap, y_in, y_out, y_add):
    """Stream -> right -> down into a block (top y_in); block bottom y_out -> down -> left into an add."""
    line(ax, [XS, XB, XB], [y_tap, y_tap, y_in + 0.25])
    arrow(ax, XB, y_in + 0.25, XB, y_in)
    line(ax, [XB, XB], [y_out, y_add])
    arrow(ax, XB, y_add, XS + 0.13, y_add)
    add(ax, y_add)


def main():
    x0, x1, y0, y1 = 0.0, 4.6, 0.55, 8.9
    fig = plt.figure(figsize=(x1 - x0, y1 - y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")

    # input and patch embedding
    ax.text(XS, 8.72, "ECG + PPG", fontsize=FS_NAME + 1, ha="center", va="center")
    dims(ax, XS, 8.49, "(B, 1250, 2)", ha="center")
    dims(ax, XS, 8.29, "10 s at 125 Hz", ha="center")
    arrow(ax, XS, 8.14, XS, 7.6)
    dims(ax, XS + 0.15, 7.87, "← audit shifts the PPG here", style="italic")
    poly(ax, [(XS - 0.65, 7.6), (XS + 0.65, 7.6), (XS + 0.13, 6.95), (XS - 0.13, 6.95)])
    dims(ax, XS + 0.8, 7.38, "25 samples (200 ms) → 64")
    dims(ax, XS + 0.8, 7.17, "50 tokens × 200 ms = 10 s")

    # residual stream
    line(ax, [XS, XS], [6.95, 2.25], lw=2.6)
    dims(ax, XS - 0.12, 6.6, "(B, 50, 64)", ha="right")

    # attention
    branch(ax, 6.55, 6.05, 5.15, 4.85)
    ax.add_patch(Rectangle((XB - 0.45, 5.15), 0.9, 0.9, fc="white", ec=INK, lw=1.2, zorder=3))
    ax.text(XB, 5.6, "Attn", fontsize=FS_NAME, ha="center", va="center", zorder=4)
    dims(ax, XB + 0.6, 5.72, "4 heads")
    dims(ax, XB + 0.6, 5.5, "(B, 4, 50, 50)")

    # MLP: expand, nonlinearity, contract
    branch(ax, 4.5, 4.0, 2.95, 2.65)
    poly(ax, [(XB - 0.28, 4.0), (XB + 0.28, 4.0), (XB + 0.62, 3.53), (XB - 0.62, 3.53)])
    ax.add_patch(Rectangle((XB - 0.62, 3.44), 1.24, 0.06, fc="white", ec=INK, lw=1.0, zorder=3))
    poly(ax, [(XB - 0.62, 3.41), (XB + 0.62, 3.41), (XB + 0.28, 2.95), (XB - 0.28, 2.95)])
    dims(ax, XB + 0.75, 3.47, "(B, 50, 128)")

    # the two blocks repeat 3 times
    line(ax, [4.35, 4.45, 4.45, 4.35], [6.4, 6.4, 2.75, 2.75], lw=0.9)
    ax.text(4.4, 4.55, "× 3", fontsize=FS_NAME, ha="right", va="center")

    # read-out
    poly(ax, [(XS - 0.45, 2.25), (XS + 0.45, 2.25), (XS, 1.6)])
    dims(ax, XS + 0.6, 1.95, "mean over tokens → (B, 64)")
    dims(ax, XS + 0.6, 1.75, "linear → (B,)")
    arrow(ax, XS, 1.6, XS, 1.25)
    ax.text(XS, 1.08, "BP", fontsize=FS_NAME + 1, ha="center", va="center")

    dims(ax, 0.05, 0.7, "B = batch (128 segments)  ·  50 = tokens (200 ms each)  ·  64 = width", fontsize=7.5)

    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig_arch.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG / "fig_arch.pdf", bbox_inches="tight", pad_inches=0.05)
    print("wrote", FIG / "fig_arch.png")


if __name__ == "__main__":
    main()
