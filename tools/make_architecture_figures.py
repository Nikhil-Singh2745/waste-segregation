"""Draws the two architecture diagrams used in the README.

    python tools/make_architecture_figures.py

Writes docs/arch_detector.png and docs/arch_classifier.png. The shapes and layer counts
are the ones defined in wastesort/models.py; if you change the networks, update them here.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs"

INK = "#1f2937"
MUTED = "#6b7280"
# (fill, edge) per role
DATA = ("#f3f4f6", "#9ca3af")
BACKBONE = ("#e6effd", "#3b6fd4")
NECK = ("#e5f5ec", "#2e8b57")
HEAD = ("#fdeedd", "#d9792b")
OUTPUT = ("#f1e9fd", "#7c4ddb")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, style=DATA, size=9, bold=True):
    """Rounded box centred on (x, y) with a title and an optional smaller second line."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=style[0], ec=style[1], lw=1.1, zorder=2))
    two = bool(sub) and "\n" in sub
    dy = (0.2 if two else 0.13) if sub else 0
    ax.text(x, y + dy, title, ha="center", va="center", color=INK, fontsize=size,
            fontweight="bold" if bold else "normal", zorder=3)
    if sub:
        ax.text(x, y - (0.1 if two else 0.17), sub, ha="center", va="center", color=MUTED, fontsize=size - 1.5, zorder=3,
                linespacing=1.25)


def arrow(ax, p, q, label=None, color=INK, off=(0, 0.14), ha="center", size=7.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color,
                                 shrinkA=0, shrinkB=0, zorder=1, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, ha=ha, va="center",
                color=MUTED, fontsize=size, zorder=3)


def plus(ax, x, y, style=NECK):
    ax.add_patch(Circle((x, y), 0.17, fc="white", ec=style[1], lw=1.2, zorder=2))
    ax.text(x, y - 0.005, "+", ha="center", va="center", fontsize=11, color=style[1], fontweight="bold", zorder=3)


def panel(ax, x0, y0, x1, y1, label, style):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="none", ec=style[1], lw=0.9, ls=(0, (4, 3)), zorder=0))
    ax.text(x0 + 0.12, y1 - 0.2, label, ha="left", va="center", color=style[1], fontsize=8.5, fontweight="bold")


# ------------------------------------------------------------------------------- detector
def detector():
    fig, ax = canvas(11.0, 7.6)
    bw, bh = 2.0, 0.78
    xb, xl, xp, xo, xh = 1.5, 4.25, 5.75, 7.35, 9.75       # columns: backbone, lateral, +, pyramid, head
    y_in, y_stem, y3, y4, y5 = 6.8, 5.4, 4.0, 2.55, 1.1  # rows

    panel(ax, 0.3, 0.45, 2.7, 6.3, "BACKBONE", BACKBONE)
    panel(ax, 3.2, 0.45, 8.5, 4.9, "NECK  (feature pyramid, 96 channels)", NECK)
    panel(ax, 8.7, 0.45, 10.8, 4.9, "HEAD", HEAD)

    box(ax, xb, y_in, bw, bh, "Input photo", "3 × 416 × 416")
    box(ax, xb, y_stem, bw, bh, "Stem", "2 conv (stride 2) + 1 ResBlock\n64 × 104 × 104   (1/4)", BACKBONE)
    box(ax, xb, y3, bw, bh, "Stage C3", "2 ResBlocks\n128 × 52 × 52   (1/8)", BACKBONE)
    box(ax, xb, y4, bw, bh, "Stage C4", "2 ResBlocks\n192 × 26 × 26   (1/16)", BACKBONE)
    box(ax, xb, y5, bw, bh, "Stage C5", "2 ResBlocks\n256 × 13 × 13   (1/32)", BACKBONE)
    for a, b in [(y_in, y_stem), (y_stem, y3), (y3, y4), (y4, y5)]:
        arrow(ax, (xb, a - bh / 2), (xb, b + bh / 2))

    lw_, lh = 1.25, 0.6
    pw = 1.6
    # lateral 1x1 convolutions
    for y in (y3, y4):
        box(ax, xl, y, lw_, lh, "1×1 conv", "→ 96 ch", NECK, size=8.5)
        arrow(ax, (xb + bw / 2, y), (xl - lw_ / 2, y))
        arrow(ax, (xl + lw_ / 2, y), (xp - 0.17, y))
        plus(ax, xp, y)
        arrow(ax, (xp + 0.17, y), (xo - pw / 2, y))
    box(ax, xo, y5, pw, bh, "P5", "1×1 conv → 96 ch\n96 × 13 × 13", NECK)
    arrow(ax, (xb + bw / 2, y5), (xo - pw / 2, y5))
    box(ax, xo, y4, pw, bh, "P4", "3×3 conv\n96 × 26 × 26", NECK)
    box(ax, xo, y3, pw, bh, "P3", "3×3 conv\n96 × 52 × 52", NECK)
    # top-down pathway
    arrow(ax, (xo - 0.35, y5 + bh / 2), (xp + 0.1, y4 - 0.15), "upsample ×2", off=(-0.62, -0.02))
    arrow(ax, (xo - 0.35, y4 + bh / 2), (xp + 0.1, y3 - 0.15), "upsample ×2", off=(-0.62, -0.02))

    # head
    box(ax, xh, y3, 1.7, bh, "Prediction head", "2 × 3×3 conv, 1×1 conv\n5 × 52 × 52", HEAD)
    arrow(ax, (xo + pw / 2, y3), (xh - 0.85, y3))
    outs = [("Centre heat-map", "1 channel · sigmoid", 2.75),
            ("Centre offset", "2 channels · sigmoid", 1.9),
            ("Box size (w, h)", "2 channels · exp", 1.05)]
    for title, sub, y in outs:
        box(ax, xh, y, 1.7, 0.66, title, sub, OUTPUT, size=8.5)
    arrow(ax, (xh, y3 - bh / 2), (xh, 2.75 + 0.33))
    ax.text(xh, 0.62, "per cell of the 52 × 52 grid", ha="center", va="center", color=MUTED, fontsize=7.5)

    # post-processing strip along the top
    box(ax, 6.85, y_in, 7.5, bh, "Decoding  (no learned weights)",
        "keep heat-map peaks above the confidence threshold  →  build boxes from offset + size  →  NMS (IoU 0.5)  →  crop each box for the classifier",
        size=9)
    arrow(ax, (xh + 0.55, y3 + bh / 2), (xh + 0.55, y_in - bh / 2))

    ax.text(0.3, 7.38, "WasteDetector", fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(2.02, 7.38, "anchor-free, class-agnostic, CenterNet-style  ·  4.51 M parameters", fontsize=9,
            color=MUTED, va="center")
    fig.savefig(OUT / "arch_detector.png", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- classifier
def classifier():
    fig, ax = canvas(11.0, 6.3)
    ax.text(0.3, 6.03, "WasteClassifier", fontsize=11.5, fontweight="bold", color=INK, va="center")
    ax.text(2.2, 6.03, "compact ResNet  ·  4.01 M parameters", fontsize=9, color=MUTED, va="center")

    # feature maps drawn as slabs: height follows spatial size, width follows channel count
    yc = 4.35
    maps = [  # x, width, height, shape label, style
        (0.75, 0.16, 2.3, "3 ch\n128 × 128", DATA),
        (2.25, 0.26, 1.85, "32 ch\n64 × 64", BACKBONE),
        (3.75, 0.34, 1.45, "64 ch\n32 × 32", BACKBONE),
        (5.2, 0.44, 1.1, "128 ch\n16 × 16", BACKBONE),
        (6.65, 0.56, 0.8, "256 ch\n8 × 8", BACKBONE),
        (8.05, 0.56, 0.55, "256 ch\n4 × 4", BACKBONE),
    ]
    ops = ["Stem\n2 × 3×3 conv", "Stage 1\n2 ResBlocks", "Stage 2\n2 ResBlocks", "Stage 3\n2 ResBlocks",
           "Stage 4\n1 ResBlock"]
    for x, w, h, label, style in maps:
        ax.add_patch(FancyBboxPatch((x - w / 2, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=style[0], ec=style[1], lw=1.1, zorder=2))
        ax.text(x, 2.95, label, ha="center", va="top", fontsize=8, color=INK, linespacing=1.3)
    for (x0, w0, *_), (x1, w1, *_), op in zip(maps, maps[1:], ops):
        arrow(ax, (x0 + w0 / 2 + 0.06, yc), (x1 - w1 / 2 - 0.06, yc))
        ax.text((x0 + x1) / 2, yc + 0.3, op, ha="center", va="center", fontsize=8, color=INK, linespacing=1.3)
    ax.text(0.75, 5.66, "Input crop", ha="center", va="center", fontsize=8, color=INK)

    # head
    box(ax, 9.05, yc, 0.78, 0.7, "GAP", "→ 256", HEAD, size=8.5)
    box(ax, 10.2, yc, 0.95, 0.7, "Linear", "256 → 4", HEAD, size=8.5)
    arrow(ax, (8.05 + 0.34, yc), (9.05 - 0.39, yc))
    arrow(ax, (9.05 + 0.39, yc), (10.2 - 0.475, yc), "dropout\n0.3", off=(0, -0.62), size=7.5)
    ax.text(9.05, 5.05, "Global average\npooling", ha="center", va="center", fontsize=8, color=INK, linespacing=1.3)
    ax.text(10.2, 5.05, "Classifier\nlayer", ha="center", va="center", fontsize=8, color=INK, linespacing=1.3)
    ax.text(10.2, 3.2, "4 class scores\n→ softmax", ha="center", va="top", fontsize=8, color=INK, linespacing=1.3)
    ax.text(4.4, 2.25, "every stage halves height and width (stride 2 in its first block)", ha="center",
            va="center", fontsize=7.5, color=MUTED)

    # building blocks
    ax.plot([0.3, 10.7], [1.85, 1.85], color="#e5e7eb", lw=1)
    ax.text(0.3, 1.55, "Building blocks  (shared by both networks)", fontsize=9, fontweight="bold", color=INK,
            va="center")
    y = 0.72
    ax.text(0.3, y, "ConvBN", fontsize=8.5, fontweight="bold", color=BACKBONE[1], va="center")
    xs = [1.45, 2.5, 3.4]
    for x, t, w in zip(xs, ["Conv k×k", "BatchNorm", "SiLU"], [0.85, 0.9, 0.6]):
        box(ax, x, y, w, 0.42, t, style=DATA, size=8, bold=False)
    arrow(ax, (1.45 + 0.425, y), (2.5 - 0.45, y))
    arrow(ax, (2.5 + 0.45, y), (3.4 - 0.3, y))

    x0 = 4.45
    ax.text(x0, y, "ResBlock", fontsize=8.5, fontweight="bold", color=BACKBONE[1], va="center")
    xa, xb_, xp_, xs_ = 6.2, 7.75, 8.85, 9.65
    box(ax, xa, y, 1.15, 0.42, "ConvBN 3×3", style=BACKBONE, size=8, bold=False)
    box(ax, xb_, y, 1.35, 0.42, "Conv 3×3 + BN", style=BACKBONE, size=8, bold=False)
    plus(ax, xp_, y, BACKBONE)
    box(ax, xs_, y, 0.6, 0.42, "SiLU", style=DATA, size=8, bold=False)
    arrow(ax, (5.25, y), (xa - 0.575, y))
    arrow(ax, (xa + 0.575, y), (xb_ - 0.675, y))
    arrow(ax, (xb_ + 0.675, y), (xp_ - 0.17, y))
    arrow(ax, (xp_ + 0.17, y), (xs_ - 0.3, y))
    arrow(ax, (xs_ + 0.3, y), (xs_ + 0.75, y))
    # skip connection
    ax.plot([5.4, 5.4, xp_], [y, y + 0.48, y + 0.48], color=BACKBONE[1], lw=1.1, zorder=1)
    arrow(ax, (xp_, y + 0.48), (xp_, y + 0.17), color=BACKBONE[1])
    ax.text((5.4 + xp_) / 2, y + 0.62, "skip connection  (identity, or 1×1 conv + BN when the shape changes)",
            ha="center", va="center", fontsize=7.5, color=BACKBONE[1])
    fig.savefig(OUT / "arch_classifier.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    detector()
    classifier()
    print("wrote", OUT / "arch_detector.png", "and", OUT / "arch_classifier.png")
