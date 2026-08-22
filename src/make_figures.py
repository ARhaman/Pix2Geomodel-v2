#!/usr/bin/env python3
"""
make_figures.py -- new figures for RINENG-D-26-10842.

Outputs, for each figure, three forms:
    figures_new/<name>.pdf   vector, for the journal
    figures_new/<name>.png   900 dpi raster
    figures_new/Figures_editable.pptx   native PowerPoint shapes, fully editable

Design constraints, in order of priority:
  1. Must survive greyscale printing. Every series therefore carries a hatch
     pattern as well as a hue, so identity never depends on colour alone.
  2. Validated categorical palette (dataviz six checks, light surface: PASS).
     Slots used in fixed order, never cycled.
  3. Direct value labels on every bar. The palette's contrast check returns a
     WARN for the lighter slots, and visible labels are the required relief.
  4. One axis per panel. No dual scales anywhere.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_new")
os.makedirs(OUT, exist_ok=True)

# Validated categorical slots, fixed order.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
HATCH = ["", "///", "...", "xxx"]
LBL = 6.4

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "hatch.linewidth": 0.6,
})


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 900})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.pdf / {name}.png")


def bars(ax, groups, series, values, colors, ylabel, ylim=None, fmt="{:.3f}",
         label_rot=0, inside=False):
    """Grouped bars with hatch as secondary encoding and direct labels."""
    n_s = len(series)
    width = 0.8 / n_s
    xs = range(len(groups))
    for j, s in enumerate(series):
        off = -0.4 + width * (j + 0.5)
        pos = [x + off for x in xs]
        b = ax.bar(pos, values[j], width * 0.92, label=s,
                   color=colors[j], edgecolor="white", linewidth=0.8,
                   hatch=HATCH[j % len(HATCH)])
        for rect, v in zip(b, values[j]):
            if inside:
                ax.annotate(fmt.format(v),
                            (rect.get_x() + rect.get_width() / 2,
                             rect.get_height()),
                            ha="center", va="top", xytext=(0, -3),
                            textcoords="offset points", fontsize=LBL,
                            color="white", rotation=label_rot, zorder=6)
            else:
                ax.annotate(fmt.format(v),
                            (rect.get_x() + rect.get_width() / 2,
                             rect.get_height()),
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points",
                            fontsize=LBL, color=INK2, rotation=label_rot)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


# ===================================================================== FIG A
def fig_benchmark():
    global LBL
    tasks = ["VCL", "Porosity", "Permeability"]
    gan_pa = [0.843, 0.615, 0.468]
    bay_pa = [0.956, 0.648, 0.478]
    rf_pa = [0.756, 0.600, 0.474]
    gan_f1 = [0.493, 0.272, 0.193]
    bay_f1 = [0.602, 0.312, 0.209]
    rf_f1 = [0.621, 0.283, 0.221]

    LBL = 5.8
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1))
    inv = ["Pix2Geomodel", "Bayes classifier", "Random forest"]

    bars(axes[0], tasks, inv, [gan_pa, bay_pa, rf_pa], [C1, C2, C3],
         "Pixel accuracy", (0, 1.34), label_rot=90)
    axes[0].set_title("(a) Inverse — pixel accuracy", fontsize=8, pad=6)

    bars(axes[1], tasks, inv, [gan_f1, bay_f1, rf_f1], [C1, C2, C3],
         "Macro-F1", (0, 0.95), label_rot=90)
    axes[1].set_title("(b) Inverse — macro-F1", fontsize=8, pad=6)

    ftasks = ["VCL", "Porosity", "Permeability\n(log₁₀)"]
    gan_r2 = [0.854, 0.575, 0.165]
    look_r2 = [0.654, 0.366, 0.175]
    bars(axes[2], ftasks, ["Pix2Geomodel", "Class-mean lookup"],
         [gan_r2, look_r2], [C1, C4], "R²", (0, 1.15), label_rot=90)
    axes[2].set_title("(c) Forward — R²", fontsize=8, pad=6)

    for ax in axes:
        ax.set_xlabel("Input property", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7.2)
    axes[2].set_xlabel("Target property", fontsize=7.5)
    for ax in axes:
        for lab in ax.get_xticklabels():
            lab.set_rotation(18); lab.set_ha("right")
    axes[0].legend(frameon=False, fontsize=6.2, loc="upper right",
                   handlelength=1.3, borderpad=0.1, handletextpad=0.4)
    axes[2].legend(frameon=False, fontsize=6.2, loc="upper right",
                   handlelength=1.3, borderpad=0.1, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "Fig_A_benchmark")
    LBL = 6.4



# ===================================================================== FIG E
def fig_ceiling():
    """Achieved R2 against the information ceiling eta^2, and partition test."""
    global LBL
    LBL = 6.4
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    tasks = ["Facies →\nVCL", "Facies →\nPorosity", "Facies → Perm.\n(log$_{10}$)"]
    ceil = [0.8814, 0.7145, 0.2130]
    got = [0.8543, 0.5754, 0.1652]
    pct = [100 * g / c for g, c in zip(got, ceil)]

    ax = axes[0]
    xs = range(len(tasks))
    ax.bar(xs, ceil, 0.62, color="#e8e8e4", edgecolor=INK2, linewidth=0.8,
           label="Ceiling: variance available from facies (η²)")
    ax.bar(xs, got, 0.62, color=C1, edgecolor="white", linewidth=0.8,
           label="Attained by Pix2Geomodel (R²)")
    for i, (c, g, p_) in enumerate(zip(ceil, got, pct)):
        ax.annotate("%.3f" % c, (i, c), ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points",
                    fontsize=LBL, color=INK2)
        ax.annotate("%.0f%%" % p_, (i, g / 2), ha="center", va="center",
                    fontsize=7.4, color="white", weight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels(tasks, fontsize=7.2)
    ax.set_ylabel("Fraction of variance")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="x", visible=False)
    ax.set_title("(a) Attained against available", fontsize=8, pad=6)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right",
              handlelength=1.3, borderpad=0.1, handletextpad=0.4)

    ax = axes[1]
    labs = ["Facies class\n(7 classes)", "Porosity\nseptile", "Flow-unit\nseptile"]
    vals = [0.2130, 0.2421, 0.8239]
    cols = [C1, C3, "#bdbdb8"]
    hats = ["", "...", "xxx"]
    for i, (v, c, h) in enumerate(zip(vals, cols, hats)):
        ax.bar([i], [v], 0.6, color=c, edgecolor="white", linewidth=0.8, hatch=h)
        ax.annotate("%.3f" % v, (i, v), ha="center", va="bottom",
                    xytext=(0, 2), textcoords="offset points",
                    fontsize=LBL, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_ylabel("η² of log$_{10}$ permeability")
    ax.set_ylim(0, 0.98)
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Which partition organises permeability", fontsize=8, pad=6)
    fig.tight_layout()
    save(fig, "Fig_E_information_ceiling")


# ===================================================================== FIG B
def fig_ladder():
    tasks = ["Porosity → Facies", "VCL → Facies", "Permeability → Facies"]
    corrected = [0.6151, 0.8434, 0.4675]
    plus_bg = [0.7141, 0.8824, 0.6044]
    plus_metric = [0.8843, 0.9506, 0.8153]
    submitted = [0.9060, 0.8440, 0.7948]
    NOINFO = 0.8746

    fig, ax = plt.subplots(figsize=(7.3, 3.5))
    bars(ax, tasks,
         ["Corrected: 7 classes, reservoir cells only",
          "+ inactive background scored as a class",
          "+ submitted metric: dominant RGB channel"],
         [corrected, plus_bg, plus_metric], [C1, C3, C2],
         "Pixel accuracy", (0, 1.30), inside=True)
    ax.set_xlim(-0.58, 3.18)

    LAB_Y = 1.15
    for i, s_val in enumerate(submitted):
        ax.plot([i - 0.46, i + 0.46], [s_val, s_val], color=INK, linewidth=1.5,
                linestyle=(0, (4, 2)), zorder=5)
        ax.plot([i, i], [s_val, LAB_Y - 0.035], color=INK2, linewidth=0.6,
                zorder=5)
        ax.annotate(f"submitted {s_val:.4f}", (i, LAB_Y), fontsize=6.8,
                    color=INK, ha="center", va="bottom", zorder=6)

    ax.axhline(NOINFO, color=INK2, linewidth=1.0, linestyle=(0, (1, 2)), zorder=4)
    ax.annotate("no-information\nprediction\n0.8746", (2.60, NOINFO),
                fontsize=6.8, color=INK2, ha="left", va="bottom", zorder=6,
                xytext=(0, 4), textcoords="offset points")

    ax.legend(frameon=False, fontsize=7.0, loc="upper center",
              bbox_to_anchor=(0.45, 1.19), handlelength=1.5,
              borderpad=0.2, ncol=1)
    fig.tight_layout()
    save(fig, "SUPP_defect_ladder_response_letter_only")


# ===================================================================== FIG C
def fig_positional():
    tasks = ["VCL →\nFacies", "Porosity →\nFacies", "Permeability →\nFacies"]
    matched_pa = [0.8435, 0.6151, 0.4675]
    shuffled_pa = [0.4205, 0.4293, 0.3682]
    constant_pa = [0.5103, 0.2105, 0.1384]
    matched_iou = [0.4768, 0.2191, 0.1449]
    shuffled_iou = [0.1315, 0.1165, 0.1006]
    constant_iou = [0.0954, 0.0600, 0.0330]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ser = ["Matched input", "Shuffled input (derangement)", "Constant input"]

    bars(axes[0], tasks, ser, [matched_pa, shuffled_pa, constant_pa],
         [C1, C2, C3], "Pixel accuracy", (0, 1.05))
    axes[0].set_title("(a) Pixel accuracy", fontsize=8.5, pad=6)

    bars(axes[1], tasks, ser, [matched_iou, shuffled_iou, constant_iou],
         [C1, C2, C3], "mIoU", (0, 0.62))
    axes[1].set_title("(b) mIoU", fontsize=8.5, pad=6)

    axes[0].legend(frameon=False, fontsize=6.8, loc="upper right",
                   handlelength=1.4, borderpad=0.2)
    fig.tight_layout()
    save(fig, "Fig_C_positional_control")


# ===================================================================== FIG D
def fig_dispersion():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    props = ["Porosity", "VCL", "Permeability"]
    sill = [73.5, 55.0, 15.0]          # variance retained, midpoint of range
    sd = [71.7, 82.0, 29.4]            # standard deviation retained
    bars(axes[0], props, ["Variance retained (sill)", "SD retained (marginal)"],
         [sill, sd], [C1, C3], "Percentage of reference (%)", (0, 118),
         fmt="{:.0f}")
    axes[0].axhline(100, color=INK2, linewidth=0.9, linestyle=(0, (1, 2)))
    axes[0].set_title("(a) Under-dispersion in value", fontsize=8.5, pad=6)

    cls = ["Tight", "V Poor", "Poor", "Fair", "Good", "G Clean", "V Good"]
    ratio = [964 / 116, 647 / 325, 1385 / 487, 4175 / 13,
             3524 / 749, 1128 / 169, 2487 / 444]
    axes[1].bar(range(len(cls)), ratio, 0.66, color=C2, edgecolor="white",
                linewidth=0.8, hatch="///")
    for i, r in enumerate(ratio):
        axes[1].annotate(f"{r:.0f}×", (i, r), ha="center", va="bottom",
                         fontsize=6.4, color=INK2, xytext=(0, 2),
                         textcoords="offset points")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(cls)))
    axes[1].set_xticklabels(cls, rotation=35, ha="right")
    axes[1].set_ylabel("Predicted / reference body count")
    axes[1].axhline(1, color=INK2, linewidth=0.9, linestyle=(0, (1, 2)))
    axes[1].grid(axis="x", visible=False)
    axes[1].set_ylim(0.8, 900)
    axes[1].set_title("(b) Fragmentation in space (VCL → facies)",
                      fontsize=8.5, pad=6)

    axes[0].legend(frameon=False, fontsize=6.4, loc="upper right",
                   handlelength=1.3, borderpad=0.1, handletextpad=0.4)
    fig.tight_layout()
    save(fig, "Fig_D_dispersion_fragmentation")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig_benchmark()
    fig_ceiling()
    fig_ladder()
    fig_positional()
    fig_dispersion()
    print("done")
