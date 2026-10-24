#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_plot_error_maps.py -- build the pixel-wise error-map figure (Figure 10).

Reads the arrays written by 31_error_maps.py and draws, for each forward task:

    column 1  the reference property on the representative test layer
    column 2  where the network is better, where the forest is better
              (|e_forest| - |e_network|, diverging; blue = network better)
    column 3  mean absolute error against distance to the nearest facies-class
              boundary, over all nine test layers

Column 3 is the point of the figure. The maps show that the difference is
spatially organised rather than noise; the curves say what organises it, and
they say something different for each property:

  porosity   the network's advantage is concentrated within ~8 cells of a
             class boundary and vanishes in the interior, where the two
             estimators are indistinguishable;
  VCL        the opposite -- the forest is better near boundaries, where the
             class label almost determines VCL, and the network is better deep
             inside bodies, where only a wide receptive field can follow the
             within-class trend;
  log10 k    the network is slightly ahead everywhere, and both are poor,
             because little is recoverable from the label at any distance.

Design follows src/make_figures.py: serif, hatch as a secondary encoding,
direct value labels, one axis per panel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

IN_DIR = "/mnt/user-data/uploads/Revision/results"
OUT = "/home/claude/r3/figs"
os.makedirs(OUT, exist_ok=True)

C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "savefig.facecolor": "white", "hatch.linewidth": 0.6,
})

TASKS = [
    ("_errmap_facies_to_porosity_linear.npz", "Facies → Porosity",
     "Porosity (v/v)", "MAE (v/v)"),
    ("_errmap_facies_to_vcl_linear.npz", "Facies → VCL",
     "VCL (v/v)", "MAE (v/v)"),
    ("_errmap_facies_to_permeability_log10.npz",
     "Facies → Permeability (log₁₀)", "log₁₀ k (mD)", "MAE (log₁₀ mD)"),
]


def crop(*arrs):
    """Trim all arrays to the bounding box of the valid (non-NaN) area."""
    m = np.isfinite(arrs[0])
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arrs
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return tuple(a[r0:r1, c0:c1] for a in arrs)


def main() -> int:
    data = []
    for fn, title, cb1, ylab in TASKS:
        p = os.path.join(IN_DIR, fn)
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run 31_error_maps.py first")
        data.append((np.load(p, allow_pickle=True), title, cb1, ylab))

    # The reservoir crop is exactly 2:1 (tall), so a square panel wastes half
    # its width. The map columns are sized to that aspect and the figure is
    # made taller, which enlarges the maps instead of padding them.
    fig, axes = plt.subplots(3, 3, figsize=(7.1, 8.5),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 2.25]})

    for r, (z, title, cb1, ylab) in enumerate(data):
        ref, diff = crop(z["ref"], z["e_rf"] - z["e_net"])  # >0: net better
        rep = int(z["rep_layer"])
        # full-resolution fraction, computed by 31_error_maps.py before
        # downsampling -- not re-derived from the displayed map
        frac = 100.0 * float(z["frac_net_better"])

        # ---- column 1: reference -----------------------------------------
        ax = axes[r, 0]
        im = ax.imshow(ref, cmap="viridis", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(title, fontsize=7.6)
        if r == 0:
            ax.set_title("Reference, representative\ntest layer",
                         fontsize=7.8, pad=5)
        ax.set_xlabel(f"layer {rep}", fontsize=6.2, labelpad=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
        cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.6)

        # ---- column 2: who is better, where ------------------------------
        ax = axes[r, 1]
        v = np.nanpercentile(np.abs(diff), 98)
        im = ax.imshow(diff, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0:
            ax.set_title("|error| forest − |error| network\n"
                         "(blue: network better)", fontsize=7.8, pad=5)
        ax.set_xlabel(f"network better on {frac:.1f} % of cells",
                      fontsize=6.2, labelpad=1.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.015)
        cb.ax.tick_params(labelsize=5.6); cb.outline.set_linewidth(0.6)

        # ---- column 3: error against distance to boundary ----------------
        ax = axes[r, 2]
        labs = [str(x) for x in z["bin_labels"]]
        xs = np.arange(len(labs))
        mn, mr = z["mae_net_by_dist"], z["mae_rf_by_dist"]
        ax.plot(xs, mr, "-s", color=C2, lw=1.3, ms=3.4,
                mfc="white", mew=1.0, label="Random forest")
        ax.plot(xs, mn, "-o", color=C1, lw=1.3, ms=3.4,
                mfc="white", mew=1.0, label="Pix2Geomodel")
        ax.fill_between(xs, mn, mr, where=(mn <= mr), color=C1, alpha=0.13,
                        interpolate=True, lw=0)
        ax.fill_between(xs, mn, mr, where=(mn > mr), color=C2, alpha=0.13,
                        interpolate=True, lw=0)
        ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=6.6)
        ax.set_ylabel(ylab, fontsize=7.2)
        ax.tick_params(axis="y", labelsize=6.6)
        ax.grid(axis="x", visible=False)
        if r == 0:
            ax.set_title("Error against distance to the\n"
                         "nearest facies boundary", fontsize=7.8, pad=5)
        if r == 2:
            ax.set_xlabel("Distance to nearest facies-class boundary (cells)",
                          fontsize=7.0)
        ax.legend(frameon=False, fontsize=6.0, handlelength=1.5,
                  borderpad=0.1, handletextpad=0.4, loc="best")

    fig.tight_layout(h_pad=1.0, w_pad=0.45)
    for ext, kw in (("png", {"dpi": 900}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"Fig_error_maps.{ext}"),
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"wrote {OUT}/Fig_error_maps.png and .pdf")

    # numbers for the manuscript text
    print("\nMAE by distance to the nearest facies boundary:")
    for z, title, _, _ in data:
        print(f"  {title}")
        for lb, a, b in zip(z["bin_labels"], z["mae_net_by_dist"],
                            z["mae_rf_by_dist"]):
            better = "network" if a < b else "forest"
            print(f"     {str(lb):>6}  network {a:.4g}   forest {b:.4g}   "
                  f"-> {better}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
