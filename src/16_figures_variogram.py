#!/usr/bin/env python3
"""
16_figures_variogram.py
=======================

Publication figures for the spatial validation, built from the CSVs written by
09_spatial_validation.py.

REPLACES the variogram figure in the submitted manuscript
(Figures/Viogram/Variogram_Validation_Q1). That figure was not salvageable:

  * it was computed from rendered figure PNGs, not from model output
    (the generating script wrote to a directory named validation_from_pngs);
  * the "original" and "generated" screenshots are different sizes -- 740x744
    against 736x749 for porosity, 600x1206 against 601x1209 for facies -- so
    the two rasters have no cell correspondence and 449,282 valid cells were
    compared against 449,864. Nothing paired can be computed from that;
  * one later variant compared an original of (12045, 8184) against a
    generated map of (1207, 603);
  * lags were counted in screenshot pixels. One pixel of a 600x1206 rendering
    of a 248x365 grid is about 13.6 cells across and 10 cells down, so the
    plotted curve mixes two different physical scales and manufactures
    anisotropy;
  * the extracted facies composition (Tight 35 %, Very Good Sand 0.2 %) does
    not match the test layers (Tight 7.6 %, Very Good Sand 49.4 %), so the
    figure was not describing the evaluated data at all.

Two unaligned screenshots of similar-looking maps have similar marginal
statistics, which is why the curves appeared to agree almost perfectly. That
agreement was an artifact of the procedure, not evidence of spatial fidelity.

What this script draws instead, from cell-aligned model output on the nine
held-out test layers:

  FIGURE 1  directional semivariograms of the continuous properties, reference
            against prediction, with 95 % bootstrap envelopes over layers,
            lags in GRID CELLS, one column per property and one row per
            direction pair.
  FIGURE 2  indicator semivariograms per facies class, which is the correct
            tool for a categorical variable.

    python 16_figures_variogram.py
    python 16_figures_variogram.py --format pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures_revised")

# Colourblind-safe (Wong). Reference is the darker blue, prediction the orange;
# the pair is distinguishable in greyscale by the dashed prediction line too,
# so the figure survives a black-and-white print of the journal.
C_REF, C_PRED = "#0072B2", "#D55E00"

# Units differ per column, so every column carries its own y-label. The first
# draft labelled only column 0, which silently applied porosity units to a
# permeability panel whose gamma is in log10(mD) squared -- a different
# physical quantity entirely.
# The unit goes in the COLUMN HEADING and the y-axis carries only the word
# "Semivariance". Repeating a full unit string on twelve axes is what made the
# left-hand side unreadable: the labels were long enough to collide with the
# row annotations and with each other.
CONT_TASKS = [("facies_to_porosity", "Porosity",
               r"$\gamma$ in (fraction)$^2$"),
              ("facies_to_permeability", "Permeability",
               r"$\gamma$ in (log$_{10}$ mD)$^2$"),
              ("facies_to_vcl", "VCL",
               r"$\gamma$ in (fraction)$^2$")]
DIR_PAIRS = [("0 (along grid j)", "90 (along grid i)"),
             ("45 (diagonal)", "135 (anti-diagonal)")]


def read_csv(name: str) -> List[dict]:
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def panel(ax, rows: List[dict], task: str, direction: str, ylabel: str,
          show_legend: bool) -> None:
    sel = [r for r in rows if r["Task"] == task and r["Direction"] == direction]
    if not sel:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_xticks([]); ax.set_yticks([])
        return
    sel.sort(key=lambda r: int(r["Lag (cells)"]))
    h = np.array([int(r["Lag (cells)"]) for r in sel], float)
    rm = np.array([fnum(r["Reference gamma"]) for r in sel])
    pm = np.array([fnum(r["Predicted gamma"]) for r in sel])
    rl = np.array([fnum(r["Ref 95% low"]) for r in sel])
    rh = np.array([fnum(r["Ref 95% high"]) for r in sel])
    pl = np.array([fnum(r["Pred 95% low"]) for r in sel])
    ph = np.array([fnum(r["Pred 95% high"]) for r in sel])

    ax.fill_between(h, rl, rh, color=C_REF, alpha=0.18, linewidth=0)
    ax.fill_between(h, pl, ph, color=C_PRED, alpha=0.18, linewidth=0)
    ax.plot(h, rm, color=C_REF, lw=1.8, label="Reference")
    ax.plot(h, pm, color=C_PRED, lw=1.8, ls="--", label="Predicted")

    ax.set_xlim(h.min(), h.max())
    ax.set_ylim(bottom=0)
    ax.grid(True, lw=0.4, alpha=0.35)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.yaxis.get_offset_text().set_fontsize(7)
    if show_legend:
        ax.legend(fontsize=9, frameon=False, loc="lower right")


def figure_continuous(fmt: str) -> None:
    rows = read_csv("9_Directional_variograms.csv")
    if not rows:
        print("  ! 9_Directional_variograms.csv not found -- run 09 first")
        return
    directions = [d for pair in DIR_PAIRS for d in pair]
    n_r = len(directions)
    fig, axes = plt.subplots(n_r, len(CONT_TASKS),
                             figsize=(3.3 * len(CONT_TASKS), 2.15 * n_r),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, direction in enumerate(directions):
        for c, (task, title, unit) in enumerate(CONT_TASKS):
            ax = axes[r, c]
            panel(ax, rows, task, direction,
                  "Semivariance" if c == 0 else "",
                  show_legend=(r == 0 and c == 0))
            if r == 0:
                ax.set_title(f"{title}\n{unit}", fontsize=10, pad=8)
            if c == len(CONT_TASKS) - 1:
                ax.text(1.03, 0.5, direction, transform=ax.transAxes,
                        rotation=270, va="center", ha="left", fontsize=9,
                        color="0.25")
            if r == n_r - 1:
                ax.set_xlabel("Lag (grid cells)", fontsize=9)
    # No figure title: it belongs in the caption, where the journal expects it.
    for ext in dict.fromkeys([fmt, "pdf"]):
        p = os.path.join(FIGS, f"Fig_variograms_continuous.{ext}")
        fig.savefig(p, dpi=600, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def figure_indicator(fmt: str) -> None:
    rows = read_csv("9_Indicator_variograms.csv")
    if not rows:
        print("  ! 9_Indicator_variograms.csv not found -- run 09 first")
        return
    # h50 per class and direction; one grouped bar chart is far more legible
    # than 35 tiny variogram panels, and h50 is the quantity being compared.
    tasks = sorted({r["Task"] for r in rows})
    d0 = "0 (along grid j)"
    for task in tasks:
        sel = [r for r in rows if r["Task"] == task and r["Direction"] == d0]
        if not sel:
            continue
        classes, ref, pred, flag = [], [], [], []
        for name in dict.fromkeys(r["Class"] for r in sel):
            rr = [r for r in sel if r["Class"] == name and r["Field"] == "reference"]
            pr = [r for r in sel if r["Class"] == name and r["Field"] == "prediction"]
            if not rr:
                continue
            classes.append(name)
            ref.append(fnum(rr[0]["Indicator h50 (cells)"]))
            pred.append(fnum(pr[0]["Indicator h50 (cells)"]) if pr else np.nan)
            flag.append((rr[0]["Resolved"] != "yes") or
                        (bool(pr) and pr[0]["Resolved"] != "yes"))
        if not classes:
            continue
        x = np.arange(len(classes)); w = 0.38
        fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 3.0, 3.4))
        ax.bar(x - w / 2, ref, w, color=C_REF, label="Reference")
        ax.bar(x + w / 2, pred, w, color=C_PRED, label="Predicted")
        for i, f in enumerate(flag):
            if f:
                top = np.nanmax([ref[i], pred[i]])
                ax.text(x[i], top * 1.03, "unresolved", ha="center",
                        fontsize=7, color="0.35")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Indicator $h_{50}$ (grid cells)", fontsize=10)

        ax.grid(True, axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, frameon=False)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        for ext in dict.fromkeys([fmt, "pdf"]):
            p = os.path.join(FIGS, f"Fig_indicator_h50_{task}.{ext}")
            fig.savefig(p, dpi=600, bbox_inches="tight")
            print(f"  wrote {p}")
        plt.close(fig)


def variance_deficit() -> None:
    """
    Ratio of predicted to reference sill, per task and direction.

    The half-sill range h50 anchors each curve to ITS OWN variance, so a field
    with the right correlation length but a fraction of the true variability
    scores well on h50. The figure makes the deficit obvious; this puts a
    number on it. Sill is taken as gamma at the widest lag, which is where both
    curves are closest to their plateau.
    """
    rows = read_csv("9_Directional_variograms.csv")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["Task"], r["Direction"])].append(r)
    out = [["Task", "Direction", "Reference sill", "Predicted sill",
            "Predicted / reference", "Variance retained (%)"]]
    print("\n" + "=" * 74)
    print("VARIANCE DEFICIT  (gamma at the widest lag)")
    print("=" * 74)
    print(f"{'task':<24}{'direction':<22}{'ref':>10}{'pred':>10}{'retained':>10}")
    for (task, direction), rs in sorted(by.items()):
        rs.sort(key=lambda r: int(r["Lag (cells)"]))
        ref, pred = fnum(rs[-1]["Reference gamma"]), fnum(rs[-1]["Predicted gamma"])
        if not (np.isfinite(ref) and ref > 0):
            continue
        ratio = pred / ref
        out.append([task, direction, f"{ref:.6g}", f"{pred:.6g}",
                    f"{ratio:.3f}", f"{100*ratio:.1f}"])
        print(f"{task:<24}{direction:<22}{ref:>10.4g}{pred:>10.4g}"
              f"{100*ratio:>9.1f}%")
    with open(os.path.join(OUT, "15_Variance_deficit.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(out)
    print(f"\nwritten to {os.path.join(OUT, '15_Variance_deficit.csv')}")
    print("A value well below 100 % means the predicted field is markedly less")
    print("variable than the reference -- under-dispersion, not a range error.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default="png", choices=["png", "pdf"])
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8,
                         "savefig.facecolor": "white"})
    print("building figures from results/9_*.csv")
    figure_continuous(args.format)
    figure_indicator(args.format)
    variance_deficit()
    print(f"\nfigures in {FIGS}")
    print("These replace Figures/Viogram/Variogram_Validation_Q1, which was")
    print("computed from unaligned screenshots and cannot be corrected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
