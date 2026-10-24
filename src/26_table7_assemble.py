#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_table7_assemble.py
=====================

Builds the corrected Table 7 from the three validated result files, so the
table in the manuscript is generated rather than transcribed.

Inputs
------
  results/24_FULL_forward_metrics.csv   full (GAN) arm, per seed, both conventions
  results/24_L1_forward_metrics.csv     no-adversarial arm, per seed, both conventions
  results/25_Baselines_both_conventions.csv   class mean/median + random forest

Each of those was produced by a script that first reproduced the corresponding
published numbers exactly (24_* reproduced results/4_Physical_units.csv and
L1_4_Physical_units.csv; 25_* reproduced results/13_Conventional_baselines.csv),
so every number below traces to a validated code path.

Why the table had to be rebuilt
-------------------------------
As submitted, Table 7 mixed three aggregations:
  * the network's linear columns were LAYERWISE three-seed score means;
  * every baseline was POOLED;
  * the network's log10 permeability value (0.1652) was a three-seed
    PREDICTION ENSEMBLE, pooled (13_permeability_error.py line 139,
    `pv = np.mean(pv_seeds, axis=0)`), i.e. a different estimator again.
This script reports one convention -- LAYERWISE -- for every cell, which is
the convention 02_evaluation.evaluate_task and Table 4 already use, and which
treats the layer as the unit of independence (the same argument made for the
clustered bootstrap).

Usage
-----
    python 26_table7_assemble.py
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")

FULL_CSV = os.path.join(OUT_DIR, "24_FULL_forward_metrics.csv")
L1_CSV = os.path.join(OUT_DIR, "24_L1_forward_metrics.csv")
BASE_CSV = os.path.join(OUT_DIR, "25_Baselines_both_conventions.csv")
OUT_CSV = os.path.join(OUT_DIR, "26_Table7_layerwise.csv")

ROW_ORDER = [
    ("facies_to_porosity", "linear", "Facies -> Porosity"),
    ("facies_to_vcl", "linear", "Facies -> VCL"),
    ("facies_to_permeability", "linear", "Facies -> Permeability (linear)"),
    ("facies_to_permeability", "log10", "Facies -> Permeability (log10)"),
]

CONV = "layerwise"   # the single convention this table reports


def network_stats(path: str) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """{(task, space): (mean, sd, n_seeds)} of R2 in the chosen convention."""
    if not os.path.exists(path):
        sys.exit(f"missing input: {path}")
    acc: Dict[Tuple[str, str], List[float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            acc.setdefault((r["task"], r["space"]), []).append(
                float(r[f"R2_{CONV}"]))
    out = {}
    for k, v in acc.items():
        a = np.array(v, dtype=float)
        out[k] = (float(a.mean()),
                  float(a.std(ddof=1)) if a.size > 1 else 0.0,
                  int(a.size))
    return out


def baseline_stats(path: str) -> Dict[Tuple[str, str, str], float]:
    """{(task, space, short_name): R2} in the chosen convention."""
    if not os.path.exists(path):
        sys.exit(f"missing input: {path}")
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            short = r["baseline"].split(" [")[0]
            out[(r["task"], r["space"], short)] = float(r[f"R2_{CONV}"])
    return out


def main() -> int:
    full = network_stats(FULL_CSV)
    l1 = network_stats(L1_CSV)
    base = baseline_stats(BASE_CSV)

    header = ["Task", "Pix2Geomodel", "No-adversarial (L1 only)",
              "Class-mean lookup", "Class-median lookup", "Random forest",
              "Best"]
    rows: List[List[str]] = []
    raw: List[dict] = []

    for task, space, label in ROW_ORDER:
        if (task, space) not in full:
            print(f"  [skip] {label}: absent from {os.path.basename(FULL_CSV)}")
            continue
        fm, fsd, fn = full[(task, space)]
        lm, lsd, ln = l1.get((task, space), (float("nan"),) * 3)
        cmean = base.get((task, space, "class mean"), float("nan"))
        cmed = base.get((task, space, "class median"), float("nan"))
        rf = base.get((task, space, "random forest"), float("nan"))

        cands = {"Pix2Geomodel": fm, "No-adversarial": lm,
                 "Class-mean": cmean, "Class-median": cmed,
                 "Random forest": rf}
        finite = {k: v for k, v in cands.items() if np.isfinite(v)}
        best = max(finite, key=finite.get)
        bestv = finite[best]
        # a rival is "tied" when it sits inside the winner's seed SD
        tied = [k for k, v in finite.items()
                if k != best and abs(v - bestv) <= max(fsd, lsd)]
        best_s = best + (" (tie: " + ", ".join(tied) + ")" if tied else "")

        rows.append([label,
                     f"{fm:+.4f} ± {fsd:.4f}",
                     f"{lm:+.4f} ± {lsd:.4f}",
                     f"{cmean:+.4f}", f"{cmed:+.4f}", f"{rf:+.4f}",
                     best_s])
        raw.append({"task": task, "space": space, "label": label,
                    "pix2geomodel_mean": fm, "pix2geomodel_sd": fsd,
                    "pix2geomodel_n_seeds": fn,
                    "l1_only_mean": lm, "l1_only_sd": lsd,
                    "l1_only_n_seeds": ln,
                    "class_mean": cmean, "class_median": cmed,
                    "random_forest": rf, "best": best_s,
                    "convention": CONV})

    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(header)]
    def line(cells): return "| " + " | ".join(
        c.ljust(w) for c, w in zip(cells, widths)) + " |"

    print(f"\nCorrected Table 7 -- every cell is the {CONV.upper()} R2")
    print("(network columns: mean ± SD over 3 seeds; baselines are "
          "deterministic given random_state=2025)\n")
    print(line(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(line(r))

    if raw:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(raw[0].keys()))
            w.writeheader()
            w.writerows(raw)
        print(f"\nwrote {OUT_CSV}  ({len(raw)} rows)")

    print("\nContribution of the adversarial term "
          "(Pix2Geomodel minus no-adversarial):")
    for d in raw:
        delta = d["pix2geomodel_mean"] - d["l1_only_mean"]
        noise = max(d["pix2geomodel_sd"], d["l1_only_sd"])
        verdict = ("within seed noise" if abs(delta) <= noise
                   else ("helps" if delta > 0 else "hurts"))
        print(f"  {d['label']:<34} {delta:+.4f}   "
              f"(seed SD {noise:.4f}) -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
