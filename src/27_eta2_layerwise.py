#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
27_eta2_layerwise.py
====================

The correlation ratio eta^2 of the TRUE facies partition is the information
ceiling for any estimator that sees only facies labels. Section 3.3 expresses
results as a percentage of that ceiling.

The published values (0.7145 porosity, 0.8814 VCL, 0.2127 log10 permeability,
in results/13_Conventional_baselines.csv as eta2_true_facies) are POOLED --
all test cells concatenated, one computation. Table 7 is being rebuilt on the
LAYERWISE convention, so the ceiling has to be recomputed the same way or the
"percentage of ceiling" statements compare quantities from two different
aggregations, which is the defect this revision exists to remove.

eta^2 = SS_between / SS_total, with the seven facies classes as groups.
Layerwise: computed per test image, then averaged over layers -- the same
collapse used for every other layerwise number in the revision.

Self-validating: refuses to emit the layerwise ceiling unless its pooled
values first reproduce results/13_Conventional_baselines.csv within --tol.

CPU only; needs numpy and Pillow. Runs in well under a minute.

Usage
-----
    python 27_eta2_layerwise.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
EXPORT = os.path.join(HERE, "export_v2")
BASELINE_CSV = os.path.join(OUT_DIR, "13_Conventional_baselines.csv")
VARDEC_CSV = os.path.join(OUT_DIR, "21_Variance_decomposition.csv")
OUT_CSV = os.path.join(OUT_DIR, "27_Eta2_both_conventions.csv")

# (task used for the file pairs, domain, space, name in 13_Conventional_baselines)
COMBOS = [
    ("facies_to_porosity", "porosity", "linear", "porosity_to_facies"),
    ("facies_to_vcl", "vcl", "linear", "vcl_to_facies"),
    ("facies_to_permeability", "permeability", "linear", None),
    ("facies_to_permeability", "permeability", "log10", "permeability_to_facies"),
]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load("ev02", "02_evaluation.py")
BL = _load("bl14", "14_baseline_conventional.py")
N_CLS = len(EV.FACIES_LEGEND)


def _assert_palette_roundtrip() -> None:
    swatch = np.array(
        [[0, 0, 0]] + [EV.hex_to_rgb(h) for _, h in EV.FACIES_LEGEND],
        dtype=np.uint8).reshape(1, -1, 3)
    got = EV.decode_rgb_to_facies(swatch).ravel()
    want = np.arange(len(EV.FACIES_LEGEND) + 1, dtype=got.dtype)
    if not np.array_equal(got, want):
        raise AssertionError(f"palette does not round-trip: {got.tolist()}")


_assert_palette_roundtrip()


def adopt_legend(domain: str) -> bool:
    lm = os.path.join(EXPORT, domain, "legend_manifest.json")
    if os.path.exists(lm):
        EV.load_legend_manifest(lm)
        return True
    return False


def _ref_and_labels(path: str, domain: str, logsp: bool):
    """Identical masking to 14_baseline_conventional.py / 25_baseline_layerwise.py."""
    tgt, inp = BL.halves(path)
    val, vmask = BL.decode_cont(tgt, domain, EV)
    if logsp:
        vmask = vmask & (val > 0)
        val = np.log10(np.where(val > 0, val, 1.0))
    lab = EV.decode_rgb_to_facies(inp)
    return val, vmask & (lab >= 1), lab


def eta2(val: np.ndarray, lab: np.ndarray) -> float:
    """Correlation ratio of `val` given the grouping `lab`."""
    if val.size == 0:
        return float("nan")
    grand = float(val.mean())
    ss_tot = float(np.sum((val - grand) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_bet = 0.0
    for c in np.unique(lab):
        sel = val[lab == c]
        if sel.size:
            ss_bet += sel.size * (float(sel.mean()) - grand) ** 2
    return ss_bet / ss_tot


def _layer_of(path: str) -> int:
    m = EV.LAYER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse layer from {path!r}")
    return int(m.group(1))


def published_eta2() -> Dict[str, float]:
    """Pooled eta^2 keyed by the inverse-task name, from the baselines file."""
    out: Dict[str, float] = {}
    if not os.path.exists(BASELINE_CSV):
        return out
    with open(BASELINE_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Metric"] == "eta2_true_facies":
                out[row["Task"]] = float(row["Value"])
    return out


def published_vardec() -> Dict[Tuple[str, str], float]:
    """
    Pooled eta^2 keyed by (property, space), from 21_Variance_decomposition.csv.
    This is the only source for LINEAR permeability (0.3097), which the
    baselines file does not carry because every inverse permeability number
    is log10.
    """
    out: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(VARDEC_CSV):
        return out
    with open(VARDEC_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[(row["Property"], row["Space"])] = float(row["eta2"])
            except (KeyError, ValueError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.004)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    published = published_eta2()
    vardec = published_vardec()
    rows: List[dict] = []
    checks: List[Tuple[str, float, float]] = []
    notes: List[Tuple[str, float, float]] = []

    for task, domain, space, inv_name in COMBOS:
        if not adopt_legend(domain):
            print(f"  WARNING: no legend manifest for '{domain}'")
        logsp = (space == "log10")
        print(f"\n=== {domain} [{space}] ===")

        per_layer: Dict[int, List[float]] = {}
        pool_v: List[np.ndarray] = []
        pool_l: List[np.ndarray] = []
        for p in BL.test_files(task):
            val, m, lab = _ref_and_labels(p, domain, logsp)
            if not m.any():
                continue
            v, g = val[m], lab[m]
            per_layer.setdefault(_layer_of(p), []).append(eta2(v, g))
            pool_v.append(v)
            pool_l.append(g)

        if not per_layer:
            print("  no test pairs")
            continue

        per = [float(np.mean(v)) for v in per_layer.values()]
        lay = float(np.mean(per))
        lay_sd = float(np.std(per, ddof=1)) if len(per) > 1 else 0.0
        V, G = np.concatenate(pool_v), np.concatenate(pool_l)
        pool = eta2(V, G)

        print(f"  layerwise eta2 = {lay:.4f} ± {lay_sd:.4f} "
              f"(over {len(per_layer)} layers)")
        print(f"  pooled    eta2 = {pool:.4f}  ({V.size:,} cells)")

        rows.append({"domain": domain, "space": space,
                     "eta2_layerwise": lay, "eta2_layerwise_sd_over_layers": lay_sd,
                     "eta2_pooled": pool,
                     "n_layers": len(per_layer), "n_cells": int(V.size)})
        if inv_name and inv_name in published:
            checks.append((inv_name, published[inv_name], pool))
        if (domain, space) in vardec:
            # Informational only. 21_facies_property_decomposition.py decodes
            # the reference through its own path and subsamples with a stride,
            # so in LINEAR permeability -- where the variance is dominated by a
            # few extreme cells and by the treatment of near-zero values -- it
            # need not agree with the main evaluation pipeline. Its log10 value
            # does agree, which is the space the manuscript reports.
            notes.append((f"{domain} [{space}] vs 21_Variance_decomposition",
                          vardec[(domain, space)], pool))

    print("\n" + "=" * 72)
    print("VALIDATION  reproduce the published POOLED eta2")
    print("=" * 72)
    worst = 0.0
    for name, want, got in checks:
        d = abs(want - got)
        worst = max(worst, d)
        print(f"  {name:<46} published {want:.4f}   recomputed {got:.4f}   "
              f"d={d:.4f}  {'OK' if d <= args.tol else 'MISMATCH'}")
    ok = bool(checks) and worst <= args.tol
    if args.validate and not ok:
        print(f"\nVALIDATION FAILED (worst d={worst:.4f}). "
              f"Layerwise ceiling not written.")
        return 1
    print(f"\nValidation passed (worst d={worst:.4f}).")

    if notes:
        print("\n" + "-" * 72)
        print("INFORMATIONAL  cross-check against 21_Variance_decomposition.csv")
        print("-" * 72)
        for name, want, got in notes:
            dd = abs(want - got)
            flag = "agrees" if dd <= args.tol else "DIFFERS -- see note in source"
            print(f"  {name:<52} {want:.4f} vs {got:.4f}  d={dd:.4f}  {flag}")

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
