#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_ceiling_test.py
==================

Settles whether Pix2Geomodel actually exceeds the facies-label information
ceiling, rather than asserting it from two means that were never compared on
a common footing.

The question
------------
Section 3.3 reports the network's layerwise R2 (VCL: 0.8543) next to the
correlation ratio of the true facies partition (VCL: 0.8247). Taken at face
value the network appears to beat the ceiling. But the ceiling has its own
layer-to-layer spread (+/- 0.0388), and the apparent excess (0.0296) sits
inside it. Comparing a mean against a mean therefore cannot support the claim.

The test
--------
Both quantities are defined per layer on the same nine held-out layers, so
they are paired. This script computes, for each layer, the network's R2 and
the partition's eta^2, and applies the exact sign-permutation test already
used in 11_GAN_vs_L1.csv (12_compare_arms.exact_sign_permutation) to the
paired differences. That is the same unit of independence used everywhere
else in the revision.

eta^2 is computed on the test layers themselves, so it is an IN-SAMPLE bound,
while the network's R2 is out-of-sample. A network that beats it is therefore
beating a bound measured under more favourable conditions than its own, which
makes a positive result meaningful rather than an artefact.

Interpretation of a positive result: eta^2 bounds estimators that map a facies
LABEL to one value per class. The network sees the spatial arrangement of the
classes, so exceeding the bound indicates it is using geometry the label alone
cannot express. A null result means the network reaches, but does not provably
surpass, what the labels carry -- and the manuscript must say only that.

Self-validation
---------------
Refuses to report any test unless the per-layer series it builds reproduce
both published means: the network R2 from results/26_Table7_layerwise.csv and
the layerwise eta^2 from results/27_Eta2_both_conventions.csv.

Usage
-----
    python 29_ceiling_test.py
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
T7_CSV = os.path.join(OUT_DIR, "26_Table7_layerwise.csv")
ETA_CSV = os.path.join(OUT_DIR, "27_Eta2_both_conventions.csv")
OUT_CSV = os.path.join(OUT_DIR, "29_Ceiling_test.csv")

# (task, domain, space)
COMBOS = [("facies_to_porosity", "porosity", "linear"),
          ("facies_to_vcl", "vcl", "linear"),
          ("facies_to_permeability", "permeability", "log10")]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M27 = _load("m27", "27_eta2_layerwise.py")        # eta2, masking, layer parse
M28 = _load("m28", "28_gan_vs_l1_log10.py")       # cached per-layer network R2
CMP = _load("cmp12b", "12_compare_arms.py")       # exact_sign_permutation


def per_layer_eta2(task: str, domain: str, space: str) -> Dict[int, float]:
    """{layer: eta^2} of the true facies partition, averaged within a layer."""
    M27.adopt_legend(domain)
    logsp = (space == "log10")
    acc: Dict[int, List[float]] = {}
    for p in M27.BL.test_files(task):
        val, m, lab = M27._ref_and_labels(p, domain, logsp)
        if not m.any():
            continue
        acc.setdefault(M27._layer_of(p), []).append(M27.eta2(val[m], lab[m]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def published(path: str, key_cols: Tuple[str, ...], val_col: str
              ) -> Dict[Tuple[str, ...], float]:
    out: Dict[Tuple[str, ...], float] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[tuple(r[c] for c in key_cols)] = float(r[val_col])
            except (KeyError, ValueError):
                continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-root", default="splits_v2")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--tol", type=float, default=0.004)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pub_net = published(T7_CSV, ("task", "space"), "pix2geomodel_mean")
    pub_eta = published(ETA_CSV, ("domain", "space"), "eta2_layerwise")

    rows: List[dict] = []
    checks: List[Tuple[str, str, float, float]] = []

    for task, domain, space in COMBOS:
        print(f"\n=== {domain} [{space}] ===")
        net = M28.seed_mean_per_layer(task, domain, space, "full", seeds,
                                      args.splits_root, args.runs_root)
        eta = per_layer_eta2(task, domain, space)
        common = sorted(set(net) & set(eta))
        if len(common) < 2:
            print("  too few paired layers")
            continue

        a = np.array([net[k] for k in common], dtype=float)
        b = np.array([eta[k] for k in common], dtype=float)
        d = a - b
        p = CMP.exact_sign_permutation(d)
        n_above = int(np.sum(d > 0))

        print(f"  network R2  mean {a.mean():+.4f}")
        print(f"  eta^2       mean {b.mean():+.4f}")
        print(f"  difference  mean {d.mean():+.4f}   "
              f"(network above ceiling on {n_above}/{len(common)} layers)")
        print(f"  exact paired sign-permutation p = {p:.4f}")
        verdict = ("exceeds the ceiling" if p < 0.05 and d.mean() > 0 else
                   ("below the ceiling" if p < 0.05 and d.mean() < 0 else
                    "indistinguishable from the ceiling"))
        print(f"  -> {verdict}")

        rows.append({"domain": domain, "space": space,
                     "network_R2_layerwise": f"{a.mean():.4f}",
                     "eta2_layerwise": f"{b.mean():.4f}",
                     "difference": f"{d.mean():+.4f}",
                     "layers_above_ceiling": n_above,
                     "paired_layers": len(common),
                     "exact_paired_permutation_p": f"{p:.4f}",
                     "verdict": verdict})

        if (task, space) in pub_net:
            checks.append(("network " + domain, "26_Table7",
                           pub_net[(task, space)], float(a.mean())))
        if (domain, space) in pub_eta:
            checks.append(("eta2 " + domain, "27_Eta2",
                           pub_eta[(domain, space)], float(b.mean())))

    print("\n" + "=" * 74)
    print("VALIDATION  per-layer series must reproduce the published means")
    print("=" * 74)
    worst = 0.0
    for what, src, want, got in checks:
        dd = abs(want - got)
        worst = max(worst, dd)
        print(f"  {what:<22} {src:<10} published {want:+.4f}   "
              f"recomputed {got:+.4f}   d={dd:.4f}  "
              f"{'OK' if dd <= args.tol else 'MISMATCH'}")

    ok = bool(checks) and worst <= args.tol
    if args.validate and not ok:
        print(f"\nVALIDATION FAILED (worst d={worst:.4f}). Nothing written.")
        return 1
    print(f"\nValidation passed (worst d={worst:.4f}).")

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
