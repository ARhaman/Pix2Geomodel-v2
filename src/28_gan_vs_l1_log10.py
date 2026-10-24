#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
28_gan_vs_l1_log10.py
=====================

Extends results/11_GAN_vs_L1.csv with the facies -> permeability comparison in
LOG10 space, which that table does not contain.

Why it is missing: 12_compare_arms.py reads per-layer metrics out of
raw_results.json / L1_raw_results.json, and those files store permeability in
linear units only. The log10 R2 of the no-adversarial arm therefore has to be
computed from the prediction images before the two arms can be compared in
that space.

Why it must be the paired permutation test and not a seed standard deviation:
the nine held-out layers are the unit of independence (the argument already
made for the clustered bootstrap), and 11_GAN_vs_L1.csv compares the arms with
an exact sign-permutation test over those layers. Comparing the new log10 row
on a different basis would reintroduce the mixed-convention problem this
revision exists to remove. This script therefore reuses
12_compare_arms.exact_sign_permutation unchanged.

Self-validation
---------------
Before reporting the new log10 row, the script recomputes the three forward
R2 rows that 11_GAN_vs_L1.csv already contains -- facies_to_porosity,
facies_to_permeability (linear) and facies_to_vcl -- straight from the
prediction images, and checks both the mean difference and the exact p-value
against that file. If the published rows do not reproduce, nothing is written.

CPU only; needs numpy and Pillow. Per-layer scores are cached so a re-run is
immediate.

Usage
-----
    python 28_gan_vs_l1_log10.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
GAN_L1_CSV = os.path.join(OUT_DIR, "11_GAN_vs_L1.csv")
OUT_CSV = os.path.join(OUT_DIR, "28_GAN_vs_L1_log10.csv")
CACHE = os.environ.get(
    "P2G_CACHE", os.path.join(os.path.expanduser("~"), ".p2g_cache"))

FORWARD = [("facies_to_porosity", "porosity", "linear"),
           ("facies_to_permeability", "permeability", "linear"),
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


M24 = _load("m24", "24_l1_forward_metrics.py")   # helpers + palette assertion
CMP = _load("cmp12", "12_compare_arms.py")       # exact_sign_permutation
EV = M24.EV


def per_layer_r2(pred_dir: str, ref_dir: str, domain: str,
                 logsp: bool) -> Dict[int, float]:
    """
    {layer: R2} for one seed. R2 is computed per image and averaged within the
    layer, which is the collapse every layerwise number in this revision uses.
    """
    pairs = EV.collect_pairs(pred_dir, ref_dir)
    if not pairs:
        return {}
    acc: Dict[int, List[float]] = {}
    for spec in pairs:
        pred_rgb = M24._read_rgb(spec.pred_path)
        ref_img = M24._read_rgb(spec.ref_path)
        w = ref_img.shape[1] // 2
        ref_rgb = ref_img[:, :w]
        if pred_rgb.shape != ref_rgb.shape:
            raise ValueError(f"shape mismatch {os.path.basename(spec.pred_path)}")
        pv = M24._decode_property(pred_rgb, domain)
        rv = M24._decode_property(ref_rgb, domain)
        valid = ~np.all(ref_rgb == 0, axis=-1)
        if logsp:
            dm = EV.CONTINUOUS_DOMAINS[domain]
            pv = np.log10(np.clip(pv, dm["vmin"], dm["vmax"]))
            rv = np.log10(np.clip(rv, dm["vmin"], dm["vmax"]))
        p, r = pv[valid], rv[valid]
        if p.size:
            acc.setdefault(spec.layer, []).append(M24._r2(p, r))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def seed_mean_per_layer(task: str, domain: str, space: str, arm: str,
                        seeds: List[int], splits_root: str,
                        runs_root: str) -> Dict[int, float]:
    """Per-layer R2 averaged over seeds, mirroring 12_compare_arms."""
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, f"perlayer_{task}_{space}_{arm}.json")
    if os.path.exists(cpath):
        try:
            with open(cpath, encoding="utf-8") as fh:
                return {int(k): float(v) for k, v in json.load(fh).items()}
        except Exception:                                   # noqa: BLE001
            pass

    suffix = "_l1" if arm == "l1" else ""
    ref_dir = os.path.join(splits_root, task, "test")
    logsp = (space == "log10")
    M24.adopt_legend(domain)

    acc: Dict[int, List[float]] = {}
    for s in seeds:
        pdir = os.path.join(runs_root, f"{task}_seed{s}{suffix}", "predictions")
        if not os.path.isdir(pdir):
            print(f"    seed{s}: missing {pdir}")
            continue
        for layer, v in per_layer_r2(pdir, ref_dir, domain, logsp).items():
            acc.setdefault(layer, []).append(v)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    if out:
        with open(cpath, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
    return out


def published_rows() -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
    """{(task, metric): (GAN, L1, diff, p)} from 11_GAN_vs_L1.csv."""
    out = {}
    if not os.path.exists(GAN_L1_CSV):
        return out
    with open(GAN_L1_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                out[(r["Task"], r["Metric"])] = (
                    float(r["GAN"]), float(r["L1"]),
                    float(r["Difference (GAN - L1)"]),
                    float(r["Exact paired permutation p"]))
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
    ap.add_argument("--tol-p", type=float, default=0.002)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    published = published_rows()
    rows: List[dict] = []

    for task, domain, space in FORWARD:
        label = f"{task} [{space}]"
        print(f"\n=== {label} ===")
        gan = seed_mean_per_layer(task, domain, space, "full", seeds,
                                  args.splits_root, args.runs_root)
        l1 = seed_mean_per_layer(task, domain, space, "l1", seeds,
                                 args.splits_root, args.runs_root)
        common = sorted(set(gan) & set(l1))
        if len(common) < 2:
            print("  too few paired layers")
            continue
        d = np.array([gan[k] - l1[k] for k in common], dtype=float)
        p = CMP.exact_sign_permutation(d)
        g_mean = float(np.mean([gan[k] for k in common]))
        l_mean = float(np.mean([l1[k] for k in common]))
        diff = float(d.mean())
        better = "GAN" if diff > 0 else ("L1" if diff < 0 else "tie")
        print(f"  GAN {g_mean:+.4f}   L1 {l_mean:+.4f}   "
              f"diff {diff:+.4f}   p={p:.4f}   ({len(common)} layers)")
        rows.append({"Task": task, "Space": space, "Metric": "R2",
                     "GAN": f"{g_mean:.4f}", "L1": f"{l_mean:.4f}",
                     "Difference (GAN - L1)": f"{diff:+.4f}",
                     "Better arm": better, "Paired layers": len(common),
                     "Exact paired permutation p": f"{p:.4f}"})

    # ---------------- validation --------------------------------------
    print("\n" + "=" * 74)
    print("VALIDATION  reproduce the three forward R2 rows of 11_GAN_vs_L1.csv")
    print("=" * 74)
    checks = 0
    worst_v = worst_p = 0.0
    for r in rows:
        if r["Space"] != "linear":
            continue
        key = (r["Task"], "R2")
        if key not in published:
            print(f"  {r['Task']:<26} not present in 11_GAN_vs_L1.csv")
            continue
        pg, pl, pd, pp = published[key]
        dv = max(abs(pg - float(r["GAN"])), abs(pl - float(r["L1"])),
                 abs(pd - float(r["Difference (GAN - L1)"])))
        dp = abs(pp - float(r["Exact paired permutation p"]))
        worst_v, worst_p = max(worst_v, dv), max(worst_p, dp)
        checks += 1
        ok = dv <= args.tol and dp <= args.tol_p
        print(f"  {r['Task']:<26} published GAN {pg:+.4f} L1 {pl:+.4f} "
              f"p={pp:.4f} | recomputed GAN {float(r['GAN']):+.4f} "
              f"L1 {float(r['L1']):+.4f} p={float(r['Exact paired permutation p']):.4f}"
              f"  {'OK' if ok else 'MISMATCH'}")

    ok_all = checks >= 3 and worst_v <= args.tol and worst_p <= args.tol_p
    if args.validate and not ok_all:
        print(f"\nVALIDATION FAILED (worst value d={worst_v:.4f}, "
              f"worst p d={worst_p:.4f}, {checks} rows checked). "
              f"The log10 row was not written.")
        return 1
    print(f"\nValidation passed ({checks} rows; worst value d={worst_v:.4f}, "
          f"worst p d={worst_p:.4f}).")

    new = [r for r in rows if r["Space"] == "log10"]
    if new:
        print("\nNEW ROW for the facies -> permeability comparison in log10 space:")
        for r in new:
            print(f"  GAN {r['GAN']}   L1 {r['L1']}   "
                  f"diff {r['Difference (GAN - L1)']}   "
                  f"better {r['Better arm']}   "
                  f"p={r['Exact paired permutation p']}")

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
