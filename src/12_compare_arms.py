#!/usr/bin/env python3
"""
12_compare_arms.py
==================

Adversarial (pix2pix) vs non-adversarial (L1-only U-Net): does the
discriminator earn its place?

A CONTROLLED ablation: same script, same U-Net generator, same 15,000 steps,
same blocked split, same augmentation, same optimiser, same seeds {1,2,3}, same
validation-based checkpoint selection. The only difference is --gan-weight,
which at zero removes the adversarial term from the graph entirely.

TEST, NOT VALIDATION. The validation score is what selected each checkpoint, so
quoting it reports the maximum of a noisy quantity over 15 evaluations. That
bias is real and it is large here: on validation the adversarial arm looked
roughly twice as good on every facies task, and almost all of that advantage
disappears on the held-out layers.

PAIRING AT THE LAYER LEVEL. Seeds are averaged per test layer, then the nine
layers are paired between arms. Pairing on the three seeds instead would leave
an exact sign-permutation test with 2^3 = 8 arrangements and a smallest
attainable p of 0.25 -- unable to detect anything, whatever the data. Nine
layers give 2^9 = 512 arrangements and a floor of 0.0039.

METRIC LOOKUP IS EXPLICIT. An earlier version searched the result tree
recursively for a metric name and silently matched `encoding_floor.MAE`, the
error the RGB palette imposes before any model is involved. That value is
identical in both arms by construction, so all three continuous tasks reported
a perfect tie. Metrics are now read from exactly one path and a missing key is
an error, not a fallback.

    python 12_compare_arms.py
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

FACIES_TASKS = ["porosity_to_facies", "permeability_to_facies", "vcl_to_facies"]
CONT_TASKS = ["facies_to_porosity", "facies_to_permeability", "facies_to_vcl"]

FACIES_METRICS = [("pixel_accuracy", True), ("macro_F1", True), ("mIoU", True)]
CONT_METRICS = [("MAE", False), ("RMSE", False), ("R2", True)]


def load_arm(prefix: str) -> Optional[dict]:
    p = os.path.join(OUT, prefix + "raw_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def point(rec: dict, key: str) -> Optional[float]:
    """rec['metrics'][key]['point_estimate'] -- one path, no searching."""
    m = rec.get("metrics", {}).get(key)
    if not isinstance(m, dict):
        return None
    v = m.get("point_estimate")
    return float(v) if isinstance(v, (int, float)) else None


def per_layer(rec: dict, key: str) -> Dict[str, float]:
    """rec['metrics'][key]['per_layer'] -> {layer_id: value}."""
    m = rec.get("metrics", {}).get(key)
    if not isinstance(m, dict):
        return {}
    pl = m.get("per_layer")
    return {str(k): float(v) for k, v in pl.items()} if isinstance(pl, dict) else {}


def seed_mean_per_layer(task_rec: dict, key: str, seeds: List[str]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for s in seeds:
        for layer, v in per_layer(task_rec[s], key).items():
            acc.setdefault(layer, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def exact_sign_permutation(diffs: np.ndarray) -> float:
    """Exact two-sided paired sign-permutation test on the mean difference."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    if n > 20:                      # 2^20 already over a million; sample instead
        rng = np.random.default_rng(2025)
        obs = abs(float(d.mean()))
        draws = rng.choice([-1.0, 1.0], size=(20000, n))
        return float(np.mean(np.abs((draws * d).mean(axis=1)) >= obs - 1e-15))
    obs = abs(float(d.mean()))
    hits = sum(1 for signs in itertools.product((1.0, -1.0), repeat=n)
               if abs(float(np.mean(d * np.asarray(signs)))) >= obs - 1e-15)
    return hits / (2 ** n)


def main() -> int:
    gan, l1 = load_arm(""), load_arm("L1_")
    if gan is None or l1 is None:
        print("missing results/raw_results.json or results/L1_raw_results.json")
        print("  python 07_report.py --arm gan")
        print("  python 07_report.py --arm l1")
        return 2

    rows = [["Task", "Metric", "GAN", "L1", "Difference (GAN - L1)", "Better arm",
             "Paired layers", "Exact paired permutation p"]]
    print(f"{'task':<24}{'metric':<16}{'GAN':>10}{'L1':>10}{'diff':>10}"
          f"{'better':>8}{'layers':>8}{'p':>8}")

    missing = []
    for task in FACIES_TASKS + CONT_TASKS:
        metrics = FACIES_METRICS if task in FACIES_TASKS else CONT_METRICS
        g_t, l_t = gan.get(task, {}), l1.get(task, {})
        seeds = sorted(set(g_t) & set(l_t), key=int)
        if not seeds:
            continue
        for key, higher_better in metrics:
            gm = [point(g_t[s], key) for s in seeds]
            lm = [point(l_t[s], key) for s in seeds]
            if any(v is None for v in gm + lm):
                missing.append(f"{task}/{key}")
                continue
            g_layers = seed_mean_per_layer(g_t, key, seeds)
            l_layers = seed_mean_per_layer(l_t, key, seeds)
            shared = sorted(set(g_layers) & set(l_layers), key=int)
            if shared:
                d = np.array([g_layers[k] - l_layers[k] for k in shared])
                p = exact_sign_permutation(d)
            else:
                d, p = np.array([]), float("nan")
            g, l = float(np.mean(gm)), float(np.mean(lm))
            dm = g - l
            better = "tie" if dm == 0 else ("GAN" if (dm > 0) == higher_better else "L1")
            rows.append([task, key, f"{g:.4f}", f"{l:.4f}", f"{dm:+.4f}", better,
                         len(shared), f"{p:.4f}"])
            print(f"{task:<24}{key:<16}{g:>10.4f}{l:>10.4f}{dm:>+10.4f}"
                  f"{better:>8}{len(shared):>8}{p:>8.4f}")

    if missing:
        print("\n! metric not found in results (NOT silently substituted): "
              + ", ".join(missing))

    with open(os.path.join(OUT, "11_GAN_vs_L1.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    print("\n" + "=" * 78)
    print("Pairs are the 9 held-out test layers, seeds averaged within each layer.")
    print("Exact two-sided sign-permutation over 2^9 = 512 arrangements, so the")
    print("smallest attainable p is 0.0039.")
    print(f"written to {os.path.join(OUT, '11_GAN_vs_L1.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
