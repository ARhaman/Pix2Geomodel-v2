#!/usr/bin/env python3
"""
18_score_ablation.py
====================

Score the ablation arms on the held-out test layers (Reviewer 5 #16, Reviewer 1 #5).

Each arm changes exactly one factor against the trained baseline, which is
runs/<task>_seed<n> and is NOT re-trained. Comparison is paired on the nine
test layers with seeds averaged within each layer, and tested with an exact
two-sided sign-permutation over 2^9 = 512 arrangements, so the smallest
attainable p is 0.0039.

    baseline   as published: augmentation on, 2048 x 1024, lambda 100, GAN on
    noaug      augmentation off
    halfres    1024 x 512
    lambda10   L1 weight 10
    lambda1    L1 weight 1
    l1         adversarial term removed entirely (gan_weight 0)

Reading the output. A negative difference means the arm is WORSE than the
baseline, i.e. the factor that was removed or changed was contributing. An arm
that is statistically indistinguishable from the baseline is the more
interesting result here: it means the manuscript's choice of that setting was
not load-bearing, and for resolution it directly addresses the claim that
downsampling caused "a significant loss of information".

    python 18_score_ablation.py
    python 18_score_ablation.py --task vcl_to_facies
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")

ARMS = [("", "baseline (as published)"),
        ("_noaug", "no augmentation"),
        ("_halfres", "half resolution (1024x512)"),
        ("_lambda10", "lambda = 10"),
        ("_lambda1", "lambda = 1"),
        ("_l1", "no adversarial term")]
METRICS = ["pixel_accuracy", "macro_F1", "mIoU"]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def exact_sign_permutation(d: np.ndarray) -> float:
    d = np.asarray([x for x in d if np.isfinite(x)], dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    obs = abs(float(d.mean()))
    if n > 20:
        rng = np.random.default_rng(2025)
        draws = rng.choice([-1.0, 1.0], size=(20000, n))
        return float(np.mean(np.abs((draws * d).mean(axis=1)) >= obs - 1e-15))
    hits = sum(1 for s in itertools.product((1.0, -1.0), repeat=n)
               if abs(float(np.mean(d * np.asarray(s)))) >= obs - 1e-15)
    return hits / (2 ** n)


def per_layer_metrics(task: str, seed: int, suffix: str, ev,
                      files: List[str], test_dir: str) -> Optional[Dict[str, Dict[str, float]]]:
    """{metric: {layer: value}} for one run, or None if predictions are absent."""
    pdir = os.path.join(RUNS, f"{task}_seed{seed}{suffix}", "predictions")
    if not os.path.isdir(pdir):
        return None
    n_cls = len(ev.FACIES_LEGEND)
    out = {m: {} for m in METRICS}
    got = 0
    for fn in files:
        pp = os.path.join(pdir, fn)
        if not os.path.exists(pp):
            continue
        pair = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
        ref = ev.decode_rgb_to_facies(pair[:, : pair.shape[1] // 2])
        pred_img = np.array(Image.open(pp).convert("RGB"))
        # An arm trained at reduced resolution writes smaller predictions.
        # Nearest-neighbour resize to the reference grid keeps the comparison
        # on identical cells; any other interpolation would invent classes.
        if pred_img.shape[:2] != ref.shape:
            pred_img = np.array(Image.fromarray(pred_img).resize(
                (ref.shape[1], ref.shape[0]), Image.NEAREST))
        pred = ev.decode_rgb_to_facies(pred_img)
        cm = ev.confusion_matrix(pred, ref, n_cls)
        m = ev.categorical_metrics(cm, [n for n, _ in ev.FACIES_LEGEND])
        layer = fn.split("_")[0]
        for k in METRICS:
            out[k][layer] = float(m[k])
        got += 1
    return out if got else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="vcl_to_facies")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)
    test_dir = os.path.join(SPLITS, args.task, "test")
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))

    # arm -> metric -> layer -> mean over seeds
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for suffix, label in ARMS:
        acc = {m: {} for m in METRICS}
        n_seeds = 0
        for s in args.seeds:
            r = per_layer_metrics(args.task, s, suffix, ev, files, test_dir)
            if r is None:
                continue
            n_seeds += 1
            for m in METRICS:
                for layer, v in r[m].items():
                    acc[m].setdefault(layer, []).append(v)
        if n_seeds == 0:
            print(f"  ! no predictions for arm '{label}' -- run 06_predict.py --all")
            continue
        data[suffix] = {m: {k: float(np.mean(v)) for k, v in acc[m].items()}
                        for m in METRICS}
        print(f"  {label:<32} {n_seeds} seed(s)")

    if "" not in data:
        print("baseline arm missing; nothing to compare against")
        return 2

    rows = [["Task", "Arm", "Metric", "Baseline", "Arm value", "Difference",
             "Paired layers", "Exact permutation p", "Verdict"]]
    print(f"\n{'arm':<26}{'metric':<16}{'baseline':>10}{'arm':>10}"
          f"{'diff':>10}{'p':>9}  verdict")
    print("=" * 92)

    for suffix, label in ARMS:
        if suffix == "" or suffix not in data:
            continue
        for m in METRICS:
            base, arm = data[""][m], data[suffix][m]
            shared = sorted(set(base) & set(arm))
            if not shared:
                continue
            d = np.array([arm[k] - base[k] for k in shared])
            p = exact_sign_permutation(d)
            bm = float(np.mean([base[k] for k in shared]))
            am = float(np.mean([arm[k] for k in shared]))
            dm = float(d.mean())
            if p >= 0.05:
                verdict = "no detectable difference"
            else:
                verdict = "arm is better" if dm > 0 else "arm is worse"
            rows.append([args.task, label, m, f"{bm:.4f}", f"{am:.4f}",
                         f"{dm:+.4f}", len(shared), f"{p:.4f}", verdict])
            print(f"{label:<26}{m:<16}{bm:>10.4f}{am:>10.4f}{dm:>+10.4f}"
                  f"{p:>9.4f}  {verdict}")

    with open(os.path.join(OUT, "16_Ablation.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    print("\n" + "=" * 92)
    print("Paired on the 9 held-out layers, seeds averaged within each layer.")
    print("Exact two-sided sign-permutation over 2^9 = 512 arrangements; the")
    print("smallest attainable p is 0.0039. 'No detectable difference' means")
    print("the factor was not load-bearing at this sample size, which for the")
    print("resolution arm is itself the answer to the downsampling objection.")
    print(f"written to {os.path.join(OUT, '16_Ablation.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
