#!/usr/bin/env python3
"""
15_encoding_penalty.py
======================

How much of the gap between the network and the 1-D classifier is the RGB
representation rather than the model? (Reviewer 5 #11, and the mechanism behind
the R5-15 baseline result.)

The conventional baselines beat the conditional GAN on all three inverse tasks.
Before the manuscript attributes that to the model, it has to separate two very
different explanations:

  (a) the network is genuinely worse at the mapping; or
  (b) the network is competitive, but it must emit a CONTINUOUS RGB image that
      is then snapped to the nearest of seven legend colours, and that snapping
      destroys accuracy the classifier never risks because it outputs a class
      label directly.

(b) is a property of the image-based formulation itself -- the paper's own
method -- so if it is large it is a finding about the approach, not a scoring
artifact to be corrected away.

This script measures it three ways on the predicted facies images:

  1. COLOUR AMBIGUITY. Distance from each predicted pixel to its nearest
     legend colour, and to its second nearest. Pixels far from every legend
     entry are ones where the snap is close to arbitrary.

  2. MARGIN. The gap between nearest and second-nearest distance. A small
     margin means a slightly different RGB output would have produced a
     different facies class.

  3. RECOVERABLE ACCURACY. Accuracy if every ambiguous pixel (margin below a
     threshold, or nearest distance above one) were decided correctly. This is
     an upper bound on what better decoding alone could buy, and therefore
     bounds explanation (b).

    python 15_encoding_penalty.py            (CPU, a few minutes)
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from typing import Dict, List

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")

TASKS = ["porosity_to_facies", "permeability_to_facies", "vcl_to_facies"]
SEEDS = [1, 2, 3]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)

    pal = np.array([[0, 0, 0]] + [list(ev.hex_to_rgb(h)) for _, h in ev.FACIES_LEGEND],
                   dtype=np.int32)
    rows = [["Task", "Seed", "Cells", "Median nearest dist", "Mean nearest dist",
             "% beyond 60 RGB", "Median margin", "% margin < 30",
             "Pixel accuracy", "Accuracy if ambiguous resolved", "Headroom"]]

    print(f"{'task':<24}{'seed':>5}{'med d':>8}{'%far':>8}{'med marg':>10}"
          f"{'%tight':>8}{'PA':>9}{'PA ideal':>10}{'headroom':>10}")

    for task in TASKS:
        test_dir = os.path.join(SPLITS, task, "test")
        files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
        for seed in SEEDS:
            pdir = os.path.join(RUNS, f"{task}_seed{seed}", "predictions")
            if not os.path.isdir(pdir):
                continue
            d1s, margins, correct, amb, tot = [], [], 0, 0, 0
            for fn in files:
                pp = os.path.join(pdir, fn)
                if not os.path.exists(pp):
                    continue
                pair = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
                ref = ev.decode_rgb_to_facies(pair[:, : pair.shape[1] // 2])
                img = np.array(Image.open(pp).convert("RGB")).astype(np.int32)

                flat = img.reshape(-1, 3)
                # Distinct colours only: predictions hold far fewer unique RGB
                # triples than pixels, and the distance matrix is 8 columns wide.
                uniq, inv = np.unique(flat, axis=0, return_inverse=True)
                d = np.sqrt(((uniq[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2))
                order = np.argsort(d, axis=1)
                near = d[np.arange(d.shape[0]), order[:, 0]]
                second = d[np.arange(d.shape[0]), order[:, 1]]
                cls = order[:, 0]

                pred = cls[inv].reshape(img.shape[:2])
                d1 = near[inv].reshape(img.shape[:2])
                mg = (second - near)[inv].reshape(img.shape[:2])

                m = ref >= 1
                d1s.append(d1[m][::17]); margins.append(mg[m][::17])
                correct += int(np.sum(pred[m] == ref[m]))
                amb += int(np.sum((mg[m] < 30) | (d1[m] > 60)))
                tot += int(m.sum())

            if tot == 0:
                continue
            d1a, mga = np.concatenate(d1s), np.concatenate(margins)
            pa = correct / tot
            # Upper bound: every ambiguous cell decided correctly.
            wrong_amb = min(amb, tot - correct)
            pa_ideal = (correct + wrong_amb) / tot
            far = 100.0 * float(np.mean(d1a > 60))
            tight = 100.0 * float(np.mean(mga < 30))
            rows.append([task, seed, tot, f"{np.median(d1a):.1f}",
                         f"{d1a.mean():.1f}", f"{far:.2f}",
                         f"{np.median(mga):.1f}", f"{tight:.2f}",
                         f"{pa:.4f}", f"{pa_ideal:.4f}", f"{pa_ideal - pa:+.4f}"])
            print(f"{task:<24}{seed:>5}{np.median(d1a):>8.1f}{far:>8.2f}"
                  f"{np.median(mga):>10.1f}{tight:>8.2f}{pa:>9.4f}"
                  f"{pa_ideal:>10.4f}{pa_ideal-pa:>+10.4f}")

    with open(os.path.join(OUT, "14_Encoding_penalty.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nwritten to {os.path.join(OUT, '14_Encoding_penalty.csv')}")
    print("\n'Headroom' is an UPPER BOUND on what perfect decoding could add.")
    print("If it is small, the gap to the 1-D classifier is the model, not the")
    print("colour encoding. If it is large, the image formulation itself is")
    print("costing the accuracy and that is the finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
