#!/usr/bin/env python3
"""
10_class_support.py
===================

Facies class occurrence per split, per layer.

Why this exists. The spatial validation flags two classes -- Good Clean Sand
and Very Good Sand -- as present in the PREDICTIONS but absent from the
reference test layers. There are two very different explanations, and they
call for opposite statements in the paper:

  (a) the class does not occur in the reservoir model at all, so the network
      is inventing a rock type; or

  (b) the class occurs in the training depth interval but not in the test
      depth interval, so the network has learned a real class and is placing
      it where the test layers happen not to contain it.

(b) is a direct consequence of blocked, contiguous-depth splitting, which was
adopted precisely to remove the leakage in the original design. If (b) holds,
the honest statement is that the leakage-free split creates a train/test class
imbalance, and the spurious predictions are a measurable cost of the stricter
protocol -- not evidence of hallucination.

The script reports, for every class and every split, the number of layers in
which the class occurs and its mean area fraction, so the two cases can be
distinguished by inspection rather than assumed.

    python 10_class_support.py            (CPU; run in base or tf210)
"""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
OUT = os.path.join(HERE, "results")

# Any facies task works: the reference (target) half of the pair is the same
# facies image in every one of them.
TASK = "porosity_to_facies"
SPLIT_NAMES = ("train", "val", "test")


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
    n_cls = len(ev.FACIES_LEGEND)
    os.makedirs(OUT, exist_ok=True)

    # frac[split][cls] = list of per-layer area fractions
    frac = {s: defaultdict(list) for s in SPLIT_NAMES}
    n_layers = {}

    for split in SPLIT_NAMES:
        d = os.path.join(SPLITS, TASK, split)
        if not os.path.isdir(d):
            print(f"  {split}: missing -- skipped")
            n_layers[split] = 0
            continue
        # Originals only. The training split holds 1116 images that are
        # augmentations of 36 original layers, and flips/rotations cannot
        # change which classes occur in a layer -- counting them would just
        # multiply each original by 31 and inflate "layers with class" into a
        # number that means nothing geologically.
        files = sorted(f for f in os.listdir(d)
                       if f.endswith(".png") and "_aug00" in f)
        n_layers[split] = len(files)
        print(f"  {split}: {len(files)} original layers")
        for fn in files:
            pair = np.array(Image.open(os.path.join(d, fn)).convert("RGB"))
            w = pair.shape[1] // 2
            # TARGET IS THE LEFT HALF, input is the right (03_train.load_pair,
            # target_on_left=True -- the original convention, preserved).
            # Reading the wrong half is silent rather than fatal: decoding a
            # porosity image through the facies LUT maps every pixel to its
            # nearest facies colour and yields a histogram that sums to 100 %
            # and looks entirely plausible. The only way to catch it is to
            # cross-check against an independent read of the same layers,
            # which is how this bug was found -- 09_spatial_validation
            # reported Tight at 12.6 % where this script reported 50.5 %.
            lab = ev.decode_rgb_to_facies(pair[:, :w])
            valid = lab >= 1
            tot = max(int(valid.sum()), 1)
            for cls in range(1, n_cls + 1):
                frac[split][cls].append(float((lab == cls).sum()) / tot)

    # ---- consistency check against 09_spatial_validation -------------------
    # Both scripts read the same test layers, so their test-split area
    # fractions must agree. A mismatch means one of them is reading the wrong
    # half of the pair, which produces plausible-looking numbers rather than
    # an error. Comparing them is the only cheap way to detect that.
    ref_csv = os.path.join(OUT, "9_Indicator_variograms.csv")
    if os.path.exists(ref_csv):
        want = {}
        with open(ref_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["Field"] == "reference":
                    want[r["Class"]] = float(r["Mean area fraction"])
        bad = []
        for cls in range(1, n_cls + 1):
            name = ev.FACIES_LEGEND[cls - 1][0]
            if name not in want:
                continue
            got = float(np.mean(frac["test"][cls])) if frac["test"][cls] else 0.0
            if abs(got - want[name]) > 0.01:
                bad.append((name, got, want[name]))
        if bad:
            print("\n" + "!" * 72)
            print("INCONSISTENT with 09_spatial_validation on the test split.")
            print("One of the two scripts is reading the wrong half of the pair.")
            for name, got, exp in bad:
                print(f"   {name:<18} here {100*got:6.2f} %   "
                      f"09_spatial_validation {100*exp:6.2f} %")
            print("!" * 72 + "\n")
        else:
            print("\n  consistency check vs 09_spatial_validation: test split agrees\n")

    rows = [["Class index", "Class", "Split", "Layers with class",
             "Layers total", "Mean area fraction", "Max area fraction"]]
    print(f"\n{'class':<18}{'split':<8}{'layers with':>12}{'of':>5}"
          f"{'mean area':>12}{'max area':>11}")
    verdict = {}
    for cls in range(1, n_cls + 1):
        name = ev.FACIES_LEGEND[cls - 1][0]
        present = {}
        for split in SPLIT_NAMES:
            f = frac[split][cls]
            n_with = int(sum(1 for v in f if v > 0))
            present[split] = n_with
            mean = float(np.mean(f)) if f else 0.0
            mx = float(np.max(f)) if f else 0.0
            rows.append([cls, name, split, n_with, n_layers.get(split, 0),
                         f"{mean:.6f}", f"{mx:.6f}"])
            print(f"{name:<18}{split:<8}{n_with:>12}{n_layers.get(split,0):>5}"
                  f"{100*mean:>11.3f} %{100*mx:>10.3f} %")
        if present.get("train", 0) > 0 and present.get("test", 0) == 0:
            verdict[name] = ("in TRAIN but not TEST -- blocked-split imbalance, "
                             "not an invented class")
        elif present.get("train", 0) == 0 and present.get("test", 0) == 0:
            verdict[name] = "absent from BOTH -- genuinely not in the exported model"
        print()

    with open(os.path.join(OUT, "10_Class_support.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    print("=" * 72)
    print("CLASSES NOT PRESENT IN THE TEST LAYERS")
    print("=" * 72)
    if not verdict:
        print("  none -- every class occurring in training also occurs in test")
    for k, v in verdict.items():
        print(f"  {k:<18} {v}")
    print(f"\nwritten to {os.path.join(OUT, '10_Class_support.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
