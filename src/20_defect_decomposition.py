#!/usr/bin/env python3
"""
20_defect_decomposition.py
==========================

Why are the corrected numbers lower than the submitted ones? Answer it with a
ladder rather than an assertion.

The submitted Table 4 reported, for the inverse tasks:

    porosity   -> facies   PA 0.9060  mPA 0.5898  mIoU 0.5129  FWIoU 0.8418
    VCL        -> facies   PA 0.8440  mPA 0.8506  mIoU 0.7049  FWIoU 0.7429
    permeability-> facies  PA 0.7948  mPA 0.4274  mIoU 0.3500  FWIoU 0.6955

The corrected protocol reports macro-F1 0.493 and mIoU 0.412 for the best of
them. A drop that large has to be EXPLAINED, not merely announced, or a reviewer
is entitled to suspect the new code is the broken one. This script rebuilds the
submitted number from the corrected one, one defect at a time, using the SAME
trained models and the SAME predictions throughout. Nothing is retrained, so
every difference below is a scoring or data-selection difference and nothing else.

    STEP 1  corrected            7 facies classes, cells outside the reservoir
                                 excluded, held-out layers 46-54
    STEP 2  + background scored  the 25.78 % of each image that is inactive is
                                 counted as an eighth class the model gets right
    STEP 3  + 3-channel argmax   the submitted definition: agreement in which of
                                 R, G, B is the largest channel, num_classes=3.
                                 This is not a facies metric at all.
    STEP 4  + augmented training the submitted "test set of 252 images" drew
            images (optional)    augmented copies of layers the model trained on.
                                 Requires predictions on the training split.

If STEP 3 lands near the submitted values, the gap is the metric definition and
the leakage, not the model and not the new evaluation code. That is a far
stronger statement than "we fixed some bugs", and it is the honest answer to the
question every reviewer will ask.

    python 20_defect_decomposition.py --self-test
    python 20_defect_decomposition.py
    python 20_defect_decomposition.py --with-train-split      (needs STEP 4 preds)

For STEP 4, first generate predictions on the training split, e.g.
    python 06_predict.py --task vcl_to_facies --seed 1 --split train --limit 60
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")

TASKS = ["porosity_to_facies", "vcl_to_facies", "permeability_to_facies"]
SEEDS = [1, 2, 3]

# Submitted Table 4, for direct comparison. These are the values being explained.
SUBMITTED = {
    "porosity_to_facies":     dict(PA=0.9060, mPA=0.5898, mIoU=0.5129, FWIoU=0.8418),
    "vcl_to_facies":          dict(PA=0.8440, mPA=0.8506, mIoU=0.7049, FWIoU=0.7429),
    "permeability_to_facies": dict(PA=0.7948, mPA=0.4274, mIoU=0.3500, FWIoU=0.6955),
}


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# The three scoring conventions
# --------------------------------------------------------------------------

def cm_of(pred: np.ndarray, ref: np.ndarray, n: int,
          mask: Optional[np.ndarray] = None) -> np.ndarray:
    p, r = pred.ravel(), ref.ravel()
    if mask is not None:
        m = mask.ravel()
        p, r = p[m], r[m]
    keep = (r >= 0) & (r < n) & (p >= 0) & (p < n)
    return np.bincount(r[keep] * n + p[keep], minlength=n * n).reshape(n, n)


def metrics_of(cm: np.ndarray) -> Dict[str, float]:
    """PA, mPA, mIoU, FWIoU exactly as the submitted manuscript defined them."""
    total = cm.sum()
    if total == 0:
        return dict(PA=float("nan"), mPA=float("nan"),
                    mIoU=float("nan"), FWIoU=float("nan"))
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    valid = support > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        union = support + predicted - tp
        iou = np.where(union > 0, tp / union, np.nan)
    freq = support / total
    return dict(
        PA=float(tp.sum() / total),
        mPA=float(np.nanmean(recall[valid])),
        mIoU=float(np.nanmean(iou[valid])),
        FWIoU=float(np.nansum(freq[valid] * iou[valid])),
    )


def argmax3(rgb: np.ndarray) -> np.ndarray:
    """The submitted definition: which of R, G, B is largest. Three classes."""
    return np.argmax(rgb.astype(np.int32), axis=-1)


# --------------------------------------------------------------------------

def score_dir(task: str, split: str, ev, seeds: List[int],
              limit: Optional[int] = None) -> Optional[Dict[str, Dict[str, float]]]:
    ref_dir = os.path.join(SPLITS, task, split)
    if not os.path.isdir(ref_dir):
        return None
    files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))
    if limit:
        files = files[:limit]

    acc: Dict[str, List[Dict[str, float]]] = {"step0": [], "step1": [], "step2": [], "step3": []}
    n_used = 0
    n_cls = len(ev.FACIES_LEGEND)

    for seed in seeds:
        pdir = os.path.join(RUNS, f"{task}_seed{seed}", "predictions")
        if split != "test":
            alt = os.path.join(RUNS, f"{task}_seed{seed}", f"predictions_{split}")
            pdir = alt if os.path.isdir(alt) else pdir
        if not os.path.isdir(pdir):
            continue
        for fn in files:
            pp = os.path.join(pdir, fn)
            if not os.path.exists(pp):
                continue
            pair = np.array(Image.open(os.path.join(ref_dir, fn)).convert("RGB"))
            ref_rgb = pair[:, : pair.shape[1] // 2]
            pred_rgb = np.array(Image.open(pp).convert("RGB"))
            if pred_rgb.shape != ref_rgb.shape:
                continue

            ref = ev.decode_rgb_to_facies(ref_rgb)
            pred = ev.decode_rgb_to_facies(pred_rgb)

            # STEP 1 -- seven true classes, reservoir cells only.
            inside = ref >= 1
            c1 = cm_of(pred - 1, ref - 1, n_cls, inside)
            acc["step1"].append(metrics_of(c1))

            # STEP 2 -- background scored as an eighth class the model "gets right".
            c2 = cm_of(pred, ref, n_cls + 1)
            acc["step2"].append(metrics_of(c2))

            # STEP 3 -- the submitted metric: dominant colour channel, 3 classes.
            ref3 = argmax3(ref_rgb)
            c3 = cm_of(argmax3(pred_rgb), ref3, 3)
            acc["step3"].append(metrics_of(c3))

            # STEP 0 -- the trivial baseline under the submitted metric: emit a
            # constant image whose red channel dominates, and predict nothing.
            # Five of the seven legend colours have red as their largest channel,
            # and so does the black background, so this scores whatever fraction
            # of the image falls in that group. If it approaches the submitted
            # value, the submitted metric was not measuring facies recovery.
            acc["step0"].append(metrics_of(cm_of(np.zeros_like(ref3), ref3, 3)))
            n_used += 1

    if n_used == 0:
        return None
    out = {}
    for step, rows in acc.items():
        out[step] = {k: float(np.mean([r[k] for r in rows if np.isfinite(r[k])]))
                     for k in ("PA", "mPA", "mIoU", "FWIoU")}
    out["_n"] = {"images": float(n_used)}
    return out


# --------------------------------------------------------------------------

def self_test() -> int:
    ok = True

    # A perfect prediction scores 1 on every metric.
    ref = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    m = metrics_of(cm_of(ref, ref, 3))
    if not all(abs(m[k] - 1.0) < 1e-12 for k in m):
        print(f"  FAIL perfect prediction: {m}"); ok = False
    else:
        print("  ok   a perfect prediction scores 1 on PA, mPA, mIoU and FWIoU")

    # Adding a large always-correct class inflates PA and FWIoU but not mPA.
    # This is the whole mechanism of STEP 2, so it is worth pinning down.
    r = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 0])       # PA 0.5
    base = metrics_of(cm_of(p, r, 2))
    rb = np.concatenate([r, np.full(16, 2)])
    pb = np.concatenate([p, np.full(16, 2)])
    withbg = metrics_of(cm_of(pb, rb, 3))
    if not (withbg["PA"] > base["PA"] + 0.3 and withbg["mPA"] > base["mPA"]):
        print(f"  FAIL background inflation: {base} -> {withbg}"); ok = False
    else:
        print(f"  ok   an always-correct background class lifts PA "
              f"{base['PA']:.3f} -> {withbg['PA']:.3f}")

    # argmax3 collapses the seven-colour legend onto three channels, so
    # distinct facies become indistinguishable. Demonstrate the collapse.
    ev = _load("pix2geo_eval", "02_evaluation.py")
    pal = np.array([list(ev.hex_to_rgb(h)) for _, h in ev.FACIES_LEGEND],
                   dtype=np.uint8).reshape(1, -1, 3)
    lab = argmax3(pal).ravel()
    distinct = len(set(lab.tolist()))
    if distinct >= len(ev.FACIES_LEGEND):
        print(f"  FAIL argmax3 kept {distinct} classes"); ok = False
    else:
        names = [n for n, _ in ev.FACIES_LEGEND]
        groups: Dict[int, List[str]] = {}
        for nm, c in zip(names, lab.tolist()):
            groups.setdefault(int(c), []).append(nm)
        print(f"  ok   argmax3 collapses 7 facies onto {distinct} channel classes")
        for c in sorted(groups):
            print(f"         channel {'RGB'[c]}: {', '.join(groups[c])}")

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--with-train-split", action="store_true",
                    help="also score STEP 4 on the training split (needs predictions)")
    ap.add_argument("--train-limit", type=int, default=60)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)

    rows = [["Task", "Step", "What changes", "PA", "mPA", "mIoU", "FWIoU",
             "Submitted PA", "Remaining PA gap"]]
    labels = {
        "step1": "corrected: 7 classes, reservoir cells only, held-out layers",
        "step2": "+ inactive background scored as a class",
        "step3": "+ submitted metric: dominant RGB channel, 3 classes",
        "step4": "+ augmented copies of training layers as the test set",
        "step0": "TRIVIAL BASELINE under the submitted metric: predict nothing",
    }

    for task in args.tasks:
        sub = SUBMITTED.get(task, {})
        res = score_dir(task, "test", ev, args.seeds)
        if res is None:
            print(f"  ! no predictions for {task}")
            continue
        print(f"\n{task}   ({int(res['_n']['images'])} predictions)")
        print(f"{'step':<7}{'PA':>9}{'mPA':>9}{'mIoU':>9}{'FWIoU':>9}   what changes")
        for step in ("step0", "step1", "step2", "step3"):
            m = res[step]
            gap = sub.get("PA", float("nan")) - m["PA"]
            print(f"{step:<7}{m['PA']:>9.4f}{m['mPA']:>9.4f}{m['mIoU']:>9.4f}"
                  f"{m['FWIoU']:>9.4f}   {labels[step]}")
            rows.append([task, step, labels[step], f"{m['PA']:.4f}",
                         f"{m['mPA']:.4f}", f"{m['mIoU']:.4f}", f"{m['FWIoU']:.4f}",
                         f"{sub.get('PA', float('nan')):.4f}", f"{gap:+.4f}"])

        if args.with_train_split:
            tr = score_dir(task, "train", ev, args.seeds, limit=args.train_limit)
            if tr is None:
                print("  (step4 skipped: no predictions on the training split)")
            else:
                m = tr["step3"]
                gap = sub.get("PA", float("nan")) - m["PA"]
                print(f"{'step4':<7}{m['PA']:>9.4f}{m['mPA']:>9.4f}{m['mIoU']:>9.4f}"
                      f"{m['FWIoU']:>9.4f}   {labels['step4']}")
                rows.append([task, "step4", labels["step4"], f"{m['PA']:.4f}",
                             f"{m['mPA']:.4f}", f"{m['mIoU']:.4f}",
                             f"{m['FWIoU']:.4f}",
                             f"{sub.get('PA', float('nan')):.4f}", f"{gap:+.4f}"])
        if sub:
            print(f"{'SUBMIT':<7}{sub['PA']:>9.4f}{sub['mPA']:>9.4f}"
                  f"{sub['mIoU']:>9.4f}{sub['FWIoU']:>9.4f}   submitted Table 4")

    with open(os.path.join(OUT, "20_Defect_decomposition.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nwritten {os.path.join(OUT, '20_Defect_decomposition.csv')}")
    print("\n" + "=" * 74)
    print("Read the ladder upward. Each step re-introduces ONE defect of the")
    print("submitted evaluation, using the same models and the same predictions.")
    print("Whatever distance remains at the last step is the only part that could")
    print("be attributed to the evaluation code itself -- and if it is small, the")
    print("corrected numbers are confirmed rather than merely asserted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
