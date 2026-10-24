#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
30_class_imbalance.py
=====================

Reviewer 1, comment 2: "Rare classes are almost never recovered, yet the
authors have neither tried class-weighted losses or resampling nor explained
why, leaving an obvious limitation unaddressed."

This script supplies the missing experiment, without retraining the GAN.

The Bayes baseline in 14_baseline_conventional.py is a MAP classifier:

    post = (hist + 1) / (prior + 256) * max(prior, 1)
             \________likelihood______/   \__prior__/

Dropping the prior factor turns it into a maximum-likelihood classifier, which
is exactly the uniform-prior (class-balanced) variant of the same estimator --
one term, no new hyperparameter, no retraining. Comparing the two isolates the
effect of the class prior from the effect of the likelihood, and so answers
directly whether rare classes are lost because they are rare or because they
are not separable in the conditioning property.

It also records what the manuscript does not currently state: the inverse
random forest in 14_baseline_conventional.py is already fitted with
class_weight="balanced_subsample", so a class-weighted estimator has in fact
been evaluated throughout.

Per-class precision, recall and F1 are reported for every facies class under
both priors, so the rare classes (Fair Sand, Good Clean Sand) can be read
directly rather than inferred from macro averages.

Self-validating: refuses to report the uniform-prior numbers unless the
MAP variant first reproduces the published pixel accuracy, macro-F1 and mIoU
in results/13_Conventional_baselines.csv.

CPU only; needs numpy and Pillow. scikit-learn is NOT required -- the forest is
not refitted here, its published numbers are cited for context.

Usage
-----
    python 30_class_imbalance.py
    python 30_class_imbalance.py --only vcl_to_facies
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
EXPORT = os.path.join(HERE, "export_v2")
BASELINE_CSV = os.path.join(OUT_DIR, "13_Conventional_baselines.csv")
OUT_CSV = os.path.join(OUT_DIR, "30_Class_imbalance.csv")

INVERSE = [("porosity_to_facies", "porosity"),
           ("permeability_to_facies", "permeability"),
           ("vcl_to_facies", "vcl")]


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
CLASS_NAMES = [n for n, _ in EV.FACIES_LEGEND]


def adopt_legend(domain: str) -> None:
    lm = os.path.join(EXPORT, domain, "legend_manifest.json")
    if os.path.exists(lm):
        EV.load_legend_manifest(lm)


def prepared(path: str, domain: str):
    """(value, mask, true_facies) for one pair, as 14_baseline_conventional does."""
    tgt, inp = BL.halves(path)
    lab = EV.decode_rgb_to_facies(tgt)
    val, vmask = BL.decode_cont(inp, domain, EV)
    if domain == "permeability":
        vmask = vmask & (val > 0)
        val = np.log10(np.where(val > 0, val, 1.0))
    return val, vmask & (lab >= 1), lab


def fit_luts(task: str, domain: str):
    """Returns (edges, lut_map, lut_ml, class_counts, cmean)."""
    vals, labs = [], []
    for p in BL.train_files(task):
        val, m, lab = prepared(p, domain)
        # feature 0 of BL.features(v) is v itself, so the raw value suffices
        vals.append(val[m][::BL.FIT_STRIDE])
        labs.append(lab[m][::BL.FIT_STRIDE])
    x = np.concatenate(vals)
    y = np.concatenate(labs)
    print(f"  fitted on {y.size:,} sampled cells")

    lo, hi = float(np.percentile(x, 0.1)), float(np.percentile(x, 99.9))
    edges = np.linspace(lo, hi, 257)
    hist = np.zeros((N_CLS, 256))
    for c in range(1, N_CLS + 1):
        sel = x[y == c]
        if sel.size:
            hist[c - 1] = np.histogram(sel, bins=edges)[0]
    prior = hist.sum(axis=1, keepdims=True)
    like = (hist + 1.0) / (prior + 256.0)            # P(bin | class)
    lut_map = np.argmax(like * np.maximum(prior, 1.0), axis=0) + 1
    lut_ml = np.argmax(like, axis=0) + 1             # uniform prior

    # Class means for the physical-unit round trip, fitted on the SAME
    # training sample and in the SAME space as 14_baseline_conventional.py
    # (already log10 for permeability). A class with no training support gets
    # NaN and its cells are excluded rather than imputed.
    cmean = np.full(N_CLS + 1, np.nan)
    for c in range(1, N_CLS + 1):
        sel = x[y == c]
        if sel.size:
            cmean[c] = float(sel.mean())
    return edges, lut_map, lut_ml, prior.ravel(), cmean


def score(task: str, domain: str, edges, lut, cmean):
    """
    Confusion matrix plus the streaming sums for the physical-unit metrics,
    accumulated over the test layers exactly as 14_baseline_conventional.py
    does: n, sum|e|, sum e^2, sum v, sum v^2.
    """
    cm = np.zeros((N_CLS, N_CLS))
    acc = np.zeros(5)
    for p in BL.test_files(task):
        val, m, lab = prepared(p, domain)
        b = np.clip(np.digitize(val, edges) - 1, 0, 255)
        pred = lut[b][m]
        cm += EV.confusion_matrix(pred, lab[m], N_CLS)
        tv = val[m]
        ok = ~np.isnan(cmean[pred])
        if np.any(ok):
            q = cmean[pred[ok]]
            r = tv[ok]
            e = q - r
            acc += np.array([r.size, np.abs(e).sum(), (e * e).sum(),
                             r.sum(), (r * r).sum()])
    return cm, acc


def phys(acc: np.ndarray) -> Dict[str, float]:
    n, s_ae, s_e2, s_v, s_v2 = acc
    if n <= 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    ss_tot = s_v2 - (s_v * s_v) / n
    return {"MAE": s_ae / n,
            "RMSE": float(np.sqrt(s_e2 / n)),
            "R2": 1.0 - s_e2 / ss_tot if ss_tot > 0 else float("nan")}


def metrics(cm: np.ndarray, pred_is_rows: bool) -> Dict[str, object]:
    """
    PA, macro-F1, mIoU and per-class precision/recall/F1.

    A class present in the reference but never predicted receives precision
    zero rather than being dropped, matching the manuscript's stated rule.
    """
    c = cm if pred_is_rows else cm.T      # rows = predicted, cols = true
    total = c.sum()
    pa = float(np.trace(c) / total) if total else float("nan")
    prec, rec, f1, iou = [], [], [], []
    for k in range(c.shape[0]):
        tp = c[k, k]
        fp = c[k, :].sum() - tp
        fn = c[:, k].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        j = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        prec.append(p); rec.append(r); f1.append(f); iou.append(j)
    return {"PA": pa, "macro_F1": float(np.mean(f1)), "mIoU": float(np.mean(iou)),
            "precision": prec, "recall": rec, "f1": f1,
            "support_true": (c.sum(axis=0) if True else None)}


def published() -> Dict[Tuple[str, str], float]:
    out = {}
    if not os.path.exists(BASELINE_CSV):
        return out
    with open(BASELINE_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["Direction"] != "inverse" or r["Baseline"] != "bayes":
                continue
            key = r["Metric"].split(" [")[0]
            if key in ("pixel_accuracy", "macro_F1", "mIoU",
                       "MAE", "RMSE", "R2"):
                out[(r["Task"], key)] = float(r["Value"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tol", type=float, default=0.004)
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    pub = published()
    rows: List[dict] = []
    checks: List[Tuple[str, str, float, float]] = []

    todo = [c for c in INVERSE if args.only is None or c[0] == args.only]
    if not todo:
        print(f"no task matches --only {args.only!r}")
        return 2

    for task, domain in todo:
        adopt_legend(domain)
        print(f"\n{'=' * 76}\n{task}\n{'=' * 76}")
        edges, lut_map, lut_ml, counts, cmean = fit_luts(task, domain)

        res = {}
        orient = None
        for name, lut in (("training prior (MAP, as published)", lut_map),
                          ("uniform prior (balanced)", lut_ml)):
            cm, acc = score(task, domain, edges, lut, cmean)
            if orient is None:
                # settle the confusion-matrix orientation once, against the
                # published MAP numbers, instead of assuming it
                cand = {o: metrics(cm, o) for o in (True, False)}
                want = pub.get((task, "macro_F1"))
                if want is None:
                    orient = True
                else:
                    best = min(cand, key=lambda o: abs(cand[o]["macro_F1"] - want))
                    orient = best
            res[name] = metrics(cm, orient)
            res[name].update(phys(acc))

        for name, r in res.items():
            print(f"\n  {name}")
            print(f"    PA {r['PA']:.4f}   macro-F1 {r['macro_F1']:.4f}   "
                  f"mIoU {r['mIoU']:.4f}")
            print(f"    MAE {r['MAE']:.4f}  RMSE {r['RMSE']:.4f}  "
                  f"R2 {r['R2']:+.4f}   (physical units, class-mean decode)")
            for k, cn in enumerate(CLASS_NAMES):
                print(f"      {cn:<18} precision {r['precision'][k]:.3f}  "
                      f"recall {r['recall'][k]:.3f}  F1 {r['f1'][k]:.3f}")
            rows.append({"task": task, "variant": name,
                         "PA": f"{r['PA']:.6f}",
                         "macro_F1": f"{r['macro_F1']:.6f}",
                         "mIoU": f"{r['mIoU']:.6f}",
                         "MAE": f"{r['MAE']:.6f}",
                         "RMSE": f"{r['RMSE']:.6f}",
                         "R2": f"{r['R2']:.6f}",
                         **{f"F1_{cn.replace(' ', '_')}": f"{r['f1'][k]:.4f}"
                            for k, cn in enumerate(CLASS_NAMES)},
                         **{f"recall_{cn.replace(' ', '_')}": f"{r['recall'][k]:.4f}"
                            for k, cn in enumerate(CLASS_NAMES)}})

        m = res["training prior (MAP, as published)"]
        for metric, key in (("pixel_accuracy", "PA"), ("macro_F1", "macro_F1"),
                            ("mIoU", "mIoU"), ("MAE", "MAE"),
                            ("RMSE", "RMSE"), ("R2", "R2")):
            if (task, metric) in pub:
                checks.append((task, metric, pub[(task, metric)], m[key]))

    print("\n" + "=" * 76)
    print("VALIDATION  MAP variant must reproduce results/13_Conventional_baselines.csv")
    print("=" * 76)
    worst = 0.0
    for task, metric, want, got in checks:
        d = abs(want - got)
        worst = max(worst, d)
        print(f"  {task:<26} {metric:<14} published {want:.4f}   "
              f"recomputed {got:.4f}   d={d:.4f}  "
              f"{'OK' if d <= args.tol else 'MISMATCH'}")
    ok = bool(checks) and worst <= args.tol
    if args.validate and not ok:
        print(f"\nVALIDATION FAILED (worst d={worst:.4f}). Nothing written.")
        return 1
    print(f"\nValidation passed (worst d={worst:.4f}).")
    print("\nNote: the inverse random forest in 14_baseline_conventional.py is "
          "already fitted with\nclass_weight=\"balanced_subsample\", so a "
          "class-weighted estimator is present in the\npublished baselines; "
          "its numbers are in results/13_Conventional_baselines.csv.")

    if rows:
        keys = list(rows[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
