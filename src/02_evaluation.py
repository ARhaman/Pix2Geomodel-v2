#!/usr/bin/env python3
"""
02_evaluation.py
================

Corrected evaluation for Pix2Geomodel V2 (RINENG-D-26-10842).

WHY THIS SCRIPT EXISTS
----------------------
The original evaluation in every training notebook was:

    def evaluate_segmentation_performance(generator, test_dataset, num_classes=3):
        ...
        pred_labels = tf.argmax(pred, axis=-1)     # argmax over the RGB axis
        tar_labels  = tf.argmax(tar, axis=-1)

`axis=-1` is the colour-channel axis and `num_classes=3` is never overridden,
so PA / mPA / mIoU / FWIoU were measuring agreement in *which of R, G or B is
the largest channel*. They were not measuring the seven facies classes, and
they were not measuring porosity, permeability or VCL. Reviewer 1 (Results)
and Reviewer 5 (#8, #10, #11) all point at this from different directions.

This module replaces that with two evaluations that measure what the
manuscript claims to measure:

  CATEGORICAL (facies targets)
      Decode the palette index -> true 7-class label. Report per-class
      precision / recall / F1, the full confusion matrix, class frequencies,
      and correctly defined PA / mPA / mIoU / FWIoU over 7 classes.
      -> Reviewer 5 #10, #26e

  CONTINUOUS (porosity, permeability, VCL targets)
      Invert the fixed export legend analytically to recover physical values,
      then report MAE, RMSE, normalised RMSE, R^2, Pearson, Spearman and bias.
      Permeability is additionally evaluated in log10 space.
      -> Reviewer 1 (Results), Reviewer 5 #8, #11, #12, #26d

It also quantifies the ENCODING FLOOR: the error the RGB representation
imposes before the network is involved at all. Reviewer 5 #11 asks for this
explicitly, and separating encoding error from model error is the only way to
answer it honestly.

Uncertainty is reported as clustered bootstrap 95 % confidence intervals,
resampled at the ORIGINAL-LAYER level, because pixels and slices within a
layer are not independent (Reviewer 5 #5, #7, #9, #26c).

USAGE
-----
    python 02_evaluation.py --self-test
    python 02_evaluation.py --domain permeability --encoding-report
    python 02_evaluation.py --task facies_to_porosity \
        --pred-dir  runs/f2p_seed1/predictions \
        --ref-dir   splits/facies_to_porosity/test \
        --manifest  splits/facies_to_porosity/split_manifest.json \
        --out       results/f2p_seed1.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

# --------------------------------------------------------------------------
# Legend definitions -- copied verbatim from
# `1. Extract orginal ayers for whole model.ipynb` so that decoding is the
# exact analytical inverse of the export, not an approximation.
# --------------------------------------------------------------------------

LUT_SIZE = 255              # ramp occupies indices 0..254
BACKGROUND_INDEX = 255      # reserved for NODATA, rendered black

FACIES_LEGEND: List[Tuple[str, str]] = [
    ("Tight",           "#8B4513"),
    ("Very Poor Sand",  "#00FF00"),
    ("Poor Sand",       "#ADFF2F"),
    ("Fair Sand",       "#FFFF00"),
    ("Good Sand",       "#FFA500"),
    ("Good Clean Sand", "#FF00FF"),
    ("Very Good Sand",  "#FF0000"),
]

CONTINUOUS_DOMAINS: Dict[str, dict] = {
    "porosity": {
        "vmin": 0.0, "vmax": 0.225, "scale": "linear", "unit": "fraction",
        "stops": ((0.00, "#FF00FF"), (0.10, "#0000FF"), (0.20, "#00FFFF"),
                  (0.40, "#00FF00"), (0.70, "#FFFF00"), (0.85, "#FFA500"),
                  (1.00, "#FF0000")),
    },
    "permeability": {
        "vmin": 0.01, "vmax": 1000.0, "scale": "log10", "unit": "mD",
        "stops": ((0.00, "#0000FF"), (0.25, "#00FFFF"), (0.50, "#00FF00"),
                  (0.75, "#FFFF00"), (0.90, "#FFA500"), (1.00, "#FF0000")),
    },
    "vcl": {
        "vmin": 0.0, "vmax": 0.50, "scale": "linear", "unit": "fraction",
        "stops": ((0.00, "#FF0000"), (0.10, "#FFA500"), (0.30, "#FFFF00"),
                  (0.50, "#00FF00"), (0.60, "#00FFFF"), (0.90, "#0000FF"),
                  (1.00, "#FF00FF")),
    },
}


def load_legend_manifest(path: str) -> dict:
    """
    Adopt the exact legend that `04_export_common_legend.py` used for an export.

    Decoding must be the analytical inverse of the encoding that actually
    produced the images. When the extended legends are in use -- wider ranges
    plus a reserved palette index for zero permeability -- decoding with the
    original hard-coded ranges would silently mis-scale every predicted value.
    Reading the manifest removes that whole class of error.

    Returns the domain spec dict that was installed.
    """
    with open(path) as fh:
        man = json.load(fh)
    prop = man.get("property")
    legend = man.get("legend")
    if prop in CONTINUOUS_DOMAINS and legend:
        spec = dict(CONTINUOUS_DOMAINS[prop])
        spec.update({k: legend[k] for k in ("vmin", "vmax", "scale", "unit")
                     if k in legend})
        if legend.get("zero_class"):
            spec["zero_class"] = True
        if man.get("gradient_stops"):
            spec["stops"] = tuple(tuple(x) for x in man["gradient_stops"])
        CONTINUOUS_DOMAINS[prop] = spec
        print(f"[legend] {prop}: {spec['vmin']} .. {spec['vmax']} ({spec['scale']})"
              f"{', reserved zero class' if spec.get('zero_class') else ''}"
              f"  <- {os.path.basename(path)}")
    return CONTINUOUS_DOMAINS.get(prop, {})


def ramp_range(spec: dict) -> Tuple[int, int]:
    """Palette indices occupied by the ramp; index 0 is reserved when
    a zero class is in use (see 04_export_common_legend.py)."""
    return (1, LUT_SIZE - 1) if spec.get("zero_class") else (0, LUT_SIZE - 1)


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def build_lut(stops: Sequence[Tuple[float, str]], lut_size: int = LUT_SIZE) -> np.ndarray:
    """Exact reproduction of the export-side LUT construction."""
    positions = np.array([p for p, _ in stops], dtype=float)
    colors = np.array([hex_to_rgb(c) for _, c in stops], dtype=float)
    x = np.linspace(0.0, 1.0, lut_size, dtype=float)
    r = np.interp(x, positions, colors[:, 0])
    g = np.interp(x, positions, colors[:, 1])
    b = np.interp(x, positions, colors[:, 2])
    return np.stack([r, g, b], axis=1).round().astype(np.uint8)


# --------------------------------------------------------------------------
# Encode / decode
# --------------------------------------------------------------------------

def value_to_index(values: np.ndarray, domain: str) -> np.ndarray:
    """Physical value -> palette index (the export-side mapping)."""
    d = CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = ramp_range(d)
    v = np.asarray(values, dtype=float)
    if d["scale"] == "log10":
        vc = np.clip(v, d["vmin"], d["vmax"])
        norm = (np.log10(vc) - math.log10(d["vmin"])) / (
            math.log10(d["vmax"]) - math.log10(d["vmin"]))
    else:
        norm = (v - d["vmin"]) / max(d["vmax"] - d["vmin"], 1e-9)
    idx = (lo_i + np.clip(norm, 0.0, 1.0) * (hi_i - lo_i)).round().astype(np.uint8)
    if d.get("zero_class"):
        idx = np.where(v <= 0, np.uint8(0), idx)
    return idx


def index_to_value(idx: np.ndarray, domain: str) -> np.ndarray:
    """Palette index -> physical value (analytical inverse)."""
    d = CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = ramp_range(d)
    idx = np.asarray(idx)
    norm = np.clip((idx.astype(float) - lo_i) / (hi_i - lo_i), 0.0, 1.0)
    if d["scale"] == "log10":
        lo, hi = math.log10(d["vmin"]), math.log10(d["vmax"])
        out = np.power(10.0, lo + norm * (hi - lo))
    else:
        out = d["vmin"] + norm * (d["vmax"] - d["vmin"])
    if d.get("zero_class"):
        out = np.where(idx == 0, 0.0, out)
    return out


def decode_rgb_to_index(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Nearest-LUT-entry decoding of an RGB image.

    A generated image is not guaranteed to contain exact legend colours, so
    nearest-neighbour in RGB space is the honest inverse.

    PERFORMANCE: a 2048x1024 image is 2.1 M pixels; brute force against 254
    palette entries is ~1.6 G distance terms per image, which made in-training
    validation cost ~28 s per image. Two accelerations, both exact:
      1. decode only the DISTINCT colours present (a generated slice typically
         holds a few thousand, not millions), then scatter the result back;
      2. exact int32 arithmetic with deterministic (lowest-index) tie-breaking.
    """
    flat = rgb.reshape(-1, 3)
    # collapse to distinct colours -- the dominant win
    packed = (flat[:, 0].astype(np.uint32) << 16 |
              flat[:, 1].astype(np.uint32) << 8 |
              flat[:, 2].astype(np.uint32))
    uniq, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255],
                       axis=1).astype(np.int32)

    # int32, NOT int16: a channel difference of 255 squares to 65025 and the
    # three-channel sum reaches 195075, past the int16 maximum of 32767. An
    # int16 formulation wraps around and silently returns the WRONG nearest
    # colour for widely separated pixels.
    #
    # Deterministic tie-breaking: np.argmin returns the LOWEST index among
    # equidistant candidates. A KD-tree is not used here even though SciPy is
    # available -- it breaks ties arbitrarily, which made repeated evaluations
    # of the same image disagree on ~6 % of pixels. After collapsing to
    # distinct colours the brute-force search is also the faster of the two.
    lut32 = lut.astype(np.int32)
    nearest = np.empty(colours.shape[0], dtype=np.uint8)
    step = 200_000
    for s0 in range(0, colours.shape[0], step):
        chunk = colours[s0:s0 + step]
        d2 = ((chunk[:, None, :] - lut32[None, :, :]) ** 2).sum(axis=2)
        nearest[s0:s0 + step] = np.argmin(d2, axis=1).astype(np.uint8)

    return nearest[inverse].reshape(rgb.shape[:2])


def decode_rgb_to_facies(rgb: np.ndarray) -> np.ndarray:
    """
    RGB -> facies class 1..7 (0 = background/unassigned).

    Uses nearest legend colour. This is what the metrics should always have
    been computed on: seven real classes, not three colour channels.

    int32, NOT int16 -- the same overflow documented for decode_rgb_to_index
    above. It was left in place here and is worth spelling out because the
    failure is silent and total rather than noisy and partial. A channel
    difference of 255 squares to 65025, which wraps to -511 as an int16, so a
    LARGE distance reads as the SMALLEST one and argmin selects the colour
    that is furthest away. Feeding the palette back into the int16 version,
    only one of the eight entries decoded to itself: black background came
    back as Fair Sand (its true squared distance of 130050 wrapping to -1022),
    Very Poor Sand as Tight, Good Clean Sand as Poor Sand. Because the
    corruption is deterministic, reference and prediction were mislabelled the
    same way, so confusion matrices stayed plausible and nothing downstream
    raised an error -- but every per-class name was wrong and 25.8 % of each
    image (the background) was being scored as a real facies class.
    """
    legend_rgb = np.array([hex_to_rgb(h) for _, h in FACIES_LEGEND], dtype=np.int32)
    black = np.array([[0, 0, 0]], dtype=np.int32)
    palette = np.vstack([black, legend_rgb])           # row 0 = background
    flat = rgb.reshape(-1, 3).astype(np.int32)
    out = np.empty(flat.shape[0], dtype=np.uint8)
    step = 200_000
    for s in range(0, flat.shape[0], step):
        chunk = flat[s:s + step]
        d2 = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        out[s:s + step] = np.argmin(d2, axis=1).astype(np.uint8)
    return out.reshape(rgb.shape[:2])


def _selftest_metrics() -> None:
    """
    macro-F1 must penalise a class the model never predicts.

    Reference has three equal-ish classes; the model predicts only class 1.
    Correct macro-F1 is (0.667 + 0 + 0) / 3 = 0.222. The NaN-dropping version
    returned 0.667 -- three times too high, and higher than the model's own
    matched-condition score in the positional control.
    """
    cm = np.array([[100.0, 0.0, 0.0], [50.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    m = categorical_metrics(cm, ["a", "b", "c"])
    if abs(m["macro_F1"] - 2.0 / 9.0) > 1e-9:
        raise AssertionError(
            f"macro_F1 is {m['macro_F1']:.4f}, expected {2/9:.4f}: a class "
            "present in the reference but never predicted must score F1 = 0, "
            "not be dropped from the average")
    if m["per_class"][1]["f1"] != 0.0 or m["per_class"][2]["f1"] != 0.0:
        raise AssertionError("unpredicted classes must report f1 = 0, not None")
    # A class absent from the reference must not enter the macro average.
    cm2 = np.array([[100.0, 0.0], [0.0, 0.0]])
    m2 = categorical_metrics(cm2, ["a", "b"])
    if abs(m2["macro_F1"] - 1.0) > 1e-9:
        raise AssertionError("a zero-support class must be excluded, not scored 0")


def _selftest_decoders() -> None:
    """
    Round-trip guard: every palette colour must decode to its own index.

    This is the check that would have caught the int16 overflow immediately.
    It is cheap, so it runs on import rather than waiting to be remembered.
    """
    pal = [(0, 0, 0)] + [hex_to_rgb(h) for _, h in FACIES_LEGEND]
    img = np.array(pal, dtype=np.uint8).reshape(1, -1, 3)
    got = decode_rgb_to_facies(img).ravel()
    want = np.arange(len(pal), dtype=got.dtype)
    if not np.array_equal(got, want):
        names = ["background"] + [n for n, _ in FACIES_LEGEND]
        bad = "; ".join(f"{names[i]} -> {names[int(got[i])]}"
                        for i in range(len(pal)) if got[i] != i)
        raise AssertionError(
            "decode_rgb_to_facies does not round-trip the palette: " + bad)




# --------------------------------------------------------------------------
# Encoding-floor analysis  (Reviewer 5 #11)
# --------------------------------------------------------------------------

def lut_injectivity(domain: str) -> dict:
    """Are two different palette indices rendered as the same RGB triple?"""
    lut = build_lut(CONTINUOUS_DOMAINS[domain]["stops"])
    uniq, counts = np.unique(lut, axis=0, return_counts=True)
    dup = int((counts > 1).sum())
    return {
        "lut_entries": int(lut.shape[0]),
        "distinct_colours": int(uniq.shape[0]),
        "colliding_colours": dup,
        "injective": dup == 0,
    }


def encoding_floor(domain: str, n: int = 200_000, seed: int = 2025) -> dict:
    """
    Round-trip error imposed by the RGB representation alone:
    value -> palette index -> value. This is the error floor the network
    cannot beat, and it must be reported alongside model error.
    """
    d = CONTINUOUS_DOMAINS[domain]
    rng = np.random.default_rng(seed)
    if d["scale"] == "log10":
        lo, hi = math.log10(d["vmin"]), math.log10(d["vmax"])
        true = np.power(10.0, rng.uniform(lo, hi, n))
    else:
        true = rng.uniform(d["vmin"], d["vmax"], n)

    rec = index_to_value(value_to_index(true, domain), domain)
    err = rec - true
    out = {
        "domain": domain, "unit": d["unit"], "scale": d["scale"],
        "range": [d["vmin"], d["vmax"]],
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.max(np.abs(err))),
    }
    if d["scale"] == "log10":
        rel = rec / true
        out["max_relative_error_pct"] = float(100 * np.max(np.abs(rel - 1)))
        out["log10_RMSE"] = float(np.sqrt(np.mean((np.log10(rec) - np.log10(true)) ** 2)))
    else:
        span = d["vmax"] - d["vmin"]
        out["RMSE_pct_of_range"] = float(100 * out["RMSE"] / span)
    out.update(lut_injectivity(domain))
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def continuous_metrics(pred: np.ndarray, ref: np.ndarray,
                       domain: str, mask: Optional[np.ndarray] = None) -> dict:
    """Physical-unit error metrics. Permeability additionally in log10 space."""
    p = np.asarray(pred, float).ravel()
    r = np.asarray(ref, float).ravel()
    if mask is not None:
        m = np.asarray(mask, bool).ravel()
        p, r = p[m], r[m]
    if p.size == 0:
        return {"n": 0}

    err = p - r
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    d = CONTINUOUS_DOMAINS[domain]

    out = {
        "n": int(p.size),
        "unit": d["unit"],
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "nRMSE_pct_of_range": float(100 * np.sqrt(np.mean(err ** 2)) / (d["vmax"] - d["vmin"])),
        "bias": float(np.mean(err)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson": float(np.corrcoef(p, r)[0, 1]) if p.size > 1 and p.std() > 0 and r.std() > 0 else float("nan"),
        "spearman": _spearman(p, r),
    }
    if d["scale"] == "log10":
        lp = np.log10(np.clip(p, d["vmin"], d["vmax"]))
        lr = np.log10(np.clip(r, d["vmin"], d["vmax"]))
        le = lp - lr
        out["log10_MAE"] = float(np.mean(np.abs(le)))
        out["log10_RMSE"] = float(np.sqrt(np.mean(le ** 2)))
        out["log10_bias"] = float(np.mean(le))
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # average ties
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def confusion_matrix(pred: np.ndarray, ref: np.ndarray, n_classes: int,
                     ignore_background: bool = True) -> np.ndarray:
    """Rows = reference class, columns = predicted class. Classes are 1..n."""
    p = np.asarray(pred).ravel().astype(np.int64)
    r = np.asarray(ref).ravel().astype(np.int64)
    valid = (r >= 1) & (r <= n_classes) & (p >= 1) & (p <= n_classes)
    if not ignore_background:
        valid = np.ones_like(r, dtype=bool)
    p, r = p[valid] - 1, r[valid] - 1
    idx = r * n_classes + p
    return np.bincount(idx, minlength=n_classes ** 2).reshape(n_classes, n_classes).astype(float)


def categorical_metrics(cm: np.ndarray, class_names: Sequence[str]) -> dict:
    """
    Correctly defined segmentation metrics over the TRUE class set,
    plus the per-class breakdown Reviewer 5 #10 asks for.
    """
    n = cm.shape[0]
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)          # reference counts per class
    predicted = cm.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        # A class that IS in the reference but that the model never predicts
        # has undefined precision (0/0). Leaving it NaN propagates to F1, and
        # np.nanmean then DROPS that class from macro-F1 instead of scoring it
        # zero -- so a model is rewarded for refusing to predict a class it
        # cannot get right. The convention (and sklearn's zero_division=0) is
        # precision 0, hence F1 0, hence the class counts against the macro
        # average. mIoU never had this problem because its denominator
        # support+predicted-tp is positive whenever support is, which is why
        # the two metrics disagreed: under a constant input one model scored
        # macro-F1 0.6561 against mIoU 0.1399, and the macro-F1 was higher
        # than its own matched-condition score. That contradiction is what
        # exposed this.
        precision = np.where(predicted > 0, tp / predicted,
                             np.where(support > 0, 0.0, np.nan))
        denom = precision + recall
        f1 = np.where(np.isfinite(denom) & (denom > 0),
                      2 * precision * recall / np.where(denom > 0, denom, 1.0),
                      np.where(support > 0, 0.0, np.nan))
        union = support + predicted - tp
        iou = np.where(union > 0, tp / union, np.nan)

    freq = support / total if total > 0 else np.zeros(n)
    valid = support > 0

    return {
        "n_classes": n,
        "n_pixels": int(total),
        "pixel_accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "mean_pixel_accuracy": float(np.nanmean(recall[valid])) if valid.any() else float("nan"),
        "mIoU": float(np.nanmean(iou[valid])) if valid.any() else float("nan"),
        "FWIoU": float(np.nansum(freq[valid] * iou[valid])) if valid.any() else float("nan"),
        "macro_F1": float(np.nanmean(f1[valid])) if valid.any() else float("nan"),
        "per_class": [
            {
                "class": int(i + 1),
                "name": class_names[i] if i < len(class_names) else f"class_{i+1}",
                "support_px": int(support[i]),
                "frequency": float(freq[i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
                "f1": None if np.isnan(f1[i]) else float(f1[i]),
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
            }
            for i in range(n)
        ],
        "confusion_matrix": cm.astype(int).tolist(),
    }


# --------------------------------------------------------------------------
# Clustered bootstrap  (Reviewer 5 #7, #9, #26c)
# --------------------------------------------------------------------------

def clustered_bootstrap_ci(per_layer_values: Dict[int, float],
                           n_boot: int = 10_000, alpha: float = 0.05,
                           seed: int = 2025) -> dict:
    """
    Resample LAYERS with replacement, not pixels or slices.

    With 54 original layers -- of which only a handful are in the test set --
    pixel-level or image-level intervals would be wildly overconfident.
    Resampling at the layer level is the correct unit of independence.
    """
    layers = sorted(per_layer_values)
    vals = np.array([per_layer_values[k] for k in layers], dtype=float)
    n = vals.size
    if n == 0:
        return {"n_layers": 0}
    if n == 1:
        return {"n_layers": 1, "point_estimate": float(vals[0]),
                "ci_low": None, "ci_high": None,
                "note": "single layer -- no interval estimable"}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return {
        "n_layers": int(n),
        "point_estimate": float(vals.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "layer_sd": float(vals.std(ddof=1)),
        "n_boot": int(n_boot),
        "per_layer": {int(k): float(v) for k, v in zip(layers, vals)},
    }


def paired_permutation_test(a: Dict[int, float], b: Dict[int, float],
                            n_perm: int = 10_000, seed: int = 2025) -> dict:
    """
    Compare two tasks/models on the SAME layers by sign-flipping the paired
    differences. Ordinary two-sample tests are invalid here because the
    observations share geological layers (Reviewer 5 #9).
    """
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return {"n_layers": len(common), "note": "too few paired layers"}
    d = np.array([a[k] - b[k] for k in common], dtype=float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"n_layers": len(common), "mean_difference": float(obs), "p_value": p,
            "n_perm": int(n_perm)}


# --------------------------------------------------------------------------
# Image-pair driver
# --------------------------------------------------------------------------

LAYER_RE = re.compile(r"layer(\d+)_aug(\d+)", re.IGNORECASE)


@dataclass
class PairSpec:
    pred_path: str
    ref_path: str
    layer: int


def collect_pairs(pred_dir: str, ref_dir: str) -> List[PairSpec]:
    pairs: List[PairSpec] = []
    for fn in sorted(os.listdir(pred_dir)):
        if not fn.lower().endswith(".png"):
            continue
        m = LAYER_RE.search(fn)
        if not m:
            continue
        ref = os.path.join(ref_dir, fn)
        if not os.path.exists(ref):
            continue
        pairs.append(PairSpec(os.path.join(pred_dir, fn), ref, int(m.group(1))))
    return pairs


def evaluate_task(pred_dir: str, ref_dir: str, target_domain: str,
                  ref_is_pair: bool = True, target_on_left: bool = True) -> dict:
    """
    target_domain: 'facies' or one of CONTINUOUS_DOMAINS.

    ref_is_pair=True means the reference file is the concatenated pair written
    by 01_layer_level_split.py, so the target half must be sliced out first.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image evaluation")

    pairs = collect_pairs(pred_dir, ref_dir)
    if not pairs:
        raise RuntimeError(f"No matching prediction/reference pairs found")

    per_layer: Dict[int, dict] = {}
    per_layer_cm: Dict[int, np.ndarray] = {}
    cm_total = np.zeros((len(FACIES_LEGEND),) * 2) if target_domain == "facies" else None

    for spec in pairs:
        pred = np.array(Image.open(spec.pred_path).convert("RGB"))
        ref_img = np.array(Image.open(spec.ref_path).convert("RGB"))
        if ref_is_pair:
            w = ref_img.shape[1] // 2
            ref_img = ref_img[:, :w] if target_on_left else ref_img[:, w:]
        if pred.shape != ref_img.shape:
            raise ValueError(f"shape mismatch for {os.path.basename(spec.pred_path)}: "
                             f"{pred.shape} vs {ref_img.shape}")

        if target_domain == "facies":
            p = decode_rgb_to_facies(pred)
            r = decode_rgb_to_facies(ref_img)
            cm = confusion_matrix(p, r, len(FACIES_LEGEND))
            cm_total += cm
            # Keep the per-layer confusion matrix: it lets any metric be
            # recomputed later over a subset of classes, with the layer-level
            # bootstrap still available, without re-running inference.
            per_layer_cm.setdefault(spec.layer, np.zeros_like(cm))
            per_layer_cm[spec.layer] += cm
            m = categorical_metrics(cm, [n for n, _ in FACIES_LEGEND])
            per_layer.setdefault(spec.layer, {"pixel_accuracy": [], "mIoU": [], "macro_F1": []})
            for k in ("pixel_accuracy", "mIoU", "macro_F1"):
                per_layer[spec.layer][k].append(m[k])
        else:
            dspec = CONTINUOUS_DOMAINS[target_domain]
            lo_i, hi_i = ramp_range(dspec)
            lut = build_lut(dspec["stops"], hi_i - lo_i + 1)
            pv = index_to_value(decode_rgb_to_index(pred, lut) + lo_i, target_domain)
            rv = index_to_value(decode_rgb_to_index(ref_img, lut) + lo_i, target_domain)
            valid = ~(np.all(ref_img == 0, axis=-1))       # exclude NODATA black
            m = continuous_metrics(pv, rv, target_domain, mask=valid)
            per_layer.setdefault(spec.layer, {"MAE": [], "RMSE": [], "R2": []})
            for k in ("MAE", "RMSE", "R2"):
                per_layer[spec.layer][k].append(m[k])

    # collapse repeated slices within a layer, then bootstrap over layers
    summary: Dict[str, dict] = {}
    metric_names = list(next(iter(per_layer.values())).keys())
    for name in metric_names:
        collapsed = {k: float(np.mean(v[name])) for k, v in per_layer.items()}
        summary[name] = clustered_bootstrap_ci(collapsed)

    out = {
        "target_domain": target_domain,
        "n_pairs": len(pairs),
        "n_layers": len(per_layer),
        "metrics": summary,
    }
    if cm_total is not None:
        out["pooled_categorical"] = categorical_metrics(
            cm_total, [n for n, _ in FACIES_LEGEND])
        out["per_layer_confusion"] = {int(k): v.astype(int).tolist()
                                      for k, v in per_layer_cm.items()}
    if target_domain in CONTINUOUS_DOMAINS:
        out["encoding_floor"] = encoding_floor(target_domain)
    return out


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    print("=" * 70)
    print("ENCODING FLOOR -- error imposed by the RGB representation alone")
    print("=" * 70)
    for dom in CONTINUOUS_DOMAINS:
        r = encoding_floor(dom)
        print(f"\n{dom.upper():14s} range {r['range']} {r['unit']}  ({r['scale']})")
        print(f"  LUT entries {r['lut_entries']}, distinct colours {r['distinct_colours']}, "
              f"injective={r['injective']}")
        print(f"  round-trip MAE  {r['MAE']:.6g}   RMSE {r['RMSE']:.6g}   "
              f"max |err| {r['max_abs_error']:.6g}")
        if "max_relative_error_pct" in r:
            print(f"  max relative error {r['max_relative_error_pct']:.2f} %   "
                  f"log10 RMSE {r['log10_RMSE']:.5f}")
        else:
            print(f"  RMSE = {r['RMSE_pct_of_range']:.3f} % of range")

    print("\n" + "=" * 70)
    print("ROUND-TRIP CONSISTENCY (encode -> decode must be exact on grid values)")
    print("=" * 70)
    ok = True
    for dom in CONTINUOUS_DOMAINS:
        idx = np.arange(255, dtype=np.uint8)
        back = value_to_index(index_to_value(idx, dom), dom)
        good = bool(np.array_equal(idx, back))
        ok &= good
        print(f"  {dom:14s} index->value->index exact: {good}")

    print("\n" + "=" * 70)
    print("CATEGORICAL METRICS -- synthetic 7-class check")
    print("=" * 70)
    rng = np.random.default_rng(0)
    ref = rng.integers(1, 8, size=(200, 200))
    pred = ref.copy()
    flip = rng.random(ref.shape) < 0.20
    pred[flip] = rng.integers(1, 8, size=flip.sum())
    cm = confusion_matrix(pred, ref, 7)
    m = categorical_metrics(cm, [n for n, _ in FACIES_LEGEND])
    print(f"  pixel accuracy {m['pixel_accuracy']:.4f}  (expected ~0.83)")
    print(f"  mIoU           {m['mIoU']:.4f}")
    print(f"  macro F1       {m['macro_F1']:.4f}")
    print(f"  classes present: {sum(1 for c in m['per_class'] if c['support_px'] > 0)} of 7")
    ok &= abs(m["pixel_accuracy"] - 0.83) < 0.03

    print("\n  contrast: the OLD metric (argmax over RGB, num_classes=3)")
    old_pred = np.argmax(np.stack([pred, 7 - pred, pred // 2], -1), -1)
    old_ref = np.argmax(np.stack([ref, 7 - ref, ref // 2], -1), -1)
    old_cm = confusion_matrix(old_pred + 1, old_ref + 1, 3)
    old_m = categorical_metrics(old_cm, ["chan0", "chan1", "chan2"])
    print(f"  same data scored by 3-channel argmax: PA {old_m['pixel_accuracy']:.4f}, "
          f"mIoU {old_m['mIoU']:.4f}")
    print("  -> different number, different quantity: not comparable to the above.")

    print("\n" + "=" * 70)
    print("CLUSTERED BOOTSTRAP -- layer-level intervals")
    print("=" * 70)
    per_layer = {46: 0.81, 47: 0.79, 48: 0.88, 49: 0.62, 50: 0.90,
                 51: 0.77, 52: 0.85, 53: 0.71, 54: 0.83}
    ci = clustered_bootstrap_ci(per_layer)
    print(f"  n_layers={ci['n_layers']}  mean={ci['point_estimate']:.4f}  "
          f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  sd={ci['layer_sd']:.4f}")
    print(f"  interval width {ci['ci_high'] - ci['ci_low']:.4f} -- this is the honest")
    print("  precision available from 9 independent layers.")

    other = {k: v - 0.06 for k, v in per_layer.items()}
    pt = paired_permutation_test(per_layer, other)
    print(f"\n  paired permutation test vs a -0.06 shifted model: "
          f"diff={pt['mean_difference']:.4f}, p={pt['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    print("=" * 70)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--encoding-report", action="store_true")
    ap.add_argument("--domain", choices=sorted(CONTINUOUS_DOMAINS))
    ap.add_argument("--task")
    ap.add_argument("--pred-dir")
    ap.add_argument("--ref-dir")
    ap.add_argument("--target-domain", help="'facies' or porosity/permeability/vcl")
    ap.add_argument("--out")
    ap.add_argument("--legend-file",
                    help="legend_manifest.json written by 04_export_common_legend.py; "
                         "makes decoding the exact inverse of the encoding used")
    args = ap.parse_args()

    if args.legend_file:
        load_legend_manifest(args.legend_file)

    if args.self_test:
        return self_test()

    if args.encoding_report:
        doms = [args.domain] if args.domain else sorted(CONTINUOUS_DOMAINS)
        print(json.dumps({d: encoding_floor(d) for d in doms}, indent=2))
        return 0

    if not (args.pred_dir and args.ref_dir and args.target_domain):
        ap.error("--pred-dir, --ref-dir and --target-domain are required")

    res = evaluate_task(args.pred_dir, args.ref_dir, args.target_domain)
    res["task"] = args.task
    text = json.dumps(res, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"written: {args.out}")
    else:
        print(text)
    return 0


# Import-time guards. Both of these encode a bug that actually shipped in this
# project and produced plausible, wrong numbers rather than an error.
_selftest_decoders()
_selftest_metrics()

if __name__ == "__main__":
    sys.exit(main())
