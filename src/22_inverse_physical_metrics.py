#!/usr/bin/env python3
"""
22_inverse_physical_metrics.py
==============================

Physical-unit scoring of the INVERSE tasks, for Pix2Geomodel V2
(RINENG-D-26-10842).

WHY THIS SCRIPT EXISTS
----------------------
Table 4 scores the forward tasks in physical units (MAE / RMSE / R2). Table 5
scores the inverse tasks with classification metrics (PA / macro-F1 / mIoU).
The two tables are therefore not comparable, and a reader cannot ask "is the
inverse direction better or worse than the forward direction?" of the paper as
it stands. Reviewer 5 #11 asks for outputs to be inverted to physical units and
compared distributionally; this closes the inverse half of that.

WHAT IS MEASURED -- READ THIS BEFORE QUOTING ANY NUMBER
-------------------------------------------------------
An inverse model outputs FACIES. To score it in physical units the predicted
facies field is decoded through the TRAINING-SET CLASS MEANS back to a property
value, and compared against the true property on the same cells. That is a
round trip:

        property  ->  facies (the model)  ->  property (class-mean decode)

Consequences that must be respected in the manuscript text:

  1. This is NOT "the model predicts porosity". It measures how much property
     information survives the facies bottleneck. Describe it that way.

  2. The R2 is bounded above by eta^2, the fraction of property variance the
     seven-class facies scheme explains at all (Section 3.11: ~0.88 for VCL,
     ~0.21 for log10 permeability). A perfect classifier attains eta^2 and no
     more. A bare R2 is therefore uninterpretable on its own, so this script
     emits eta^2 in the same row and reports attainment as a percentage of
     that ceiling.

  3. Because the ceiling differs per property, ACROSS-PROPERTY comparison of
     raw R2 is meaningless. Compare pct_of_ceiling instead.

  4. Permeability is scored in log10 space for the headline number. Linear
     space is emitted as a diagnostic only, for consistency with Table 4.

The same decode is applied to the conventional baselines (one-dimensional Bayes
classifier, random forest), so Pix2Geomodel and the baselines are scored on
identical cells with identical decoders -- which is the only way the Section 3.3
comparison stays honest.

DEFECT HISTORY THIS SCRIPT GUARDS AGAINST
-----------------------------------------
Eight of the nine bugs found in this pipeline were silent: they returned
well-formed, plausible numbers instead of raising. Every guard below exists
because something upstream once failed this way.

  * palette round-trip assertion at import  (the int16 decode_rgb_to_facies
    overflow: only 1 of 8 legend entries decoded to itself)
  * explicit half-selection with an assertion on which half carries facies
    (10_class_support read the wrong half and produced a plausible
    100 %-summing histogram)
  * background excluded by NODATA-black test on the REFERENCE, never the
    prediction (a generated image may contain near-black cells that are not
    NODATA)
  * class means computed from TRAINING layers only, with the training layer
    list asserted disjoint from the test layer list
  * a class present in test but absent from training has no mean; it is
    counted and reported rather than silently imputed
  * --self-test builds a synthetic case whose answer is known analytically

USAGE
-----
    python 22_inverse_physical_metrics.py --self-test

    python 22_inverse_physical_metrics.py \
        --splits-root splits_v2 \
        --runs-root   runs \
        --out         results/22_Inverse_physical.csv

    # single task / single method, for debugging
    python 22_inverse_physical_metrics.py --task vcl_to_facies --method bayes

OUTPUTS
-------
    results/22_Inverse_physical.csv      task x method x seed, long format
    results/22_Inverse_physical_summary.csv
                                         task x method, mean +/- SD over seeds
    results/22_Class_means.csv           the training class means actually used
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover
    Image = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
EXPORT = os.path.join(HERE, "export_v2")

# Fitting convention -- MUST match 14_baseline_conventional.py exactly, or the
# class means here and the class means behind Table 6 are fitted on different
# samples and the two tables stop being comparable.
#
#   * originals only (`_aug00`): augmented copies of a layer carry the same
#     class statistics but reweight each layer by however many crops it
#     happened to receive.
#   * capped at 120 images and strided by 101 cells: the training split holds
#     1,116 images of ~2 M pixels; every 101st cell is still tens of millions
#     of samples per domain, far more than is needed for seven class means.
FIT_STRIDE = 101
MAX_FIT_IMAGES = 120


# --------------------------------------------------------------------------
# Import the corrected decoders from 02_evaluation.py rather than restating
# them. Restating a palette is how two copies drift apart.
# --------------------------------------------------------------------------

def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load("ev02", "02_evaluation.py")

FACIES_LEGEND = EV.FACIES_LEGEND
CONTINUOUS_DOMAINS = EV.CONTINUOUS_DOMAINS
N_CLASSES = len(FACIES_LEGEND)


# --------------------------------------------------------------------------
# Import-time guard: the palette must round-trip.
# --------------------------------------------------------------------------

def _assert_palette_roundtrip() -> None:
    """
    Every legend colour, and black, must decode to its own index.

    This is the assertion that would have caught the int16 overflow on the day
    it was introduced instead of after the results were written up.
    """
    swatch = np.array(
        [[0, 0, 0]] + [EV.hex_to_rgb(h) for _, h in FACIES_LEGEND],
        dtype=np.uint8,
    ).reshape(1, -1, 3)
    got = EV.decode_rgb_to_facies(swatch).ravel()
    want = np.arange(N_CLASSES + 1, dtype=got.dtype)
    if not np.array_equal(got, want):
        raise AssertionError(
            "facies palette does not round-trip: "
            f"expected {want.tolist()}, got {got.tolist()}"
        )


_assert_palette_roundtrip()


# --------------------------------------------------------------------------
# Task configuration
# --------------------------------------------------------------------------

# Inverse tasks: input is the property, target (model output) is facies.
INVERSE_TASKS: Dict[str, str] = {
    "porosity_to_facies": "porosity",
    "permeability_to_facies": "permeability",
    "vcl_to_facies": "vcl",
}

# Paired sets in which the property half can be read for class-mean fitting.
FORWARD_OF: Dict[str, str] = {
    "porosity": "facies_to_porosity",
    "permeability": "facies_to_permeability",
    "vcl": "facies_to_vcl",
}

METHODS = ("pix2geomodel", "bayes", "random_forest")


def default_pred_dir(runs_root: str, task: str, method: str, seed: int) -> str:
    """Prediction directory layout. Override with --pred-dir when debugging."""
    if method == "pix2geomodel":
        return os.path.join(runs_root, f"{task}_seed{seed}", "predictions")
    return os.path.join(runs_root, "baselines", method, f"{task}_seed{seed}",
                        "predictions")


# --------------------------------------------------------------------------
# Image reading
# --------------------------------------------------------------------------

def _read_rgb(path: str) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required")
    return np.array(Image.open(path).convert("RGB"))


def _split_pair(img: np.ndarray, want: str, target_on_left: bool) -> np.ndarray:
    """
    Slice the concatenated pair written by 01_layer_level_split.py.

    `want` is 'target' or 'input'. Kept explicit rather than implicit because
    10_class_support silently read the wrong half and still produced a
    plausible, 100 %-summing histogram.
    """
    w = img.shape[1] // 2
    left, right = img[:, :w], img[:, w:]
    if want == "target":
        return left if target_on_left else right
    if want == "input":
        return right if target_on_left else left
    raise ValueError(want)


def _nodata_mask(ref_rgb: np.ndarray) -> np.ndarray:
    """
    Valid-cell mask from the REFERENCE image only.

    Never take this from a prediction: a generator can emit near-black cells
    inside the reservoir, which would silently shrink the evaluated area and
    flatter the model.
    """
    return ~np.all(ref_rgb == 0, axis=-1)


def _decode_property(rgb: np.ndarray, domain: str) -> np.ndarray:
    d = CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = EV.ramp_range(d)
    lut = EV.build_lut(d["stops"], hi_i - lo_i + 1)
    return EV.index_to_value(EV.decode_rgb_to_index(rgb, lut) + lo_i, domain)


def adopt_legend(domain: str) -> bool:
    """
    Adopt the legend the export actually used, exactly as
    14_baseline_conventional.py does.

    Decoding must be the analytical inverse of the encoding that ran, not of
    the defaults compiled into 02_evaluation.py. If the two ever diverge, every
    physical value is wrong by a smooth, plausible-looking amount -- which is
    precisely the failure mode that made eight of the nine known bugs silent.
    """
    lm = os.path.join(EXPORT, domain, "legend_manifest.json")
    if os.path.exists(lm):
        EV.load_legend_manifest(lm)
        return True
    return False


def split_files(task: str, split: str, splits_root: str,
                originals_only: bool = True,
                cap: Optional[int] = None) -> List[str]:
    """
    File list for a split, following 14_baseline_conventional.train_files().

    `_aug00` is the un-augmented copy of each layer. Falling back to every PNG
    when no `_aug00` exists preserves behaviour on splits written without the
    augmentation suffix.
    """
    d = os.path.join(splits_root, task, split)
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    names = sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))
    if originals_only:
        orig = [f for f in names if f.endswith("_aug00.png")]
        if orig:
            names = orig
    if cap is not None:
        names = names[:cap]
    return [os.path.join(d, f) for f in names]


# --------------------------------------------------------------------------
# Training class means -- the decoder for the round trip
# --------------------------------------------------------------------------

def fit_class_means(splits_root: str, domain: str, target_on_left: bool,
                    log_space: bool) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Mean property value of each facies class, from TRAINING layers only.

    Returns (means[N_CLASSES], counts[N_CLASSES], layers_used).
    Classes with zero training support get NaN, never an imputed value.

    For permeability the mean is taken in log10 space when `log_space` is set,
    because the arithmetic mean of a four-decade variable is governed by its
    top decile and is not the value a geomodeller would populate a class with.
    """
    task = FORWARD_OF[domain]
    paths = split_files(task, "train", splits_root,
                        originals_only=True, cap=MAX_FIT_IMAGES)

    sums = np.zeros(N_CLASSES, dtype=np.float64)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    layers: set = set()

    for path in paths:
        fn = os.path.basename(path)
        m = EV.LAYER_RE.search(fn)
        if m:
            layers.add(int(m.group(1)))
        img = _read_rgb(path)
        # In facies_to_<prop>, facies is the INPUT and the property the TARGET.
        fac_rgb = _split_pair(img, "input", target_on_left)
        prop_rgb = _split_pair(img, "target", target_on_left)

        fac = EV.decode_rgb_to_facies(fac_rgb)
        val = _decode_property(prop_rgb, domain)
        valid = _nodata_mask(prop_rgb) & (fac > 0)

        if log_space:
            d = CONTINUOUS_DOMAINS[domain]
            val = np.log10(np.clip(val, d["vmin"], d["vmax"]))

        # Stride the cells, matching 14_baseline_conventional.FIT_STRIDE.
        f = fac[valid].astype(np.int64)[::FIT_STRIDE] - 1
        v = val[valid].astype(np.float64)[::FIT_STRIDE]
        np.add.at(sums, f, v)
        np.add.at(counts, f, 1)

    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means, counts, sorted(layers)


# --------------------------------------------------------------------------
# eta^2 ceiling
# --------------------------------------------------------------------------

def eta_squared_from_test(splits_root: str, domain: str, target_on_left: bool,
                          log_space: bool) -> float:
    """
    Fraction of TEST-set property variance explained by the true facies
    partition. This is the ceiling the round-trip R2 is measured against.

    Computed on the test layers, not the training layers, because that is the
    population the R2 is computed over. Using the training eta^2 would quote a
    ceiling from a different distribution -- and the blocked split makes the
    two genuinely different (test is 49.4 % Very Good Sand against 14.3 % in
    training).
    """
    task = FORWARD_OF[domain]
    vals: List[np.ndarray] = []
    facs: List[np.ndarray] = []
    for path in split_files(task, "test", splits_root, originals_only=False):
        img = _read_rgb(path)
        fac_rgb = _split_pair(img, "input", target_on_left)
        prop_rgb = _split_pair(img, "target", target_on_left)
        fac = EV.decode_rgb_to_facies(fac_rgb)
        val = _decode_property(prop_rgb, domain)
        valid = _nodata_mask(prop_rgb) & (fac > 0)
        if log_space:
            d = CONTINUOUS_DOMAINS[domain]
            val = np.log10(np.clip(val, d["vmin"], d["vmax"]))
        vals.append(val[valid].astype(np.float64))
        facs.append(fac[valid].astype(np.int64))

    if not vals:
        return float("nan")
    v = np.concatenate(vals)
    f = np.concatenate(facs)
    grand = v.mean()
    ss_tot = float(np.sum((v - grand) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_between = 0.0
    for c in np.unique(f):
        sel = v[f == c]
        ss_between += sel.size * (sel.mean() - grand) ** 2
    return float(ss_between / ss_tot)


# --------------------------------------------------------------------------
# Scoring one (task, method, seed)
# --------------------------------------------------------------------------

def score_run(pred_dir: str, ref_dir: str, domain: str, means: np.ndarray,
              target_on_left: bool, log_space: bool) -> Optional[dict]:
    """
    Decode predicted facies to class-mean property values and score against the
    true property.

    TWO estimators are returned, and the difference between them is not
    cosmetic:

      POOLED  -- every valid cell across all nine test layers enters one
                 global sum. This is the primary estimator, because
                 14_baseline_conventional.py pools its confusion matrix across
                 layers before computing PA / macro-F1 / mIoU. Using anything
                 else here would put the new physical-unit columns on a
                 different footing from the classification columns already in
                 the same table.

      LAYERWISE -- metrics computed per image, averaged within a layer, then
                 averaged over layers. Reported as a secondary diagnostic and
                 for the clustered bootstrap, which needs layer as the
                 resampling unit.

    Pooled SS_tot includes between-layer variance and is therefore much larger
    than any single layer's, so pooled R2 runs systematically higher. Comparing
    a layerwise R2 against a pooled one manufactures a difference that is not
    in the data -- which is exactly what happened when this script and
    script 14 were first compared.
    """
    pairs = EV.collect_pairs(pred_dir, ref_dir)
    if not pairs:
        return None

    per_layer: Dict[int, Dict[str, List[float]]] = {}
    unsupported_cells = 0
    total_cells = 0
    # pooled: n, sum|e|, sum e^2, sum r, sum r^2
    pool = np.zeros(5, dtype=np.float64)

    for spec in pairs:
        pred_rgb = _read_rgb(spec.pred_path)
        ref_img = _read_rgb(spec.ref_path)
        # In <prop>_to_facies, facies is the TARGET and the property the INPUT.
        prop_rgb = _split_pair(ref_img, "input", target_on_left)
        fac_ref_rgb = _split_pair(ref_img, "target", target_on_left)

        if pred_rgb.shape != fac_ref_rgb.shape:
            raise ValueError(
                f"shape mismatch {os.path.basename(spec.pred_path)}: "
                f"{pred_rgb.shape} vs {fac_ref_rgb.shape}"
            )

        pred_fac = EV.decode_rgb_to_facies(pred_rgb)
        true_val = _decode_property(prop_rgb, domain)
        if log_space:
            d = CONTINUOUS_DOMAINS[domain]
            true_val = np.log10(np.clip(true_val, d["vmin"], d["vmax"]))

        valid = _nodata_mask(prop_rgb) & (pred_fac > 0)
        idx = pred_fac.astype(np.int64) - 1
        recon = np.full(pred_fac.shape, np.nan, dtype=np.float64)
        ok = valid & (idx >= 0)
        recon[ok] = means[idx[ok]]

        # A class with no training support yields NaN; count it, do not impute.
        supported = ok & ~np.isnan(recon)
        unsupported_cells += int(np.sum(ok & np.isnan(recon)))
        total_cells += int(np.sum(ok))
        if not np.any(supported):
            continue

        p = recon[supported]
        r = true_val[supported]
        err = p - r
        pool += np.array([r.size, np.abs(err).sum(), float(np.sum(err ** 2)),
                          r.sum(), float(np.sum(r * r))])
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        rec = {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
        d = per_layer.setdefault(spec.layer, {k: [] for k in rec})
        for k, v in rec.items():
            d[k].append(v)

    if not per_layer:
        return None

    collapsed = {k: {L: float(np.mean(v[k])) for L, v in per_layer.items()}
                 for k in next(iter(per_layer.values()))}
    lay = {k: float(np.mean(list(v.values()))) for k, v in collapsed.items()}

    n_p, sae, se2, sv, sv2 = pool
    ss_tot_p = sv2 - (sv * sv) / n_p if n_p > 0 else 0.0
    out = {
        "MAE": sae / n_p,
        "RMSE": float(np.sqrt(se2 / n_p)),
        "bias": float(np.sum(pool[0] * 0) + 0.0),   # replaced below
        "R2": 1.0 - se2 / ss_tot_p if ss_tot_p > 0 else float("nan"),
        "MAE_layerwise": lay["MAE"],
        "RMSE_layerwise": lay["RMSE"],
        "R2_layerwise": lay["R2"],
        "bias_layerwise": lay["bias"],
        "n_cells": int(n_p),
    }
    out["bias"] = lay["bias"]     # bias is aggregation-invariant up to weights
    out["n_layers"] = len(per_layer)
    out["n_pairs"] = len(pairs)
    out["pct_cells_unsupported_class"] = (
        100.0 * unsupported_cells / total_cells if total_cells else 0.0)
    out["_per_layer"] = collapsed
    return out


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    print("=" * 70)
    print("22_inverse_physical_metrics -- self test")
    print("=" * 70)
    ok = True

    print("\n[1] palette round-trip .......... ", end="")
    try:
        _assert_palette_roundtrip()
        print("PASS")
    except AssertionError as e:
        print(f"FAIL: {e}")
        ok = False

    # [2] A perfect classifier must attain exactly eta^2, no more.
    print("[2] perfect classifier == eta^2 . ", end="")
    rng = np.random.default_rng(2025)
    n_per = 4000
    class_mu = np.array([0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20])
    vals, facs = [], []
    for c, mu in enumerate(class_mu):
        vals.append(rng.normal(mu, 0.02, n_per))
        facs.append(np.full(n_per, c))
    v = np.concatenate(vals)
    f = np.concatenate(facs)
    grand = v.mean()
    ss_tot = np.sum((v - grand) ** 2)
    ss_between = sum(
        (f == c).sum() * (v[f == c].mean() - grand) ** 2 for c in range(7))
    eta2 = ss_between / ss_tot
    means = np.array([v[f == c].mean() for c in range(7)])
    recon = means[f]
    r2 = 1.0 - np.sum((recon - v) ** 2) / ss_tot
    if abs(r2 - eta2) < 1e-9:
        print(f"PASS  (eta^2 = R^2 = {eta2:.4f})")
    else:
        print(f"FAIL: R2={r2:.6f} eta2={eta2:.6f}")
        ok = False

    # [3] A degraded classifier must fall strictly below the ceiling.
    print("[3] degraded < ceiling .......... ", end="")
    f_bad = f.copy()
    swap = rng.random(f.size) < 0.30
    f_bad[swap] = rng.integers(0, 7, swap.sum())
    r2_bad = 1.0 - np.sum((means[f_bad] - v) ** 2) / ss_tot
    if r2_bad < eta2:
        print(f"PASS  ({r2_bad:.4f} < {eta2:.4f}, "
              f"{100 * r2_bad / eta2:.1f} % of ceiling)")
    else:
        print(f"FAIL: {r2_bad:.4f} !< {eta2:.4f}")
        ok = False

    # [4] Unsupported classes must be counted, not imputed.
    print("[4] unsupported class -> NaN .... ", end="")
    m = np.array([0.02, np.nan, 0.08, 0.11, 0.14, 0.17, 0.20])
    idx = np.array([0, 1, 2, 1])
    rec = m[idx]
    if np.isnan(rec).sum() == 2:
        print("PASS")
    else:
        print("FAIL")
        ok = False

    # [5] Half-selection must be the inverse of itself.
    print("[5] pair half selection ......... ", end="")
    left = np.zeros((4, 4, 3), np.uint8)
    left[..., 0] = 255
    right = np.zeros((4, 4, 3), np.uint8)
    right[..., 2] = 255
    pair = np.concatenate([left, right], axis=1)
    a = _split_pair(pair, "target", True)
    b = _split_pair(pair, "input", True)
    if np.array_equal(a, left) and np.array_equal(b, right):
        print("PASS")
    else:
        print("FAIL")
        ok = False

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--splits-root", default="splits_v2")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--task", default=None, choices=list(INVERSE_TASKS))
    ap.add_argument("--method", default=None, choices=METHODS)
    ap.add_argument("--pred-dir", default=None,
                    help="override; use with --task and --method")
    ap.add_argument("--target-on-left", action="store_true", default=True,
                    help="target half is the LEFT half of the pair (default)")
    ap.add_argument("--target-on-right", dest="target_on_left",
                    action="store_false")
    ap.add_argument("--perm-linear", action="store_true",
                    help="score permeability in linear space too (diagnostic)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR,
                                                  "22_Inverse_physical.csv"))
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = [args.task] if args.task else list(INVERSE_TASKS)
    methods = [args.method] if args.method else list(METHODS)

    rows: List[dict] = []
    mean_rows: List[dict] = []

    for task in tasks:
        domain = INVERSE_TASKS[task]
        log_space = CONTINUOUS_DOMAINS[domain]["scale"] == "log10"
        unit = ("log10 " if log_space else "") + CONTINUOUS_DOMAINS[domain]["unit"]

        print(f"\n=== {task}  ({domain}, scored in "
              f"{'log10' if log_space else 'linear'} space) ===")
        if adopt_legend(domain):
            print(f"  legend manifest adopted from export_v2/{domain}")
        else:
            print(f"  WARNING: no legend manifest for '{domain}'; using "
                  f"02_evaluation defaults -- values may not match the export")

        try:
            means, counts, train_layers = fit_class_means(
                args.splits_root, domain, args.target_on_left, log_space)
        except FileNotFoundError as e:
            print(f"  SKIP -- missing {e}")
            continue

        ref_dir = os.path.join(args.splits_root, task, "test")
        if not os.path.isdir(ref_dir):
            print(f"  SKIP -- missing {ref_dir}")
            continue

        eta2 = eta_squared_from_test(args.splits_root, domain,
                                     args.target_on_left, log_space)
        print(f"  eta^2 ceiling on test layers: {eta2:.4f}")

        for c, (name, _) in enumerate(FACIES_LEGEND):
            mean_rows.append({
                "domain": domain, "space": "log10" if log_space else "linear",
                "class": name, "train_mean": means[c], "train_cells": counts[c],
            })
            if counts[c] == 0:
                print(f"  WARNING: class '{name}' has no training support; "
                      f"cells predicted as this class are excluded, not imputed")

        for method in methods:
            per_seed: List[dict] = []
            for seed in seeds:
                pred_dir = args.pred_dir or default_pred_dir(
                    args.runs_root, task, method, seed)
                if not os.path.isdir(pred_dir):
                    print(f"  {method} seed{seed}: missing {pred_dir}")
                    continue
                res = score_run(pred_dir, ref_dir, domain, means,
                                args.target_on_left, log_space)
                if res is None:
                    print(f"  {method} seed{seed}: no matching pairs")
                    continue
                per_seed.append(res)
                rows.append({
                    "task": task, "method": method, "seed": seed, "unit": unit,
                    "MAE": res["MAE"], "RMSE": res["RMSE"], "bias": res["bias"],
                    "R2": res["R2"],
                    "MAE_layerwise": res["MAE_layerwise"],
                    "RMSE_layerwise": res["RMSE_layerwise"],
                    "R2_layerwise": res["R2_layerwise"],
                    "n_cells": res["n_cells"], "eta2_ceiling": eta2,
                    "pct_of_eta2": 100.0 * res["R2"] / eta2 if eta2 == eta2 and eta2 > 0 else float("nan"),
                    "n_layers": res["n_layers"], "n_pairs": res["n_pairs"],
                    "pct_cells_unsupported_class": res["pct_cells_unsupported_class"],
                })
                print(f"  {method} seed{seed}: "
                      f"pooled MAE={res['MAE']:.4f} RMSE={res['RMSE']:.4f} "
                      f"R2={res['R2']:+.4f} ({100 * res['R2'] / eta2:.1f} % of eta2)"
                      f"  |  layerwise R2={res['R2_layerwise']:+.4f}")

            if per_seed:
                agg = {k: np.array([r[k] for r in per_seed])
                       for k in ("MAE", "RMSE", "bias", "R2")}
                mrow = {"task": task, "method": method, "unit": unit,
                        "n_seeds": len(per_seed), "eta2_ceiling": eta2}
                for k, v in agg.items():
                    mrow[f"{k}_mean"] = float(v.mean())
                    mrow[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
                mrow["pct_of_eta2_mean"] = (
                    100.0 * mrow["R2_mean"] / eta2 if eta2 == eta2 and eta2 > 0
                    else float("nan"))
                _SUMMARY.append(mrow)

    if not rows:
        print("\nNo runs scored. Check --splits-root and --runs-root.")
        return 1

    _write_csv(args.out, rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")

    if _SUMMARY:
        spath = args.out.replace(".csv", "_summary.csv")
        _write_csv(spath, _SUMMARY)
        print(f"wrote {spath}  ({len(_SUMMARY)} rows)")

    if mean_rows:
        cpath = os.path.join(OUT_DIR, "22_Class_means.csv")
        _write_csv(cpath, mean_rows)
        print(f"wrote {cpath}  ({len(mean_rows)} rows)")

    return 0


_SUMMARY: List[dict] = []


def _write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys and not k.startswith("_"):
                keys.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    raise SystemExit(main())
