#!/usr/bin/env python3
"""
09_spatial_validation.py
========================

Spatial-structure validation of the generated fields.

Addresses the spatial-realism comments: variogram agreement alone does not
establish geological realism, because realisations with similar second-order
statistics can differ in connectivity and architecture. This module computes,
for reference and prediction alike:

  1. DIRECTIONAL semivariograms at 0, 45, 90 and 135 degrees, with 95 %
     uncertainty envelopes obtained by bootstrapping over layers.
  2. Per-direction length scales, so anisotropy is compared numerically
     rather than visually: the half-sill range h50 (non-parametric, the
     primary measure) and a fitted exponential nugget/sill/range (parametric,
     secondary). Both carry an explicit resolved/unresolved flag, and the
     anisotropy ratio is suppressed rather than quoted when either direction
     fails to reach its threshold inside the lag window.
  3. CONNECTED-COMPONENT statistics per facies class: body count, mean body
     size, and the fraction of the class held in its largest body.
  4. TRANSITION-PROBABILITY matrices between adjacent facies cells,
     horizontally and vertically.
  5. WASSERSTEIN distance between the marginal distributions of reference and
     prediction.

LAG UNITS. Lags are expressed in GRID CELLS, not pixels. The export resizes
each slice from 8184 x 12045 px to 1024 x 2048 px, which is not an isotropic
rescaling: the horizontal axis is compressed roughly 26 % more than the
vertical. Measuring lags in pixels would therefore manufacture apparent
anisotropy. Pixels-per-cell is derived separately for each axis from the grid
dimensions recorded in the legend manifest.

CPU only. Run in the `base` environment.

    python 09_spatial_validation.py
    python 09_spatial_validation.py --tasks facies_to_porosity --seeds 1
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
EXPORT = os.path.join(HERE, "export_v2")
OUT = os.path.join(HERE, "results")

TASKS = [
    ("facies_to_porosity",     "porosity"),
    ("facies_to_permeability", "permeability"),
    ("facies_to_vcl",          "vcl"),
    ("porosity_to_facies",     "facies"),
    ("permeability_to_facies", "facies"),
    ("vcl_to_facies",          "facies"),
]

# DIRECTION LABELS. Each image is a constant-depth (constant-k) map slice of
# the cube, so BOTH image axes are lateral: image columns run along grid j,
# image rows along grid i (cube_shape (i, j, k) = (365, 248, 55); the export
# does `cube[:, :, k]` with no transpose, and PIL reads (rows, cols) as
# height x width). Neither axis is depth. Calling 90 degrees "vertical" would
# be read as across-bedding, and would make an areal ratio below 1 look
# physically impossible for a layered reservoir; it is not, because this is
# lateral anisotropy in the map plane.
DIRECTIONS = {"0 (along grid j)": 0.0, "45 (diagonal)": 45.0,
              "90 (along grid i)": 90.0, "135 (anti-diagonal)": 135.0}

# Lag window, in GRID CELLS. The Yemen grid is 248 x 365 cells, so the
# conventional ceiling of about one third of the field extent puts the maximum
# usable lag near 80. A 1-25 window (the first version of this module) is
# shorter than the correlation length of the well-connected directions, so
# their variograms never reach the sill inside the window, their ranges are
# unresolvable, and any anisotropy ratio computed from them is compressed
# towards 1 -- on a synthetic field with a true 4:1 anisotropy the measured
# ratio came out 1.6. The stride widens beyond lag 20 because the near-origin
# part of the curve carries the nugget and the far part only the plateau.
LAGS_CELLS = np.concatenate([np.arange(1, 21), np.arange(22, 81, 2)])


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
# Variography
# --------------------------------------------------------------------------

def directional_variogram(field: np.ndarray, valid: np.ndarray,
                          lags: Sequence[int], angle_deg: float,
                          px_per_cell_x: float, px_per_cell_y: float) -> np.ndarray:
    """
    Semivariance gamma(h) along `angle_deg`, with h in GRID CELLS.

    gamma(h) = 0.5 * mean over valid pairs of (z(x) - z(x+h))^2
    """
    th = np.deg2rad(angle_deg)
    out = np.full(len(lags), np.nan)
    ny, nx = field.shape
    for i, h in enumerate(lags):
        ox = int(round(h * np.cos(th) * px_per_cell_x))
        oy = int(round(h * np.sin(th) * px_per_cell_y))
        if ox == 0 and oy == 0:
            continue
        if abs(ox) >= nx or abs(oy) >= ny:
            continue
        a = field[max(0, -oy):ny - max(0, oy), max(0, -ox):nx - max(0, ox)]
        b = field[max(0, oy):ny - max(0, -oy), max(0, ox):nx - max(0, -ox)]
        va = valid[max(0, -oy):ny - max(0, oy), max(0, -ox):nx - max(0, ox)]
        vb = valid[max(0, oy):ny - max(0, -oy), max(0, ox):nx - max(0, -ox)]
        m = va & vb
        if m.sum() < 100:
            continue
        d = a[m].astype(np.float64) - b[m].astype(np.float64)
        out[i] = 0.5 * np.mean(d * d)
    return out


def fit_exponential(lags: np.ndarray, gamma: np.ndarray,
                    sill_hint: Optional[float] = None) -> Dict[str, float]:
    """
    Least-squares fit of gamma(h) = nugget + (sill - nugget) * (1 - exp(-3h/range))
    by grid search. Robust and dependency-free; the fitted numbers are used for
    comparison between reference and prediction, not for kriging.

    `sill_hint` should be the VARIANCE of the field, which is the theoretical
    sill of a stationary variogram. It matters: without it the sill is inferred
    from the maximum of each directional curve, and a direction whose
    correlation length exceeds the lag range never plateaus, so it is handed a
    sill far below the true variance and the fit compensates by inflating the
    range. On a test field with a true 4:1 anisotropy that error compressed the
    measured ratio to 1.3.
    """
    ok = np.isfinite(gamma)
    if ok.sum() < 4:
        return {"nugget": float("nan"), "sill": float("nan"),
                "range": float("nan"), "range_unresolved": True}
    x, y = lags[ok].astype(float), gamma[ok].astype(float)
    best, params = np.inf, (np.nan, np.nan, np.nan)
    anchor = float(sill_hint) if (sill_hint and np.isfinite(sill_hint) and sill_hint > 0) \
        else max(y.max(), 1e-12)
    lo_s, hi_s = (0.7 * anchor, 1.3 * anchor) if sill_hint else (0.4 * anchor, 1.6 * anchor)
    smax = anchor
    # The search may run to twice the widest lag so that an over-long range is
    # visibly over-long rather than silently pinned to the window edge, but
    # anything beyond the widest lag is extrapolation and is flagged below.
    h_max = float(lags.max())
    r_hi = h_max * 2.0
    # Log spacing: the fit is far more sensitive to the range at short
    # correlation lengths than at long ones, and a linear grid over a 200-cell
    # interval leaves only ~2-cell resolution where it matters most.
    for rng in np.geomspace(0.5, r_hi, 160):
        for sill in np.linspace(lo_s, hi_s, 25):
            for nug in np.linspace(0.0, 0.6 * smax, 13):
                if nug >= sill:
                    continue
                pred = nug + (sill - nug) * (1.0 - np.exp(-3.0 * x / rng))
                sse = float(np.sum((pred - y) ** 2))
                if sse < best:
                    best, params = sse, (nug, sill, rng)
    # A fitted range beyond the widest lag actually sampled is an
    # extrapolation, not a measurement: the data contain no evidence of where
    # that curve plateaus. It is reported as a lower bound and every quantity
    # derived from it (anisotropy ratio, reference-vs-prediction difference)
    # is suppressed rather than quoted.
    unresolved = bool(params[2] >= h_max)
    return {"nugget": params[0], "sill": params[1], "range": params[2],
            "range_unresolved": unresolved}


def practical_range(lags: np.ndarray, gamma: np.ndarray,
                    sill_hint: Optional[float] = None,
                    frac: float = 0.50) -> Dict[str, float]:
    """
    Half-sill range h50: the lag at which gamma(h) first reaches `frac` of the
    sill, by linear interpolation between the two bracketing lags.

    This is reported alongside the exponential fit, and is the quantity used
    for the anisotropy ratio. It assumes no model shape, so it does not
    inherit the fitted model's sensitivity to the nugget/sill trade-off, and
    it cannot silently extrapolate: if the curve never reaches the threshold
    inside the lag window the result is flagged `unresolved` and the returned
    range is the widest lag, i.e. an explicit lower bound.

    Why the threshold is half the sill and not 95 % of it. Near the sill the
    empirical variogram is flat, so a small vertical error maps to a large
    horizontal one, and at long lags it is estimated from the fewest pairs.
    Measured on synthetic fields with a known 4:1 anisotropy, a 0.95 threshold
    recovers the ratio to within 5 % in one orientation but errs by 88 % in
    the transposed one, purely from which noisy lag first pokes above the
    line; at 0.50 the same three controls (ratios 4.0, 1.0, 0.25) return
    4.15, 1.08 and 0.27 -- within 8 % and, importantly, self-consistent under
    transposition. h50 is proportional to the correlation length for a fixed
    model shape, so the RATIO between directions is the meaningful quantity;
    the absolute value is a robust length scale in grid cells and is NOT the
    exponential range (converting between them would assume the model shape).
    """
    ok = np.isfinite(gamma)
    if ok.sum() < 3:
        return {"range": float("nan"), "unresolved": True, "target": float("nan"),
                "rel_nugget": float("nan"), "nugget_dominated": True}
    x, y = lags[ok].astype(float), gamma[ok].astype(float)
    sill = float(sill_hint) if (sill_hint and np.isfinite(sill_hint) and sill_hint > 0) \
        else float(np.max(y))
    target = frac * sill
    # Relative nugget, measured directly as gamma at the shortest lag over the
    # sill. When most of the variance is already present at one cell the field
    # has no resolvable structure, and h50 stops being a correlation length --
    # it becomes an arbitrary crossing of a nearly flat curve. Reporting it as
    # a length scale in that regime would be a category error, so it is
    # flagged and every derived ratio is suppressed.
    rel_nugget = float(y[0] / sill) if sill > 0 else float("nan")
    nugget_dominated = bool(np.isfinite(rel_nugget) and rel_nugget >= 0.75)
    # Enforce monotonicity before crossing detection: an empirical variogram is
    # noisy at long lags (few pairs) and can dip back below the threshold after
    # crossing it, which would make the first crossing depend on noise.
    ymono = np.maximum.accumulate(y)
    base = {"target": target, "rel_nugget": rel_nugget,
            "nugget_dominated": nugget_dominated}
    above = np.nonzero(ymono >= target)[0]
    if above.size == 0:
        return {"range": float(x.max()), "unresolved": True, **base}
    j = int(above[0])
    if j == 0:
        return {"range": float(x[0]), "unresolved": nugget_dominated, **base}
    x0, x1, y0, y1 = x[j - 1], x[j], ymono[j - 1], ymono[j]
    r = x0 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return {"range": float(r), "unresolved": nugget_dominated, **base}


# --------------------------------------------------------------------------
# Facies architecture
# --------------------------------------------------------------------------

def connected_components(labels: np.ndarray, cls: int) -> Dict[str, float]:
    """Body count, mean body size and largest-body fraction for one class."""
    mask = labels == cls
    n_px = int(mask.sum())
    if n_px == 0:
        return {"n_bodies": 0, "mean_body_px": 0.0, "largest_fraction": 0.0,
                "class_px": 0}
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        sizes = np.bincount(lab.ravel())[1:]
    except Exception:                                  # dependency-free fallback
        sizes, n = _components_bfs(mask)
    if len(sizes) == 0:
        return {"n_bodies": 0, "mean_body_px": 0.0, "largest_fraction": 0.0,
                "class_px": n_px}
    return {"n_bodies": int(len(sizes)), "mean_body_px": float(np.mean(sizes)),
            "largest_fraction": float(np.max(sizes) / n_px), "class_px": n_px}


def _components_bfs(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    ny, nx = mask.shape
    for sy in range(ny):
        for sx in range(nx):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack, size = [(sy, sx)], 0
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            sizes.append(size)
    return np.array(sizes), len(sizes)


def transition_matrix(labels: np.ndarray, n_cls: int, axis: int) -> np.ndarray:
    """Row-normalised transition probabilities between adjacent cells."""
    a = labels[:-1, :] if axis == 0 else labels[:, :-1]
    b = labels[1:, :] if axis == 0 else labels[:, 1:]
    m = (a >= 1) & (a <= n_cls) & (b >= 1) & (b <= n_cls)
    idx = (a[m] - 1) * n_cls + (b[m] - 1)
    cm = np.bincount(idx, minlength=n_cls ** 2).reshape(n_cls, n_cls).astype(float)
    return cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)


def wasserstein1(a: np.ndarray, b: np.ndarray, n: int = 4000) -> float:
    """1-D Wasserstein distance via quantile matching."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def px_per_cell(img_shape: Tuple[int, int], grid: Optional[Sequence[int]]) -> Tuple[float, float]:
    """Pixels per grid cell on each axis, so lags can be expressed in cells."""
    ny, nx = img_shape
    if not grid or len(grid) < 2:
        return 1.0, 1.0
    ni, nj = int(grid[0]), int(grid[1])       # cube is (i, j, k); image is (i rows, j cols)
    return nx / max(nj, 1), ny / max(ni, 1)   # x uses j, y uses i


def decode(img: np.ndarray, domain: str, ev,
           log_perm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode a rendered slice to physical values plus a validity mask.

    Permeability is returned as log10(k) by default. Two reasons, and the
    second is the one that matters here:

      * permeability is log-normal over four decades in this reservoir, so a
        variogram computed on raw mD is dominated by a handful of high-k
        cells and is not the quantity geostatistics normally models;
      * the colour ramp for permeability is itself logarithmic, so ONE index
        of colour error near the top of the ramp is worth thousands of mD.
        Computing the variogram in linear mD therefore converts ordinary
        quantisation error into what looks like a pure nugget. Any claim that
        a predicted permeability field is spatially unstructured has to be
        made in log space or it may be an artifact of the decode rather than
        a property of the model.

    Pass log_perm=False to reproduce the linear-mD behaviour for comparison.
    """
    if domain == "facies":
        lab = ev.decode_rgb_to_facies(img)
        return lab.astype(np.float64), lab >= 1
    spec = ev.CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = ev.ramp_range(spec)
    lut = ev.build_lut(spec["stops"], hi_i - lo_i + 1)
    idx = ev.decode_rgb_to_index(img, lut) + lo_i
    val = ev.index_to_value(idx, domain).astype(np.float64)
    valid = ~np.all(img == 0, axis=-1)
    if domain == "permeability" and log_perm:
        # The reserved zero class (index 0, impermeable) has no logarithm and
        # is excluded from the mask rather than clipped to a floor, which
        # would plant a spike of identical values at the low end.
        valid = valid & (val > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.log10(np.where(val > 0, val, 1.0))
    return val, valid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=[t for t, _ in TASKS])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ev = _load("pix2geo_eval", "02_evaluation.py")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    vario_rows = [["Task", "Domain", "Direction", "Lag (cells)",
                   "Reference gamma", "Predicted gamma",
                   "Ref 95% low", "Ref 95% high",
                   "Pred 95% low", "Pred 95% high"]]
    fit_rows = [["Task", "Direction", "Field", "Nugget", "Sill",
                 "Exponential range (cells; '>' means longer than the lags examined)",
                 "h50 (cells)", "h50 resolved",
                 "Relative nugget gamma(1)/sill"]]
    aniso_rows = [["Task", "Field", "h50 0 deg (cells)", "h50 90 deg (cells)",
                   "Anisotropy ratio (h50 0/90)", "Status"]]
    ind_rows = [["Task", "Class", "Direction", "Field", "Mean area fraction",
                 "Indicator h50 (cells)", "Resolved",
                 "Relative nugget gamma(1)/sill"]]
    spur_rows = [["Task", "Class", "Reference area fraction",
                  "Predicted area fraction", "Predicted bodies",
                  "Predicted mean body (px)", "Status"]]
    comp_rows = [["Task", "Class", "Field", "Bodies", "Mean body (px)",
                  "Largest body fraction", "Bodies pred/ref"]]
    trans_rows = [["Task", "Axis", "Field", "From class", "To class", "Probability"]]
    dist_rows = [["Task", "Domain", "Wasserstein distance", "Unit"]]

    for task, domain in TASKS:
        if task not in args.tasks:
            continue
        test_dir = os.path.join(SPLITS, task, "test")
        if not os.path.isdir(test_dir):
            print(f"  {task}: no test split -- skipped"); continue

        grid = None
        if domain != "facies":
            lm = os.path.join(EXPORT, domain, "legend_manifest.json")
            if os.path.exists(lm):
                man = ev.load_legend_manifest(lm)
                try:
                    grid = json.load(open(lm)).get("cube_shape")
                except Exception:
                    grid = None
        else:
            lm = os.path.join(EXPORT, "facies", "legend_manifest.json")
            if os.path.exists(lm):
                try:
                    grid = json.load(open(lm)).get("cube_shape")
                except Exception:
                    grid = None

        files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
        print(f"\n{task} ({domain}) -- {len(files)} test layers")

        ref_g = {d: [] for d in DIRECTIONS}
        pred_g = {d: [] for d in DIRECTIONS}
        ref_var, pred_var = [], []
        ref_vals, pred_vals = [], []
        comp_acc = {"reference": defaultdict(list), "prediction": defaultdict(list)}
        trans_acc = {"reference": {0: [], 1: []}, "prediction": {0: [], 1: []}}
        # Indicator variograms, per facies class. A semivariogram of the CLASS
        # INDEX treats the labels as an interval scale, so it weights a
        # 1 -> 7 transition thirty-six times a 1 -> 2 one purely because of
        # how the classes happen to be numbered. That makes an index-field
        # range uninterpretable for a categorical variable: a model that
        # speckles class 6 into a class 2 matrix scores a large gamma at short
        # lag for a reason that has nothing to do with body size. Indicator
        # variograms on the 0/1 field of each class are the standard tool and
        # are invariant to the numbering.
        ind_g = {"reference": defaultdict(lambda: {d: [] for d in DIRECTIONS}),
                 "prediction": defaultdict(lambda: {d: [] for d in DIRECTIONS})}
        ind_var = {"reference": defaultdict(list), "prediction": defaultdict(list)}
        support = {"reference": defaultdict(list), "prediction": defaultdict(list)}

        for fn in files:
            pair = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
            w = pair.shape[1] // 2
            ref_img = pair[:, :w]
            rv, rmask = decode(ref_img, domain, ev)
            ppx, ppy = px_per_cell(rv.shape, grid)

            preds = []
            for seed in args.seeds:
                pth = os.path.join(RUNS, f"{task}_seed{seed}", "predictions", fn)
                if os.path.exists(pth):
                    preds.append(np.array(Image.open(pth).convert("RGB")))
            if not preds:
                continue
            pv, pmask = decode(preds[0], domain, ev)

            if domain != "facies":
                for dname, ang in DIRECTIONS.items():
                    ref_g[dname].append(directional_variogram(rv, rmask, LAGS_CELLS,
                                                              ang, ppx, ppy))
                    pred_g[dname].append(directional_variogram(pv, pmask, LAGS_CELLS,
                                                               ang, ppx, ppy))

            ref_vals.append(rv[rmask][::37])
            pred_vals.append(pv[pmask][::37])
            ref_var.append(float(np.var(rv[rmask])))
            pred_var.append(float(np.var(pv[pmask])))

            if domain == "facies":
                n_cls = len(ev.FACIES_LEGEND)
                for cls in range(1, n_cls + 1):
                    comp_acc["reference"][cls].append(connected_components(rv.astype(int), cls))
                    comp_acc["prediction"][cls].append(connected_components(pv.astype(int), cls))
                for label, fld, msk in (("reference", rv, rmask),
                                        ("prediction", pv, pmask)):
                    tot = max(int(msk.sum()), 1)
                    for cls in range(1, n_cls + 1):
                        ind = ((fld == cls) & msk).astype(np.float64)
                        frac = float(ind.sum()) / tot
                        support[label][cls].append(frac)
                        if frac < 0.01 or frac > 0.99:
                            continue
                        ind_var[label][cls].append(float(np.var(ind[msk])))
                        for dname, ang in DIRECTIONS.items():
                            ind_g[label][cls][dname].append(
                                directional_variogram(ind, msk, LAGS_CELLS,
                                                      ang, ppx, ppy))
                for ax in (0, 1):
                    trans_acc["reference"][ax].append(transition_matrix(rv.astype(int), n_cls, ax))
                    trans_acc["prediction"][ax].append(transition_matrix(pv.astype(int), n_cls, ax))

        # ---- variograms with layer bootstrap ----
        rng = np.random.default_rng(2025)
        h50: Dict[str, Dict[str, dict]] = {"reference": {}, "prediction": {}}
        for dname in DIRECTIONS:
            R = np.array(ref_g[dname]); P = np.array(pred_g[dname])
            if R.size == 0:
                continue
            def boot(M):
                idx = rng.integers(0, M.shape[0], size=(args.n_boot, M.shape[0]))
                means = np.nanmean(M[idx], axis=1)
                return (np.nanpercentile(means, 2.5, axis=0),
                        np.nanpercentile(means, 97.5, axis=0))
            rl, rh = boot(R); pl, ph = boot(P)
            rm, pm = np.nanmean(R, axis=0), np.nanmean(P, axis=0)
            for i, h in enumerate(LAGS_CELLS):
                vario_rows.append([task, domain, dname, int(h),
                                   f"{rm[i]:.6g}", f"{pm[i]:.6g}",
                                   f"{rl[i]:.6g}", f"{rh[i]:.6g}",
                                   f"{pl[i]:.6g}", f"{ph[i]:.6g}"])
            v_ref = float(np.mean(ref_var)) if ref_var else None
            v_pred = float(np.mean(pred_var)) if pred_var else None
            for label, series, hint in (("reference", rm, v_ref),
                                        ("prediction", pm, v_pred)):
                f = fit_exponential(LAGS_CELLS, series, hint)
                p = practical_range(LAGS_CELLS, series, hint)
                fit_rows.append([task, dname, label, f"{f['nugget']:.6g}",
                                 f"{f['sill']:.6g}",
                                 (f">{f['range']:.0f} (unresolved)"
                                  if f.get("range_unresolved") else f"{f['range']:.2f}"),
                                 f"{p['range']:.2f}",
                                 ("no (nugget-dominated)" if p.get("nugget_dominated")
                                  else "no (lower bound)" if p["unresolved"] else "yes"),
                                 f"{p.get('rel_nugget', float('nan')):.3f}"])
                h50[label][dname] = p
                tag = ("NUGGET" if p.get("nugget_dominated")
                       else ">" if p["unresolved"] else " ")
                print(f"  {dname:22s} {label:10s} h50 {p['range']:6.2f} cells "
                      f"{tag:<7}| nugget/sill {p.get('rel_nugget', float('nan')):5.2f}"
                      f" | exp range {'>' if f.get('range_unresolved') else ' '}"
                      f"{f['range']:6.2f}")

        # ---- anisotropy ratio, horizontal over vertical --------------------
        # Quoted only when BOTH directions resolve inside the lag window. If
        # either is a lower bound the ratio is a bound too, and a bound is not
        # a measurement -- it is reported as suppressed rather than printed as
        # a number that would be read as one.
        d0, d90 = "0 (along grid j)", "90 (along grid i)"
        for label in ("reference", "prediction") if domain != "facies" else ():
            a, b = h50[label].get(d0), h50[label].get(d90)
            if not a or not b:
                continue
            if a.get("nugget_dominated") or b.get("nugget_dominated"):
                aniso_rows.append([task, label, f"{a['range']:.2f}", f"{b['range']:.2f}",
                                   "n/a",
                                   "suppressed: nugget-dominated, no resolvable structure"])
            elif a["unresolved"] or b["unresolved"] or b["range"] <= 0:
                aniso_rows.append([task, label, f"{a['range']:.2f}", f"{b['range']:.2f}",
                                   "n/a", "suppressed: a direction exceeds the lag window"])
            else:
                aniso_rows.append([task, label, f"{a['range']:.2f}", f"{b['range']:.2f}",
                                   f"{a['range'] / b['range']:.2f}", "resolved"])

        # ---- marginal distance ----
        if ref_vals:
            unit = ("class" if domain == "facies"
                    else ev.CONTINUOUS_DOMAINS[domain].get("unit", ""))
            dist_rows.append([task, domain,
                              f"{wasserstein1(np.concatenate(ref_vals), np.concatenate(pred_vals)):.6g}",
                              unit])

        # ---- indicator variograms and spurious classes ---------------------
        if domain == "facies":
            n_cls = len(ev.FACIES_LEGEND)
            print("  indicator variograms (h50 per class, cells):")
            for cls in range(1, n_cls + 1):
                cname = ev.FACIES_LEGEND[cls - 1][0]
                rfrac = float(np.mean(support["reference"][cls])) if support["reference"][cls] else 0.0
                pfrac = float(np.mean(support["prediction"][cls])) if support["prediction"][cls] else 0.0
                # A class the model paints that is simply not present in the
                # reference test layers is not a range error, it is an
                # invented rock type; it is recorded separately so it cannot
                # be averaged away inside a macro statistic.
                if rfrac < 1e-6 and pfrac > 0:
                    recs = [r for r in comp_acc["prediction"][cls] if r["class_px"] > 0]
                    nb = float(np.mean([r["n_bodies"] for r in recs])) if recs else 0.0
                    mb = float(np.mean([r["mean_body_px"] for r in recs])) if recs else 0.0
                    spur_rows.append([task, cname, f"{rfrac:.6f}", f"{pfrac:.6f}",
                                      f"{nb:.1f}", f"{mb:.0f}",
                                      "SPURIOUS: absent from the reference test layers"])
                    print(f"    {cname:<17} SPURIOUS -- absent from reference, "
                          f"predicted at {100*pfrac:.3f} % area in "
                          f"{nb:.0f} bodies of {mb:.0f} px")
                    continue
                for dname in DIRECTIONS:
                    for label, frac in (("reference", rfrac), ("prediction", pfrac)):
                        curves = ind_g[label][cls][dname]
                        if not curves:
                            continue
                        m = np.nanmean(np.array(curves), axis=0)
                        hint = float(np.mean(ind_var[label][cls])) if ind_var[label][cls] else None
                        pr = practical_range(LAGS_CELLS, m, hint)
                        ind_rows.append([
                            task, cname, dname, label, f"{frac:.4f}",
                            f"{pr['range']:.2f}",
                            ("no (nugget-dominated)" if pr.get("nugget_dominated")
                             else "no (lower bound)" if pr["unresolved"] else "yes"),
                            f"{pr.get('rel_nugget', float('nan')):.3f}"])
                        if dname == "0 (along grid j)":
                            print(f"    {cname:<17} {label:<10} area "
                                  f"{100*frac:5.2f} %  h50(j) {pr['range']:6.2f}"
                                  f"  nugget/sill {pr.get('rel_nugget', float('nan')):5.2f}")

        # ---- architecture ----
        if domain == "facies":
            for cls in sorted(comp_acc["reference"]):
                nb = {}
                for label in ("reference", "prediction"):
                    recs = [r for r in comp_acc[label][cls] if r["class_px"] > 0]
                    if not recs:
                        continue
                    nb[label] = float(np.mean([r["n_bodies"] for r in recs]))
                    ratio = ("" if label == "reference" or not nb.get("reference")
                             else f"{nb['prediction'] / nb['reference']:.1f}x")
                    comp_rows.append([task, ev.FACIES_LEGEND[cls - 1][0], label,
                                      f"{nb[label]:.1f}",
                                      f"{np.mean([r['mean_body_px'] for r in recs]):.0f}",
                                      f"{np.mean([r['largest_fraction'] for r in recs]):.3f}",
                                      ratio])
            for ax, axn in ((0, "along grid i (image rows)"),
                            (1, "along grid j (image cols)")):
                for label in ("reference", "prediction"):
                    if not trans_acc[label][ax]:
                        continue
                    M = np.mean(trans_acc[label][ax], axis=0)
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            if M[i, j] > 0.001:
                                trans_rows.append([task, axn, label,
                                                   ev.FACIES_LEGEND[i][0],
                                                   ev.FACIES_LEGEND[j][0],
                                                   f"{M[i, j]:.4f}"])

    def write(name, rows):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"  {name}: {len(rows) - 1} rows")

    print("\nwriting:")
    write("9_Directional_variograms.csv", vario_rows)
    write("9_Variogram_fits.csv", fit_rows)
    write("9_Anisotropy.csv", aniso_rows)
    write("9_Indicator_variograms.csv", ind_rows)
    write("9_Spurious_classes.csv", spur_rows)
    write("9_Connected_components.csv", comp_rows)
    write("9_Transition_probabilities.csv", trans_rows)
    write("9_Marginal_distances.csv", dist_rows)
    print(f"\nLags are in GRID CELLS; pixels-per-cell differs by axis because the "
          f"export is not an isotropic resize.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
