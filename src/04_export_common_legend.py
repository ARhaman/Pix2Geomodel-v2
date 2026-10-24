#!/usr/bin/env python3
"""
04_export_common_legend.py
==========================

Re-export the reservoir property grids to PNG using EXTENDED legends, and build
the Yemen -> Netherlands cross-reservoir transfer datasets.

WHY THIS SCRIPT EXISTS
----------------------
Two independent problems with the original export.

**1. The legends clip a large fraction of the data.** Measured on the Yemen
grid:

    property      legend used        actual range          clipped
    porosity      0     - 0.225      0.080  - 0.263        10.80 %  (all above cap)
    permeability  0.01  - 1000 mD    0      - 7168.89      14.20 %  (7.66 low + 6.54 high)
    VCL           0     - 0.50       0      - 0.9349        2.95 %  (all above cap)

    (measured on the full verified grid, 3,627,990 cells, k = 1..54)

Every permeability cell at or below 0.01 mD is rendered the same blue and every
cell above 1000 mD the same red, so ~14 % of the permeability field carries no
information in the training images. The network cannot learn it and cannot be
scored on it. This is a substantial part of why permeability tasks underperform
-- a preprocessing choice, not only geology (Reviewer 1 #6, Reviewer 5 #11, #12).

A further 1.85 % of permeability cells are EXACTLY zero. A log scale cannot
represent zero, so the original export clipped impermeable rock into the same
colour as 0.01 mD rock; the extended legend reserves palette index 0 for it.

**2. Cross-reservoir transfer needs one shared colour language.** The
Netherlands grid reaches 0.293 porosity and 20,333 mD, well outside the Yemen
legends. Testing a Yemen-trained model on Netherlands images exported with
different ramps would measure nothing (Reviewer 5 #4).

The extended legends below cover both reservoirs with no clipping:

    porosity      0.0    - 0.30           linear
    permeability  0.0001 - 25000 mD       log10
    VCL           0.0    - 1.00           linear

Cost of widening permeability from 5 to 8.4 decades: quantisation coarsens from
+/-2.29 % to +/-3.90 % relative error. Removing 14.20 % clipping and recovering
the zero class is worth 1.6 points of quantisation, and both numbers are
reported in the manifest so the choice is auditable.

A `legend_manifest.json` is written beside every export and is read back by
`02_evaluation.py --legend-file` so that decoding is always the exact inverse
of the encoding actually used.

USAGE
-----
    # what would be clipped, under old and new legends -- no files written
    python 04_export_common_legend.py --saturation-report --dataset yemen

    # re-export Yemen with extended legends
    python 04_export_common_legend.py --dataset yemen --property porosity \
        --out "D:/.../Revision/export_v2/yemen/porosity"

    # cross-reservoir tiles (both reservoirs, identical cell geometry)
    python 04_export_common_legend.py --dataset yemen        --property porosity \
        --tile-cells 128 --tile-px 1024 --out .../tiles/yemen/porosity
    python 04_export_common_legend.py --dataset netherlands  --property permeability \
        --tile-cells 128 --tile-px 1024 --out .../tiles/netherlands/permeability
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------
# Legends
# --------------------------------------------------------------------------

LUT_SIZE = 255
BACKGROUND_INDEX = 255

# Gradient stops are UNCHANGED from the original export -- only the value
# ranges are widened, so the visual appearance of the maps stays familiar.
STOPS = {
    "porosity": ((0.00, "#FF00FF"), (0.10, "#0000FF"), (0.20, "#00FFFF"),
                 (0.40, "#00FF00"), (0.70, "#FFFF00"), (0.85, "#FFA500"),
                 (1.00, "#FF0000")),
    "permeability": ((0.00, "#0000FF"), (0.25, "#00FFFF"), (0.50, "#00FF00"),
                     (0.75, "#FFFF00"), (0.90, "#FFA500"), (1.00, "#FF0000")),
    "vcl": ((0.00, "#FF0000"), (0.10, "#FFA500"), (0.30, "#FFFF00"),
            (0.50, "#00FF00"), (0.60, "#00FFFF"), (0.90, "#0000FF"),
            (1.00, "#FF00FF")),
}

ORIGINAL_LEGEND = {
    "porosity":     {"vmin": 0.0,  "vmax": 0.225,  "scale": "linear", "unit": "fraction"},
    "permeability": {"vmin": 0.01, "vmax": 1000.0, "scale": "log10",  "unit": "mD"},
    "vcl":          {"vmin": 0.0,  "vmax": 0.50,   "scale": "linear", "unit": "fraction"},
}

EXTENDED_LEGEND = {
    "porosity":     {"vmin": 0.0,    "vmax": 0.30,    "scale": "linear", "unit": "fraction"},
    # zero_class: 1.85 % of Yemen permeability cells are EXACTLY zero, and a
    # logarithmic scale cannot represent zero -- the original export silently
    # clipped them to the legend minimum, so genuinely impermeable rock was
    # rendered identically to 0.01 mD rock. Index 0 is reserved as a distinct
    # "zero / impermeable" class so the network can learn it as its own state;
    # the ramp then occupies indices 1..254.
    "permeability": {"vmin": 0.0001, "vmax": 25000.0, "scale": "log10",
                     "unit": "mD", "zero_class": True},
    "vcl":          {"vmin": 0.0,    "vmax": 1.00,    "scale": "linear", "unit": "fraction"},
}

# Colour for the reserved zero class. White is absent from every ramp
# (blue->cyan->green->yellow->orange->red), so it cannot be confused with a
# ramp value, and black is already taken by NODATA.
ZERO_CLASS_INDEX = 0
ZERO_CLASS_COLOUR = "#FFFFFF"

FACIES_LEGEND_YEMEN = [
    ("Tight", "#8B4513"), ("Very Poor Sand", "#00FF00"), ("Poor Sand", "#ADFF2F"),
    ("Fair Sand", "#FFFF00"), ("Good Sand", "#FFA500"),
    ("Good Clean Sand", "#FF00FF"), ("Very Good Sand", "#FF0000"),
]


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def build_lut(stops: Sequence[Tuple[float, str]],
              n: int = LUT_SIZE) -> np.ndarray:
    pos = np.array([p for p, _ in stops], float)
    col = np.array([hex_to_rgb(c) for _, c in stops], float)
    x = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(x, pos, col[:, i]) for i in range(3)],
                    axis=1).round().astype(np.uint8)


def ramp_range(spec: dict) -> Tuple[int, int]:
    """First and last palette index occupied by the colour ramp."""
    return (1, LUT_SIZE - 1) if spec.get("zero_class") else (0, LUT_SIZE - 1)


def value_to_index(v: np.ndarray, spec: dict) -> np.ndarray:
    v = np.asarray(v, float)
    lo_i, hi_i = ramp_range(spec)
    if spec["scale"] == "log10":
        vc = np.clip(v, spec["vmin"], spec["vmax"])
        norm = (np.log10(vc) - math.log10(spec["vmin"])) / (
            math.log10(spec["vmax"]) - math.log10(spec["vmin"]))
    else:
        norm = (v - spec["vmin"]) / max(spec["vmax"] - spec["vmin"], 1e-12)
    idx = (lo_i + np.clip(norm, 0.0, 1.0) * (hi_i - lo_i)).round().astype(np.uint8)
    if spec.get("zero_class"):
        idx = np.where(v <= 0, np.uint8(ZERO_CLASS_INDEX), idx)
    return idx


def index_to_value(idx: np.ndarray, spec: dict) -> np.ndarray:
    """Analytical inverse of value_to_index. Used by the evaluation module."""
    idx = np.asarray(idx)
    lo_i, hi_i = ramp_range(spec)
    norm = (idx.astype(float) - lo_i) / (hi_i - lo_i)
    norm = np.clip(norm, 0.0, 1.0)
    if spec["scale"] == "log10":
        lo, hi = math.log10(spec["vmin"]), math.log10(spec["vmax"])
        out = np.power(10.0, lo + norm * (hi - lo))
    else:
        out = spec["vmin"] + norm * (spec["vmax"] - spec["vmin"])
    if spec.get("zero_class"):
        out = np.where(idx == ZERO_CLASS_INDEX, 0.0, out)
    return out


def quantisation_error(spec: dict) -> dict:
    """Error floor imposed by 255 levels over this range."""
    lo_i, hi_i = ramp_range(spec)
    levels = hi_i - lo_i
    if spec["scale"] == "log10":
        decades = math.log10(spec["vmax"]) - math.log10(spec["vmin"])
        half_step = decades / levels / 2
        return {"decades": decades, "ramp_levels": levels + 1,
                "zero_class": bool(spec.get("zero_class")),
                "max_relative_error_pct": 100 * (10 ** half_step - 1)}
    span = spec["vmax"] - spec["vmin"]
    return {"span": span, "ramp_levels": levels + 1,
            "half_step": span / levels / 2,
            "max_error_pct_of_range": 100 / levels / 2}


# --------------------------------------------------------------------------
# Grid loading
# --------------------------------------------------------------------------

def load_yemen_cube(csv_path: str, grid_shape: Optional[Sequence[int]] = None,
                    corruption_tolerance: int = 50) -> np.ndarray:
    """
    Yemen Petrel export: whitespace-delimited `i j k X Y Z value`, CRLF endings.
    Returns a float cube indexed [i, j, k] with NaN where undefined.

    NOTE ON FILE INTEGRITY
    ----------------------
    `perm.csv` in this project is TRUNCATED AND CORRUPTED: rows 1..2,111,499
    parse cleanly (covering k = 23..54 only), after which the file degrades into
    binary noise. `Facies.csv`, `Porosity.csv` and `Vshale_ns.csv` are intact
    (3,627,990 rows each, every row 7 fields).

    Two consequences this loader guards against:
      * fields must be parsed into locals and appended together, never appended
        one at a time -- a mid-row failure otherwise desynchronises the arrays
        and produces a silent index error much later;
      * a run of consecutive unparseable rows means corruption, not stray bad
        values, so we stop and report the row number instead of silently
        returning a partial grid.
    """
    print(f"  reading {os.path.basename(csv_path)} ...")
    ii: List[int] = []; jj: List[int] = []; kk: List[int] = []; vv: List[float] = []
    consecutive_bad = 0
    first_bad_row: Optional[int] = None
    n_bad = 0
    stopped_at: Optional[int] = None

    with open(csv_path, "r", errors="replace") as fh:
        for row, line in enumerate(fh, start=1):
            p = line.split()
            ok = False
            if len(p) >= 7:
                try:
                    a, b, c = int(p[0]), int(p[1]), int(p[2])
                    val = float(p[6])
                    ok = True
                except ValueError:
                    ok = False
            if ok:
                ii.append(a); jj.append(b); kk.append(c); vv.append(val)
                consecutive_bad = 0
            else:
                n_bad += 1
                consecutive_bad += 1
                if first_bad_row is None:
                    first_bad_row = row
                if consecutive_bad >= corruption_tolerance:
                    stopped_at = row
                    print(f"    !! {consecutive_bad} consecutive unparseable rows "
                          f"ending at line {row}: file appears CORRUPTED from "
                          f"line ~{row - consecutive_bad + 1} onward. Stopping.")
                    break

    if not ii:
        raise SystemExit(f"no parseable rows in {csv_path}")

    i = np.array(ii, np.int32); j = np.array(jj, np.int32)
    k = np.array(kk, np.int32); v = np.array(vv, np.float64)

    shape = (tuple(grid_shape) if grid_shape
             else (int(i.max()) + 1, int(j.max()) + 1, int(k.max()) + 1))
    cube = np.full(shape, np.nan)
    inside = (i < shape[0]) & (j < shape[1]) & (k < shape[2])
    cube[i[inside], j[inside], k[inside]] = v[inside]

    k_present = np.unique(k)
    k_lo, k_hi = int(k_present.min()), int(k_present.max())
    # Petrel exports index k from 1, so an absent layer 0 is normal, not a gap.
    interior_gaps = sorted(set(range(k_lo, k_hi + 1)) - set(k_present.tolist()))

    print(f"    cube {cube.shape}, {np.isfinite(cube).sum():,} defined cells")
    print(f"    parsed {len(ii):,} rows; skipped {n_bad:,}")
    print(f"    k layers present: {k_present.size} (contiguous {k_lo}..{k_hi})"
          f"{'  [1-based indexing]' if k_lo == 1 else ''}")

    if stopped_at or interior_gaps:
        print(f"    !! WARNING: incomplete grid.")
        if interior_gaps:
            print(f"    !!   {len(interior_gaps)} missing layers inside the range: "
                  f"{interior_gaps[:10]}{'...' if len(interior_gaps) > 10 else ''}")
        if stopped_at:
            print(f"    !!   parsing stopped early at line {stopped_at:,} (corruption).")
        print(f"    !! Statistics below cover only the layers present. Recover or "
              f"re-export this property before using it as ground truth.")
    return cube


def load_netherlands_cube(npy_path: str, null_value: float = -99.0) -> np.ndarray:
    """GeoAgent grids: float32 .npy of shape (i, j, k), -99 marks NODATA."""
    print(f"  reading {os.path.basename(npy_path)} ...")
    cube = np.load(npy_path).astype(np.float64)
    cube[cube <= null_value + 1e-6] = np.nan
    print(f"    cube {cube.shape}, {np.isfinite(cube).sum():,} defined cells")
    return cube


DATASETS = {
    "yemen": {
        "loader": "csv",
        "files": {"facies": "Facies.csv", "porosity": "Porosity.csv",
                  "permeability": "perm.csv", "vcl": "Vshale_ns.csv"},
        "default_root": "D:/First year/Papers/2. Pix2Geomdel.V2/2D Yemen Model",
    },
    "netherlands": {
        "loader": "npy",
        "files": {"facies": "grids/facies.npy", "porosity": "grids/porosity.npy",
                  "permeability": "grids/perm.npy"},
        "default_root": "D:/First year/Papers/GeoAgent",
    },
}


def load_cube(dataset: str, prop: str, root: str) -> np.ndarray:
    spec = DATASETS[dataset]
    if prop not in spec["files"]:
        raise SystemExit(f"{dataset} has no '{prop}' grid "
                         f"(available: {sorted(spec['files'])})")
    path = os.path.join(root, spec["files"][prop])
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    return (load_yemen_cube(path) if spec["loader"] == "csv"
            else load_netherlands_cube(path))


# --------------------------------------------------------------------------
# Saturation report
# --------------------------------------------------------------------------

def saturation(cube: np.ndarray, spec: dict) -> dict:
    v = cube[np.isfinite(cube)]
    n = v.size
    if n == 0:
        return {"n": 0}
    zeros = int((v <= 0).sum())
    if spec.get("zero_class"):
        low = int(((v > 0) & (v < spec["vmin"])).sum())
    else:
        low = int((v < spec["vmin"]).sum())
    high = int((v > spec["vmax"]).sum())
    return {
        "n_cells": n,
        "data_min": float(v.min()), "data_max": float(v.max()),
        "legend_min": spec["vmin"], "legend_max": spec["vmax"],
        "clipped_low": low, "clipped_low_pct": 100.0 * low / n,
        "clipped_high": high, "clipped_high_pct": 100.0 * high / n,
        "clipped_total_pct": 100.0 * (low + high) / n,
        "exact_zeros": zeros, "exact_zeros_pct": 100.0 * zeros / n,
        "zero_class_reserved": bool(spec.get("zero_class")),
    }


def saturation_report(dataset: str, root: str, props: Sequence[str]) -> dict:
    out = {}
    for prop in props:
        if prop not in DATASETS[dataset]["files"] or prop == "facies":
            continue
        cube = load_cube(dataset, prop, root)
        rep = {"original": saturation(cube, ORIGINAL_LEGEND[prop]),
               "extended": saturation(cube, EXTENDED_LEGEND[prop]),
               "quantisation_original": quantisation_error(ORIGINAL_LEGEND[prop]),
               "quantisation_extended": quantisation_error(EXTENDED_LEGEND[prop])}
        out[prop] = rep
        o, e = rep["original"], rep["extended"]
        print(f"\n{prop.upper()}")
        print(f"  data range      : {o['data_min']:.6g} .. {o['data_max']:.6g}")
        print(f"  ORIGINAL legend {o['legend_min']} .. {o['legend_max']}  ->  "
              f"clipped {o['clipped_total_pct']:.2f} % "
              f"({o['clipped_low_pct']:.2f} % low, {o['clipped_high_pct']:.2f} % high)")
        print(f"  EXTENDED legend {e['legend_min']} .. {e['legend_max']}  ->  "
              f"clipped {e['clipped_total_pct']:.2f} %")
        qo, qe = rep["quantisation_original"], rep["quantisation_extended"]
        key = "max_relative_error_pct" if "max_relative_error_pct" in qo else "max_error_pct_of_range"
        print(f"  quantisation    : {qo[key]:.3f} %  ->  {qe[key]:.3f} %")
    return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def palette_for(prop: str, facies_legend=FACIES_LEGEND_YEMEN) -> List[int]:
    pal = [0] * (256 * 3)
    pal[BACKGROUND_INDEX * 3: BACKGROUND_INDEX * 3 + 3] = [0, 0, 0]
    if prop == "facies":
        for idx, (_, hx) in enumerate(facies_legend, start=1):
            pal[idx * 3: idx * 3 + 3] = list(hex_to_rgb(hx))
    else:
        spec = EXTENDED_LEGEND.get(prop, {})
        lo_i, hi_i = ramp_range(spec) if spec else (0, LUT_SIZE - 1)
        lut = build_lut(STOPS[prop], hi_i - lo_i + 1)
        flat = lut.reshape(-1).tolist()
        pal[lo_i * 3: lo_i * 3 + len(flat)] = flat
        if spec.get("zero_class"):
            pal[ZERO_CLASS_INDEX * 3: ZERO_CLASS_INDEX * 3 + 3] = list(
                hex_to_rgb(ZERO_CLASS_COLOUR))
    return pal


def encode_layer(layer: np.ndarray, prop: str, spec: Optional[dict],
                 facies_codes: Optional[Dict[int, int]] = None) -> Image.Image:
    """One k-slice -> palette image. NODATA becomes BACKGROUND_INDEX."""
    valid = np.isfinite(layer)
    idx = np.full(layer.shape, BACKGROUND_INDEX, np.uint8)
    if prop == "facies":
        codes = np.rint(layer[valid]).astype(int)
        mapped = np.array([facies_codes.get(int(c), 0) for c in codes], np.uint8) \
            if facies_codes else (codes + 1).astype(np.uint8)
        idx[valid] = mapped
    else:
        idx[valid] = value_to_index(layer[valid], spec)
    # ORIENTATION -- do not "fix" this by transposing.
    # The original export does `layer = cube[:, :, k]` then `Image.fromarray(...)`
    # with no transpose. PIL reads an array of shape (rows, cols) as
    # height x width, so a (365, 248) Yemen layer becomes a 248 x 365 image,
    # upscaled x33 to the 8184 x 12045 files on disk. Transposing here would
    # rotate every re-exported slice 90 degrees relative to the published
    # figures and relative to the paired images the model was trained on.
    img = Image.fromarray(idx, mode="P")
    img.putpalette(palette_for(prop))
    return img


def upscale(img: Image.Image, long_side: Optional[int]) -> Image.Image:
    if not long_side:
        return img
    scale = max(1, int(math.ceil(long_side / max(img.width, img.height))))
    if scale == 1:
        return img
    pal = img.getpalette()
    out = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    out.putpalette(pal)
    return out


def tiles_of(layer: np.ndarray, cells: int) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Non-overlapping cells x cells tiles; partial edges are skipped."""
    ni, nj = layer.shape
    for a in range(0, ni - cells + 1, cells):
        for b in range(0, nj - cells + 1, cells):
            yield a, b, layer[a:a + cells, b:b + cells]


def export(dataset: str, prop: str, root: str, out_dir: str,
           legend_set: str, long_side: Optional[int],
           tile_cells: Optional[int], tile_px: Optional[int],
           min_valid_frac: float) -> dict:
    spec = (EXTENDED_LEGEND if legend_set == "extended" else ORIGINAL_LEGEND).get(prop)
    cube = load_cube(dataset, prop, root)
    os.makedirs(out_dir, exist_ok=True)

    facies_codes = None
    if prop == "facies":
        codes = np.rint(cube[np.isfinite(cube)]).astype(int)
        uniq = sorted(set(codes.tolist()))
        facies_codes = {c: i + 1 for i, c in enumerate(uniq)}
        print(f"  facies codes {uniq} -> legend indices {list(facies_codes.values())}")

    n_written = 0
    manifest_records: List[dict] = []
    nk = cube.shape[2]

    for k in range(nk):
        layer = cube[:, :, k]
        if not np.isfinite(layer).any():
            continue

        if tile_cells:
            for a, b, tile in tiles_of(layer, tile_cells):
                frac = float(np.isfinite(tile).mean())
                if frac < min_valid_frac:
                    continue
                img = encode_layer(tile, prop, spec, facies_codes)
                img = upscale(img, tile_px)
                name = f"{dataset}_{prop}_k{k:04d}_i{a:04d}_j{b:04d}.png"
                img.save(os.path.join(out_dir, name), compress_level=6)
                manifest_records.append({"k": k, "i0": a, "j0": b,
                                         "valid_frac": frac, "file": name})
                n_written += 1
        else:
            img = encode_layer(layer, prop, spec, facies_codes)
            img = upscale(img, long_side)
            name = f"{prop}_layer_{k}_pal_{img.width}x{img.height}.png"
            img.save(os.path.join(out_dir, name), compress_level=6)
            manifest_records.append({"k": k, "file": name,
                                     "px": [img.width, img.height]})
            n_written += 1

        if (k + 1) % 20 == 0:
            print(f"    layer {k+1}/{nk}  ({n_written} images so far)")

    sat = saturation(cube, spec) if spec else None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "dataset": dataset,
        "property": prop,
        "legend_set": legend_set,
        "legend": spec,
        "gradient_stops": list(STOPS[prop]) if prop != "facies" else None,
        "facies_legend": FACIES_LEGEND_YEMEN if prop == "facies" else None,
        "facies_code_map": facies_codes,
        "lut_size": LUT_SIZE,
        "background_index": BACKGROUND_INDEX,
        "cube_shape": list(cube.shape),
        "tiling": {"tile_cells": tile_cells, "tile_px": tile_px,
                   "min_valid_frac": min_valid_frac} if tile_cells else None,
        "saturation": sat,
        "quantisation": quantisation_error(spec) if spec else None,
        "n_images": n_written,
        "records": manifest_records,
    }
    with open(os.path.join(out_dir, "legend_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n  wrote {n_written} images -> {out_dir}")
    if sat:
        print(f"  clipped by this legend: {sat['clipped_total_pct']:.2f} %")
    print(f"  legend manifest: {os.path.join(out_dir, 'legend_manifest.json')}")
    return manifest


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--property", choices=["facies", "porosity", "permeability", "vcl"])
    ap.add_argument("--root", help="dataset root (defaults per dataset)")
    ap.add_argument("--out")
    ap.add_argument("--legend-set", choices=["original", "extended"], default="extended")
    ap.add_argument("--long-side", type=int, default=4096,
                    help="upscale so the long side is at least this many px "
                         "(original export used 12000; 4096 is 2x the training "
                         "height and much faster). 0 disables upscaling.")
    ap.add_argument("--tile-cells", type=int,
                    help="emit square tiles of this many grid cells instead of "
                         "whole layers -- use for the cross-reservoir test so "
                         "both grids share identical cell geometry")
    ap.add_argument("--tile-px", type=int, default=1024,
                    help="upscale each tile to this many pixels per side")
    ap.add_argument("--min-valid-frac", type=float, default=0.60,
                    help="skip tiles with less than this fraction of defined cells")
    ap.add_argument("--saturation-report", action="store_true",
                    help="report clipping under both legends and exit")
    args = ap.parse_args()

    root = args.root or DATASETS[args.dataset]["default_root"]

    if args.saturation_report:
        props = ([args.property] if args.property
                 else [p for p in DATASETS[args.dataset]["files"] if p != "facies"])
        rep = saturation_report(args.dataset, root, props)
        print("\n" + json.dumps(rep, indent=2)[:400] + " ...")
        return 0

    if not (args.property and args.out):
        ap.error("--property and --out are required unless --saturation-report")

    print(f"dataset     : {args.dataset}   root {root}")
    print(f"property    : {args.property}")
    print(f"legend set  : {args.legend_set}")
    if args.tile_cells:
        print(f"tiling      : {args.tile_cells} cells -> {args.tile_px} px, "
              f"min valid {args.min_valid_frac:.0%}")
    print()

    export(args.dataset, args.property, root, args.out, args.legend_set,
           args.long_side or None, args.tile_cells, args.tile_px,
           args.min_valid_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
