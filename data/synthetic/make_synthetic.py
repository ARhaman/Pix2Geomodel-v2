#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synthetic.py -- generate a small synthetic geomodel export.

The reservoir model behind the paper is proprietary and cannot be
redistributed. This script builds a synthetic stand-in with the same
*structure*, so the whole pipeline -- splitting, pairing, training,
prediction, decoding and evaluation -- can be run end to end by anyone, and so
the split logic and the decoders can be checked without access to the real
model.

What is reproduced
------------------
  * four aligned property rasters per layer (facies, porosity, permeability,
    VCL), written as RGB PNGs under the same fixed legends the paper uses;
  * the same facies rule: seven reservoir-quality classes obtained by
    thresholding VCL at 0.12, 0.20, 0.30 and 0.40 with porosity resolving
    ties, so Fair Sand is again a thin porosity-tie-break class and is again
    rare -- the behaviour Reviewer 1 asks about is reproducible here;
  * an irregular inactive boundary occupying roughly a quarter of each image,
    encoded as NODATA black, matching the 25.78 % of the real export;
  * a depth trend, so contiguous-depth splitting is a meaningful operation and
    train/test distribution shift is present rather than absent.

What is NOT reproduced
----------------------
The geology. These are smoothed random fields with a plausible
porosity-permeability relation, not an interpretation of a real reservoir.
Numbers obtained from this dataset are not comparable with the paper's and are
not meant to be; the purpose is to make the code runnable and auditable.

Encoding uses src/02_evaluation.py itself -- value_to_index and build_lut --
rather than a reimplementation, so a round trip through the published decoder
is exact by construction. The script asserts that round trip before writing.

Usage
-----
    python data/synthetic/make_synthetic.py
    python data/synthetic/make_synthetic.py --layers 18 --size 256 --out export

Then, from the repository root:

    python src/01_layer_level_split.py --export data/synthetic/export ...

Requires numpy and Pillow only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:                                  # pragma: no cover
    sys.exit("Pillow is required:  pip install pillow")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL_PY = os.path.join(ROOT, "src", "02_evaluation.py")

# The legend ranges the PAPER used. These come from the legend manifests in
# data/legends/ and OVERRIDE the fallback defaults compiled into
# 02_evaluation.CONTINUOUS_DOMAINS -- which are narrower (0.225 / 1000 / 0.50)
# and would encode differently. Anything generated or evaluated without these
# manifests is not on the paper's scale.
LEGENDS: Dict[str, dict] = {
    "porosity":     {"vmin": 0.0,    "vmax": 0.3,     "scale": "linear",
                     "unit": "fraction"},
    "permeability": {"vmin": 0.0001, "vmax": 25000.0, "scale": "log10",
                     "unit": "mD", "zero_class": True},
    "vcl":          {"vmin": 0.0,    "vmax": 1.0,     "scale": "linear",
                     "unit": "fraction"},
}

# VCL thresholds defining the seven reservoir-quality classes, with porosity
# resolving ties -- the rule stated in Section 2.1 of the paper.
VCL_CUTS = (0.12, 0.20, 0.30, 0.40)


def _load_eval():
    if not os.path.exists(EVAL_PY):
        sys.exit(f"cannot find {EVAL_PY}\n"
                 f"run this script from a full checkout of the repository.")
    spec = importlib.util.spec_from_file_location("ev02", EVAL_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ev02"] = mod
    spec.loader.exec_module(mod)
    return mod


def smooth(a: np.ndarray, k: int) -> np.ndarray:
    """
    Separable box blur, repeated, so no SciPy dependency is introduced.

    The kernel is forced odd and the running sum carries a leading zero, so
    the output keeps the input shape exactly.
    """
    k = max(3, int(k) | 1)
    pad = k // 2
    out = a.astype(float)
    for _ in range(3):
        p = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
        c = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
        out = (c[k:, :] - c[:-k, :]) / k
        p = np.pad(out, ((0, 0), (pad, pad)), mode="reflect")
        c = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)],
                           axis=1)
        out = (c[:, k:] - c[:, :-k]) / k
    if out.shape != a.shape:                      # guard against drift
        raise AssertionError(f"smooth changed shape {a.shape} -> {out.shape}")
    return out


def unit(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def make_layer(rng: np.random.Generator, n: int, depth: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray]:
    """Returns (active, vcl, phie, perm, facies) for one layer."""
    # --- inactive boundary: an irregular blob covering ~25 % of the image ---
    yy, xx = np.mgrid[0:n, 0:n] / float(n)
    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    wob = smooth(rng.normal(size=(n, n)), max(3, n // 12))
    active = (r + 0.25 * unit(wob) - 0.125) < 0.46
    # guarantee a usable active fraction even for odd draws
    if active.mean() < 0.55:
        active = r < 0.46

    # --- VCL: smoothed field, cleaner upward (shallower = lower depth) ------
    v = unit(smooth(rng.normal(size=(n, n)), max(3, n // 8)))
    vcl = np.clip(0.08 + 0.75 * v - 0.18 * (1.0 - depth), 0.0, 1.0)

    # --- porosity: anti-correlated with VCL, with its own texture ----------
    p = unit(smooth(rng.normal(size=(n, n)), max(3, n // 16)))
    phie = np.clip(0.26 * (1.0 - vcl) ** 1.3 + 0.035 * (p - 0.5)
                   + 0.02 * depth, 0.0, 0.3)

    # --- permeability: Kozeny-Carman-like, lognormal scatter ---------------
    eps = 1e-6
    logk = (4.3 + 7.0 * np.log10(phie + eps) - 2.2 * np.log10(1.0 - vcl + eps)
            + 0.45 * rng.normal(size=(n, n)))
    perm = np.clip(10.0 ** np.clip(logk, -4.0, 4.39), 1e-4, 25000.0)

    # --- facies: VCL thresholds, porosity breaks the ties ------------------
    fac = np.full((n, n), 7, dtype=np.int16)          # Very Good Sand
    fac[vcl >= VCL_CUTS[0]] = 6                       # Good Clean Sand
    fac[vcl >= VCL_CUTS[1]] = 5                       # Good Sand
    fac[vcl >= VCL_CUTS[2]] = 3                       # Poor Sand
    fac[vcl >= VCL_CUTS[3]] = 1                       # Tight
    # porosity tie-break: a thin high-porosity sliver inside Poor Sand
    # becomes Fair Sand, and a low-porosity sliver becomes Very Poor Sand.
    band = (vcl >= VCL_CUTS[1]) & (vcl < VCL_CUTS[2])
    fac[band & (phie > np.quantile(phie[band], 0.985))] = 4   # Fair Sand
    fac[(vcl >= VCL_CUTS[3]) & (phie < np.quantile(phie, 0.10))] = 2
    fac[~active] = 0
    return active, vcl, phie, perm, fac


def write_png(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default=os.path.join(HERE, "export"))
    args = ap.parse_args()

    ev = _load_eval()
    rng = np.random.default_rng(args.seed)
    n = args.size

    # Adopt the paper's legends for the whole run.
    for dom, leg in LEGENDS.items():
        ev.CONTINUOUS_DOMAINS[dom].update(leg)

    luts = {}
    for dom in LEGENDS:
        spec = ev.CONTINUOUS_DOMAINS[dom]
        lo_i, hi_i = ev.ramp_range(spec)
        luts[dom] = (ev.build_lut(spec["stops"], hi_i - lo_i + 1), lo_i)

    fac_rgb = np.array([[0, 0, 0]]
                       + [ev.hex_to_rgb(h) for _, h in ev.FACIES_LEGEND],
                       dtype=np.uint8)

    counts = np.zeros(8, dtype=np.int64)
    worst = 0.0
    for i in range(1, args.layers + 1):
        depth = (i - 1) / max(args.layers - 1, 1)
        active, vcl, phie, perm, fac = make_layer(rng, n, depth)
        counts += np.bincount(fac.ravel(), minlength=8)

        # Filenames follow the convention src/01_layer_level_split.py
        # discovers with LAYER_RE = r"_layer_(\d+)_".
        write_png(os.path.join(args.out, "facies",
                               f"synthetic_layer_{i:02d}_facies.png"),
                  fac_rgb[fac])

        for dom, val in (("porosity", phie), ("permeability", perm),
                         ("vcl", vcl)):
            lut, lo_i = luts[dom]
            idx = ev.value_to_index(val, dom)
            rgb = lut[np.clip(idx.astype(int) - lo_i, 0, len(lut) - 1)]
            rgb[~active] = 0
            write_png(os.path.join(args.out, dom,
                                   f"synthetic_layer_{i:02d}_{dom}.png"), rgb)

            # round trip through the PUBLISHED decoder must be exact
            back = ev.index_to_value(
                ev.decode_rgb_to_index(rgb, lut) + lo_i, dom)
            ref = ev.index_to_value(idx, dom)
            d = float(np.max(np.abs(back[active] - ref[active])))
            worst = max(worst, d)

        got = ev.decode_rgb_to_facies(fac_rgb[fac])
        if not np.array_equal(got, fac):
            sys.exit(f"facies round trip failed on layer {i}")

    if worst > 1e-6:
        sys.exit(f"continuous round trip failed: max |decode-encode| = {worst}")

    for dom, leg in LEGENDS.items():
        man = dict(leg)
        man["note"] = ("synthetic dataset; these are the legend ranges used "
                       "in the paper and they override the fallback defaults "
                       "in src/02_evaluation.py")
        p = os.path.join(args.out, dom, "legend_manifest.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)

    total = counts[1:].sum()
    names = [nm for nm, _ in ev.FACIES_LEGEND]
    print(f"wrote {args.layers} layers of {n} x {n} to {args.out}")
    print(f"round trip through the published decoder is exact "
          f"(max error {worst:.2e})")
    print(f"active cells: {100.0 * total / counts.sum():.1f} % "
          f"(real export: 74.2 %)")
    print("facies areas:")
    for k, nm in enumerate(names, start=1):
        print(f"  {nm:<18} {100.0 * counts[k] / total:6.3f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
