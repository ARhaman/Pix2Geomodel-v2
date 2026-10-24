#!/usr/bin/env python3
"""
21_facies_property_decomposition.py
===================================
 
WHY IS PERMEABILITY UNPREDICTABLE FROM FACIES?
 
Every method tried in this study fails on permeability together: the conditional
GAN (R2 = -0.28 linear, +0.17 log10), the seven-value class-mean lookup
(R2 = +0.001) and the histogram classifier. When a deterministic lookup, a
non-parametric classifier and a 54-million-parameter generator all fail on the
same target, the limitation is not in the models. It is in how much of the
target the conditioning variable determines.
 
This script measures that directly, with no model involved.
 
  1. CORRELATION RATIO (eta squared). For each property, the fraction of its
     total variance explained by facies class:
 
         eta2 = SS_between / SS_total
 
     computed on the reference maps only. eta2 is the theoretical ceiling on the
     R2 attainable by ANY estimator that sees only the facies class -- which is
     what the class-mean lookup is. If eta2 for permeability is near zero, the
     lookup's R2 of 0.001 is not a poor result; it is the best possible one.
 
  2. WITHIN-CLASS DISPERSION. Mean, SD and coefficient of variation of each
     property inside each facies class. This shows WHERE the variance sits.
     Permeability is additionally treated in log10, because a coefficient of
     variation on a four-decade variable is not interpretable in linear units.
 
  3. HYDRAULIC FLOW UNITS as an alternative conditioning variable. Facies here
     are a static reservoir-quality scheme built from VCL and PHIE. Permeability
     is governed by pore-throat geometry, which those two variables do not
     determine: through Kozeny-Carman, k depends on specific surface area and
     tortuosity, so two cells with identical porosity and clay volume can differ
     by orders of magnitude. The standard petrophysical response is the flow
     zone indicator (Amaefule et al., 1993):
 
         RQI  = 0.0314 * sqrt(k / phi)        [micrometres]
         phiz = phi / (1 - phi)               [normalised porosity]
         FZI  = RQI / phiz
 
     Cells with similar FZI belong to one hydraulic flow unit. Binning log10 FZI
     into the same number of classes as the facies scheme gives a like-for-like
     comparison: eta2 of log10 k by facies class against eta2 of log10 k by flow
     unit, with the same number of degrees of freedom. If the flow-unit
     partition explains far more, then the correct recommendation is not a better
     generator but a better conditioning variable -- and that is a finding of
     use to anyone modelling permeability, independent of this workflow.
 
  4. A FAIR CEILING FOR EACH TASK. eta2 is reported for all three properties, so
     the ordering VCL > porosity > permeability can be stated as a property of
     the labelling scheme rather than inferred from model scores.
 
Outputs
    results/21_Variance_decomposition.csv   eta2 per property, both spaces
    results/21_Within_class.csv             per-class mean, SD, CV
    results/21_Flow_units.csv               facies vs FZI partition of log10 k
 
    python 21_facies_property_decomposition.py --self-test
    python 21_facies_property_decomposition.py
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
OUT = os.path.join(HERE, "results")
 
# The reference half of these paired sets carries the property; the facies field
# is the reference half of the matching inverse task on the same layers.
PROP_TASK = {"porosity": "facies_to_porosity",
             "permeability": "facies_to_permeability",
             "vcl": "facies_to_vcl"}
FACIES_TASK = "vcl_to_facies"
 
 
def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
 
 
def eta_squared(values: np.ndarray, groups: np.ndarray) -> Tuple[float, int]:
    """Fraction of variance of `values` explained by the partition `groups`."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if v.size < 2:
        return float("nan"), 0
    grand = v.mean()
    ss_tot = float(((v - grand) ** 2).sum())
    if ss_tot <= 0:
        return float("nan"), 0
    ss_bet = 0.0
    n_grp = 0
    for c in np.unique(g):
        m = g == c
        if m.sum() == 0:
            continue
        ss_bet += float(m.sum()) * (float(v[m].mean()) - grand) ** 2
        n_grp += 1
    return ss_bet / ss_tot, n_grp
 
 
def quantile_bins(x: np.ndarray, n: int) -> np.ndarray:
    """Split finite values of x into n equal-count bins, labelled 1..n."""
    out = np.zeros(x.shape, dtype=np.int32)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    qs = np.quantile(x[ok], np.linspace(0, 1, n + 1)[1:-1])
    out[ok] = np.digitize(x[ok], qs) + 1
    return out
 
 
def self_test() -> int:
    ok = True
    rng = np.random.default_rng(3)
 
    # A partition that fully determines the value gives eta2 = 1.
    g = np.repeat(np.arange(4), 500)
    v = g.astype(float) * 10.0
    e, n = eta_squared(v, g)
    if abs(e - 1.0) > 1e-12 or n != 4:
        print(f"  FAIL deterministic partition: eta2={e}, groups={n}"); ok = False
    else:
        print("  ok   a partition that determines the value gives eta2 = 1")
 
    # A partition unrelated to the value gives eta2 near 0.
    v2 = rng.normal(size=g.size)
    e2, _ = eta_squared(v2, g)
    if not (e2 < 0.02):
        print(f"  FAIL unrelated partition: eta2={e2} not near 0"); ok = False
    else:
        print(f"  ok   an unrelated partition gives eta2 = {e2:.4f}")
 
    # eta2 is exactly the R2 of the group-mean predictor: that identity is what
    # licenses reading eta2 as the ceiling for the class-mean lookup.
    v3 = g.astype(float) * 3.0 + rng.normal(scale=2.0, size=g.size)
    pred = np.zeros_like(v3)
    for c in np.unique(g):
        pred[g == c] = v3[g == c].mean()
    r2 = 1.0 - ((v3 - pred) ** 2).sum() / ((v3 - v3.mean()) ** 2).sum()
    e3, _ = eta_squared(v3, g)
    if abs(r2 - e3) > 1e-10:
        print(f"  FAIL eta2 != R2 of group means: {e3} vs {r2}"); ok = False
    else:
        print("  ok   eta2 equals the R2 of the group-mean predictor exactly")
 
    # Quantile bins are equal-count.
    b = quantile_bins(rng.normal(size=7000), 7)
    counts = np.bincount(b)[1:]
    if counts.max() - counts.min() > 5:
        print(f"  FAIL quantile bins uneven: {counts}"); ok = False
    else:
        print(f"  ok   quantile binning is equal-count ({counts.min()}–{counts.max()})")
 
    # FZI is scale-consistent: doubling k multiplies FZI by sqrt(2).
    phi = np.full(100, 0.2)
    k = np.full(100, 50.0)
    rqi = 0.0314 * np.sqrt(k / phi)
    fzi1 = rqi / (phi / (1 - phi))
    rqi2 = 0.0314 * np.sqrt(2 * k / phi)
    fzi2 = rqi2 / (phi / (1 - phi))
    if abs(float(fzi2[0] / fzi1[0]) - np.sqrt(2)) > 1e-12:
        print("  FAIL FZI scaling"); ok = False
    else:
        print("  ok   FZI scales as sqrt(k) at fixed porosity")
 
    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
 
 
def load_reference(task: str, domain: str, ev, split: str,
                   stride: int) -> Dict[str, np.ndarray]:
    """Reference (left-half) field per layer, decoded to physical units."""
    d = os.path.join(SPLITS, task, split)
    if not os.path.isdir(d):
        return {}
    out = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".png")):
        img = np.array(Image.open(os.path.join(d, fn)).convert("RGB"))
        img = img[:, : img.shape[1] // 2]
        layer = fn.split("_")[0]
        if domain == "facies":
            out[layer] = ev.decode_rgb_to_facies(img)[::stride, ::stride]
        else:
            spec = ev.CONTINUOUS_DOMAINS[domain]
            lo, hi = ev.ramp_range(spec)
            lut = ev.build_lut(spec["stops"], hi - lo + 1)
            val = ev.index_to_value(ev.decode_rgb_to_index(img, lut) + lo, domain)
            val = np.where(np.all(img == 0, axis=-1), np.nan, val)
            out[layer] = val[::stride, ::stride]
    return out
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=3,
                    help="spatial subsample; variance ratios are insensitive to it")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
 
    if args.self_test:
        return self_test()
 
    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)
    names = [n for n, _ in ev.FACIES_LEGEND]
    n_cls = len(names)
 
    facies = load_reference(FACIES_TASK, "facies", ev, args.split, args.stride)
    if not facies:
        print("no facies reference found"); return 2
    props = {p: load_reference(t, p, ev, args.split, args.stride)
             for p, t in PROP_TASK.items()}
 
    layers = sorted(set(facies) & set.intersection(*[set(v) for v in props.values()]))
    print(f"{len(layers)} layers, stride {args.stride}, split '{args.split}'")
 
    fac = np.concatenate([facies[l].ravel() for l in layers])
    dat = {p: np.concatenate([props[p][l].ravel() for l in layers]) for p in props}
    inside = fac >= 1
    fac = fac[inside]
    dat = {p: v[inside] for p, v in dat.items()}
    print(f"{fac.size:,} reservoir cells")
 
    # ---- 1. correlation ratio -----------------------------------------
    rows = [["Property", "Space", "eta^2 by facies class", "Classes",
             "Interpretation"]]
    print(f"\n{'property':<16}{'space':<10}{'eta^2':>10}   ceiling on R2 of any "
          f"facies-only estimator")
    for p in ("vcl", "porosity", "permeability"):
        for space in ("linear", "log10"):
            if space == "log10" and p != "permeability":
                continue
            v = dat[p]
            if space == "log10":
                v = np.log10(np.clip(v, 1e-4, None))
            e, ng = eta_squared(v, fac)
            interp = ("facies almost fully determines this property" if e > 0.7 else
                      "facies partly determines this property" if e > 0.3 else
                      "facies carries little information about this property")
            rows.append([p, space, f"{e:.4f}", ng, interp])
            print(f"{p:<16}{space:<10}{e:>10.4f}   {interp}")
 
    with open(os.path.join(OUT, "21_Variance_decomposition.csv"), "w",
              newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
 
    # ---- 2. within-class dispersion -----------------------------------
    wrows = [["Property", "Space", "Class", "Facies name", "Cells", "Mean",
              "SD", "Coefficient of variation"]]
    for p in ("vcl", "porosity", "permeability"):
        for space in ("linear", "log10"):
            if space == "log10" and p != "permeability":
                continue
            v = dat[p]
            if space == "log10":
                v = np.log10(np.clip(v, 1e-4, None))
            for c in range(1, n_cls + 1):
                m = (fac == c) & np.isfinite(v)
                if m.sum() < 10:
                    continue
                mu, sd = float(v[m].mean()), float(v[m].std())
                cv = sd / abs(mu) if mu != 0 else float("nan")
                wrows.append([p, space, c, names[c - 1], int(m.sum()),
                              f"{mu:.5g}", f"{sd:.5g}", f"{cv:.4f}"])
    with open(os.path.join(OUT, "21_Within_class.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(wrows)
 
    # ---- 3. hydraulic flow units --------------------------------------
    phi, k = dat["porosity"], dat["permeability"]
    good = np.isfinite(phi) & np.isfinite(k) & (phi > 1e-4) & (phi < 0.999) & (k > 0)
    logk = np.full(phi.shape, np.nan)
    logk[good] = np.log10(k[good])
    fzi = np.full(phi.shape, np.nan)
    rqi = 0.0314 * np.sqrt(k[good] / phi[good])
    fzi[good] = rqi / (phi[good] / (1.0 - phi[good]))
    logfzi = np.full(phi.shape, np.nan)
    logfzi[good] = np.log10(np.clip(fzi[good], 1e-6, None))
 
    hfu = quantile_bins(logfzi, n_cls)
    e_fac, _ = eta_squared(logk[good], fac[good])
    e_hfu, _ = eta_squared(logk[good], hfu[good])
    e_phi, _ = eta_squared(logk[good], quantile_bins(phi, n_cls)[good])
 
    frows = [["Partition of log10 permeability", "Number of groups",
              "eta^2", "Note"],
             ["Facies class (VCL + PHIE thresholds)", n_cls, f"{e_fac:.4f}",
              "the conditioning variable used throughout this study"],
             ["Porosity septile", n_cls, f"{e_phi:.4f}",
              "porosity alone, same degrees of freedom"],
             ["Hydraulic flow unit (log10 FZI septile)", n_cls, f"{e_hfu:.4f}",
              "Amaefule et al. (1993); RQI = 0.0314 sqrt(k/phi), FZI = RQI / (phi/(1-phi))"]]
    with open(os.path.join(OUT, "21_Flow_units.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(frows)
 
    print("\n" + "=" * 74)
    print(f"log10 permeability, variance explained with {n_cls} groups each:")
    print(f"  by facies class      {e_fac:.4f}")
    print(f"  by porosity septile  {e_phi:.4f}")
    print(f"  by flow-unit septile {e_hfu:.4f}")
    print("""
HOW TO READ THIS. eta^2 is exactly the R2 that a group-mean predictor attains,
so the facies row is the ceiling for the class-mean lookup and for any model
that sees only the facies class. If it is near zero, the lookup's R2 of 0.001
is not a poor result but the best attainable one, and the conclusion is about
the classification rather than about any model. The flow-unit row says whether
a petrophysically motivated partition of the same data, with the same number of
groups, would have done better -- which is the actionable recommendation.""")
    print(f"written to {OUT}")
    return 0
 