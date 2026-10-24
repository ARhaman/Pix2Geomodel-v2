#!/usr/bin/env python3
"""
19_distributional_connectivity.py
=================================
 
Distributional and connectivity validation (Reviewer 5 #14, and the extrema /
marginal part of #11).
 
R5-14 asks for what a variogram cannot see. A semivariogram is a two-point
second-order statistic: two fields can match on it and still differ in which
cells are connected to which, in how often one facies sits next to another, and
in the shape of the value histogram. This script measures those three things
directly.
 
  1. MARGINAL DISTRIBUTION (continuous targets).
     Quantiles P1..P99, extrema, mean and SD for reference and prediction, plus
     two distances between the two marginals:
        Wasserstein-1  mean |sorted(pred) - sorted(ref)|, in physical units.
                       Reference and prediction are cell-aligned and identically
                       masked, so the two samples are the same size and W1 is
                       exactly the mean absolute difference of the order
                       statistics. Reported raw and as a fraction of the
                       reference interquartile range.
        Energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'|, subsampled, which is
                       sensitive to differences W1 can understate.
     Permeability is additionally reported in log10 space, because a linear-space
     marginal is dominated by the top decile.
 
  2. CONNECTED COMPONENTS (facies targets).
     Per class: number of connected bodies (4-connectivity), mean and median
     body size, and the fraction of the class area held by its largest body.
     A model that produces the right proportions in the wrong arrangement --
     many small patches instead of a few coherent bodies, or the reverse --
     shows up here and nowhere else in the current evaluation.
 
  3. TRANSITION PROBABILITIES (facies targets).
     Row-normalised lateral transition matrices along both image axes, for
     reference and prediction, restricted to cells inside the reservoir.
     The DIAGONAL of this matrix is the probability that a cell's neighbour is
     the same class, i.e. how "sticky" the field is. An over-smoothed
     prediction inflates it. This is a categorical analogue of the variogram
     sill deficit already reported for the continuous targets, obtained by a
     completely independent route -- which is the point of running it.
 
Outputs
    results/17_Marginals.csv      per task, per layer, seeds averaged
    results/18_Connectivity.csv   per task, per class
    results/19_Transitions.csv    per task: diagonal comparison + matrix distance
    results/19_Transition_matrices.csv   the full matrices, reference and predicted
 
    python 19_distributional_connectivity.py --self-test
    python 19_distributional_connectivity.py
    python 19_distributional_connectivity.py --tasks vcl_to_facies
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
 
CONTINUOUS_TASKS = {"facies_to_porosity": "porosity",
                    "facies_to_permeability": "permeability",
                    "facies_to_vcl": "vcl"}
FACIES_TASKS = ["porosity_to_facies", "permeability_to_facies", "vcl_to_facies"]
SEEDS = [1, 2, 3]
 
QUANTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
 
 
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
# Distances between marginals
# --------------------------------------------------------------------------
 
def wasserstein1(p: np.ndarray, r: np.ndarray) -> float:
    """Equal-size paired samples: W1 is the mean gap between order statistics."""
    if p.size == 0 or p.size != r.size:
        return float("nan")
    return float(np.mean(np.abs(np.sort(p) - np.sort(r))))
 
 
def energy_distance(p: np.ndarray, r: np.ndarray, n: int = 4000,
                    seed: int = 2025) -> float:
    """2E|X-Y| - E|X-X'| - E|Y-Y'|, subsampled to keep the O(n^2) term finite."""
    if p.size < 2 or r.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.choice(p, size=min(n, p.size), replace=False)
    b = rng.choice(r, size=min(n, r.size), replace=False)
    exy = float(np.mean(np.abs(a[:, None] - b[None, :])))
    exx = float(np.mean(np.abs(a[:, None] - a[None, :])))
    eyy = float(np.mean(np.abs(b[:, None] - b[None, :])))
    return 2.0 * exy - exx - eyy
 
 
# --------------------------------------------------------------------------
# Connected components
# --------------------------------------------------------------------------
 
def _label_scipy(binary: np.ndarray) -> Optional[np.ndarray]:
    try:
        from scipy import ndimage
    except Exception:
        return None
    lab, _ = ndimage.label(binary, structure=np.array([[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]]))
    return lab
 
 
def _label_unionfind(binary: np.ndarray) -> np.ndarray:
    """4-connectivity two-pass labelling; fallback when SciPy is unavailable."""
    h, w = binary.shape
    parent: List[int] = [0]
 
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
 
    lab = np.zeros((h, w), dtype=np.int32)
    nxt = 1
    for i in range(h):
        row, prev = binary[i], binary[i - 1] if i else None
        for j in range(w):
            if not row[j]:
                continue
            up = lab[i - 1, j] if i and prev[j] else 0
            left = lab[i, j - 1] if j and row[j - 1] else 0
            if up and left:
                lab[i, j] = min(up, left)
                union(up, left)
            elif up or left:
                lab[i, j] = up or left
            else:
                parent.append(nxt)
                lab[i, j] = nxt
                nxt += 1
    for i in range(h):
        for j in range(w):
            if lab[i, j]:
                lab[i, j] = find(lab[i, j])
    return lab
 
 
def components(binary: np.ndarray) -> Tuple[int, float, float, float]:
    """(count, mean size, median size, largest body as fraction of class area)."""
    total = int(binary.sum())
    if total == 0:
        return 0, float("nan"), float("nan"), float("nan")
    lab = _label_scipy(binary)
    if lab is None:
        lab = _label_unionfind(binary)
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    return (int(sizes.size), float(sizes.mean()), float(np.median(sizes)),
            float(sizes.max()) / total)
 
 
# --------------------------------------------------------------------------
# Transition probabilities
# --------------------------------------------------------------------------
 
def transition_matrix(cls: np.ndarray, valid: np.ndarray, n_cls: int) -> np.ndarray:
    """
    Row-normalised lateral transition probabilities over both image axes.
 
    cls holds 1..n_cls inside the reservoir; valid marks cells to count. A pair
    contributes only when BOTH of its cells are valid, so the reservoir boundary
    never registers as a facies transition.
    """
    counts = np.zeros((n_cls, n_cls), dtype=np.int64)
    for axis in (0, 1):
        a = np.take(cls, np.arange(cls.shape[axis] - 1), axis=axis)
        b = np.take(cls, np.arange(1, cls.shape[axis]), axis=axis)
        va = np.take(valid, np.arange(valid.shape[axis] - 1), axis=axis)
        vb = np.take(valid, np.arange(1, valid.shape[axis]), axis=axis)
        m = va & vb
        if not m.any():
            continue
        np.add.at(counts, (a[m] - 1, b[m] - 1), 1)
        np.add.at(counts, (b[m] - 1, a[m] - 1), 1)   # symmetric: undirected pairs
    row = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(row > 0, counts / np.maximum(row, 1), np.nan)
 
 
# --------------------------------------------------------------------------
# Self-test  (see the bug register: every new statistic gets a known answer)
# --------------------------------------------------------------------------
 
def self_test() -> int:
    ok = True
 
    # W1 between a sample and the same sample shifted by a constant is that
    # constant, exactly.
    x = np.random.default_rng(0).normal(size=5000)
    w = wasserstein1(x + 3.0, x)
    if abs(w - 3.0) > 1e-9:
        print(f"  FAIL W1 shift: {w} != 3.0"); ok = False
    else:
        print("  ok   W1 of a constant shift equals the shift")
 
    # Energy distance of a sample against itself is exactly 0 when the whole
    # sample is used: both draws are then permutations of the same set. With
    # n below the sample size the estimator is a seeded subsample and carries
    # sampling noise, so the identity is checked at full size and the noise
    # floor is reported rather than asserted away.
    e = energy_distance(x, x, n=x.size)
    if abs(e) > 1e-9:
        print(f"  FAIL energy self-distance at full size: {e} != 0"); ok = False
    else:
        print("  ok   energy distance of a sample against itself is zero")
    noise = abs(energy_distance(x, x, n=4000))
    shifted = energy_distance(x + 1.0, x, n=4000)
    if not (shifted > 20 * max(noise, 1e-12)):
        print(f"  FAIL energy separation: shift {shifted} vs noise {noise}"); ok = False
    else:
        print(f"  ok   energy distance separates a unit shift ({shifted:.4f}) "
              f"from subsampling noise ({noise:.4f})")
 
    # A cross of 5 cells is one 4-connected body; the same cells split
    # diagonally are four.
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    n, _, _, frac = components(cross)
    if n != 1 or abs(frac - 1.0) > 1e-12:
        print(f"  FAIL components(cross): {n} bodies, largest {frac}"); ok = False
    else:
        print("  ok   4-connected cross is one body holding all of the area")
    diag = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], bool)
    n2, _, _, _ = components(diag)
    if n2 != 4:
        print(f"  FAIL components(diagonal): {n2} != 4"); ok = False
    else:
        print("  ok   diagonal neighbours are not 4-connected")
 
    # A field of one class has persistence 1; perfect stripes of two classes
    # alternate every step along one axis, so along that axis persistence is 0
    # and the off-diagonal is 1. Counting both axes, a vertical stripe pattern
    # is constant down columns, so the diagonal is exactly 1/2.
    one = np.ones((6, 6), np.int32)
    v = np.ones((6, 6), bool)
    tm = transition_matrix(one, v, 2)
    if abs(tm[0, 0] - 1.0) > 1e-12:
        print(f"  FAIL persistence of a constant field: {tm[0,0]}"); ok = False
    else:
        print("  ok   constant field has persistence 1")
    stripes = np.tile(np.array([1, 2], np.int32), (6, 3))
    tm2 = transition_matrix(stripes, v, 2)
    if abs(tm2[0, 0] - 0.5) > 1e-12 or abs(tm2[0, 1] - 0.5) > 1e-12:
        print(f"  FAIL stripe persistence: {tm2[0,0]} / {tm2[0,1]}"); ok = False
    else:
        print("  ok   vertical stripes give persistence 1/2 over both axes")
 
    # The boundary mask must suppress pairs that touch an invalid cell.
    half = np.zeros((4, 4), bool); half[:, :2] = True
    fld = np.ones((4, 4), np.int32); fld[:, 2:] = 2
    tm3 = transition_matrix(fld, half, 2)
    if not (abs(tm3[0, 0] - 1.0) < 1e-12 and np.isnan(tm3[1, 1])):
        print(f"  FAIL masked transitions: {tm3}"); ok = False
    else:
        print("  ok   pairs crossing the reservoir boundary are not counted")
 
    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1
 
 
# --------------------------------------------------------------------------
 
def continuous_task(task: str, domain: str, ev, seeds: List[int]) -> List[list]:
    test_dir = os.path.join(SPLITS, task, "test")
    if not os.path.isdir(test_dir):
        return []
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
    dspec = ev.CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = ev.ramp_range(dspec)
    lut = ev.build_lut(dspec["stops"], hi_i - lo_i + 1)
    log_space = dspec["scale"] == "log10"
 
    acc: Dict[str, List[float]] = {}
    def push(k, v):
        if np.isfinite(v):
            acc.setdefault(k, []).append(float(v))
 
    n_used = 0
    for seed in seeds:
        pdir = os.path.join(RUNS, f"{task}_seed{seed}", "predictions")
        if not os.path.isdir(pdir):
            continue
        for fn in files:
            pp = os.path.join(pdir, fn)
            if not os.path.exists(pp):
                continue
            ref_img = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
            ref_img = ref_img[:, : ref_img.shape[1] // 2]
            pred_img = np.array(Image.open(pp).convert("RGB"))
            valid = ~np.all(ref_img == 0, axis=-1)
            if not valid.any():
                continue
            pv = ev.index_to_value(ev.decode_rgb_to_index(pred_img, lut) + lo_i, domain)[valid]
            rv = ev.index_to_value(ev.decode_rgb_to_index(ref_img, lut) + lo_i, domain)[valid]
 
            for q in QUANTILES:
                push(f"ref_P{q}", np.percentile(rv, q))
                push(f"pred_P{q}", np.percentile(pv, q))
            for nm, arr in (("ref", rv), ("pred", pv)):
                push(f"{nm}_min", arr.min()); push(f"{nm}_max", arr.max())
                push(f"{nm}_mean", arr.mean()); push(f"{nm}_sd", arr.std())
            iqr = float(np.percentile(rv, 75) - np.percentile(rv, 25))
            w1 = wasserstein1(pv, rv)
            push("W1", w1)
            push("W1_over_IQR", w1 / iqr if iqr > 0 else np.nan)
            push("energy", energy_distance(pv, rv))
            if log_space:
                lp = np.log10(np.clip(pv, dspec["vmin"], dspec["vmax"]))
                lr = np.log10(np.clip(rv, dspec["vmin"], dspec["vmax"]))
                push("log10_W1", wasserstein1(lp, lr))
                push("log10_energy", energy_distance(lp, lr))
                push("log10_ref_sd", lr.std()); push("log10_pred_sd", lp.std())
            n_used += 1
 
    if n_used == 0:
        print(f"  ! no predictions for {task}")
        return []
    print(f"  {task:<26} {n_used} prediction(s)")
    unit = dspec["unit"]
    rows = []
    for k in sorted(acc):
        v = np.array(acc[k])
        rows.append([task, domain, unit, k, f"{v.mean():.6g}", f"{v.std():.4g}", v.size])
    return rows
 
 
def facies_task(task: str, ev, seeds: List[int]) -> Tuple[List[list], List[list], List[list]]:
    test_dir = os.path.join(SPLITS, task, "test")
    if not os.path.isdir(test_dir):
        return [], [], []
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
    names = [n for n, _ in ev.FACIES_LEGEND]
    n_cls = len(names)
 
    comp: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    tms = {"ref": [], "pred": []}
    n_used = 0
 
    for seed in seeds:
        pdir = os.path.join(RUNS, f"{task}_seed{seed}", "predictions")
        if not os.path.isdir(pdir):
            continue
        for fn in files:
            pp = os.path.join(pdir, fn)
            if not os.path.exists(pp):
                continue
            pair = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
            ref = ev.decode_rgb_to_facies(pair[:, : pair.shape[1] // 2])
            pred = ev.decode_rgb_to_facies(np.array(Image.open(pp).convert("RGB")))
            valid = ref >= 1
            if not valid.any():
                continue
            for c in range(1, n_cls + 1):
                for nm, fld in (("ref", ref), ("pred", pred)):
                    cnt, mean, med, frac = components((fld == c) & valid)
                    d = comp.setdefault((nm, c), {"count": [], "mean": [],
                                                  "median": [], "largest": []})
                    if cnt:
                        d["count"].append(cnt); d["mean"].append(mean)
                        d["median"].append(med); d["largest"].append(frac)
            tms["ref"].append(transition_matrix(np.where(valid, ref, 1), valid, n_cls))
            tms["pred"].append(transition_matrix(
                np.where(valid, np.clip(pred, 1, n_cls), 1), valid, n_cls))
            n_used += 1
 
    if n_used == 0:
        print(f"  ! no predictions for {task}")
        return [], [], []
    print(f"  {task:<26} {n_used} prediction(s)")
 
    crows = []
    for c in range(1, n_cls + 1):
        r, p = comp.get(("ref", c)), comp.get(("pred", c))
        if not r or not r["count"]:
            continue
        pc = (lambda k: f"{np.mean(p[k]):.4g}" if p and p[k] else "-")
        crows.append([task, c, names[c - 1],
                      f"{np.mean(r['count']):.4g}", pc("count"),
                      f"{np.mean(r['mean']):.4g}", pc("mean"),
                      f"{np.mean(r['largest']):.4f}", pc("largest")])
 
    ref_m = np.nanmean(np.stack(tms["ref"]), axis=0)
    pred_m = np.nanmean(np.stack(tms["pred"]), axis=0)
    trows, mrows = [], []
    for c in range(n_cls):
        if np.isnan(ref_m[c, c]):
            continue
        trows.append([task, c + 1, names[c], f"{ref_m[c, c]:.4f}",
                      f"{pred_m[c, c]:.4f}", f"{pred_m[c, c] - ref_m[c, c]:+.4f}"])
    fin = np.isfinite(ref_m) & np.isfinite(pred_m)
    trows.append([task, "-", "MEAN ABSOLUTE MATRIX DIFFERENCE", "-", "-",
                  f"{np.mean(np.abs(pred_m[fin] - ref_m[fin])):.4f}"])
    for nm, M in (("reference", ref_m), ("predicted", pred_m)):
        for c in range(n_cls):
            mrows.append([task, nm, names[c]] +
                         ["-" if np.isnan(v) else f"{v:.4f}" for v in M[c]])
    return crows, trows, mrows
 
 
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+",
                    default=list(CONTINUOUS_TASKS) + FACIES_TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
 
    if args.self_test:
        return self_test()
 
    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)
    names = [n for n, _ in ev.FACIES_LEGEND]
 
    print("MARGINAL DISTRIBUTIONS (Reviewer 5 #11, #14)")
    mrows = [["Task", "Domain", "Unit", "Statistic", "Mean over layers/seeds",
              "SD", "n"]]
    for t, dom in CONTINUOUS_TASKS.items():
        if t in args.tasks:
            mrows += continuous_task(t, dom, ev, args.seeds)
 
    print("\nCONNECTIVITY AND TRANSITIONS (Reviewer 5 #14)")
    crows = [["Task", "Class", "Facies name", "Ref bodies", "Pred bodies",
              "Ref mean size", "Pred mean size", "Ref largest frac",
              "Pred largest frac"]]
    trows = [["Task", "Class", "Facies name", "Ref persistence",
              "Pred persistence", "Difference"]]
    fullm = [["Task", "Field", "From \\ To"] + names]
    for t in FACIES_TASKS:
        if t in args.tasks:
            a, b, c = facies_task(t, ev, args.seeds)
            crows += a; trows += b; fullm += c
 
    for fn, rows in (("17_Marginals.csv", mrows),
                     ("18_Connectivity.csv", crows),
                     ("19_Transitions.csv", trows),
                     ("19_Transition_matrices.csv", fullm)):
        with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        print(f"  written {os.path.join(OUT, fn)}")
 
    print("\n" + "=" * 74)
    print("How to read these. A predicted persistence HIGHER than the reference")
    print("means neighbouring cells agree more often than they should, i.e. the")
    print("facies field is over-smoothed -- the categorical counterpart of the")
    print("variogram sill deficit already reported for the continuous targets.")
    print("Fewer, larger connected bodies than the reference says the same thing")
    print("in a different currency. If both point the same way, the")
    print("under-dispersion finding rests on three independent measurements.")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())