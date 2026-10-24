#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_baseline_layerwise.py
========================

Why this script exists
----------------------
Table 7 of the R2 manuscript compares Pix2Geomodel against the conventional
baselines, but the two sides were aggregated differently:

  * the Pix2Geomodel linear columns (0.5754 / 0.8543 / -0.2818) are LAYERWISE
    three-seed means, produced by 02_evaluation.evaluate_task;
  * every baseline value in results/13_Conventional_baselines.csv is POOLED
    (all test cells concatenated into one array, scored once -- the
    "14009256" in its last column);
  * the Pix2Geomodel log10 value (0.1652) is neither: 13_permeability_error.py
    averages the three seeds' PREDICTED FIELDS pixel-by-pixel and scores that
    ensemble once, pooled.

24_l1_forward_metrics.py showed the two conventions are not interchangeable
(porosity, L1 arm: 0.4528 layerwise vs 0.5663 pooled). So Table 7 cannot be
repaired by editing one cell; the baselines have to be rescored on the same
convention as the network.

This script does exactly that and nothing else. It reuses the estimators from
14_baseline_conventional.py by importing them, so the class-lookup tables and
the random forest are identical by construction -- same features, same
FIT_STRIDE, same 400k subsample under rng(2025), same forest hyperparameters
and random_state. Only the aggregation changes.

Self-validation
---------------
The script recomputes the POOLED numbers first and refuses to print or write
its layerwise numbers unless the pooled ones reproduce
results/13_Conventional_baselines.csv within --tol. That file is what Table 7
was built from, so reproducing it proves this script is scoring the same
estimator on the same cells, and the only remaining difference is the
aggregation this script was written to change.

Runtime
-------
CPU only. No TensorFlow, no GPU. Needs numpy, Pillow and scikit-learn.
The random forest refit dominates; expect a few minutes for all four
task/space combinations.

Usage
-----
    python 25_baseline_layerwise.py                      # all four
    python 25_baseline_layerwise.py --only facies_to_vcl # one task
    python 25_baseline_layerwise.py --no-rf              # lookups only (fast)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
EXPORT = os.path.join(HERE, "export_v2")
BASELINE_CSV = os.path.join(OUT_DIR, "13_Conventional_baselines.csv")

# Fitted estimators are cached OUTSIDE the manuscript folder, so repeated
# invocations do not litter it. The device shell caps a call at 180 s and the
# forest alone takes ~115 s, so fitting and scoring have to be able to happen
# in separate calls.
CACHE_DIR = os.environ.get(
    "P2G_CACHE", os.path.join(os.path.expanduser("~"), ".p2g_cache"))

# (task, domain, space) -- mirrors FORWARD_SPACES in 14_baseline_conventional.py
FORWARD_SPACES = [
    ("facies_to_porosity", "porosity", "linear"),
    ("facies_to_permeability", "permeability", "linear"),
    ("facies_to_vcl", "vcl", "linear"),
    ("facies_to_permeability", "permeability", "log10"),
]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 14_baseline_conventional runs _selftest_neighbourhood() and
# _selftest_facies_features() at import time. That is deliberate: if either
# helper has drifted, the import fails here rather than producing wrong
# numbers downstream.
EV = _load("ev02", "02_evaluation.py")
BL = _load("bl14", "14_baseline_conventional.py")

N_CLS = len(EV.FACIES_LEGEND)


def _assert_palette_roundtrip() -> None:
    swatch = np.array(
        [[0, 0, 0]] + [EV.hex_to_rgb(h) for _, h in EV.FACIES_LEGEND],
        dtype=np.uint8).reshape(1, -1, 3)
    got = EV.decode_rgb_to_facies(swatch).ravel()
    want = np.arange(len(EV.FACIES_LEGEND) + 1, dtype=got.dtype)
    if not np.array_equal(got, want):
        raise AssertionError(f"palette does not round-trip: {got.tolist()}")


_assert_palette_roundtrip()


def adopt_legend(domain: str) -> bool:
    lm = os.path.join(EXPORT, domain, "legend_manifest.json")
    if os.path.exists(lm):
        EV.load_legend_manifest(lm)
        return True
    return False


def _r2(pred: np.ndarray, ref: np.ndarray) -> float:
    ss_res = float(np.sum((pred - ref) ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    err = pred - ref
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": _r2(pred, ref)}


def _layer_of(path: str) -> int:
    """Layer index from the filename, using 02_evaluation's own regex."""
    m = EV.LAYER_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse layer from {path!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# reference / mask extraction, matching 14_baseline_conventional.py exactly
# ---------------------------------------------------------------------------

def _ref_and_labels(path: str, domain: str, logsp: bool
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (value, mask, facies_labels) for one test pair, using the same
    masking rule as 14_baseline_conventional.py:
        m = vmask & (lab >= 1),  and for log10 additionally val > 0.
    """
    tgt, inp = BL.halves(path)
    val, vmask = BL.decode_cont(tgt, domain, EV)
    if logsp:
        vmask = vmask & (val > 0)
        val = np.log10(np.where(val > 0, val, 1.0))
    lab = EV.decode_rgb_to_facies(inp)
    return val, vmask & (lab >= 1), lab


def _aggregate(per_image: List[Tuple[int, Dict[str, float]]],
               pool_p: List[np.ndarray], pool_r: List[np.ndarray]
               ) -> Dict[str, object]:
    """
    Both conventions, computed the same way 24_l1_forward_metrics.py does:
    layerwise = per image -> mean within layer -> mean over layers.
    """
    per_layer: Dict[int, List[Dict[str, float]]] = {}
    for layer, m in per_image:
        per_layer.setdefault(layer, []).append(m)

    lay: Dict[str, float] = {}
    n_nan = 0
    for k in ("MAE", "RMSE", "R2"):
        per = [float(np.mean([m[k] for m in v])) for v in per_layer.values()]
        n_nan += int(np.sum(~np.isfinite(per)))
        lay[k] = float(np.mean(per))

    P, R = np.concatenate(pool_p), np.concatenate(pool_r)
    return {"layerwise": lay,
            "pooled": _metrics(P, R),
            "n_layers": len(per_layer),
            "n_images": len(per_image),
            "n_cells": int(P.size),
            "n_nonfinite_layers": n_nan}


# ---------------------------------------------------------------------------
# the three baselines
# ---------------------------------------------------------------------------

def fit_lookup_tables(task: str, domain: str, logsp: bool
                      ) -> Tuple[Dict[int, float], Dict[int, float], float]:
    """Class-mean and class-median tables, exactly as 14_baseline_conventional."""
    acc: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, N_CLS + 1)}
    train = BL.train_files(task)
    for p in train:
        val, m, lab = _ref_and_labels(p, domain, logsp)
        for c in range(1, N_CLS + 1):
            sel = val[m & (lab == c)]
            if sel.size:
                acc[c].append(sel[::BL.FIT_STRIDE])
    table_mean, table_med = {}, {}
    for c in range(1, N_CLS + 1):
        if acc[c]:
            a = np.concatenate(acc[c])
            table_mean[c], table_med[c] = float(a.mean()), float(np.median(a))
    gm = float(np.mean([v for v in table_mean.values()])) if table_mean else 0.0
    print(f"  fitted {len(table_mean)} class values from {len(train)} "
          f"training layers")
    return table_mean, table_med, gm


def _cache_path(task: str, space: str, kind: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{task}_{space}_{kind}.pkl")


def _cache_load(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:                      # noqa: BLE001
        print(f"  [cache] ignoring unreadable {os.path.basename(path)}: {exc}")
        return None


def _cache_dump(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def fit_random_forest(task: str, domain: str, logsp: bool, n_jobs: int = 1):
    """
    Random-forest regressor, exactly as 14_baseline_conventional.

    n_jobs is the ONE parameter allowed to differ. scikit-learn seeds each
    tree from random_state in a fixed sequence that does not depend on how
    many workers build them, so the fitted forest is identical; only peak
    memory and wall time change. This VM has ~3.9 GB, and 60 trees at
    depth 18 over 400k samples will not fit with the default n_jobs=-1.
    Every other hyperparameter, the feature construction, FIT_STRIDE and the
    rng(2025) subsample are untouched -- which is what the pooled validation
    at the end of this script proves.
    """
    from sklearn.ensemble import RandomForestRegressor
    Xs, ys = [], []
    for p in BL.train_files(task):
        val, m, lab = _ref_and_labels(p, domain, logsp)
        f = BL.facies_features(lab, N_CLS)
        Xs.append(np.ascontiguousarray(f[m][::BL.FIT_STRIDE]))
        ys.append(np.ascontiguousarray(val[m][::BL.FIT_STRIDE]))
        del f, val, m, lab
    Xf = np.concatenate(Xs)
    yf = np.concatenate(ys)
    del Xs, ys
    print(f"  features {Xf.shape} ({Xf.nbytes / 2**20:.0f} MB), "
          f"RSS {_rss_mb():.0f} MB")
    n = min(yf.size, 400_000)
    idx = np.random.default_rng(2025).choice(yf.size, n, replace=False)
    rfr = RandomForestRegressor(n_estimators=60, max_depth=18, n_jobs=n_jobs,
                                random_state=2025)
    rfr.fit(Xf[idx], yf[idx])
    print(f"  random-forest regressor fitted on {n:,} cells "
          f"(n_jobs={n_jobs}), RSS {_rss_mb():.0f} MB")
    return rfr


def score_lookup(task: str, domain: str, logsp: bool,
                 table: Dict[int, float], gm: float) -> Dict[str, object]:
    per_image: List[Tuple[int, Dict[str, float]]] = []
    pool_p: List[np.ndarray] = []
    pool_r: List[np.ndarray] = []
    for p in BL.test_files(task):
        val, m, lab = _ref_and_labels(p, domain, logsp)
        if not m.any():
            continue
        pred = np.full(val.shape, gm)
        for c, v in table.items():
            pred[lab == c] = v
        q, r = pred[m], val[m]
        per_image.append((_layer_of(p), _metrics(q, r)))
        pool_p.append(q)
        pool_r.append(r)
    return _aggregate(per_image, pool_p, pool_r)


def score_rf(task: str, domain: str, logsp: bool, rfr) -> Dict[str, object]:
    per_image: List[Tuple[int, Dict[str, float]]] = []
    pool_p: List[np.ndarray] = []
    pool_r: List[np.ndarray] = []
    for p in BL.test_files(task):
        val, m, lab = _ref_and_labels(p, domain, logsp)
        if not m.any():
            continue
        q = rfr.predict(BL.facies_features(lab, N_CLS)[m])
        r = val[m]
        per_image.append((_layer_of(p), _metrics(q, r)))
        pool_p.append(q)
        pool_r.append(r)
    return _aggregate(per_image, pool_p, pool_r)


# ---------------------------------------------------------------------------
# validation against the file Table 7 was built from
# ---------------------------------------------------------------------------

def published_pooled_r2() -> Dict[Tuple[str, str], float]:
    """
    {(task, 'random forest [linear]'): R2, ...} parsed from
    results/13_Conventional_baselines.csv -- forward direction, R2 rows only.
    """
    out: Dict[Tuple[str, str], float] = {}
    if not os.path.exists(BASELINE_CSV):
        return out
    with open(BASELINE_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["Direction"] != "forward":
                continue
            if not row["Metric"].startswith("R2"):
                continue
            out[(row["Task"], row["Baseline"])] = float(row["Value"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-rf", action="store_true",
                    help="skip the random forest (lookup tables only)")
    ap.add_argument("--rf-jobs", type=int, default=1,
                    help="n_jobs for the forest; does not change the fitted "
                         "model, only peak memory (default 1)")
    ap.add_argument("--fit-only", action="store_true",
                    help="fit and cache the estimators, then stop; lets the "
                         "~115 s forest fit and the scoring pass run in "
                         "separate shell calls")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore and do not write the estimator cache")
    ap.add_argument("--only", default=None,
                    help="restrict to one task, e.g. facies_to_vcl")
    ap.add_argument("--space", default=None, choices=["linear", "log10"],
                    help="restrict to one space; combine with --only to run a "
                         "single task/space per invocation (the device shell "
                         "caps a call at 180 s)")
    ap.add_argument("--tol", type=float, default=0.004,
                    help="tolerance when reproducing the published pooled R2")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=os.path.join(OUT_DIR,
                                                  "25_Baselines_both_conventions.csv"))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    published = published_pooled_r2()
    if not published:
        print(f"WARNING: no published values parsed from {BASELINE_CSV}")

    rows: List[dict] = []
    checks: List[Tuple[str, str, float, float]] = []

    combos = [c for c in FORWARD_SPACES
              if (args.only is None or c[0] == args.only)
              and (args.space is None or c[2] == args.space)]
    if not combos:
        print(f"no combination matches --only {args.only!r} "
              f"--space {args.space!r}")
        return 2

    for task, domain, space in combos:
        logsp = (space == "log10")
        if not adopt_legend(domain):
            print(f"  WARNING: no legend manifest for '{domain}'; "
                  f"using 02_evaluation defaults")
        print(f"\n{'=' * 76}")
        print(f"{task}  [{space}]")
        print(f"{'=' * 76}")

        t0 = time.time()
        lk_path = _cache_path(task, space, "lookup")
        cached = None if args.no_cache else _cache_load(lk_path)
        if cached is not None:
            table_mean, table_med, gm = cached
            print(f"  [cache] lookup tables ({len(table_mean)} classes)")
        else:
            table_mean, table_med, gm = fit_lookup_tables(task, domain, logsp)
            if not args.no_cache:
                _cache_dump(lk_path, (table_mean, table_med, gm))
            print(f"  [{time.time() - t0:6.1f}s] lookup tables fitted")

        work: List[Tuple[str, Dict[str, object]]] = []
        rfr = None
        if not args.no_rf:
            rf_path = _cache_path(task, space, "rf")
            rfr = None if args.no_cache else _cache_load(rf_path)
            if rfr is not None:
                print(f"  [cache] random forest "
                      f"({rfr.n_estimators} trees, n_jobs={rfr.n_jobs})")
            else:
                try:
                    t1 = time.time()
                    rfr = fit_random_forest(task, domain, logsp, args.rf_jobs)
                except ImportError:
                    print("  [note] scikit-learn not installed -- forest skipped")
                    rfr = None
                else:
                    if not args.no_cache:
                        _cache_dump(rf_path, rfr)
                    print(f"  [{time.time() - t1:6.1f}s] forest fitted "
                          f"and cached")

        if args.fit_only:
            print(f"  [--fit-only] estimators cached in {CACHE_DIR}; "
                  f"re-run without --fit-only to score "
                  f"({time.time() - t0:.1f}s)")
            continue

        if rfr is not None:
            t1 = time.time()
            work.append((f"random forest [{space}]",
                         score_rf(task, domain, logsp, rfr)))
            print(f"  [{time.time() - t1:6.1f}s] forest scored")
        t1 = time.time()
        work.append((f"class mean [{space}]",
                     score_lookup(task, domain, logsp, table_mean, gm)))
        work.append((f"class median [{space}]",
                     score_lookup(task, domain, logsp, table_med, gm)))
        print(f"  [{time.time() - t1:6.1f}s] lookups scored; "
              f"{time.time() - t0:.1f}s for this combination")

        for name, res in work:
            lay, pool = res["layerwise"], res["pooled"]
            print(f"  {name:<28} layerwise R2 = {lay['R2']:+.4f}   |   "
                  f"pooled R2 = {pool['R2']:+.4f}")
            if res["n_nonfinite_layers"]:
                print(f"      NOTE {res['n_nonfinite_layers']} non-finite "
                      f"layer mean(s) -- a test image had constant reference")
            rows.append({
                "task": task, "space": space, "baseline": name,
                "MAE_layerwise": lay["MAE"], "RMSE_layerwise": lay["RMSE"],
                "R2_layerwise": lay["R2"],
                "MAE_pooled": pool["MAE"], "RMSE_pooled": pool["RMSE"],
                "R2_pooled": pool["R2"],
                "n_layers": res["n_layers"], "n_images": res["n_images"],
                "n_cells": res["n_cells"],
            })
            if (task, name) in published:
                checks.append((task, name, published[(task, name)],
                               float(pool["R2"])))

    # ---------------- validation ------------------------------------------
    print("\n" + "=" * 76)
    print("VALIDATION  reproduce the published POOLED R2 "
          "(results/13_Conventional_baselines.csv)")
    print("=" * 76)
    if not checks:
        print("  nothing to check")
    worst = 0.0
    for task, name, want, got in checks:
        d = abs(want - got)
        worst = max(worst, d)
        flag = "OK" if d <= args.tol else "MISMATCH"
        print(f"  {task:<24} {name:<28} published {want:+.4f}   "
              f"recomputed {got:+.4f}   d={d:.4f}  {flag}")

    ok = bool(checks) and worst <= args.tol
    if args.validate and not ok:
        print(f"\nVALIDATION FAILED (worst d={worst:.4f} > tol={args.tol}). "
              f"The layerwise numbers are NOT trustworthy and were not written.")
        print("Do not use any number from this run in the manuscript.")
        return 1

    if not args.validate:
        print("\n[--no-validate] validation result ignored by request.")
    else:
        print(f"\nValidation passed (worst d={worst:.4f}). The layerwise "
              f"columns below come from the same code path that just "
              f"reproduced the published pooled values.")

    print("\n" + "=" * 76)
    print("LAYERWISE baselines -- directly comparable with Table 4 / Table 7")
    print("=" * 76)
    for r in rows:
        print(f"  {r['task']:<24} {r['baseline']:<28} "
              f"R2 = {r['R2_layerwise']:+.4f}   "
              f"(MAE {r['MAE_layerwise']:.4g}, RMSE {r['RMSE_layerwise']:.4g})")

    if rows:
        # Merge into any existing file, keyed by (task, space, baseline), so a
        # run restricted with --only/--space tops up the table instead of
        # truncating it. The device shell caps a call at 180 s, so the four
        # task/space combinations are driven one invocation at a time.
        fields = list(rows[0].keys())
        merged: Dict[Tuple[str, str, str], dict] = {}
        if os.path.exists(args.out):
            with open(args.out, newline="", encoding="utf-8") as fh:
                for old in csv.DictReader(fh):
                    merged[(old["task"], old["space"], old["baseline"])] = old
        n_before = len(merged)
        for r in rows:
            merged[(r["task"], r["space"], r["baseline"])] = r
        out_rows = [merged[k] for k in sorted(merged)]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {args.out}  ({len(out_rows)} rows total; "
              f"{len(rows)} from this run, {n_before} pre-existing)")

        done = {(r["task"], r["space"]) for r in out_rows}
        missing = [c for c in FORWARD_SPACES if (c[0], c[2]) not in done]
        if missing:
            print("still missing: " +
                  ", ".join(f"{t} [{s}]" for t, _, s in missing))
        else:
            print("all four task/space combinations present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
