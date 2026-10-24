#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_error_maps.py
================

Reviewer 1, comment 1: the decomposition of the forward gain into "spatial
context" and "network contribution" "needs to be supported by a U-Net without
the adversarial term or by pixel-wise error maps to demonstrate exactly where
the network outperforms the random forest."

24/28 supply the first. This script supplies the second, which is the one that
actually answers *where*: the no-adversarial arm compares the network with
itself, not with the forest.

What it computes, per forward task
----------------------------------
  * the per-pixel absolute error of Pix2Geomodel, averaged over the three
    seeds -- the expected error of one model, NOT an ensemble prediction,
    which is the construction criticised elsewhere in this revision;
  * the per-pixel absolute error of the random forest, loaded from the
    estimator 25_baseline_layerwise.py already fitted and cached, so it is
    the same forest that produced the Table 7 column;
  * their difference, |e_RF| - |e_Pix2Geomodel|, so positive means the network
    is the better of the two at that cell;
  * both errors binned by distance to the nearest facies-class boundary,
    over all nine test layers. This is the quantitative half, and it tests the
    mechanism the manuscript claims: that much of porosity's within-class
    variation is a function of position relative to class boundaries, which a
    neighbourhood descriptor already captures.

The map panels use one representative test layer, chosen as the layer whose
network R2 is the MEDIAN of the nine, so it is neither the best nor the worst
case. The layer index is recorded in the output.

Distance to boundary is computed by iterative dilation from the class
boundaries, so no SciPy dependency is introduced.

Runs one task per invocation to stay inside the shell's call limit:

    python 31_error_maps.py --only facies_to_porosity   --space linear
    python 31_error_maps.py --only facies_to_vcl        --space linear
    python 31_error_maps.py --only facies_to_permeability --space log10

Each writes results/_errmap_<task>_<space>.npz. CPU only.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
CACHE = os.environ.get(
    "P2G_CACHE", os.path.join(os.path.expanduser("~"), ".p2g_cache"))

FORWARD = {"facies_to_porosity": "porosity",
           "facies_to_vcl": "vcl",
           "facies_to_permeability": "permeability"}

# distance-to-boundary bins, in cells
BIN_EDGES = [0, 2, 4, 8, 16, 32]
BIN_LABELS = ["0-1", "2-3", "4-7", "8-15", "16-31", "32+"]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M24 = _load("m24", "24_l1_forward_metrics.py")
BL = _load("bl14", "14_baseline_conventional.py")
EV = M24.EV
N_CLS = len(EV.FACIES_LEGEND)


def boundary_distance(lab: np.ndarray, maxd: int = 32) -> np.ndarray:
    """Chebyshev-ish distance to the nearest facies-class boundary."""
    b = np.zeros(lab.shape, dtype=bool)
    d0 = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d0
    b[1:, :] |= d0
    d1 = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d1
    b[:, 1:] |= d1

    dist = np.full(lab.shape, maxd, dtype=np.int16)
    cur = b.copy()
    dist[cur] = 0
    for k in range(1, maxd):
        if not cur.any():
            break
        nxt = np.zeros_like(cur)
        nxt[:-1, :] |= cur[1:, :]
        nxt[1:, :] |= cur[:-1, :]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[:, 1:] |= cur[:, :-1]
        nxt &= dist > k
        dist[nxt] = k
        cur = nxt
    return dist


def down2(a: np.ndarray, k: int = 2) -> np.ndarray:
    """Block-mean downsample, purely to keep the saved maps small."""
    h, w = a.shape
    h -= h % k
    w -= w % k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", required=True, choices=list(FORWARD))
    ap.add_argument("--space", default="linear", choices=["linear", "log10"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--splits-root", default="splits_v2")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    task = args.only
    domain = FORWARD[task]
    space = args.space
    logsp = (space == "log10")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    M24.adopt_legend(domain)

    rf_path = os.path.join(CACHE, f"{task}_{space}_rf.pkl")
    if not os.path.exists(rf_path):
        print(f"ERROR: no cached forest at {rf_path}\n"
              f"run 25_baseline_layerwise.py --only {task} --space {space} "
              f"--fit-only first.")
        return 1
    with open(rf_path, "rb") as fh:
        rfr = pickle.load(fh)
    print(f"loaded forest ({rfr.n_estimators} trees) for {task} [{space}]")

    ref_dir = os.path.join(args.splits_root, task, "test")
    files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".png"))

    store: Dict[int, dict] = {}
    frac_better: Dict[int, float] = {}
    nb = len(BIN_EDGES)
    sum_p = np.zeros(nb)
    sum_r = np.zeros(nb)
    cnt = np.zeros(nb)

    for fn in files:
        path = os.path.join(ref_dir, fn)
        m = EV.LAYER_RE.search(fn)
        layer = int(m.group(1)) if m else -1

        # reference property (target half) and facies input (other half)
        tgt, inp = BL.halves(path)
        val, vmask = BL.decode_cont(tgt, domain, EV)
        if logsp:
            vmask = vmask & (val > 0)
            val = np.log10(np.where(val > 0, val, 1.0))
        lab = EV.decode_rgb_to_facies(inp)
        mask = vmask & (lab >= 1)
        if not mask.any():
            continue

        # ---- network: mean |error| over seeds (not an ensemble prediction) --
        errs = []
        for s in seeds:
            p = os.path.join(args.runs_root, f"{task}_seed{s}",
                             "predictions", fn)
            if not os.path.exists(p):
                continue
            pv = M24._decode_property(M24._read_rgb(p), domain)
            if logsp:
                dm = EV.CONTINUOUS_DOMAINS[domain]
                pv = np.log10(np.clip(pv, dm["vmin"], dm["vmax"]))
            errs.append(np.abs(pv - val))
        if not errs:
            print(f"  {fn}: no network predictions found")
            continue
        e_net = np.mean(errs, axis=0)

        # ---- random forest -------------------------------------------------
        q = rfr.predict(BL.facies_features(lab, N_CLS)[mask])
        e_rf_flat = np.abs(q - val[mask])
        e_rf = np.zeros_like(val)
        e_rf[mask] = e_rf_flat

        # ---- boundary-distance binning ------------------------------------
        dist = boundary_distance(lab)
        dm_ = dist[mask]
        en = e_net[mask]
        er = e_rf_flat
        idx = np.digitize(dm_, BIN_EDGES) - 1
        idx = np.clip(idx, 0, nb - 1)
        for k in range(nb):
            sel = idx == k
            if sel.any():
                sum_p[k] += en[sel].sum()
                sum_r[k] += er[sel].sum()
                cnt[k] += sel.sum()

        frac_better[layer] = float(np.mean(en < er))
        store[layer] = {
            "ref": down2(np.where(mask, val, np.nan)).astype(np.float32),
            "e_net": down2(np.where(mask, e_net, np.nan)).astype(np.float32),
            "e_rf": down2(np.where(mask, e_rf, np.nan)).astype(np.float32),
        }
        print(f"  layer {layer:2d}: net MAE {en.mean():.4g}  "
              f"rf MAE {er.mean():.4g}  net better on "
              f"{100.0 * np.mean(en < er):.1f} % of cells")

    if not store:
        print("no layers processed")
        return 1

    # representative layer = median network MAE, so neither best nor worst
    maes = {k: float(np.nanmean(v["e_net"])) for k, v in store.items()}
    order = sorted(maes, key=maes.get)
    rep = order[len(order) // 2]
    print(f"representative layer (median network MAE): {rep}")

    out = os.path.join(OUT_DIR, f"_errmap_{task}_{space}.npz")
    np.savez_compressed(
        out,
        task=task, space=space, rep_layer=rep,
        ref=store[rep]["ref"], e_net=store[rep]["e_net"],
        e_rf=store[rep]["e_rf"],
        bin_labels=np.array(BIN_LABELS),
        mae_net_by_dist=sum_p / np.maximum(cnt, 1),
        mae_rf_by_dist=sum_r / np.maximum(cnt, 1),
        n_by_dist=cnt,
        mae_net_all=float(sum_p.sum() / max(cnt.sum(), 1)),
        mae_rf_all=float(sum_r.sum() / max(cnt.sum(), 1)),
    )
    print(f"wrote {out}")
    print("  MAE by distance to facies boundary (cells):")
    for k, lb in enumerate(BIN_LABELS):
        if cnt[k]:
            print(f"    {lb:>6}  network {sum_p[k]/cnt[k]:.4g}   "
                  f"forest {sum_r[k]/cnt[k]:.4g}   n={int(cnt[k]):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
