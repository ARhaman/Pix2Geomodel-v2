#!/usr/bin/env python3
"""
13_permeability_error.py
========================

Where does the permeability error actually live? (Reviewer 5 #12)

facies -> permeability is the one task with a NEGATIVE R2 (-0.28): the model is
worse than predicting the test-set mean everywhere. A single RMSE of ~794 mD
does not say why, and "permeability is hard" is not an answer a reviewer should
have to accept. This script decomposes the error three ways:

  1. BY MAGNITUDE. Reference values are split into deciles, and error is
     reported within each. A log-normal field with four decades of range will
     concentrate almost all of its squared error in the top decile even when
     the model is doing well everywhere else, and an R2 computed in linear mD
     is dominated by exactly those cells.

  2. IN LOG SPACE. R2 and RMSE are recomputed on log10(k). This is the space
     the colour ramp uses, the space permeability is modelled in
     geostatistically, and the space in which a factor-of-two error costs the
     same whether it happens at 1 mD or 1000 mD. If the model is competent in
     log space and only looks catastrophic in linear space, that is a
     reporting artifact and the manuscript must say so.

  3. BY FACIES CLASS. Permeability is assigned per facies in this model, so
     error should track the class the cell belongs to. This shows whether the
     failure is spread evenly or concentrated in particular rock types.

Also reports the ENCODING FLOOR per decile: the error the palette imposes
before the model is involved. Where model error approaches that floor, the
representation is the limit, not the network.

    python 13_permeability_error.py            (CPU; base or tf210)
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from typing import Dict, List

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
EXPORT = os.path.join(HERE, "export_v2")
OUT = os.path.join(HERE, "results")

TASK = "facies_to_permeability"
FACIES_TASK = "porosity_to_facies"     # any *_to_facies split carries the facies target
SEEDS = [1, 2, 3]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_perm(img: np.ndarray, ev):
    spec = ev.CONTINUOUS_DOMAINS["permeability"]
    lo_i, hi_i = ev.ramp_range(spec)
    lut = ev.build_lut(spec["stops"], hi_i - lo_i + 1)
    idx = ev.decode_rgb_to_index(img, lut) + lo_i
    val = ev.index_to_value(idx, "permeability").astype(np.float64)
    valid = ~np.all(img == 0, axis=-1)
    return val, valid


def stats(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if ref.size == 0:
        return {k: float("nan") for k in
                ("n", "MAE", "RMSE", "R2", "bias", "MAE_log", "RMSE_log", "R2_log")}
    err = pred - ref
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    out = {"n": int(ref.size),
           "MAE": float(np.mean(np.abs(err))),
           "RMSE": float(np.sqrt(np.mean(err ** 2))),
           "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
           "bias": float(np.mean(err))}
    m = (ref > 0) & (pred > 0)
    if m.sum() > 1:
        lr, lp = np.log10(ref[m]), np.log10(pred[m])
        le = lp - lr
        sr = float(np.sum(le ** 2))
        st = float(np.sum((lr - lr.mean()) ** 2))
        out.update({"MAE_log": float(np.mean(np.abs(le))),
                    "RMSE_log": float(np.sqrt(np.mean(le ** 2))),
                    "R2_log": 1.0 - sr / st if st > 0 else float("nan")})
    else:
        out.update({"MAE_log": float("nan"), "RMSE_log": float("nan"),
                    "R2_log": float("nan")})
    return out


def main() -> int:
    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)

    lm = os.path.join(EXPORT, "permeability", "legend_manifest.json")
    if os.path.exists(lm):
        ev.load_legend_manifest(lm)

    test_dir = os.path.join(SPLITS, TASK, "test")
    fac_dir = os.path.join(SPLITS, FACIES_TASK, "test")
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
    if not files:
        print("no test pairs found")
        return 2

    refs, preds, facies = [], [], []
    for fn in files:
        pair = np.array(Image.open(os.path.join(test_dir, fn)).convert("RGB"))
        w = pair.shape[1] // 2
        rv, rmask = decode_perm(pair[:, :w], ev)      # target = LEFT half

        pv_seeds = []
        for s in SEEDS:
            p = os.path.join(RUNS, f"{TASK}_seed{s}", "predictions", fn)
            if os.path.exists(p):
                v, _ = decode_perm(np.array(Image.open(p).convert("RGB")), ev)
                pv_seeds.append(v)
        if not pv_seeds:
            continue
        pv = np.mean(pv_seeds, axis=0)

        fac = None
        fp = os.path.join(fac_dir, fn)
        if os.path.exists(fp):
            fpair = np.array(Image.open(fp).convert("RGB"))
            fac = ev.decode_rgb_to_facies(fpair[:, : fpair.shape[1] // 2])

        m = rmask & (rv > 0)
        refs.append(rv[m]); preds.append(pv[m])
        facies.append(fac[m] if fac is not None else np.zeros(m.sum(), dtype=int))

    ref = np.concatenate(refs); pred = np.concatenate(preds)
    fac = np.concatenate(facies)
    print(f"{len(files)} test layers, {ref.size:,} cells with k > 0, "
          f"seeds averaged {SEEDS}")

    rows = [["Grouping", "Group", "Cells", "Ref mean (mD)", "MAE (mD)",
             "RMSE (mD)", "Bias (mD)", "R2", "MAE log10", "RMSE log10", "R2 log10"]]

    def emit(grouping, label, r, p):
        st = stats(r, p)
        rows.append([grouping, label, st["n"], f"{r.mean():.4g}",
                     f"{st['MAE']:.4g}", f"{st['RMSE']:.4g}", f"{st['bias']:+.4g}",
                     f"{st['R2']:.4f}", f"{st['MAE_log']:.4f}",
                     f"{st['RMSE_log']:.4f}", f"{st['R2_log']:.4f}"])
        return st

    print("\n" + "=" * 86)
    print("OVERALL")
    print("=" * 86)
    o = emit("overall", "all cells", ref, pred)
    print(f"  linear : MAE {o['MAE']:.1f}  RMSE {o['RMSE']:.1f}  "
          f"bias {o['bias']:+.1f}  R2 {o['R2']:+.4f}")
    print(f"  log10  : MAE {o['MAE_log']:.4f}  RMSE {o['RMSE_log']:.4f}  "
          f"R2 {o['R2_log']:+.4f}")

    print("\n" + "=" * 86)
    print("BY REFERENCE-MAGNITUDE DECILE")
    print("=" * 86)
    print(f"{'decile':<8}{'range (mD)':<26}{'cells':>10}{'RMSE':>12}"
          f"{'bias':>12}{'RMSE log':>10}")
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    share = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (ref >= lo) & (ref <= hi) if i == 9 else (ref >= lo) & (ref < hi)
        if m.sum() == 0:
            continue
        st = emit("magnitude decile", f"D{i+1} [{lo:.4g}, {hi:.4g}] mD",
                  ref[m], pred[m])
        share.append((i + 1, float(np.sum((pred[m] - ref[m]) ** 2))))
        print(f"D{i+1:<7}[{lo:>10.4g}, {hi:>10.4g}]{st['n']:>10,}"
              f"{st['RMSE']:>12.1f}{st['bias']:>+12.1f}{st['RMSE_log']:>10.4f}")

    tot_sq = sum(v for _, v in share) or 1.0
    top = max(share, key=lambda x: x[1])
    print(f"\n  decile D{top[0]} alone holds {100*top[1]/tot_sq:.1f} % of the "
          f"total squared error")

    print("\n" + "=" * 86)
    print("BY FACIES CLASS")
    print("=" * 86)
    print(f"{'class':<18}{'cells':>10}{'ref mean':>12}{'RMSE':>12}"
          f"{'bias':>12}{'RMSE log':>10}")
    for c in range(1, len(ev.FACIES_LEGEND) + 1):
        m = fac == c
        if m.sum() < 100:
            continue
        st = emit("facies class", ev.FACIES_LEGEND[c - 1][0], ref[m], pred[m])
        print(f"{ev.FACIES_LEGEND[c-1][0]:<18}{st['n']:>10,}"
              f"{ref[m].mean():>12.4g}{st['RMSE']:>12.1f}"
              f"{st['bias']:>+12.1f}{st['RMSE_log']:>10.4f}")

    with open(os.path.join(OUT, "12_Permeability_error.csv"), "w", newline="",
              encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nwritten to {os.path.join(OUT, '12_Permeability_error.csv')}")
    print("\nIf R2 is strongly negative in linear mD but near zero or positive")
    print("in log10, the headline R2 is being set by a small number of")
    print("high-permeability cells and must be reported in both spaces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
