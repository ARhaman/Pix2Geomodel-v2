#!/usr/bin/env python3
"""
24_l1_forward_metrics.py
========================

Forward-task physical-unit metrics for the NO-ADVERSARIAL arm, for
Pix2Geomodel V2 (RINENG-D-26-10842), Reviewer 1 round-2 Comment 1.

WHY THIS SCRIPT EXISTS
----------------------
Reviewer 1 asks for the forward gain to be decomposed with the help of "a
U-Net without the adversarial term". That arm already exists: the `*_l1`
runs are the same U-Net generator trained with the L1 reconstruction term
alone. Its forward results are in results/L1_4_Physical_units.csv, but only
in LINEAR space, and the manuscript reports facies->permeability in log10.
This script supplies the missing log10 value and, at the same time, the
linear values that are already known -- which is the point: the known values
are the validation.

SELF-VALIDATION
---------------
Run with --validate (the default) and the script first recomputes the LINEAR
R2 for all three forward tasks and compares them against the published
L1_4_Physical_units.csv. If they do not agree to within a small tolerance,
the aggregation convention here does not match the one behind the
manuscript's tables and the log10 number it produces could not be placed
beside them. In that case it stops rather than emit a number.

Eight of the nine defects found in this pipeline produced well-formed and
plausible numbers rather than errors, so a new number is not trusted here
until an old one has been reproduced with the same code path.

TWO AGGREGATIONS ARE REPORTED, deliberately
-------------------------------------------
  LAYERWISE  per image -> mean within a layer -> mean over the nine layers,
             then mean +/- SD across the three seeds. This is what
             02_evaluation.evaluate_task does and what Table 4 reports.
  POOLED     every valid cell of every test image in one accumulation.
             This is what 12_Permeability_error.csv reports for the
             log10 row that Table 7 currently carries.

The manuscript mixes the two in one table, so both are printed and the
caller can place like with like.

USAGE
-----
    python 24_l1_forward_metrics.py                  # validate, then report
    python 24_l1_forward_metrics.py --no-validate    # report regardless
    python 24_l1_forward_metrics.py --arm full       # same metrics, GAN arm

OUTPUT
------
    results/24_L1_forward_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover
    Image = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
EXPORT = os.path.join(HERE, "export_v2")

FORWARD = [("facies_to_porosity", "porosity"),
           ("facies_to_permeability", "permeability"),
           ("facies_to_vcl", "vcl")]

# Published linear-space R2 of the L1-only arm, from results/L1_4_Physical_units.csv.
# These are the values this script must reproduce before its log10 number is used.
PUBLISHED_L1_LINEAR = {"facies_to_porosity": 0.4528,
                       "facies_to_permeability": -0.3577,
                       "facies_to_vcl": 0.8612}
PUBLISHED_FULL_LINEAR = {"facies_to_porosity": 0.5754,
                         "facies_to_permeability": -0.2818,
                         "facies_to_vcl": 0.8543}


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load("ev02", "02_evaluation.py")


def _assert_palette_roundtrip() -> None:
    """Every legend colour must decode to its own index before anything else."""
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


def _read_rgb(path: str) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required")
    return np.array(Image.open(path).convert("RGB"))


def _decode_property(rgb: np.ndarray, domain: str) -> np.ndarray:
    d = EV.CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = EV.ramp_range(d)
    lut = EV.build_lut(d["stops"], hi_i - lo_i + 1)
    return EV.index_to_value(EV.decode_rgb_to_index(rgb, lut) + lo_i, domain)


def _r2(pred: np.ndarray, ref: np.ndarray) -> float:
    ss_res = float(np.sum((pred - ref) ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score_seed(pred_dir: str, ref_dir: str, domain: str, logsp: bool,
               target_on_left: bool = True) -> Optional[dict]:
    """
    Returns both aggregations for one seed.

    The target half of the reference pair carries the property; the input half
    carries facies. Cells outside the reservoir are NODATA-black in the
    reference and are excluded -- the mask is always taken from the reference,
    never from the prediction, because a generator can emit near-black cells
    inside the reservoir and that would silently shrink the evaluated area.
    """
    pairs = EV.collect_pairs(pred_dir, ref_dir)
    if not pairs:
        return None

    per_layer: Dict[int, List[dict]] = {}
    pool_p: List[np.ndarray] = []
    pool_r: List[np.ndarray] = []

    for spec in pairs:
        pred_rgb = _read_rgb(spec.pred_path)
        ref_img = _read_rgb(spec.ref_path)
        w = ref_img.shape[1] // 2
        ref_rgb = ref_img[:, :w] if target_on_left else ref_img[:, w:]
        if pred_rgb.shape != ref_rgb.shape:
            raise ValueError(f"shape mismatch {os.path.basename(spec.pred_path)}: "
                             f"{pred_rgb.shape} vs {ref_rgb.shape}")

        pv = _decode_property(pred_rgb, domain)
        rv = _decode_property(ref_rgb, domain)
        valid = ~np.all(ref_rgb == 0, axis=-1)

        if logsp:
            dm = EV.CONTINUOUS_DOMAINS[domain]
            pv = np.log10(np.clip(pv, dm["vmin"], dm["vmax"]))
            rv = np.log10(np.clip(rv, dm["vmin"], dm["vmax"]))

        p, r = pv[valid], rv[valid]
        if p.size == 0:
            continue
        err = p - r
        per_layer.setdefault(spec.layer, []).append({
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": _r2(p, r)})
        pool_p.append(p)
        pool_r.append(r)

    if not per_layer:
        return None

    # layerwise: collapse slices within a layer, then average the layers
    lay = {}
    for k in ("MAE", "RMSE", "R2"):
        per = [float(np.mean([m[k] for m in v])) for v in per_layer.values()]
        lay[k] = float(np.mean(per))

    P, R = np.concatenate(pool_p), np.concatenate(pool_r)
    E = P - R
    pool = {"MAE": float(np.mean(np.abs(E))),
            "RMSE": float(np.sqrt(np.mean(E ** 2))),
            "R2": _r2(P, R)}
    return {"layerwise": lay, "pooled": pool,
            "n_layers": len(per_layer), "n_cells": int(P.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-root", default="splits_v2")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--arm", default="l1", choices=["l1", "full"],
                    help="l1 = no-adversarial (*_l1 runs); full = the GAN arm")
    ap.add_argument("--tol", type=float, default=0.004,
                    help="tolerance when reproducing the published linear R2")
    ap.add_argument("--no-validate", dest="validate", action="store_false")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "24_L1_forward_metrics.csv"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    suffix = "_l1" if args.arm == "l1" else ""
    published = PUBLISHED_L1_LINEAR if args.arm == "l1" else PUBLISHED_FULL_LINEAR

    rows: List[dict] = []
    summary: Dict[Tuple[str, str], dict] = {}

    for task, domain in FORWARD:
        spaces = [("linear", False)]
        if domain == "permeability":
            spaces.append(("log10", True))

        ref_dir = os.path.join(args.splits_root, task, "test")
        if not os.path.isdir(ref_dir):
            print(f"  SKIP {task}: missing {ref_dir}")
            continue

        for space, logsp in spaces:
            if not adopt_legend(domain):
                print(f"  WARNING: no legend manifest for '{domain}'; using "
                      f"02_evaluation defaults")
            print(f"\n=== {task}  [{args.arm} arm, {space}] ===")
            per_seed = []
            for s in seeds:
                pdir = os.path.join(args.runs_root, f"{task}_seed{s}{suffix}",
                                    "predictions")
                if not os.path.isdir(pdir):
                    print(f"  seed{s}: missing {pdir}")
                    continue
                res = score_seed(pdir, ref_dir, domain, logsp)
                if res is None:
                    print(f"  seed{s}: no matching pairs")
                    continue
                per_seed.append(res)
                print(f"  seed{s}: layerwise R2={res['layerwise']['R2']:+.4f}  "
                      f"pooled R2={res['pooled']['R2']:+.4f}  "
                      f"({res['n_layers']} layers, {res['n_cells']:,} cells)")
                rows.append({"arm": args.arm, "task": task, "space": space,
                             "seed": s,
                             "MAE_layerwise": res["layerwise"]["MAE"],
                             "RMSE_layerwise": res["layerwise"]["RMSE"],
                             "R2_layerwise": res["layerwise"]["R2"],
                             "MAE_pooled": res["pooled"]["MAE"],
                             "RMSE_pooled": res["pooled"]["RMSE"],
                             "R2_pooled": res["pooled"]["R2"],
                             "n_layers": res["n_layers"],
                             "n_cells": res["n_cells"]})
            if not per_seed:
                continue
            agg = {}
            for conv in ("layerwise", "pooled"):
                for k in ("MAE", "RMSE", "R2"):
                    v = np.array([r[conv][k] for r in per_seed])
                    agg[f"{conv}_{k}_mean"] = float(v.mean())
                    agg[f"{conv}_{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
            summary[(task, space)] = agg
            print(f"  MEAN:   layerwise R2 = {agg['layerwise_R2_mean']:+.4f} "
                  f"± {agg['layerwise_R2_sd']:.4f}   |   "
                  f"pooled R2 = {agg['pooled_R2_mean']:+.4f} "
                  f"± {agg['pooled_R2_sd']:.4f}")

    # ---------------- validation ------------------------------------------
    print("\n" + "=" * 72)
    print(f"VALIDATION  reproduce the published LINEAR R2 of the {args.arm} arm")
    print("=" * 72)
    ok = True
    for task, _ in FORWARD:
        key = (task, "linear")
        if key not in summary:
            print(f"  {task:26} not scored"); ok = False; continue
        got = summary[key]["layerwise_R2_mean"]
        want = published[task]
        diff = abs(got - want)
        flag = "OK" if diff <= args.tol else "** MISMATCH"
        print(f"  {task:26} published {want:+.4f}   recomputed {got:+.4f}   "
              f"d={diff:.4f}  {flag}")
        ok &= diff <= args.tol

    if not ok:
        print("\nThe published linear values were NOT reproduced. The aggregation here "
              "does not match the one behind the manuscript tables, so the log10 "
              "number must not be placed beside them. Nothing written.")
        return 1

    print("\nValidation passed. The log10 value below is produced by the same code "
          "path that just reproduced the published linear values.")
    key = ("facies_to_permeability", "log10")
    if key in summary:
        a = summary[key]
        print(f"\n  facies_to_permeability, log10, {args.arm} arm:")
        print(f"      layerwise R2 = {a['layerwise_R2_mean']:+.4f} ± {a['layerwise_R2_sd']:.4f}")
        print(f"      pooled    R2 = {a['pooled_R2_mean']:+.4f} ± {a['pooled_R2_sd']:.4f}")
        print(f"      layerwise MAE = {a['layerwise_MAE_mean']:.4f}, "
              f"RMSE = {a['layerwise_RMSE_mean']:.4f}")

    if rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        keys = list(rows[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
