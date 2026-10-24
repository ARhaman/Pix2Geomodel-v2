#!/usr/bin/env python3
"""
01_layer_level_split.py
=======================

Leakage-free dataset construction for Pix2Geomodel V2 (RINENG-D-26-10842).

WHY THIS SCRIPT EXISTS
----------------------
The original pipeline (`3.Paird images+spilit.ipynb`) did this:

    all_images = original_images + augmented_images
    random.shuffle(all_images)              # <-- no seed
    train = all_images[:int(0.7*n)] ...

Augmented copies of the same geological layer therefore appeared in both the
training and the test sets, so the 252 "test images" were not independent
samples (Reviewer 1 #4, Reviewer 5 #3). The split was also unseeded and the
files were renamed to `1.png, 2.png, ...`, which destroyed layer provenance
and made the split impossible to reconstruct.

This script fixes all of that:

  1. The split is made over ORIGINAL LAYERS, before any augmentation.
  2. Augmentation is applied to TRAINING layers only. Validation and test
     sets contain original, un-augmented slices exclusively.
  3. By default the split is CONTIGUOUS IN DEPTH with a buffer gap between
     partitions. Adjacent geological layers are highly correlated, so a
     purely random layer split still leaks neighbour information -- exactly
     the "inter-layer similarity" Reviewer 1 #4 raises. Blocked splitting
     with a gap removes that adjacency entirely.
  4. Every random draw is seeded, per layer, so the split and the
     augmentation are exactly reproducible and independent of subset size.
  5. Layer provenance is preserved in every output filename, and a manifest
     records precisely which layer went where.

USAGE
-----
    python 01_layer_level_split.py --config config.json          # normal run
    python 01_layer_level_split.py --config config.json --dry-run  # plan only

Author: prepared for the RINENG-D-26-10842 revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

# The exported slices are 8184 x 12045 px (98.6 Mpx), above Pillow's default
# decompression-bomb threshold. These are our own files, so lift the limit
# rather than let Pillow warn on every open (or refuse outright).
Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------
# Augmentation policy -- IDENTICAL to the original study so that the new
# results remain comparable. The only change is that the RNG is seeded per
# layer rather than shared across the whole file list, which makes each
# layer's augmentation reproducible regardless of how many layers precede it.
# --------------------------------------------------------------------------

AUGS_PER_LAYER = 30
ROTATION_DEG = 10          # +/- degrees
SHIFT_FRAC = 0.10          # fraction of width/height
SHEAR_X_MAX = 0.10         # tan(theta) along x
SHEAR_Y_MAX = 0.10         # tan(theta) along y
ZOOM_FRAC = 0.10           # +/- 10 %
H_FLIP = True

LAYER_RE = re.compile(r"_layer_(\d+)_", re.IGNORECASE)


# --------------------------------------------------------------------------
# Palette-safe augmentation helpers (ported from 2.High-Resolution Augmentation)
# --------------------------------------------------------------------------

def _guess_background_index(img_p: Image.Image) -> int:
    """Modal value on a 2 px border, so 'no data' keeps its colour."""
    idx = np.array(img_p, dtype=np.uint8)
    border = np.concatenate([idx[:2, :].ravel(), idx[-2:, :].ravel(),
                             idx[:, :2].ravel(), idx[:, -2:].ravel()])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[np.argmax(counts)])


def _ensure_palette(img_p: Image.Image, palette) -> None:
    if palette:
        img_p.putpalette(palette)


def _center_zoom(img_p: Image.Image, scale: float, bg: int, palette) -> Image.Image:
    w, h = img_p.size
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    res = img_p.resize((nw, nh), resample=Image.NEAREST)
    _ensure_palette(res, palette)
    if scale >= 1.0:
        left, top = (nw - w) // 2, (nh - h) // 2
        out = res.crop((left, top, left + w, top + h))
        _ensure_palette(out, palette)
        return out
    bg_img = Image.new("P", (w, h), color=bg)
    _ensure_palette(bg_img, palette)
    bg_img.paste(res, ((w - nw) // 2, (h - nh) // 2))
    return bg_img


def _shear(img_p: Image.Image, shx: float, shy: float, bg: int, palette) -> Image.Image:
    out = img_p.transform(img_p.size, Image.AFFINE,
                          (1.0, -shx, 0.0, -shy, 1.0, 0.0),
                          resample=Image.NEAREST, fillcolor=bg)
    _ensure_palette(out, palette)
    return out


def _translate(img_p: Image.Image, dx: int, dy: int, bg: int, palette) -> Image.Image:
    out = img_p.transform(img_p.size, Image.AFFINE,
                          (1.0, 0.0, -dx, 0.0, 1.0, -dy),
                          resample=Image.NEAREST, fillcolor=bg)
    _ensure_palette(out, palette)
    return out


def augment_layer_image(img_p: Image.Image, rng: random.Random) -> Image.Image:
    """One augmented realisation, using the study's original policy."""
    palette = img_p.getpalette()
    bg = _guess_background_index(img_p)
    im = img_p.copy()

    if H_FLIP and rng.random() < 0.5:
        im = ImageOps.mirror(im)
        _ensure_palette(im, palette)

    im = _center_zoom(im, rng.uniform(1.0 - ZOOM_FRAC, 1.0 + ZOOM_FRAC), bg, palette)
    im = _shear(im,
                rng.uniform(-SHEAR_X_MAX, SHEAR_X_MAX),
                rng.uniform(-SHEAR_Y_MAX, SHEAR_Y_MAX), bg, palette)

    im = im.rotate(rng.uniform(-ROTATION_DEG, ROTATION_DEG),
                   resample=Image.NEAREST, expand=False, fillcolor=bg)
    _ensure_palette(im, palette)

    w, h = im.size
    im = _translate(im,
                    int(round(rng.uniform(-SHIFT_FRAC, SHIFT_FRAC) * w)),
                    int(round(rng.uniform(-SHIFT_FRAC, SHIFT_FRAC) * h)),
                    bg, palette)
    return im


# --------------------------------------------------------------------------
# Layer discovery and splitting
# --------------------------------------------------------------------------

@dataclass
class SplitPlan:
    mode: str
    seed: int
    gap: int
    train: List[int]
    val: List[int]
    test: List[int]
    dropped_gap: List[int]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_train"], d["n_val"], d["n_test"] = len(self.train), len(self.val), len(self.test)
        return d


def discover_layers(domain_dir: str) -> Dict[int, str]:
    """Map layer number -> file path for one property domain."""
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
    out: Dict[int, str] = {}
    for fn in sorted(os.listdir(domain_dir)):
        if not fn.lower().endswith(".png"):
            continue
        m = LAYER_RE.search(fn)
        if not m:
            continue
        k = int(m.group(1))
        if k in out:
            raise ValueError(f"Duplicate layer {k} in {domain_dir}: {fn} and {out[k]}")
        out[k] = os.path.join(domain_dir, fn)
    if not out:
        raise ValueError(f"No '*_layer_<n>_*.png' files found in {domain_dir}")
    return out


def make_split(layers: Sequence[int], mode: str, seed: int,
               ratios: Tuple[float, float, float], gap: int) -> SplitPlan:
    """
    Partition ORIGINAL layers into train/val/test.

    mode='blocked'  contiguous depth blocks separated by `gap` buffer layers
                    that are discarded. Removes adjacency leakage. RECOMMENDED.
    mode='random'   seeded random assignment at layer level. Removes augmentation
                    leakage but NOT adjacency leakage; provided for comparison.
    """
    ls = sorted(layers)
    n = len(ls)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    if mode == "random":
        rng = random.Random(seed)
        shuffled = ls[:]
        rng.shuffle(shuffled)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        return SplitPlan("random", seed, 0,
                         sorted(shuffled[:n_tr]),
                         sorted(shuffled[n_tr:n_tr + n_va]),
                         sorted(shuffled[n_tr + n_va:]),
                         [])

    if mode != "blocked":
        raise ValueError("mode must be 'blocked' or 'random'")

    # Blocked: [ train | gap | val | gap | test ]
    usable = n - 2 * gap
    if usable < 6:
        raise ValueError(f"Too few layers ({n}) for gap={gap}")
    n_tr = int(math.floor(ratios[0] * usable))
    n_va = int(math.floor(ratios[1] * usable))
    n_te = usable - n_tr - n_va

    i = 0
    train = ls[i:i + n_tr]; i += n_tr
    gap_a = ls[i:i + gap]; i += gap
    val = ls[i:i + n_va]; i += n_va
    gap_b = ls[i:i + gap]; i += gap
    test = ls[i:i + n_te]

    return SplitPlan("blocked", seed, gap, train, val, test, gap_a + gap_b)


def verify_split(plan: SplitPlan) -> None:
    """Fail loudly rather than silently producing a leaky dataset."""
    s_tr, s_va, s_te = set(plan.train), set(plan.val), set(plan.test)
    assert not (s_tr & s_va), f"Layer overlap train/val: {sorted(s_tr & s_va)}"
    assert not (s_tr & s_te), f"Layer overlap train/test: {sorted(s_tr & s_te)}"
    assert not (s_va & s_te), f"Layer overlap val/test: {sorted(s_va & s_te)}"
    assert len(s_tr) and len(s_va) and len(s_te), "Empty partition"
    if plan.mode == "blocked" and plan.gap > 0:
        assert max(s_tr) + plan.gap < min(s_va), "Gap not respected between train and val"
        assert max(s_va) + plan.gap < min(s_te), "Gap not respected between val and test"


# --------------------------------------------------------------------------
# Paired-image construction
# --------------------------------------------------------------------------

def _fit_half(img: Image.Image, half_w: int, half_h: int,
              preserve_aspect: bool) -> Image.Image:
    """
    Resize one half of the pair to the training resolution.

    preserve_aspect=False  reproduces the original study exactly: a direct
        squeeze from 8184 x 12045 (aspect 0.679) to 1024 x 2048 (aspect 0.500),
        i.e. a ~26 % anisotropic horizontal compression. Keep this as the
        default so the new results remain comparable with the published ones.

    preserve_aspect=True   letterboxes instead, so cell geometry is preserved
        and directional variograms are not biased. Use this for the resolution
        ablation (Reviewer 5 #16) and to quantify how much of the reported
        anisotropy is an artefact of resizing.
    """
    if not preserve_aspect:
        return img.resize((half_w, half_h), resample=Image.NEAREST)

    scale = min(half_w / img.width, half_h / img.height)
    nw, nh = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), resample=Image.NEAREST)
    canvas = Image.new(img.mode, (half_w, half_h), color=0)
    canvas.paste(resized, ((half_w - nw) // 2, (half_h - nh) // 2))
    return canvas


def concat_pair(src_img: Image.Image, tgt_img: Image.Image,
                target_on_left: bool = True,
                pair_half_size: Optional[Sequence[int]] = None,
                preserve_aspect: bool = False) -> Image.Image:
    """
    Horizontally concatenate source and target.

    The training notebooks read the pair as:
        input_image = image[:, w:, :]   # RIGHT half
        real_image  = image[:, :w, :]   # LEFT  half
    so for a task A -> B the TARGET (B) must be on the LEFT and the SOURCE (A)
    on the RIGHT. Keeping this convention means the existing training code
    runs unchanged.

    pair_half_size = [height, width] writes each half already at the training
    resolution. The originals are 8184 x 12045 px, so a full-resolution pair is
    16368 x 12045 px (~590 MB uncompressed); 1116 of them per task is roughly
    5 GB on disk, which is what the original `*_dataset_split.zip` archives
    weigh. Writing at [2048, 1024] instead costs about 50 MB per task, does the
    downsampling once instead of every epoch, and produces identical training
    input because the loader resizes to exactly this size anyway.
    """
    a, b = (tgt_img, src_img) if target_on_left else (src_img, tgt_img)
    a, b = a.convert("RGB"), b.convert("RGB")
    if a.size != b.size:
        raise ValueError(f"Size mismatch in pair: {a.size} vs {b.size}")
    if pair_half_size:
        h, w = int(pair_half_size[0]), int(pair_half_size[1])
        a = _fit_half(a, w, h, preserve_aspect)
        b = _fit_half(b, w, h, preserve_aspect)
    out = Image.new("RGB", (a.width + b.width, a.height))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    return out


def build_dataset(cfg: dict, plan: SplitPlan, dry_run: bool) -> dict:
    src_layers = discover_layers(cfg["source_dir"])
    tgt_layers = discover_layers(cfg["target_dir"])

    common = sorted(set(src_layers) & set(tgt_layers))
    missing = (set(src_layers) ^ set(tgt_layers))
    if missing:
        print(f"  ! layers present in only one domain, ignored: {sorted(missing)}")

    out_root = cfg["output_dir"]
    counts = {"train": 0, "val": 0, "test": 0}
    records: List[dict] = []

    for split_name, layer_ids in (("train", plan.train), ("val", plan.val), ("test", plan.test)):
        split_dir = os.path.join(out_root, split_name)
        if not dry_run:
            os.makedirs(split_dir, exist_ok=True)

        # Augmentation is applied to TRAINING layers only.
        n_aug = AUGS_PER_LAYER if split_name == "train" else 0

        for k in layer_ids:
            if k not in common:
                continue
            src = Image.open(src_layers[k])
            tgt = Image.open(tgt_layers[k])
            if src.mode != "P":
                src = src.convert("P")
            if tgt.mode != "P":
                tgt = tgt.convert("P")

            # --- original slice (always included) -------------------------
            name = f"layer{k:03d}_aug00.png"
            counts[split_name] += 1
            records.append({"split": split_name, "layer": k, "aug": 0, "file": name})
            if not dry_run:
                concat_pair(src, tgt, cfg.get("target_on_left", True),
                            cfg.get("pair_half_size"),
                            cfg.get("preserve_aspect", False)).save(
                    os.path.join(split_dir, name), format="PNG", compress_level=6)

            # --- augmented realisations (training only) -------------------
            for j in range(1, n_aug + 1):
                # Per-layer, per-realisation seed: reproducible and independent
                # of how many layers precede this one.
                rng_s = random.Random(cfg["seed"] * 1_000_000 + k * 1_000 + j)
                rng_t = random.Random(cfg["seed"] * 1_000_000 + k * 1_000 + j)
                name = f"layer{k:03d}_aug{j:02d}.png"
                counts[split_name] += 1
                records.append({"split": split_name, "layer": k, "aug": j, "file": name})
                if not dry_run:
                    # Same seed for both domains => identical geometric transform
                    # => pixel-wise alignment is preserved.
                    a_src = augment_layer_image(src, rng_s)
                    a_tgt = augment_layer_image(tgt, rng_t)
                    concat_pair(a_src, a_tgt, cfg.get("target_on_left", True),
                                cfg.get("pair_half_size"),
                                cfg.get("preserve_aspect", False)).save(
                        os.path.join(split_dir, name), format="PNG", compress_level=6)

            print(f"  {split_name:5s} layer {k:3d}  -> 1 original + {n_aug} augmented")

    return {"counts": counts, "records": records, "common_layers": common}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "task": "facies_to_porosity",
    "source_dir": "/path/to/org_facies",
    "target_dir": "/path/to/org_Porosity",
    "output_dir": "/path/to/splits/facies_to_porosity",
    "seed": 2025,
    "mode": "blocked",
    "gap": 1,
    "ratios": [0.70, 0.15, 0.15],
    "target_on_left": True,
    "pair_half_size": [2048, 1024],
    "preserve_aspect": False,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config file")
    ap.add_argument("--write-default-config", metavar="PATH",
                    help="write a template config and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without writing any images")
    args = ap.parse_args()

    if args.write_default_config:
        with open(args.write_default_config, "w") as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=2)
        print(f"Template written to {args.write_default_config}")
        return 0

    if not args.config:
        ap.error("--config is required (or use --write-default-config)")

    with open(args.config) as fh:
        cfg = {**DEFAULT_CONFIG, **json.load(fh)}

    print(f"Task            : {cfg['task']}")
    print(f"Source domain   : {cfg['source_dir']}")
    print(f"Target domain   : {cfg['target_dir']}")
    print(f"Split mode      : {cfg['mode']} (gap={cfg['gap']}, seed={cfg['seed']})")
    ps = cfg.get("pair_half_size")
    print(f"Pair half size  : {'native (full resolution)' if not ps else f'{ps[1]} x {ps[0]} px'}"
          f"{'  [aspect preserved / letterboxed]' if cfg.get('preserve_aspect') else ''}")
    print()

    src_layers = discover_layers(cfg["source_dir"])
    tgt_layers = discover_layers(cfg["target_dir"])
    common = sorted(set(src_layers) & set(tgt_layers))
    print(f"Layers found    : {len(common)}  (range {min(common)}-{max(common)})")

    plan = make_split(common, cfg["mode"], cfg["seed"], tuple(cfg["ratios"]), cfg["gap"])
    verify_split(plan)

    print(f"  train : {len(plan.train):3d} layers  {plan.train[0]}-{plan.train[-1]}")
    print(f"  val   : {len(plan.val):3d} layers  {plan.val[0]}-{plan.val[-1]}")
    print(f"  test  : {len(plan.test):3d} layers  {plan.test[0]}-{plan.test[-1]}")
    if plan.dropped_gap:
        print(f"  gap   : {len(plan.dropped_gap):3d} layers discarded as buffer {plan.dropped_gap}")
    print()

    result = build_dataset(cfg, plan, args.dry_run)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "task": cfg["task"],
        "config": cfg,
        "augmentation": {
            "augs_per_training_layer": AUGS_PER_LAYER,
            "rotation_deg": ROTATION_DEG,
            "shift_frac": SHIFT_FRAC,
            "shear_x_max": SHEAR_X_MAX,
            "shear_y_max": SHEAR_Y_MAX,
            "zoom_frac": ZOOM_FRAC,
            "h_flip": H_FLIP,
            "applied_to": "training layers only",
        },
        "pair_half_size": cfg.get("pair_half_size"),
        "preserve_aspect": cfg.get("preserve_aspect", False),
        "split": plan.as_dict(),
        "counts": result["counts"],
        "n_independent_layers": {
            "train": len(plan.train), "val": len(plan.val), "test": len(plan.test),
        },
        "records": result["records"],
    }
    blob = json.dumps(manifest["split"], sort_keys=True).encode()
    manifest["split_sha256"] = hashlib.sha256(blob).hexdigest()

    print()
    print(f"Images written  : train={result['counts']['train']}  "
          f"val={result['counts']['val']}  test={result['counts']['test']}")
    print(f"Independent units: train={len(plan.train)}  val={len(plan.val)}  test={len(plan.test)} layers")
    print(f"Split SHA-256   : {manifest['split_sha256'][:16]}...")

    if not args.dry_run:
        os.makedirs(cfg["output_dir"], exist_ok=True)
        mpath = os.path.join(cfg["output_dir"], "split_manifest.json")
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=2)
        cpath = os.path.join(cfg["output_dir"], "split_manifest.csv")
        with open(cpath, "w") as fh:
            fh.write("split,layer,aug,file\n")
            for r in result["records"]:
                fh.write(f"{r['split']},{r['layer']},{r['aug']},{r['file']}\n")
        print(f"Manifest        : {mpath}")
    else:
        print("\n(dry run -- no images or manifest written)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
