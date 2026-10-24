#!/usr/bin/env python3
"""
06_predict.py
=============

Generate test-set predictions from trained checkpoints.

`03_train.py` writes checkpoints; `02_evaluation.py` scores predictions. This
is the bridge: it restores the BEST checkpoint (selected on validation, never
on test) and writes one prediction PNG per test pair, named identically to the
reference file so the evaluator pairs them automatically.

Deliberately strict about three things:

  * It restores `best/`, not `last/`. The published Table 4 numbers came from
    undocumented mid-run checkpoints; here the selection rule is fixed,
    validation-based, and recorded in `run_manifest.json`.
  * Preprocessing is imported from `03_train.py` rather than reimplemented,
    so evaluation cannot silently drift from training.
  * The test set is never shuffled and never augmented.

RUN THIS AFTER TRAINING FINISHES -- it needs the GPU, and competing with a
running training job risks out-of-memory on both.

    python 06_predict.py --all
    python 06_predict.py --run runs/facies_to_porosity_seed1
    python 06_predict.py --all --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")

TASK_DOMAIN = {
    "facies_to_porosity": "porosity",
    "facies_to_permeability": "permeability",
    "facies_to_vcl": "vcl",
    "porosity_to_facies": "facies",
    "permeability_to_facies": "facies",
    "vcl_to_facies": "facies",
}


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import re as _re

# Any run directory may carry an arm suffix after the seed: _l1, _noaug,
# _halfres, _lambda10, _lambda1, and whatever is added next.
_RUN_RE = _re.compile(r"^(?P<task>.+)_seed(?P<seed>\d+)(?P<arm>_.+)?$")


def parse_run(run_path: str) -> Tuple[str, int]:
    """
    runs/<task>_seed<N>[_arm] -> (task, seed)

    Parsed with a regex on the digits rather than by stripping a hard-coded
    list of known suffixes. The list version silently broke the moment the
    ablation added four new arms: int("1_noaug") raises, so 06_predict.py
    reported "12 problem(s)" and produced nothing for any of them.
    """
    base = os.path.basename(os.path.normpath(run_path))
    m = _RUN_RE.match(base)
    if not m:
        raise ValueError(f"cannot parse task/seed from {base}")
    return m.group("task"), int(m.group("seed"))


def discover_runs() -> List[str]:
    if not os.path.isdir(RUNS):
        return []
    out = []
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d)
        if os.path.isdir(p) and "_seed" in d and os.path.isdir(os.path.join(p, "best")):
            out.append(p)
    return out


def predict_run(run_path: str, trainer, force: bool = False,
                dry_run: bool = False) -> Optional[dict]:
    import tensorflow as tf

    task, seed = parse_run(run_path)
    domain = TASK_DOMAIN.get(task)
    if domain is None:
        print(f"  ! unknown task {task}"); return None

    test_dir = os.path.join(SPLITS, task, "test")
    out_dir = os.path.join(run_path, "predictions")
    best_dir = os.path.join(run_path, "best")

    if not os.path.isdir(test_dir):
        print(f"  ! no test split at {test_dir}"); return None
    ckpt_state = tf.train.latest_checkpoint(best_dir)
    if ckpt_state is None:
        print(f"  ! no checkpoint in {best_dir}"); return None

    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(".png"))
    if not force and os.path.isdir(out_dir) and \
            len([f for f in os.listdir(out_dir) if f.endswith(".png")]) == len(files):
        print(f"  = already predicted ({len(files)} images)")
        return {"task": task, "seed": seed, "n": len(files), "skipped": True}

    # Resolution comes from the RUN's manifest, not from module defaults.
    # Training at a different size (e.g. the resolution ablation) must not
    # silently mismatch at prediction time.
    mf = os.path.join(run_path, "run_manifest.json")
    best_step = None
    img_h, img_w = trainer.IMG_HEIGHT, trainer.IMG_WIDTH
    if os.path.exists(mf):
        try:
            man = json.load(open(mf))
            best_step = man.get("result", {}).get("best_step")
            hp = man.get("hyperparameters", {})
            img_h = int(hp.get("img_height", img_h))
            img_w = int(hp.get("img_width", img_w))
        except Exception:
            pass

    print(f"  checkpoint : {os.path.basename(ckpt_state)}"
          f"{f' (best step {best_step})' if best_step else ''}")
    print(f"  resolution : {img_w} x {img_h} (from run manifest)")
    print(f"  test images: {len(files)}   -> {out_dir}")
    if dry_run:
        return {"task": task, "seed": seed, "n": len(files), "dry_run": True}

    trainer.set_all_seeds(seed)
    generator = trainer.build_generator(img_h, img_w)
    discriminator = trainer.build_discriminator(img_h, img_w)
    ckpt = tf.train.Checkpoint(
        generator_optimizer=tf.keras.optimizers.Adam(trainer.LEARNING_RATE,
                                                     beta_1=trainer.BETA_1),
        discriminator_optimizer=tf.keras.optimizers.Adam(trainer.LEARNING_RATE,
                                                         beta_1=trainer.BETA_1),
        generator=generator, discriminator=discriminator)
    ckpt.restore(ckpt_state).expect_partial()

    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    t0 = time.time()
    for fn in files:
        path = os.path.join(test_dir, fn)
        inp, tar = trainer.load_pair(tf.constant(path), True)
        inp, tar = trainer.resize_pair(inp, tar, img_h, img_w)
        inp, _ = trainer.normalize(inp, tar)
        pred = generator(inp[tf.newaxis, ...], training=False)[0]
        arr = ((pred.numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, fn), compress_level=6)
    dt = time.time() - t0

    meta = {
        "task": task, "seed": seed, "n_images": len(files),
        "checkpoint": os.path.basename(ckpt_state), "best_step": best_step,
        "seconds": dt, "test_dir": test_dir,
        "img_height": img_h, "img_width": img_w,
        "note": "generated from the validation-selected best checkpoint; "
                "test set never shuffled, never augmented",
    }
    with open(os.path.join(out_dir, "prediction_manifest.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  wrote {len(files)} predictions in {dt:.1f} s")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a single runs/<task>_seed<N> directory")
    ap.add_argument("--all", action="store_true", help="every run with a best checkpoint")
    ap.add_argument("--force", action="store_true", help="regenerate existing predictions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = [args.run] if args.run else (discover_runs() if args.all else [])
    if not runs:
        ap.error("give --run <dir> or --all (no runs with a best/ checkpoint found)")

    print(f"runs to process: {len(runs)}\n")
    trainer = _load("pix2geo_train", "03_train.py")

    results, failed = [], 0
    for i, r in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {os.path.basename(r)}")
        try:
            res = predict_run(r, trainer, args.force, args.dry_run)
            if res:
                results.append(res)
            else:
                failed += 1
        except Exception as e:
            print(f"  ! FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"predicted {len(results)} run(s), {failed} problem(s)")
    if results and not args.dry_run:
        print("\nNext: score them with")
        print("  python 07_report.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
