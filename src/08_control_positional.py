#!/usr/bin/env python3
"""
08_control_positional.py
========================

Control experiment: is the network predicting facies from the INPUT property,
or from POSITION?

Motivation. In the test results, "Fair Sand" is recovered with F1 ~= 0.999 by
all three inverse models, including permeability->facies, where every other
class scores 0.16-0.37. A class recovered near-perfectly regardless of which
property is supplied is a warning sign: with 36 training layers of similar
geometry, a U-Net can learn "this region is always Fair Sand" and ignore the
input entirely. If that is happening, it inflates pixel accuracy, macro-F1 and
mIoU for every task, and the inflation is invisible in the headline numbers.

Three conditions, same trained model, same references:

  MATCHED    input of layer i        -> compare with reference of layer i
             (the normal evaluation)

  SHUFFLED   input of layer sigma(i) -> compare with reference of layer i,
             where sigma is a seeded derangement (no layer keeps its own
             input). Any class still recovered here is being produced from
             position, not from the property supplied.

  CONSTANT   the spatially averaged colour of input i, broadcast to a flat
             image -> compare with reference of layer i. Same mean colour,
             zero spatial structure. The strongest form of the control.

Reading the output:

  class F1 collapses under SHUFFLED/CONSTANT  -> genuinely input-driven. Good.
  class F1 survives under SHUFFLED/CONSTANT   -> positional memorisation, and
                                                 that class must be reported
                                                 with the caveat attached.

This addresses Reviewer 5 #17 ("the discussion is too confirmatory ... examine
whether the apparent predictive relationship partly reflects" something other
than genuine learning) with an experiment rather than a paragraph.

    python 08_control_positional.py            (run in tf210; needs the GPU)
    python 08_control_positional.py --seeds 1  (quick, one seed)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")

INVERSE_TASKS = ["porosity_to_facies", "permeability_to_facies", "vcl_to_facies"]


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def derangement(n: int, seed: int = 2025) -> List[int]:
    """A permutation with no fixed point, so no layer receives its own input."""
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    for _ in range(1000):
        p = rng.permutation(n).tolist()
        if all(p[i] != idx[i] for i in range(n)):
            return p
    return [(i + 1) % n for i in range(n)]   # fallback: cyclic shift


def run_task(task: str, seed: int, trainer, ev) -> Dict[str, dict]:
    import tensorflow as tf
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    run = os.path.join(RUNS, f"{task}_seed{seed}")
    best = tf.train.latest_checkpoint(os.path.join(run, "best"))
    if best is None:
        print(f"  ! no checkpoint for {task} seed {seed}")
        return {}

    man = {}
    try:
        man = json.load(open(os.path.join(run, "run_manifest.json")))
    except Exception:
        pass
    hp = man.get("hyperparameters", {})
    h = int(hp.get("img_height", trainer.IMG_HEIGHT))
    w = int(hp.get("img_width", trainer.IMG_WIDTH))

    trainer.set_all_seeds(seed)
    gen = trainer.build_generator(h, w)
    disc = trainer.build_discriminator(h, w)
    ckpt = tf.train.Checkpoint(
        generator_optimizer=tf.keras.optimizers.Adam(trainer.LEARNING_RATE,
                                                     beta_1=trainer.BETA_1),
        discriminator_optimizer=tf.keras.optimizers.Adam(trainer.LEARNING_RATE,
                                                         beta_1=trainer.BETA_1),
        generator=gen, discriminator=disc)
    # Verify the restore actually loaded weights. `.expect_partial()` silences
    # the warning that would otherwise flag a checkpoint that did not match
    # the object graph -- which is what happens if this is run under a
    # TensorFlow other than the 2.10 the runs were trained with. The generator
    # then keeps its random initialisation and emits the SAME output for every
    # input, so matched, shuffled and constant come out identical and the
    # control silently reports 100 % retention, i.e. "POSITIONAL", for
    # everything. That is the most dangerous possible failure for this
    # particular experiment: it fabricates the exact result the control exists
    # to test for. Comparing a weight fingerprint across the restore costs
    # nothing and makes the failure loud.
    before = float(sum(float(tf.reduce_sum(v)) for v in gen.trainable_variables))
    ckpt.restore(best).expect_partial()
    after = float(sum(float(tf.reduce_sum(v)) for v in gen.trainable_variables))
    if before == after:
        raise RuntimeError(
            f"checkpoint restore did not change generator weights for {task} "
            f"seed {seed} ({best}).\nThe generator is still randomly "
            f"initialised, so every condition would return an identical, "
            f"meaningless result.\nTensorFlow in this interpreter is "
            f"{tf.__version__}; the runs were trained with 2.10. Activate the "
            f"tf210 environment:  conda activate tf210")

    test_dir = os.path.join(SPLITS, task, "test")
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".png"))
    n = len(files)

    inputs, refs = [], []
    for fn in files:
        inp, tar = trainer.load_pair(tf.constant(os.path.join(test_dir, fn)), True)
        inp, tar = trainer.resize_pair(inp, tar, h, w)
        inp, tar = trainer.normalize(inp, tar)
        inputs.append(inp)
        refs.append(((tar.numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8))

    sigma = derangement(n)
    n_cls = len(ev.FACIES_LEGEND)
    out: Dict[str, dict] = {}

    for condition in ("matched", "shuffled", "constant"):
        cm_total = np.zeros((n_cls, n_cls))
        for i in range(n):
            if condition == "matched":
                x = inputs[i]
            elif condition == "shuffled":
                x = inputs[sigma[i]]
            else:
                mean = tf.reduce_mean(inputs[i], axis=[0, 1], keepdims=True)
                x = tf.broadcast_to(mean, inputs[i].shape)
            pred = gen(x[tf.newaxis, ...], training=False)[0]
            arr = ((pred.numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            cm_total += ev.confusion_matrix(ev.decode_rgb_to_facies(arr),
                                            ev.decode_rgb_to_facies(refs[i]), n_cls)
        out[condition] = ev.categorical_metrics(
            cm_total, [nm for nm, _ in ev.FACIES_LEGEND])
        print(f"    {condition:9s} PA {out[condition]['pixel_accuracy']:.4f}  "
              f"macro-F1 {out[condition]['macro_F1']:.4f}  "
              f"mIoU {out[condition]['mIoU']:.4f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tasks", nargs="+", default=INVERSE_TASKS)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    # Fail fast and legibly on the wrong environment, rather than after the
    # GPU work has produced numbers that cannot be used.
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")
    if not tf.__version__.startswith("2.10"):
        print("  ! WARNING: the runs were trained with TensorFlow 2.10.")
        print("  ! Checkpoints may not restore into a different version, and")
        print("  ! a silent restore failure makes every condition identical.")
        print("  ! Run:  conda activate tf210")
    if not tf.config.list_physical_devices("GPU"):
        print("  ! WARNING: no GPU visible -- this will be extremely slow.")

    trainer = _load("pix2geo_train", "03_train.py")
    ev = _load("pix2geo_eval", "02_evaluation.py")

    store: Dict[str, Dict[int, Dict[str, dict]]] = defaultdict(dict)
    for task in args.tasks:
        for seed in args.seeds:
            print(f"\n{task} seed {seed}")
            r = run_task(task, seed, trainer, ev)
            if r:
                store[task][seed] = r

    # ---- aggregate ------------------------------------------------------
    rows = [["Task", "Metric", "MATCHED", "SHUFFLED", "CONSTANT",
             "retained under SHUFFLED", "verdict"]]
    per_class_rows = [["Task", "Class", "Facies name",
                       "F1 matched", "F1 shuffled", "F1 constant",
                       "retained", "verdict"]]

    for task in args.tasks:
        if task not in store:
            continue
        agg = {c: defaultdict(list) for c in ("matched", "shuffled", "constant")}
        cls_agg = {c: defaultdict(list) for c in ("matched", "shuffled", "constant")}
        for seed, conds in store[task].items():
            for c, m in conds.items():
                for k in ("pixel_accuracy", "macro_F1", "mIoU"):
                    agg[c][k].append(m[k])
                for pc in m["per_class"]:
                    if pc["f1"] is not None and pc["support_px"] > 0:
                        cls_agg[c][pc["class"]].append(pc["f1"])

        for k in ("pixel_accuracy", "macro_F1", "mIoU"):
            mm = np.mean(agg["matched"][k]); sh = np.mean(agg["shuffled"][k])
            co = np.mean(agg["constant"][k])
            ret = sh / mm if mm else float("nan")
            verdict = ("POSITIONAL" if ret > 0.8 else
                       "partly positional" if ret > 0.5 else "input-driven")
            rows.append([task, k, f"{mm:.4f}", f"{sh:.4f}", f"{co:.4f}",
                         f"{100*ret:.0f}%", verdict])

        for cls in sorted(cls_agg["matched"]):
            name = ev.FACIES_LEGEND[cls - 1][0]
            mm = np.mean(cls_agg["matched"][cls])
            sh = np.mean(cls_agg["shuffled"].get(cls, [0.0]))
            co = np.mean(cls_agg["constant"].get(cls, [0.0]))
            # A retention RATIO is only meaningful when the matched score is
            # meaningful. Fair Sand under permeability->facies scored F1
            # 0.0005 matched against 0.0009 shuffled, which the ratio rule
            # turned into "188 %, POSITIONAL" -- a confident verdict about a
            # class the model essentially never gets right either way. Below
            # this floor the class is reported as not recovered at all, and no
            # verdict is offered.
            F1_FLOOR = 0.05
            if mm < F1_FLOOR:
                per_class_rows.append([task, cls, name, f"{mm:.4f}", f"{sh:.4f}",
                                       f"{co:.4f}", "n/a",
                                       f"not recovered (matched F1 < {F1_FLOOR})"])
                continue
            ret = sh / mm
            verdict = ("POSITIONAL" if ret > 0.8 else
                       "partly positional" if ret > 0.5 else "input-driven")
            per_class_rows.append([task, cls, name, f"{mm:.4f}", f"{sh:.4f}",
                                   f"{co:.4f}", f"{100*ret:.0f}%", verdict])

    import csv
    with open(os.path.join(OUT, "8_Positional_control.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerows(rows); w.writerow([]); w.writerows(per_class_rows)
    with open(os.path.join(OUT, "positional_control_raw.json"), "w") as fh:
        json.dump({t: {str(s): c for s, c in d.items()} for t, d in store.items()},
                  fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print("POSITIONAL CONTROL -- overall")
    print("=" * 78)
    print(f"{'task':<24}{'metric':<16}{'matched':>9}{'shuffled':>10}"
          f"{'constant':>10}{'retained':>10}  verdict")
    for r in rows[1:]:
        print(f"{r[0]:<24}{r[1]:<16}{r[2]:>9}{r[3]:>10}{r[4]:>10}{r[5]:>10}  {r[6]}")

    print("\n" + "=" * 78)
    print("POSITIONAL CONTROL -- per class")
    print("=" * 78)
    print(f"{'task':<24}{'class':<18}{'matched':>9}{'shuffled':>10}"
          f"{'constant':>10}{'retained':>10}  verdict")
    for r in per_class_rows[1:]:
        print(f"{r[0]:<24}{r[2]:<18}{r[3]:>9}{r[4]:>10}{r[5]:>10}{r[6]:>10}  {r[7]}")

    print(f"\nwritten to {os.path.join(OUT, '8_Positional_control.csv')}")
    print("\nA class 'retained' above ~80 % with a shuffled or constant input is")
    print("being predicted from position, not from the input property.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
