#!/usr/bin/env python3
"""
03_train.py
===========

Reproducible Pix2Pix training for Pix2Geomodel V2 (RINENG-D-26-10842).

The network is IDENTICAL to the original study -- U-Net generator with 8
downsampling and 7 upsampling blocks, PatchGAN discriminator, adversarial +
L1 loss with lambda = 100, Adam(2e-4, beta_1 = 0.5), batch size 1. Nothing
about the architecture is changed, because the paper's claim is about
transferability, not architecture, and changing the model would invalidate
the comparison with the original Pix2Geomodel study.

What IS changed is everything the reviewers asked for around it:

  * Every RNG is seeded (Python, NumPy, TensorFlow) and the seed is recorded.
    The original never called tf.random.set_seed at all.       -- R5 #6
  * Checkpoints are selected on the VALIDATION split, never on test.
    The published Table 4 values came from undocumented mid-run checkpoints.
                                                               -- R5 #6
  * The full environment (library versions, GPU model, driver, determinism
    flags) is written to a JSON manifest beside the checkpoints.
                                                               -- R5 #6, R4 #2
  * Validation uses the corrected metrics from 02_evaluation.py -- true
    seven-class facies scoring or physical-unit errors -- not argmax over
    the RGB channel axis.                                       -- R5 #8, #10
  * `--seed` makes the 3-5 independent repeats one command each. -- R5 #26b

USAGE
-----
    # smoke test first -- confirms the GPU is actually being used
    python 03_train.py --data splits/facies_to_porosity \
        --target-domain porosity --out runs/f2p_smoke --steps 200 --smoke-test

    # real run, one per seed
    for S in 1 2 3; do
      python 03_train.py --data splits/facies_to_porosity \
          --target-domain porosity --out runs/f2p_seed$S --seed $S --steps 15000
    done
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np

# Determinism flags must be set BEFORE TensorFlow is imported.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import tensorflow as tf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_eval_module():
    """Import 02_evaluation.py by path (its name is not a valid identifier)."""
    import importlib.util as ilu
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "02_evaluation.py")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = ilu.spec_from_file_location("pix2geo_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    mod = ilu.module_from_spec(spec)
    sys.modules["pix2geo_eval"] = mod          # required before exec_module
    spec.loader.exec_module(mod)
    return mod


try:
    eval02 = _load_eval_module()
except Exception as _e:                        # pragma: no cover
    eval02 = None
    print(f"[warn] 02_evaluation.py not importable ({_e}); "
          f"validation will fall back to L1 only.")

# TF_DETERMINISTIC_OPS makes every random op require a seed. Set a default now
# so that merely importing this module (e.g. for testing) cannot fail; train()
# overrides it with the run's actual --seed before anything is built.
tf.random.set_seed(1)

# --------------------------------------------------------------------------
# Fixed architecture constants -- must match the original study exactly
# --------------------------------------------------------------------------
IMG_HEIGHT = 2048
IMG_WIDTH = 1024
BATCH_SIZE = 1
BUFFER_SIZE = 400
LAMBDA = 100
LEARNING_RATE = 2e-4
BETA_1 = 0.5
OUTPUT_CHANNELS = 3


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

TF_MAJOR, TF_MINOR = (int(x) for x in tf.__version__.split(".")[:2])


def check_tf_environment() -> None:
    """
    Warn early about the two configurations that silently fall back to CPU.

    Native Windows: TensorFlow dropped GPU support after 2.10, so 2.11+ runs
    on CPU no matter what card is installed. The supported combinations are
    TF 2.10 with CUDA 11.2 / cuDNN 8.1, or any modern TF under WSL2.
    """
    import platform
    if platform.system() == "Windows" and (TF_MAJOR, TF_MINOR) >= (2, 11):
        print("\n" + "!" * 72)
        print(f"! TensorFlow {tf.__version__} on native Windows CANNOT use a GPU.")
        print("! Supported here: TF 2.10 + CUDA 11.2 + cuDNN 8.1, or any TF under WSL2.")
        print("!" * 72 + "\n")


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # tf.keras.utils.set_random_seed exists from TF 2.7; guard anyway so the
    # script runs unchanged on the 2.10 build needed for Windows GPU support.
    setter = getattr(getattr(tf.keras, "utils", None), "set_random_seed", None)
    if callable(setter):
        setter(seed)


def gpu_report() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    info = {"n_gpus": len(gpus), "devices": [g.name for g in gpus], "details": []}
    for g in gpus:
        try:
            info["details"].append(tf.config.experimental.get_device_details(g))
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            info["nvidia_smi"] = [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        pass
    return info


def environment_manifest(seed: int, args: argparse.Namespace) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "determinism": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_CUDNN_DETERMINISTIC": os.environ.get("TF_CUDNN_DETERMINISTIC"),
        },
        "gpu": gpu_report(),
        "hyperparameters": {
            "img_height": IMG_HEIGHT, "img_width": IMG_WIDTH,
            "batch_size": BATCH_SIZE, "buffer_size": BUFFER_SIZE,
            "lambda_l1": LAMBDA, "learning_rate": LEARNING_RATE, "beta_1": BETA_1,
            "steps": args.steps,
            # Arm identifiers. Recorded explicitly so a run directory can be
            # identified from its manifest alone, without trusting its name.
            "gan_weight": getattr(args, "gan_weight", 1.0),
            "augmentation": not getattr(args, "no_augment", False),
        },
        "args": {k: v for k, v in vars(args).items()},
    }


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------

def load_pair(path: tf.Tensor, target_on_left: bool = True):
    """
    Read one concatenated pair. The original convention is preserved:
        input  = RIGHT half   (source domain)
        target = LEFT  half   (target domain)
    """
    image = tf.io.decode_png(tf.io.read_file(path))
    w = tf.shape(image)[1] // 2
    if target_on_left:
        target, inp = image[:, :w, :], image[:, w:, :]
    else:
        inp, target = image[:, :w, :], image[:, w:, :]
    return tf.cast(inp, tf.float32), tf.cast(target, tf.float32)


def resize_pair(inp, tar, h, w):
    inp = tf.image.resize(inp, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    tar = tf.image.resize(tar, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return inp, tar


def normalize(inp, tar):
    return inp / 127.5 - 1.0, tar / 127.5 - 1.0


def random_jitter(inp, tar, seed_pair):
    """Online augmentation, seeded per example for reproducibility."""
    nh = int(round(IMG_HEIGHT * 1.12))
    nw = int(round(IMG_WIDTH * 1.12))
    inp, tar = resize_pair(inp, tar, nh, nw)
    stacked = tf.stack([inp, tar], axis=0)
    stacked = tf.image.stateless_random_crop(
        stacked, size=[2, IMG_HEIGHT, IMG_WIDTH, 3], seed=seed_pair)
    inp, tar = stacked[0], stacked[1]
    if tf.random.stateless_uniform([], seed=seed_pair) > 0.5:
        inp = tf.image.flip_left_right(inp)
        tar = tf.image.flip_left_right(tar)
    return inp, tar


def make_dataset(directory: str, training: bool, seed: int,
                 target_on_left: bool = True,
                 augment: bool = True) -> tf.data.Dataset:
    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".png"))
    if not files:
        raise RuntimeError(f"No PNG files in {directory}")
    paths = [os.path.join(directory, f) for f in files]
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        def _train_map(path, i):
            inp, tar = load_pair(path, target_on_left)
            if augment:
                sp = tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)])
                inp, tar = random_jitter(inp, tar, sp)
            else:
                # Deterministic resize only. random_jitter does the up-resize,
                # random crop and random mirror; switching it off must leave the
                # geometry otherwise identical or the arm is not an ablation of
                # augmentation alone.
                inp, tar = resize_pair(inp, tar, IMG_HEIGHT, IMG_WIDTH)
            return normalize(inp, tar)

        ds = tf.data.Dataset.zip((ds, tf.data.Dataset.range(len(paths))))
        ds = ds.map(_train_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(BUFFER_SIZE, seed=seed, reshuffle_each_iteration=True)
        ds = ds.repeat()
    else:
        def _eval_map(path):
            inp, tar = load_pair(path, target_on_left)
            inp, tar = resize_pair(inp, tar, IMG_HEIGHT, IMG_WIDTH)
            return normalize(inp, tar)
        ds = ds.map(_eval_map, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE), files


# --------------------------------------------------------------------------
# Model -- unchanged from the original study
# --------------------------------------------------------------------------

def downsample(filters, size, apply_batchnorm=True):
    init = tf.random_normal_initializer(0.0, 0.02)
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding="same",
                                   kernel_initializer=init, use_bias=False))
    if apply_batchnorm:
        seq.add(tf.keras.layers.BatchNormalization())
    seq.add(tf.keras.layers.LeakyReLU())
    return seq


def upsample(filters, size, apply_dropout=False):
    init = tf.random_normal_initializer(0.0, 0.02)
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same",
                                            kernel_initializer=init, use_bias=False))
    seq.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        seq.add(tf.keras.layers.Dropout(0.5))
    seq.add(tf.keras.layers.ReLU())
    return seq


def build_generator(h=IMG_HEIGHT, w=IMG_WIDTH):
    inputs = tf.keras.layers.Input(shape=[h, w, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4), downsample(256, 4), downsample(512, 4),
        downsample(512, 4), downsample(512, 4), downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4), upsample(256, 4), upsample(128, 4), upsample(64, 4),
    ]
    init = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2,
                                           padding="same", kernel_initializer=init,
                                           activation="tanh")
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    return tf.keras.Model(inputs=inputs, outputs=last(x))


def build_discriminator(h=IMG_HEIGHT, w=IMG_WIDTH):
    init = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[h, w, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[h, w, 3], name="target_image")
    x = tf.keras.layers.concatenate([inp, tar])
    x = downsample(64, 4, False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=init,
                               use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=init)(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target, gan_weight=1.0):
    """
    Pix2pix generator objective: gan_weight * adversarial + LAMBDA * L1.

    gan_weight = 0 removes the adversarial term entirely and leaves a plain
    L1-trained U-Net. That is the non-GAN baseline the reviewers asked for,
    and it is implemented as a flag on THIS function rather than as a separate
    script on purpose: the generator architecture, data pipeline, augmentation,
    seeding, optimiser, validation metric and checkpoint-selection rule are
    then provably identical between the two arms, so any difference in the
    result is attributable to the adversarial term and nothing else. A
    standalone baseline script would leave that claim resting on two files
    being kept in sync by hand.
    """
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_weight * gan_loss + LAMBDA * l1_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + gen_loss


# --------------------------------------------------------------------------
# Validation -- the checkpoint-selection criterion
# --------------------------------------------------------------------------

def validate(generator, val_ds, target_domain: str) -> dict:
    """
    Returns a dict with 'selection_score' (higher is better) plus diagnostics.

    Facies targets     -> macro F1 over the true seven classes.
    Continuous targets -> negative RMSE in physical units.

    Crucially this runs on the VALIDATION split. The original study reported
    numbers read off mid-training checkpoints with no stated rule, which is
    exactly what Reviewer 5 #6 objects to.
    """
    l1_total, n = 0.0, 0
    scores = []
    for inp, tar in val_ds:
        pred = generator(inp, training=False)
        l1_total += float(tf.reduce_mean(tf.abs(tar - pred)))
        n += 1
        if eval02 is None:
            continue
        p = ((pred[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        r = ((tar[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        if target_domain == "facies":
            cm = eval02.confusion_matrix(
                eval02.decode_rgb_to_facies(p), eval02.decode_rgb_to_facies(r), 7)
            m = eval02.categorical_metrics(cm, [x for x, _ in eval02.FACIES_LEGEND])
            scores.append(m["macro_F1"])
        else:
            lut = eval02.build_lut(eval02.CONTINUOUS_DOMAINS[target_domain]["stops"])
            pv = eval02.index_to_value(eval02.decode_rgb_to_index(p, lut), target_domain)
            rv = eval02.index_to_value(eval02.decode_rgb_to_index(r, lut), target_domain)
            valid = ~(np.all(r == 0, axis=-1))
            m = eval02.continuous_metrics(pv, rv, target_domain, mask=valid)
            scores.append(-m["RMSE"])

    mean_l1 = l1_total / max(n, 1)
    if scores:
        score = float(np.nanmean(scores))
    else:
        score = -mean_l1
    return {"selection_score": score, "val_l1": mean_l1, "n_val": n}


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------

def train(args: argparse.Namespace) -> int:
    # These are module-level constants read by the loss, the augmentation and
    # both model builders. They are rebound here, before anything is built, so
    # that one flag changes the whole run consistently rather than only the
    # place that happens to be passed an argument.
    global LAMBDA, IMG_HEIGHT, IMG_WIDTH
    LAMBDA = float(args.lambda_l1)
    IMG_HEIGHT, IMG_WIDTH = int(args.img_height), int(args.img_width)
    check_tf_environment()
    set_all_seeds(args.seed)
    os.makedirs(args.out, exist_ok=True)

    manifest = environment_manifest(args.seed, args)
    if manifest["gpu"]["n_gpus"] == 0:
        print("\n" + "!" * 70)
        print("! NO GPU DETECTED. TensorFlow will fall back to CPU.")
        print("! Several December 2025 runs did exactly this and took 20-30 hours")
        print("! per task instead of 1.5. Fix the GPU libraries before proceeding.")
        print("!" * 70 + "\n")
        if not args.allow_cpu:
            print("Refusing to start. Pass --allow-cpu to override.")
            return 2
    else:
        print(f"GPU(s): {manifest['gpu'].get('nvidia_smi', manifest['gpu']['devices'])}")

    train_ds, train_files = make_dataset(os.path.join(args.data, "train"), True,
                                         args.seed, args.target_on_left,
                                         augment=not args.no_augment)
    val_ds, val_files = make_dataset(os.path.join(args.data, "val"), False,
                                     args.seed, args.target_on_left)
    print(f"train images: {len(train_files)}   val images: {len(val_files)}")

    split_manifest_path = os.path.join(args.data, "split_manifest.json")
    if os.path.exists(split_manifest_path):
        with open(split_manifest_path) as fh:
            sm = json.load(fh)
        manifest["split"] = sm.get("split")
        manifest["split_sha256"] = sm.get("split_sha256")
        print(f"split: {sm['split']['mode']} mode, "
              f"{sm['n_independent_layers']} independent layers, "
              f"sha {sm.get('split_sha256','')[:12]}")
    else:
        print("[warn] no split_manifest.json found -- provenance will be incomplete")

    generator = build_generator(IMG_HEIGHT, IMG_WIDTH)
    discriminator = build_discriminator(IMG_HEIGHT, IMG_WIDTH)
    g_opt = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)
    d_opt = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=BETA_1)

    ckpt = tf.train.Checkpoint(generator_optimizer=g_opt,
                               discriminator_optimizer=d_opt,
                               generator=generator,
                               discriminator=discriminator)
    best_dir = os.path.join(args.out, "best")
    last_dir = os.path.join(args.out, "last")
    best_mgr = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=1)
    last_mgr = tf.train.CheckpointManager(ckpt, last_dir, max_to_keep=1)

    summary = tf.summary.create_file_writer(os.path.join(args.out, "tb"))

    gan_weight = float(args.gan_weight)
    use_gan = gan_weight > 0.0

    @tf.function
    def train_step_gan(inp, target, step):
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            gen_output = generator(inp, training=True)
            disc_real = discriminator([inp, target], training=True)
            disc_fake = discriminator([inp, gen_output], training=True)
            g_total, g_gan, g_l1 = generator_loss(disc_fake, gen_output, target,
                                                  gan_weight)
            d_loss = discriminator_loss(disc_real, disc_fake)
        g_opt.apply_gradients(zip(gt.gradient(g_total, generator.trainable_variables),
                                  generator.trainable_variables))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, discriminator.trainable_variables),
                                  discriminator.trainable_variables))
        with summary.as_default():
            tf.summary.scalar("gen_total_loss", g_total, step=step)
            tf.summary.scalar("gen_gan_loss", g_gan, step=step)
            tf.summary.scalar("gen_l1_loss", g_l1, step=step)
            tf.summary.scalar("disc_loss", d_loss, step=step)

    @tf.function
    def train_step_l1(inp, target, step):
        # The discriminator is not built into the graph at all here. Leaving it
        # to train while contributing nothing would burn roughly a third of the
        # step time and, worse, would make the "no adversarial term" arm not
        # quite what it claims to be.
        with tf.GradientTape() as gt:
            gen_output = generator(inp, training=True)
            g_l1 = tf.reduce_mean(tf.abs(target - gen_output))
            g_total = LAMBDA * g_l1
        g_opt.apply_gradients(zip(gt.gradient(g_total, generator.trainable_variables),
                                  generator.trainable_variables))
        with summary.as_default():
            tf.summary.scalar("gen_total_loss", g_total, step=step)
            tf.summary.scalar("gen_l1_loss", g_l1, step=step)

    train_step = train_step_gan if use_gan else train_step_l1
    print(f"objective: {'adversarial + L1' if use_gan else 'L1 ONLY (non-GAN baseline)'}"
          f"   gan_weight={gan_weight}  lambda_l1={LAMBDA}")
    print(f"resolution: {IMG_HEIGHT} x {IMG_WIDTH}   "
          f"augmentation: {'OFF' if args.no_augment else 'on'}")

    history = []
    best_score = -float("inf")
    best_step = -1
    t0 = time.time()
    # Training-step time only. Validation and checkpoint writing are excluded
    # so that the reported s/step is the actual optimisation rate; mixing them
    # in inflates the projection badly on short runs.
    train_seconds = 0.0
    overhead_seconds = 0.0

    for step, (inp, tar) in train_ds.take(args.steps).enumerate():
        _t = time.time()
        train_step(inp, tar, tf.cast(step, tf.int64))
        train_seconds += time.time() - _t
        s = int(step) + 1

        if s % args.log_every == 0 or s == args.steps:
            rate = train_seconds / s
            print(f"step {s:6d}/{args.steps}  train {train_seconds/3600:5.2f} h  "
                  f"({rate:.3f} s/step, projected training {rate*args.steps/3600:.2f} h)")

        if s % args.val_every == 0 or s == args.steps:
            _t = time.time()
            v = validate(generator, val_ds, args.target_domain)
            v["step"] = s
            v["elapsed_h"] = (time.time() - t0) / 3600
            history.append(v)
            with summary.as_default():
                tf.summary.scalar("val_selection_score", v["selection_score"],
                                  step=tf.cast(s, tf.int64))
                tf.summary.scalar("val_l1", v["val_l1"], step=tf.cast(s, tf.int64))
            marker = ""
            if v["selection_score"] > best_score:
                best_score, best_step = v["selection_score"], s
                best_mgr.save(checkpoint_number=s)
                marker = "  <-- new best (checkpoint saved)"
            print(f"  validation @ {s}: score {v['selection_score']:.5f}  "
                  f"L1 {v['val_l1']:.5f}{marker}")
            last_mgr.save(checkpoint_number=s)

            with open(os.path.join(args.out, "val_history.json"), "w") as fh:
                json.dump(history, fh, indent=2)
            overhead_seconds += time.time() - _t

        if args.smoke_test and s >= args.steps:
            break

    total_h = (time.time() - t0) / 3600
    n_val = max(len(history), 1)
    manifest["result"] = {
        "total_hours": total_h,
        "training_hours": train_seconds / 3600,
        "validation_checkpoint_hours": overhead_seconds / 3600,
        "seconds_per_step": train_seconds / max(args.steps, 1),
        "seconds_per_validation": overhead_seconds / n_val,
        "best_step": best_step,
        "best_selection_score": best_score,
        "selection_criterion": ("macro_F1 over 7 facies classes (validation)"
                                if args.target_domain == "facies"
                                else f"negative RMSE in {args.target_domain} units (validation)"),
        "val_history": history,
    }
    with open(os.path.join(args.out, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nfinished in {total_h:.2f} h "
          f"(training {train_seconds/3600:.2f} h, "
          f"validation+checkpoint {overhead_seconds/3600:.2f} h)")
    print(f"best checkpoint: step {best_step}, score {best_score:.5f}  -> {best_dir}")
    print(f"manifest: {os.path.join(args.out, 'run_manifest.json')}")
    if args.smoke_test:
        rate = train_seconds / max(args.steps, 1)
        val_each = overhead_seconds / n_val
        # Project against the val_every a REAL run uses, not the smoke test's.
        # A smoke test deliberately validates every few steps to exercise that
        # code path; extrapolating that cadence to 15,000 steps counts hundreds
        # of validations that will never happen and inflates the estimate
        # several-fold. With --val-every 40 this reported 56.6 GPU-hours for an
        # 18-run arm whose real cost is about 24.
        full_steps, full_val_every = 15000, 1000
        n_val_full = full_steps // full_val_every
        train_h = rate * full_steps / 3600
        val_h = val_each * n_val_full / 3600
        proj = train_h + val_h
        print(f"\nSMOKE TEST")
        print(f"  measured training rate : {rate:.3f} s/step")
        print(f"  measured validation    : {val_each:.1f} s each "
              f"({n_val} observed at --val-every {args.val_every})")
        print(f"  --- projection for a real run: {full_steps} steps, "
              f"--val-every {full_val_every} ---")
        print(f"  training               : {train_h:.2f} h")
        print(f"  validation+checkpoint  : {n_val_full} x {val_each:.1f} s "
              f"= {val_h:.2f} h")
        print(f"  PROJECTED PER TASK     : {proj:.2f} h")
        print(f"  six tasks x 3 seeds    : {proj*18:.1f} GPU-hours")
        print("  Below ~0.5 s/step means the GPU is working. 5+ s/step means CPU.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="split directory (train/ val/ test/)")
    ap.add_argument("--target-domain", required=True,
                    help="facies | porosity | permeability | vcl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--target-on-left", action="store_true", default=True)
    ap.add_argument("--lambda-l1", type=float, default=100.0,
                    help="weight on the L1 term (default 100, the pix2pix value)")
    ap.add_argument("--img-height", type=int, default=IMG_HEIGHT)
    ap.add_argument("--img-width", type=int, default=IMG_WIDTH)
    ap.add_argument("--no-augment", action="store_true",
                    help="train on unaugmented pairs, deterministic resize only")
    ap.add_argument("--gan-weight", type=float, default=1.0,
                    help="weight on the adversarial loss. 0 = plain L1 U-Net, "
                         "the non-GAN baseline. Everything else is unchanged.")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="proceed even if no GPU is visible")
    return train(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
