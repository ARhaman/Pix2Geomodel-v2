#!/usr/bin/env python3
"""
05_run_all.py
=============

Launcher for the full training matrix: 6 translation tasks x 3 seeds.

Built to survive interruption. The console is not the record -- every run
writes its own log file and its own `run_manifest.json`, and a completed run
is skipped on restart. If the machine reboots mid-matrix, rerun the same
command and it picks up where it stopped.

    python 05_run_all.py --dry-run      # show the plan, run nothing
    python 05_run_all.py                # run everything still outstanding
    python 05_run_all.py --seeds 1      # one seed across all six tasks first
    python 05_run_all.py --status       # what is done, running, or missing

Estimated cost, from the measured 0.339 s/step on an RTX 4500 Ada:
    ~1.55 h per run  x  18 runs  =  ~28 GPU-hours
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))

# task name -> target domain (the half the network must produce)
TASKS = [
    ("facies_to_porosity",     "porosity"),
    ("facies_to_permeability", "permeability"),
    ("facies_to_vcl",          "vcl"),
    ("porosity_to_facies",     "facies"),
    ("permeability_to_facies", "facies"),
    ("vcl_to_facies",          "facies"),
]

SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
LOGS = os.path.join(HERE, "logs_training")

# Set by main() from --gan-weight. The baseline arm writes to a separate
# directory tree so the two arms can never overwrite one another, and so
# --status reports on whichever arm is being launched.
ARM_SUFFIX = ""


def run_dir(task: str, seed: int) -> str:
    return os.path.join(RUNS, f"{task}_seed{seed}{ARM_SUFFIX}")


def is_complete(task: str, seed: int) -> bool:
    """A run counts as complete only if its manifest carries a result block."""
    mf = os.path.join(run_dir(task, seed), "run_manifest.json")
    if not os.path.exists(mf):
        return False
    try:
        with open(mf) as fh:
            man = json.load(fh)
        return "result" in man and man["result"].get("best_step", -1) >= 0
    except Exception:
        return False


def completed_hours(task: str, seed: int) -> float:
    mf = os.path.join(run_dir(task, seed), "run_manifest.json")
    try:
        with open(mf) as fh:
            return float(json.load(fh)["result"]["total_hours"])
    except Exception:
        return float("nan")


def status(seeds) -> int:
    print(f"{'task':<24} " + "".join(f"seed{s:<7}" for s in seeds))
    done = pending = 0
    for task, _ in TASKS:
        cells = []
        for s in seeds:
            if is_complete(task, s):
                cells.append(f"{completed_hours(task, s):.2f} h  ")
                done += 1
            elif os.path.isdir(run_dir(task, s)):
                cells.append("PARTIAL   ")
                pending += 1
            else:
                cells.append("-         ")
                pending += 1
        print(f"{task:<24} " + "".join(cells))
    print(f"\ncomplete {done} / {done + pending}   outstanding {pending}")
    if pending:
        print(f"estimated remaining: {pending * 1.55:.1f} GPU-hours "
              f"({pending * 1.55 / 24:.1f} days)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tasks", nargs="+",
                    help="subset of task names (default: all six)")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--gan-weight", type=float, default=1.0,
                    help="0 launches the non-GAN (L1-only U-Net) baseline arm "
                         "into runs/<task>_seed<n>_l1. Default 1.0 is the "
                         "adversarial arm already trained.")
    ap.add_argument("--force", action="store_true",
                    help="rerun even if a completed manifest exists")
    args = ap.parse_args()

    global ARM_SUFFIX
    ARM_SUFFIX = "" if float(args.gan_weight) > 0 else "_l1"
    if ARM_SUFFIX:
        print("ARM: non-GAN baseline (L1 only). Runs go to *_l1 directories;\n"
              "     the adversarial runs already on disk are untouched.")

    tasks = [(t, d) for t, d in TASKS if not args.tasks or t in args.tasks]
    if args.status:
        return status(args.seeds)

    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    # seed-major order: one full seed across all tasks before the next, so an
    # interrupted matrix still yields a complete single-seed result set.
    queue = [(t, d, s) for s in args.seeds for t, d in tasks]
    todo = [q for q in queue if args.force or not is_complete(q[0], q[2])]
    skipped = len(queue) - len(todo)

    print(f"matrix     : {len(tasks)} tasks x {len(args.seeds)} seeds = {len(queue)} runs")
    print(f"already done: {skipped}")
    print(f"to run      : {len(todo)}  (~{len(todo)*1.55:.1f} GPU-hours, "
          f"~{len(todo)*1.55/24:.1f} days)")
    print(f"order       : seed-major (a full seed completes before the next starts)")
    print()
    for i, (t, d, s) in enumerate(todo, 1):
        print(f"  {i:2d}. {t:<24} seed {s}  target={d}")
    if args.dry_run:
        print("\n(dry run -- nothing launched)")
        return 0
    if not todo:
        print("nothing to do.")
        return 0

    print("\n" + "=" * 70)
    t_start = time.time()
    failures = []

    for i, (task, domain, seed) in enumerate(todo, 1):
        out = run_dir(task, seed)
        log = os.path.join(LOGS, f"{task}_seed{seed}{ARM_SUFFIX}.log")
        data = os.path.join(SPLITS, task)
        if not os.path.isdir(data):
            print(f"[{i}/{len(todo)}] SKIP {task} seed {seed}: no split at {data}")
            failures.append((task, seed, "missing split"))
            continue

        elapsed = time.time() - t_start
        remaining = (len(todo) - i + 1) * 1.55 * 3600
        eta = datetime.now() + timedelta(seconds=remaining)
        print(f"\n[{i}/{len(todo)}] {task} seed {seed}  "
              f"(elapsed {elapsed/3600:.1f} h, ETA all done {eta:%Y-%m-%d %H:%M})")
        print(f"         log -> {log}")

        # -u: unbuffered. Python block-buffers stdout when it is redirected to
        # a file, so a run that dies mid-training flushes nothing and the log
        # ends at the last TensorFlow C++ message -- exactly what happened to
        # vcl_to_facies_seed1_lambda1, which died at ~step 6000 leaving a log
        # with no error in it and no manifest. Unbuffered output costs nothing
        # here and means a crash leaves evidence.
        cmd = [sys.executable, "-u", os.path.join(HERE, "03_train.py"),
               "--data", data, "--target-domain", domain, "--out", out,
               "--seed", str(seed), "--steps", str(args.steps),
               "--val-every", str(args.val_every),
               "--gan-weight", str(args.gan_weight)]
        t0 = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 3600

        if proc.returncode == 0 and is_complete(task, seed):
            print(f"         done in {dt:.2f} h")
        else:
            print(f"         FAILED (exit {proc.returncode}) after {dt:.2f} h "
                  f"-- see {log}")
            failures.append((task, seed, f"exit {proc.returncode}"))

    total = (time.time() - t_start) / 3600
    print("\n" + "=" * 70)
    print(f"finished {len(todo) - len(failures)} / {len(todo)} runs in {total:.2f} h")
    if failures:
        print("\nfailures:")
        for t, s, why in failures:
            print(f"  {t} seed {s}: {why}")
        return 1
    print("\nAll runs complete. Next: evaluate with 02_evaluation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
