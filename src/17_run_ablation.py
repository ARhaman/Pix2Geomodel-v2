#!/usr/bin/env python3
"""
17_run_ablation.py
==================

Ablation study (Reviewer 5 #16, Reviewer 1 #5).

One representative task, four things varied one at a time against the trained
baseline, three seeds each. The baseline arm is NOT re-trained: it already
exists as runs/<task>_seed{1,2,3}, produced by the same script at the same
step count, so it is a valid control.

    arm            what changes                       why it is asked for
    ---------------------------------------------------------------------
    noaug          augmentation off                   R5-16: does the 31x
                                                      augmentation help, or
                                                      does it just inflate the
                                                      apparent dataset size?
    halfres        1024 x 512 instead of 2048 x 1024   R1-5: 8184 x 12045 was
                                                      resized to 2048 x 1024,
                                                      "a significant loss of
                                                      information". If halving
                                                      again costs little, the
                                                      information argument is
                                                      weaker than it looks.
    lambda10       L1 weight 100 -> 10                 R5-16: how much of the
    lambda1        L1 weight 100 -> 1                  result is the L1 term?
                                                      With the non-GAN arm
                                                      already run, these two
                                                      bracket the trade-off.

TASK CHOICE. vcl_to_facies by default: it is the strongest inverse task, so a
degradation there is unambiguous, and it is the one where the conventional
Bayes baseline beats the network most decisively -- which makes the question
"can any hyperparameter choice close that gap?" worth answering directly.

Cost: 4 arms x 3 seeds x ~0.8-1.4 h = roughly 12-16 GPU-hours. Resumable: an
arm that already has a manifest with a result block is skipped.

    python 17_run_ablation.py --status
    python 17_run_ablation.py
    python 17_run_ablation.py --arms noaug halfres
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
LOGS = os.path.join(HERE, "logs_training")

DEFAULT_TASK = "vcl_to_facies"
DOMAIN = {"vcl_to_facies": "facies", "porosity_to_facies": "facies",
          "permeability_to_facies": "facies", "facies_to_porosity": "porosity",
          "facies_to_permeability": "permeability", "facies_to_vcl": "vcl"}

# suffix -> extra CLI arguments
ARMS: Dict[str, List[str]] = {
    "noaug":    ["--no-augment"],
    "halfres":  ["--img-height", "1024", "--img-width", "512"],
    "lambda10": ["--lambda-l1", "10"],
    "lambda1":  ["--lambda-l1", "1"],
}


def run_dir(task: str, seed: int, arm: str) -> str:
    return os.path.join(RUNS, f"{task}_seed{seed}_{arm}")


def is_complete(task: str, seed: int, arm: str) -> bool:
    mf = os.path.join(run_dir(task, seed, arm), "run_manifest.json")
    if not os.path.exists(mf):
        return False
    try:
        with open(mf) as fh:
            return int(json.load(fh).get("result", {}).get("best_step", -1)) >= 0
    except Exception:
        return False


def hours(task: str, seed: int, arm: str) -> float:
    mf = os.path.join(run_dir(task, seed, arm), "run_manifest.json")
    try:
        with open(mf) as fh:
            return float(json.load(fh).get("result", {}).get("total_hours", 0.0))
    except Exception:
        return 0.0


def status(task: str, seeds: List[int], arms: List[str]) -> int:
    print(f"task: {task}")
    print(f"{'arm':<12}" + "".join(f"seed{s:<7}" for s in seeds))
    done = 0
    for arm in arms:
        cells = []
        for s in seeds:
            if is_complete(task, s, arm):
                done += 1
                cells.append(f"{hours(task, s, arm):.2f} h ")
            else:
                cells.append("--      ")
        print(f"{arm:<12}" + "".join(f"{c:<11}" for c in cells))
    total = len(arms) * len(seeds)
    print(f"\ncomplete {done} / {total}   outstanding {total - done}")
    print("\nbaseline control = runs/<task>_seed<n> (already trained, not re-run)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    bad = [a for a in args.arms if a not in ARMS]
    if bad:
        print(f"unknown arm(s): {', '.join(bad)}   choose from {', '.join(ARMS)}")
        return 2
    if args.status:
        return status(args.task, args.seeds, args.arms)

    os.makedirs(LOGS, exist_ok=True)
    domain = DOMAIN.get(args.task)
    if domain is None:
        print(f"unknown task {args.task}")
        return 2
    data = os.path.join(SPLITS, args.task)
    if not os.path.isdir(data):
        print(f"no split at {data}")
        return 2

    todo = [(arm, s) for arm in args.arms for s in args.seeds
            if not is_complete(args.task, s, arm)]
    if not todo:
        print("nothing to do -- every requested run is already complete")
        return 0

    print(f"ABLATION on {args.task}  ({domain})")
    print(f"{len(todo)} runs to go; the baseline arm is runs/{args.task}_seed<n>, "
          f"already trained and not re-run")
    print("=" * 70)

    t0 = time.time()
    failures = []
    for i, (arm, seed) in enumerate(todo, 1):
        out = run_dir(args.task, seed, arm)
        log = os.path.join(LOGS, f"{args.task}_seed{seed}_{arm}.log")
        eta = datetime.now() + timedelta(hours=1.1 * (len(todo) - i + 1))
        print(f"\n[{i}/{len(todo)}] {arm} seed {seed}   "
              f"(elapsed {(time.time()-t0)/3600:.1f} h, ETA {eta:%Y-%m-%d %H:%M})")
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
               "--val-every", str(args.val_every)] + ARMS[arm]
        t = time.time()
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(" ".join(cmd) + "\n\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        dt = (time.time() - t) / 3600
        if proc.returncode == 0 and is_complete(args.task, seed, arm):
            print(f"         done in {dt:.2f} h")
        else:
            print(f"         FAILED (exit {proc.returncode}) after {dt:.2f} h "
                  f"-- see {log}")
            failures.append((arm, seed))

    print("\n" + "=" * 70)
    print(f"finished {len(todo) - len(failures)} / {len(todo)} in "
          f"{(time.time()-t0)/3600:.2f} h")
    if failures:
        print("failed: " + ", ".join(f"{a} seed {s}" for a, s in failures))
        print("Re-run the same command; completed runs are skipped.")
    else:
        print("\nNext: python 06_predict.py --all, then score each arm.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
