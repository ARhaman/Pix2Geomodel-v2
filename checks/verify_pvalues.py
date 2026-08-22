#!/usr/bin/env python3
"""
verify_pvalues.py — recompute the exact paired sign-permutation p-values.

Reviewer point: with nine held-out layers an exact two-sided sign-permutation
test enumerates 2^9 = 512 sign assignments, so the p-value it returns is always
a multiple of 1/512. Two-sided values are therefore 2/512 = 0.003906,
4/512 = 0.007813, 6/512 = 0.011719 ... Neither 0.0038 nor 0.0090 is attainable,
so the manuscript values must be either rounded from attainable ones or produced
by a different convention.

This script recomputes them from the per-layer paired differences and prints
the exact attainable value, so the manuscript can quote it.

Input: a CSV with one row per held-out layer and one column per task/arm, e.g.

    layer,facies_to_porosity,facies_to_vcl,facies_to_permeability
    46,0.2701,0.4908,0.1904
    47,...

Usage:
    python verify_pvalues.py per_layer_metrics.csv --a facies_to_porosity --b facies_to_vcl
    python verify_pvalues.py per_layer_metrics.csv --all
"""
import argparse, itertools, sys
import numpy as np
import pandas as pd


def exact_sign_permutation(d, alternative="two-sided"):
    """Exact paired sign-permutation test on differences d (one per layer).

    Enumerates every assignment of +/- to the observed differences and counts
    how many give a mean at least as extreme as the observed one.
    Returns (observed_mean, p_value, n_enumerated, smallest_attainable_p).
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n == 0:
        raise ValueError("no non-missing differences")
    if n > 22:
        raise ValueError(f"{n} layers would need 2^{n} enumerations; use a Monte-Carlo test")

    obs = d.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)), dtype=float)
    means = signs @ d / n

    if alternative == "two-sided":
        count = int(np.sum(np.abs(means) >= abs(obs) - 1e-12))
        smallest = 2 / 2**n
    else:
        count = int(np.sum(means >= obs - 1e-12))
        smallest = 1 / 2**n
    return obs, count / 2**n, 2**n, smallest


def attainable_grid(n, two_sided=True, k=6):
    step = 2 if two_sided else 1
    return [step * i / 2**n for i in range(1, k + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="one row per held-out layer, one column per task or arm")
    ap.add_argument("--layer-col", default="layer")
    ap.add_argument("--a", help="first column")
    ap.add_argument("--b", help="second column")
    ap.add_argument("--all", action="store_true", help="test every pair of numeric columns")
    ap.add_argument("--alternative", default="two-sided", choices=["two-sided", "greater"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = [c for c in df.columns if c != args.layer_col and pd.api.types.is_numeric_dtype(df[c])]
    n = len(df)
    print(f"{args.csv}: {n} layers, {len(cols)} numeric columns\n")
    print(f"Exact {args.alternative} test over 2^{n} = {2**n} sign assignments.")
    print("Attainable p-values:", ", ".join(f"{p:.4f}" for p in attainable_grid(n, args.alternative == 'two-sided')))
    print()

    pairs = itertools.combinations(cols, 2) if args.all else [(args.a, args.b)]
    print(f"{'comparison':<48} {'mean diff':>10} {'p':>10}  {'p as k/2^n':>12}")
    print("-" * 86)
    for a, b in pairs:
        if a is None or b is None:
            sys.exit("give --a and --b, or --all")
        obs, p, N, smallest = exact_sign_permutation(df[a] - df[b], args.alternative)
        k = round(p * N)
        flag = "  <-- at the floor" if abs(p - smallest) < 1e-12 else ""
        print(f"{a + ' vs ' + b:<48} {obs:>+10.4f} {p:>10.4f}  {k:>6d}/{N}{flag}")
    print()
    print("Quote the printed p exactly. If a manuscript value is not on the")
    print("attainable grid above, it was not produced by this test.")


if __name__ == "__main__":
    main()
