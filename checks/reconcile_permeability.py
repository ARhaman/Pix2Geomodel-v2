#!/usr/bin/env python3
"""
reconcile_permeability.py — resolve the two different linear-space permeability
results reported for facies -> permeability.

The manuscript reports, for the same task on the same nine held-out layers:

    Table 3   MAE 508.88   RMSE 794.03   R2 -0.2818
    Table 9   MAE 507.7    RMSE 846.4    R2 -0.1609

Same quantity, two values. The usual cause is that one was computed by pooling
every cell of every layer into a single vector, and the other by computing the
metric per layer and averaging. MAE is nearly identical under both because it is
linear in the residual; RMSE and R2 are not, which matches the pattern above.

This script computes both ways from the decoded predictions and reports which
convention reproduces which table, so the manuscript can state one and drop the
other, or keep both and label them.

Inputs: one .npz (or two .npy files) per held-out layer holding the decoded
reference and prediction in mD, masked to cells inside the reservoir.

    layer_46.npz  ->  arrays 'ref' and 'pred', NaN outside the reservoir

Usage:
    python reconcile_permeability.py decoded/ --pattern "layer_*.npz"
"""
import argparse, glob, os
import numpy as np


def metrics(ref, pred):
    ref, pred = np.asarray(ref, float).ravel(), np.asarray(pred, float).ravel()
    m = ~(np.isnan(ref) | np.isnan(pred))
    ref, pred = ref[m], pred[m]
    err = pred - ref
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((ref - ref.mean())**2))
    return dict(n=ref.size,
                MAE=float(np.mean(np.abs(err))),
                RMSE=float(np.sqrt(np.mean(err**2))),
                R2=1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirpath")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--ref-key", default="ref")
    ap.add_argument("--pred-key", default="pred")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dirpath, args.pattern)))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} in {args.dirpath}")

    per_layer, pool_ref, pool_pred = [], [], []
    for f in files:
        z = np.load(f)
        r, p = z[args.ref_key], z[args.pred_key]
        per_layer.append((os.path.basename(f), metrics(r, p)))
        pool_ref.append(np.asarray(r, float).ravel())
        pool_pred.append(np.asarray(p, float).ravel())

    print(f"{len(files)} layers\n")
    print(f"{'layer':<18}{'n':>10}{'MAE':>12}{'RMSE':>12}{'R2':>10}")
    print("-" * 62)
    for name, m in per_layer:
        print(f"{name:<18}{m['n']:>10d}{m['MAE']:>12.2f}{m['RMSE']:>12.2f}{m['R2']:>10.4f}")

    pooled = metrics(np.concatenate(pool_ref), np.concatenate(pool_pred))
    avg = {k: float(np.mean([m[k] for _, m in per_layer])) for k in ("MAE", "RMSE", "R2")}

    print("\nCONVENTION A — pool every cell of every layer, then compute once")
    print(f"   MAE {pooled['MAE']:.2f}   RMSE {pooled['RMSE']:.2f}   R2 {pooled['R2']:+.4f}   n = {pooled['n']:,}")
    print("\nCONVENTION B — compute per layer, then average the nine values")
    print(f"   MAE {avg['MAE']:.2f}   RMSE {avg['RMSE']:.2f}   R2 {avg['R2']:+.4f}")

    print("\nMatch against the manuscript:")
    for label, mae, rmse, r2 in [("Table 3", 508.88, 794.03, -0.2818),
                                 ("Table 9", 507.70, 846.40, -0.1609)]:
        for conv, m in [("A pooled", pooled), ("B per-layer mean", avg)]:
            close = (abs(m['MAE']-mae) < 1.0 and abs(m['RMSE']-rmse) < 5.0
                     and abs(m['R2']-r2) < 0.02)
            if close:
                print(f"   {label} reproduced by convention {conv}")
    print("\nReport one convention in the manuscript and say which it is.")
    print("R2 is not comparable across conventions; MAE nearly is.")


if __name__ == "__main__":
    main()
