#!/usr/bin/env python3
"""
add_missing_stats.py — compute the statistics Reviewer 5 asked for and the
manuscript does not yet report:

  (a) clustered bootstrap 95 % confidence intervals, resampled at original-layer
      level (comment 7)
  (b) normalised RMSE, Pearson r and Spearman rho for the continuous tasks
      (comment 8)
  (c) confusion matrices and per-class precision / recall / F1 for all three
      inverse tasks, not only VCL -> facies (comment 10)
  (d) a paired layer-level test of the network against each baseline, which the
      central claim of the paper currently lacks

Inputs
------
Continuous tasks: one .npz per held-out layer with 'ref' and 'pred' in physical
units, NaN outside the reservoir.
Inverse tasks:    one .npz per held-out layer with 'ref' and 'pred' as integer
                  class indices 0..6, and optionally 'baseline'.

Usage
-----
python add_missing_stats.py continuous decoded_perm/ --log10
python add_missing_stats.py facies decoded_vcl2fac/ --classes 7
"""
import argparse, glob, itertools, os
import numpy as np


# ----------------------------------------------------------------- utilities
def load(dirpath, pattern="*.npz"):
    out = []
    for f in sorted(glob.glob(os.path.join(dirpath, pattern))):
        z = np.load(f)
        out.append((os.path.basename(f), z))
    if not out:
        raise SystemExit(f"no files matching {pattern} in {dirpath}")
    return out


def clustered_bootstrap(per_layer_values, n_boot=10000, seed=0):
    """Resample LAYERS with replacement, not cells. This is the quantity
    Reviewer 5 comment 7 asks for; standard deviation across seeds is not."""
    v = np.asarray(per_layer_values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_permutation(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    n = d.size
    obs = d.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=n)), float)
    means = signs @ d / n
    return obs, float(np.sum(np.abs(means) >= abs(obs) - 1e-12) / 2**n)


def rank(x):
    order = np.argsort(np.argsort(x))
    return order.astype(float)


# ---------------------------------------------------------------- continuous
def continuous(dirpath, use_log10):
    rows = []
    for name, z in load(dirpath):
        r, p = np.asarray(z["ref"], float).ravel(), np.asarray(z["pred"], float).ravel()
        m = ~(np.isnan(r) | np.isnan(p))
        r, p = r[m], p[m]
        if use_log10:
            k = (r > 0) & (p > 0)
            r, p = np.log10(r[k]), np.log10(p[k])
        e = p - r
        rmse = float(np.sqrt(np.mean(e**2)))
        ss = float(np.sum((r - r.mean())**2))
        rows.append(dict(
            layer=name, n=r.size,
            MAE=float(np.mean(np.abs(e))), RMSE=rmse,
            nRMSE_range=rmse / (r.max() - r.min()) if r.max() > r.min() else np.nan,
            nRMSE_mean=rmse / abs(r.mean()) if r.mean() != 0 else np.nan,
            bias=float(np.mean(e)),
            R2=1 - float(np.sum(e**2)) / ss if ss > 0 else np.nan,
            pearson=float(np.corrcoef(r, p)[0, 1]),
            spearman=float(np.corrcoef(rank(r), rank(p))[0, 1]),
        ))
    keys = ["MAE", "RMSE", "nRMSE_range", "nRMSE_mean", "bias", "R2", "pearson", "spearman"]
    print(f"space: {'log10' if use_log10 else 'linear'}   layers: {len(rows)}\n")
    print(f"{'metric':<14}{'mean':>12}{'95% CI low':>14}{'95% CI high':>14}")
    print("-" * 54)
    for k in keys:
        mean, lo, hi = clustered_bootstrap([r[k] for r in rows])
        print(f"{k:<14}{mean:>12.4f}{lo:>14.4f}{hi:>14.4f}")
    print("\nIntervals are a clustered bootstrap over layers (Reviewer 5 comment 7).")
    print("nRMSE is given both by reference range and by reference mean; state which you quote.")


# -------------------------------------------------------------------- facies
def facies(dirpath, n_classes, names=None):
    names = names or [f"class {i}" for i in range(n_classes)]
    cm_total = np.zeros((n_classes, n_classes), dtype=np.int64)
    per_layer_pa, per_layer_f1, base_pa = [], [], []
    for name, z in load(dirpath):
        r = np.asarray(z["ref"]).ravel(); p = np.asarray(z["pred"]).ravel()
        m = (r >= 0) & (r < n_classes) & (p >= 0) & (p < n_classes)
        r, p = r[m].astype(int), p[m].astype(int)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (r, p), 1)
        cm_total += cm
        per_layer_pa.append(np.trace(cm) / cm.sum())
        tp = np.diag(cm).astype(float)
        prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
        present = cm.sum(1) > 0
        per_layer_f1.append(f1[present].mean())
        if "baseline" in z:
            b = np.asarray(z["baseline"]).ravel()[m].astype(int)
            base_pa.append(float(np.mean(b == r)))

    tp = np.diag(cm_total).astype(float)
    prec = np.divide(tp, cm_total.sum(0), out=np.zeros_like(tp), where=cm_total.sum(0) > 0)
    rec = np.divide(tp, cm_total.sum(1), out=np.zeros_like(tp), where=cm_total.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    area = cm_total.sum(1) / cm_total.sum()

    print(f"{'class':<20}{'area %':>9}{'precision':>11}{'recall':>9}{'F1':>8}")
    print("-" * 57)
    for i in range(n_classes):
        print(f"{names[i]:<20}{100*area[i]:>9.2f}{prec[i]:>11.3f}{rec[i]:>9.3f}{f1[i]:>8.3f}")

    print("\nRow-normalised confusion matrix (rows = reference)")
    hdr = "".join(f"{n[:7]:>9}" for n in names)
    print(f"{'':<20}{hdr}")
    for i in range(n_classes):
        row = cm_total[i] / cm_total[i].sum() if cm_total[i].sum() else cm_total[i]
        print(f"{names[i]:<20}" + "".join(f"{v:>9.3f}" for v in row))

    for label, vals in [("pixel accuracy", per_layer_pa), ("macro-F1", per_layer_f1)]:
        mean, lo, hi = clustered_bootstrap(vals)
        print(f"\n{label}: {mean:.4f}   95 % CI [{lo:.4f}, {hi:.4f}]  (clustered bootstrap over layers)")

    if base_pa:
        d = np.array(per_layer_pa) - np.array(base_pa)
        obs, p = exact_sign_permutation(d)
        print(f"\nNetwork minus baseline, paired on layers: {obs:+.4f}   exact p = {p:.4f}")
        print("This is the test the central claim of the paper currently lacks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["continuous", "facies"])
    ap.add_argument("dirpath")
    ap.add_argument("--log10", action="store_true")
    ap.add_argument("--classes", type=int, default=7)
    ap.add_argument("--names", default="Tight,Very Poor Sand,Poor Sand,Fair Sand,Good Sand,Good Clean Sand,Very Good Sand")
    a = ap.parse_args()
    if a.kind == "continuous":
        continuous(a.dirpath, a.log10)
    else:
        facies(a.dirpath, a.classes, [s.strip() for s in a.names.split(",")][:a.classes])


if __name__ == "__main__":
    main()
