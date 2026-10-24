#!/usr/bin/env python3
"""
14_baseline_conventional.py
===========================

Conventional property-modelling baselines (Reviewer 5 #15, Reviewer 1 #3).

R5-15 asks for "at least one conventional geostatistical / property-modelling
baseline". The L1-only U-Net in 05_run_all.py --gan-weight 0 is a non-GAN
baseline, but it is still a 54-million-parameter convolutional network; it does
not answer the question a reservoir engineer would actually ask, which is
whether the deep model beats what the industry already does.

Two baselines, both of which a geomodeller could build in an afternoon:

  FORWARD  facies -> property
    CLASS LOOKUP. Assign every cell the mean (and separately the median) of
    that facies class, computed on the TRAINING layers only. This is
    deterministic property population, the standard non-stochastic workflow.
    It uses one number per class -- seven numbers total against 54,425,859
    parameters.

  INVERSE  property -> facies
    BAYES / MAXIMUM-LIKELIHOOD. From the training layers build a 1-D
    histogram of the property for each facies class, then label each test cell
    with the class of highest posterior. This ignores spatial context entirely,
    so it isolates exactly how much of the network's performance comes from
    the pointwise value-to-class relationship and how much from spatial
    reasoning.

    RANDOM FOREST (if scikit-learn is present) on the value plus local
    neighbourhood mean and standard deviation at two window sizes, which gives
    the classical method a fair amount of spatial context to work with.

Everything is fitted on TRAIN layers and evaluated on the same nine held-out
TEST layers, decoded with the same functions and scored with the same metrics
as the networks, so the numbers drop straight into the same tables.

Honest framing: if the lookup table matches the GAN, that is the result, and
the manuscript has to say so.

    python 14_baseline_conventional.py             (CPU; a few minutes)
    python 14_baseline_conventional.py --no-rf     (skip the random forest)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
EXPORT = os.path.join(HERE, "export_v2")
OUT = os.path.join(HERE, "results")

FORWARD = [("facies_to_porosity", "porosity"),
           ("facies_to_permeability", "permeability"),
           ("facies_to_vcl", "vcl")]
INVERSE = [("porosity_to_facies", "porosity"),
           ("permeability_to_facies", "permeability"),
           ("vcl_to_facies", "vcl")]

# Subsample stride when fitting. The training split holds 1116 images of
# 2 M pixels; every hundredth cell is still ~22 million samples per domain,
# which is far more than enough to estimate seven class means or a histogram.
FIT_STRIDE = 101
MAX_FIT_IMAGES = 120


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_cont(img: np.ndarray, domain: str, ev) -> Tuple[np.ndarray, np.ndarray]:
    spec = ev.CONTINUOUS_DOMAINS[domain]
    lo_i, hi_i = ev.ramp_range(spec)
    lut = ev.build_lut(spec["stops"], hi_i - lo_i + 1)
    idx = ev.decode_rgb_to_index(img, lut) + lo_i
    val = ev.index_to_value(idx, domain).astype(np.float64)
    return val, ~np.all(img == 0, axis=-1)


def halves(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """(target, input) -- target is the LEFT half (03_train.load_pair)."""
    pair = np.array(Image.open(path).convert("RGB"))
    w = pair.shape[1] // 2
    return pair[:, :w], pair[:, w:]


def train_files(task: str) -> List[str]:
    d = os.path.join(SPLITS, task, "train")
    fs = sorted(f for f in os.listdir(d) if f.endswith("_aug00.png")) or \
         sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return [os.path.join(d, f) for f in fs[:MAX_FIT_IMAGES]]


def test_files(task: str) -> List[str]:
    d = os.path.join(SPLITS, task, "test")
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def _integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a leading zero row and column."""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return np.pad(c, ((1, 0), (1, 0)), mode="constant")


def neighbourhood(v: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Box mean and standard deviation over a k x k window (k odd), via integral
    images.

    The zero row/column in _integral is what makes the four-corner difference
    return exactly the input shape. Without it the differences are taken over
    a table one element too small in each axis and the result comes out
    (H-1, W-1) -- which then fails at np.stack rather than quietly returning
    misaligned features, but only because the other feature happened to be
    full size. Shape arithmetic like this is checked by _selftest_neighbourhood
    against a brute-force computation rather than by reading it.
    """
    r = k // 2
    p = np.pad(v.astype(np.float64), r, mode="edge")
    c1, c2 = _integral(p), _integral(p * p)

    def box(c):
        return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]

    n = float(k * k)
    m = box(c1) / n
    var = np.maximum(box(c2) / n - m * m, 0.0)
    return m, np.sqrt(var)


def _selftest_neighbourhood() -> None:
    rng = np.random.default_rng(0)
    v = rng.standard_normal((23, 31))
    for k in (3, 5, 17):
        m, sd = neighbourhood(v, k)
        assert m.shape == v.shape == sd.shape, (k, m.shape, v.shape, sd.shape)
        r = k // 2
        p = np.pad(v, r, mode="edge")
        for (i, j) in ((0, 0), (5, 7), (22, 30), (11, 3)):
            w = p[i:i + k, j:j + k]
            assert abs(m[i, j] - w.mean()) < 1e-9, (k, i, j, m[i, j], w.mean())
            assert abs(sd[i, j] - w.std()) < 1e-9, (k, i, j, sd[i, j], w.std())


_selftest_neighbourhood()


def features(v: np.ndarray) -> np.ndarray:
    m5, s5 = neighbourhood(v, 5)
    m17, s17 = neighbourhood(v, 17)
    return np.stack([v, m5, s5, m17, s17], axis=-1)


def facies_features(lab: np.ndarray, n_cls: int) -> np.ndarray:
    """
    Features for the FORWARD random-forest regressor (Reviewer 1 #6).

    The class-mean lookup uses one number per facies class and therefore cannot
    represent any within-class variation. To give the conventional side a fair
    non-linear pointwise competitor, the regressor sees the same facies label
    plus a description of the local facies neighbourhood:

        - the class label itself (ordinal, and also as one-hot fractions below)
        - the fraction of each class within a 5 x 5 and a 17 x 17 window

    The window fractions are computed with the same integral-image box filter
    used for the inverse baseline, so the two directions share one code path.
    This is exactly the information the Reviewer describes: "local
    neighbourhood statistics as a non-linear pointwise baseline".
    """
    feats = [lab.astype(np.float64)]
    for c in range(1, n_cls + 1):
        ind = (lab == c).astype(np.float64)
        for k in (5, 17):
            m, _ = neighbourhood(ind, k)
            feats.append(m)
    return np.stack(feats, axis=-1)


def _selftest_facies_features() -> None:
    lab = np.array([[1, 1, 2], [1, 2, 2], [3, 3, 3]], dtype=np.int64)
    f = facies_features(lab, 3)
    assert f.shape == (3, 3, 1 + 3 * 2), f.shape
    # class fractions over any window must sum to 1 at every cell
    for k_off in (1, 2):
        tot = f[..., k_off::2].sum(axis=-1)
        assert np.allclose(tot, 1.0), (k_off, tot)


_selftest_facies_features()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-rf", action="store_true", help="skip the random forest")
    args = ap.parse_args()

    ev = _load("pix2geo_eval", "02_evaluation.py")
    os.makedirs(OUT, exist_ok=True)
    n_cls = len(ev.FACIES_LEGEND)
    rows = [["Direction", "Task", "Baseline", "Metric", "Value", "Cells or layers"]]

    # ---------------- FORWARD: facies -> property ------------------------
    # Permeability is scored twice: linear, for continuity with Table 4, and
    # log10, because every inverse-direction permeability number is log10 and a
    # reader comparing the two tables would otherwise see the same baseline
    # score 0.0014 in one and 0.74 in the other.
    rf_fwd = not args.no_rf
    if rf_fwd:
        try:
            from sklearn.ensemble import RandomForestRegressor  # noqa: F401
        except ImportError:
            print("[note] scikit-learn not installed -- forward random forest skipped.")
            rf_fwd = False

    FORWARD_SPACES = [(t, d, "linear") for t, d in FORWARD] + \
                     [("facies_to_permeability", "permeability", "log10")]

    for task, domain, space in FORWARD_SPACES:
        lm = os.path.join(EXPORT, domain, "legend_manifest.json")
        if os.path.exists(lm):
            ev.load_legend_manifest(lm)
        logsp = (space == "log10")
        label = f"{task}  (class-lookup baseline, {space})"
        print(f"\n{'='*76}\n{label}\n{'='*76}")

        acc: Dict[int, List[np.ndarray]] = {c: [] for c in range(1, n_cls + 1)}
        for p in train_files(task):
            tgt, inp = halves(p)
            val, vmask = decode_cont(tgt, domain, ev)
            if logsp:
                vmask = vmask & (val > 0)
                val = np.log10(np.where(val > 0, val, 1.0))
            lab = ev.decode_rgb_to_facies(inp)
            m = vmask & (lab >= 1)
            for c in range(1, n_cls + 1):
                sel = val[m & (lab == c)]
                if sel.size:
                    acc[c].append(sel[::FIT_STRIDE])
        table_mean, table_med = {}, {}
        for c in range(1, n_cls + 1):
            if acc[c]:
                a = np.concatenate(acc[c])
                table_mean[c], table_med[c] = float(a.mean()), float(np.median(a))
        gm = float(np.mean([v for v in table_mean.values()])) if table_mean else 0.0
        print(f"  fitted {len(table_mean)} class values from "
              f"{len(train_files(task))} training layers")

        # ---- random-forest regressor on facies + neighbourhood (R1 #6) ----
        if rf_fwd:
            from sklearn.ensemble import RandomForestRegressor
            Xs, ys = [], []
            for p in train_files(task):
                tgt, inp = halves(p)
                val, vmask = decode_cont(tgt, domain, ev)
                if logsp:
                    vmask = vmask & (val > 0)
                    val = np.log10(np.where(val > 0, val, 1.0))
                lab = ev.decode_rgb_to_facies(inp)
                m = vmask & (lab >= 1)
                f = facies_features(lab, n_cls)
                Xs.append(f[m][::FIT_STRIDE])
                ys.append(val[m][::FIT_STRIDE])
            Xf = np.concatenate(Xs); yf = np.concatenate(ys)
            n = min(yf.size, 400_000)
            idx = np.random.default_rng(2025).choice(yf.size, n, replace=False)
            rfr = RandomForestRegressor(n_estimators=60, max_depth=18, n_jobs=-1,
                                        random_state=2025)
            rfr.fit(Xf[idx], yf[idx])
            print(f"  random-forest regressor fitted on {n:,} cells")

            R, P = [], []
            for p in test_files(task):
                tgt, inp = halves(p)
                val, vmask = decode_cont(tgt, domain, ev)
                if logsp:
                    vmask = vmask & (val > 0)
                    val = np.log10(np.where(val > 0, val, 1.0))
                lab = ev.decode_rgb_to_facies(inp)
                m = vmask & (lab >= 1)
                R.append(val[m])
                P.append(rfr.predict(facies_features(lab, n_cls)[m]))
            r, q = np.concatenate(R), np.concatenate(P)
            err = q - r
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((r - r.mean()) ** 2))
            res = {"MAE": float(np.mean(np.abs(err))),
                   "RMSE": float(np.sqrt(np.mean(err ** 2))),
                   "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}
            unit_f = ("log10 mD" if logsp else
                      ("mD" if domain == "permeability" else "v/v"))
            for k, v in res.items():
                rows.append(["forward", task, f"random forest [{space}]",
                             f"{k} [{unit_f}]", f"{v:.6g}", r.size])
            print(f"  {'random forest':<14} MAE {res['MAE']:.4g}  "
                  f"RMSE {res['RMSE']:.4g}  R2 {res['R2']:+.4f}   [{unit_f}]")

        for stat, table in (("class mean", table_mean), ("class median", table_med)):
            R, P = [], []
            for p in test_files(task):
                tgt, inp = halves(p)
                val, vmask = decode_cont(tgt, domain, ev)
                if logsp:
                    vmask = vmask & (val > 0)
                    val = np.log10(np.where(val > 0, val, 1.0))
                lab = ev.decode_rgb_to_facies(inp)
                m = vmask & (lab >= 1)
                pred = np.full(val.shape, gm)
                for c, v in table.items():
                    pred[lab == c] = v
                R.append(val[m]); P.append(pred[m])
            r, q = np.concatenate(R), np.concatenate(P)
            err = q - r
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((r - r.mean()) ** 2))
            res = {"MAE": float(np.mean(np.abs(err))),
                   "RMSE": float(np.sqrt(np.mean(err ** 2))),
                   "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")}
            unit_f = ("log10 mD" if logsp else
                      ("mD" if domain == "permeability" else "v/v"))
            for k, v in res.items():
                rows.append(["forward", task, f"{stat} [{space}]",
                             f"{k} [{unit_f}]", f"{v:.6g}", r.size])
            print(f"  {stat:<14} MAE {res['MAE']:.4g}  RMSE {res['RMSE']:.4g}  "
                  f"R2 {res['R2']:+.4f}   [{unit_f}]")

    # ---------------- INVERSE: property -> facies ------------------------
    rf_ok = not args.no_rf
    if rf_ok:
        try:
            from sklearn.ensemble import RandomForestClassifier   # noqa: F401
        except ImportError:
            print("\n[note] scikit-learn not installed -- random forest skipped.")
            print("       pip install scikit-learn --break-system-packages")
            rf_ok = False

    for task, domain in INVERSE:
        lm = os.path.join(EXPORT, domain, "legend_manifest.json")
        if os.path.exists(lm):
            ev.load_legend_manifest(lm)
        print(f"\n{'='*76}\n{task}  (Bayes{' + random forest' if rf_ok else ''})\n{'='*76}")

        Xs, ys = [], []
        for p in train_files(task):
            tgt, inp = halves(p)
            lab = ev.decode_rgb_to_facies(tgt)
            val, vmask = decode_cont(inp, domain, ev)
            if domain == "permeability":
                vmask = vmask & (val > 0)
                val = np.log10(np.where(val > 0, val, 1.0))
            m = vmask & (lab >= 1)
            f = features(val)
            Xs.append(f[m][::FIT_STRIDE]); ys.append(lab[m][::FIT_STRIDE])
        X = np.concatenate(Xs); y = np.concatenate(ys)
        print(f"  fitted on {y.size:,} sampled cells")

        # 1-D maximum-likelihood on the raw value (feature 0), 256 bins
        lo, hi = float(np.percentile(X[:, 0], 0.1)), float(np.percentile(X[:, 0], 99.9))
        edges = np.linspace(lo, hi, 257)
        hist = np.zeros((n_cls, 256))
        for c in range(1, n_cls + 1):
            sel = X[y == c, 0]
            if sel.size:
                hist[c - 1] = np.histogram(sel, bins=edges)[0]
        prior = hist.sum(axis=1, keepdims=True)
        post = (hist + 1.0) / (prior + 256.0) * np.maximum(prior, 1.0)
        bayes_lut = np.argmax(post, axis=0) + 1

        rf = None
        if rf_ok:
            from sklearn.ensemble import RandomForestClassifier
            n = min(y.size, 400_000)
            idx = np.random.default_rng(2025).choice(y.size, n, replace=False)
            rf = RandomForestClassifier(n_estimators=60, max_depth=18,
                                        n_jobs=-1, random_state=2025,
                                        class_weight="balanced_subsample")
            rf.fit(X[idx], y[idx])
            print(f"  random forest fitted on {n:,} cells")

        # ---- class means for the physical-unit round trip -----------------
        # Fitted on the SAME training sample (X, y) used above, in the SAME
        # space -- note `val` is already log10 for permeability, so these are
        # log-space means. Any class absent from training gets NaN and its
        # cells are excluded rather than imputed.
        cmean = np.full(n_cls + 1, np.nan)
        for c in range(1, n_cls + 1):
            sel = X[y == c, 0]
            if sel.size:
                cmean[c] = float(sel.mean())
        n_missing = int(np.sum(np.isnan(cmean[1:])))
        if n_missing:
            print(f"  note: {n_missing} class(es) have no training support; "
                  f"their cells are excluded from the physical-unit metrics")

        cms = {"bayes": np.zeros((n_cls, n_cls))}
        if rf is not None:
            cms["random forest"] = np.zeros((n_cls, n_cls))

        # Streaming accumulators, so nothing depends on holding 14 M cells.
        #   n, S|e|, Se2, Sv, Sv2   ->  MAE, RMSE, R2
        acc_phys = {k: np.zeros(5) for k in cms}
        # eta^2 on the TEST layers, using the TRUE facies partition.
        eta_sum = np.zeros(n_cls + 1)
        eta_cnt = np.zeros(n_cls + 1)
        eta_glob = np.zeros(3)          # n, Sv, Sv2

        def _accum(a, pred_lab, true_val):
            ok = ~np.isnan(cmean[pred_lab])
            if not np.any(ok):
                return
            q = cmean[pred_lab[ok]]
            r = true_val[ok]
            e = q - r
            a += np.array([r.size, np.abs(e).sum(), (e * e).sum(),
                           r.sum(), (r * r).sum()])

        for p in test_files(task):
            tgt, inp = halves(p)
            lab = ev.decode_rgb_to_facies(tgt)
            val, vmask = decode_cont(inp, domain, ev)
            if domain == "permeability":
                vmask = vmask & (val > 0)
                val = np.log10(np.where(val > 0, val, 1.0))
            m = vmask & (lab >= 1)
            b = np.clip(np.digitize(val, edges) - 1, 0, 255)

            pb = bayes_lut[b][m]
            cms["bayes"] += ev.confusion_matrix(pb, lab[m], n_cls)
            _accum(acc_phys["bayes"], pb, val[m])

            if rf is not None:
                f = features(val)[m]
                pr = rf.predict(f).astype(int)
                cms["random forest"] += ev.confusion_matrix(pr, lab[m], n_cls)
                _accum(acc_phys["random forest"], pr, val[m])

            tv, tl = val[m], lab[m]
            np.add.at(eta_sum, tl, tv)
            np.add.at(eta_cnt, tl, 1)
            eta_glob += np.array([tv.size, tv.sum(), (tv * tv).sum()])

        # eta^2 = between-class SS / total SS, on the test layers.
        n_g, s_g, s2_g = eta_glob
        ss_tot_g = s2_g - (s_g * s_g) / n_g if n_g > 0 else 0.0
        ss_bet = 0.0
        if n_g > 0:
            gm_t = s_g / n_g
            for c in range(1, n_cls + 1):
                if eta_cnt[c] > 0:
                    ss_bet += eta_cnt[c] * (eta_sum[c] / eta_cnt[c] - gm_t) ** 2
        eta2 = ss_bet / ss_tot_g if ss_tot_g > 0 else float("nan")
        unit = "log10 mD" if domain == "permeability" else "v/v"
        rows.append(["inverse", task, "reference", "eta2_true_facies",
                     f"{eta2:.6g}", int(n_g)])
        print(f"  eta^2 (true facies partition, test): {eta2:.4f}  [{unit}]")

        for name, cm in cms.items():
            met = ev.categorical_metrics(cm, [n for n, _ in ev.FACIES_LEGEND])
            for k in ("pixel_accuracy", "macro_F1", "mIoU"):
                rows.append(["inverse", task, name, k, f"{met[k]:.6g}",
                             int(cm.sum())])
            print(f"  {name:<15} PA {met['pixel_accuracy']:.4f}  "
                  f"macro-F1 {met['macro_F1']:.4f}  mIoU {met['mIoU']:.4f}")

            n_a, sae, se2, sv, sv2 = acc_phys[name]
            if n_a <= 0:
                continue
            mae = sae / n_a
            rmse = float(np.sqrt(se2 / n_a))
            ss_tot = sv2 - (sv * sv) / n_a
            r2 = 1.0 - se2 / ss_tot if ss_tot > 0 else float("nan")
            for k, v in (("MAE", mae), ("RMSE", rmse), ("R2", r2)):
                rows.append(["inverse", task, name, f"{k} [{unit}]",
                             f"{v:.6g}", int(n_a)])
            print(f"  {'':<15} MAE {mae:.4f}  RMSE {rmse:.4f}  "
                  f"R2 {r2:+.4f}  ({100 * r2 / eta2:.1f} % of eta^2)")

    with open(os.path.join(OUT, "13_Conventional_baselines.csv"), "w",
              newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    print(f"\nwritten to {os.path.join(OUT, '13_Conventional_baselines.csv')}")
    print("\nCompare against results/1_Main_results.csv. Where a seven-number")
    print("lookup table matches the network, the network is not earning its cost")
    print("and the manuscript must report that plainly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
