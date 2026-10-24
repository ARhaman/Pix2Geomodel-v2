#!/usr/bin/env python3
"""
07_report.py
============

Turn 18 run folders into the manuscript's results tables.

Runs the corrected evaluation over every prediction set, then aggregates:

  Table 1  Main results -- per task, per seed, mean +/- SD across seeds, and
           layer-clustered bootstrap 95 % CIs.        (R5 #5, #6, #7, #26b/c)
  Table 2  Per-class facies metrics for the three inverse tasks: precision,
           recall, F1, IoU and class frequency.              (R5 #10, #26e)
  Table 3  Confusion matrices, pooled over seeds.                  (R5 #10)
  Table 4  Physical-unit errors for the continuous tasks, with the encoding
           floor alongside so model error and representation error are
           separable.                            (R1 Results, R5 #8/#11/#26d)
  Table 5  Paired permutation tests between tasks, at the layer level.  (R5 #9)
  Table 6  Reproducibility record -- seeds, environment, checkpoint rule.
                                                          (R5 #6, R4 #2)

Outputs `results/` as JSON + CSV always, and .xlsx when openpyxl is present.

RUN THIS IN THE `base` ENVIRONMENT, not tf210 -- it needs numpy, Pillow and
(optionally) openpyxl, but no TensorFlow.

    python 07_report.py
    python 07_report.py --no-bootstrap     (faster, no CIs)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits_v2")
RUNS = os.path.join(HERE, "runs")
EXPORT = os.path.join(HERE, "export_v2")
OUT = os.path.join(HERE, "results")
OUT_PREFIX = ""

TASKS = [
    ("facies_to_porosity",     "porosity"),
    ("facies_to_permeability", "permeability"),
    ("facies_to_vcl",          "vcl"),
    ("porosity_to_facies",     "facies"),
    ("permeability_to_facies", "facies"),
    ("vcl_to_facies",          "facies"),
]

# What the ORIGINAL manuscript reported, for the before/after table.
PUBLISHED = {
    "facies_to_porosity":     dict(PA=0.9326, mPA=0.6079, mIoU=0.5594, FWIoU=0.8807),
    "facies_to_permeability": dict(PA=0.7813, mPA=0.5694, mIoU=0.4546, FWIoU=0.6530),
    "facies_to_vcl":          dict(PA=0.6608, mPA=0.7608, mIoU=0.5593, FWIoU=0.5078),
    "porosity_to_facies":     dict(PA=0.9060, mPA=0.5898, mIoU=0.5129, FWIoU=0.8418),
    "permeability_to_facies": dict(PA=0.7948, mPA=0.4274, mIoU=0.3500, FWIoU=0.6955),
    "vcl_to_facies":          dict(PA=0.8440, mPA=0.8506, mIoU=0.7049, FWIoU=0.7429),
}


def _load(name: str, filename: str):
    path = os.path.join(HERE, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Set from --arm. "" is the adversarial arm; "_l1" the non-GAN baseline.
ARM_SUFFIX = ""


def discover(task: str) -> List[int]:
    """
    Seeds with predictions, for the CURRENT arm only.

    Matching is exact on the suffix rather than by prefix: without that, the
    adversarial arm would also pick up "<task>_seed3_l1" and silently average
    two different models together.
    """
    seeds = []
    if not os.path.isdir(RUNS):
        return seeds
    for d in sorted(os.listdir(RUNS)):
        if not d.startswith(task + "_seed"):
            continue
        tail = d.rsplit("_seed", 1)[1]
        if ARM_SUFFIX:
            if not tail.endswith(ARM_SUFFIX):
                continue
            tail = tail[: -len(ARM_SUFFIX)]
        elif not tail.isdigit():
            continue
        pred = os.path.join(RUNS, d, "predictions")
        if os.path.isdir(pred) and any(f.endswith(".png") for f in os.listdir(pred)):
            seeds.append(int(tail))
    return sorted(seeds)


def run_manifest(task: str, seed: int) -> dict:
    p = os.path.join(RUNS, f"{task}_seed{seed}{ARM_SUFFIX}", "run_manifest.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}


def mean_sd(vals: List[float]) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.4f}"
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"


def evaluate_all(ev, bootstrap: bool) -> dict:
    """Score every (task, seed) prediction set."""
    all_res: Dict[str, Dict[int, dict]] = defaultdict(dict)
    per_layer_store: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(dict)

    for task, domain in TASKS:
        seeds = discover(task)
        if not seeds:
            print(f"  {task:<24} no predictions found -- skipped")
            continue

        # adopt the legend actually used for this target domain
        if domain != "facies":
            lm = os.path.join(EXPORT, domain, "legend_manifest.json")
            if os.path.exists(lm):
                ev.load_legend_manifest(lm)
            else:
                print(f"  ! no legend manifest for {domain}; using defaults")

        for seed in seeds:
            pred = os.path.join(RUNS, f"{task}_seed{seed}{ARM_SUFFIX}", "predictions")
            ref = os.path.join(SPLITS, task, "test")
            res = ev.evaluate_task(pred, ref, domain,
                                   ref_is_pair=True, target_on_left=True)
            res["seed"] = seed
            res["manifest"] = run_manifest(task, seed).get("result", {})
            all_res[task][seed] = res

            key = "macro_F1" if domain == "facies" else "RMSE"
            m = res["metrics"].get(key, {})
            if "per_layer" in m:
                per_layer_store[task][seed] = {int(k): v
                                               for k, v in m["per_layer"].items()}
            print(f"  {task:<24} seed {seed}: "
                  f"{key} = {m.get('point_estimate', float('nan')):.4f} "
                  f"[{m.get('ci_low', float('nan')):.4f}, "
                  f"{m.get('ci_high', float('nan')):.4f}]  "
                  f"n_layers={m.get('n_layers')}")
    return {"results": all_res, "per_layer": per_layer_store}



# --------------------------------------------------------------------------
# Metrics restricted to a subset of facies classes
# --------------------------------------------------------------------------

def metrics_from_cm(cm: np.ndarray, drop: List[int]) -> dict:
    """PA / macro-F1 / mIoU recomputed with `drop` (1-based classes) removed
    from both the reference and the prediction."""
    keep = [i for i in range(cm.shape[0]) if (i + 1) not in drop]
    c = cm[np.ix_(keep, keep)]
    total = c.sum()
    if total == 0:
        return {"pixel_accuracy": float("nan"), "macro_F1": float("nan"),
                "mIoU": float("nan")}
    tp = np.diag(c)
    support, predicted = c.sum(axis=1), c.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), np.nan)
        iou = np.where((support + predicted - tp) > 0,
                       tp / (support + predicted - tp), np.nan)
    v = support > 0
    return {"pixel_accuracy": float(tp.sum() / total),
            "macro_F1": float(np.nanmean(f1[v])) if v.any() else float("nan"),
            "mIoU": float(np.nanmean(iou[v])) if v.any() else float("nan")}


def find_trivial_classes(all_res, threshold: float = 0.99) -> List[int]:
    """
    Classes recovered at F1 above `threshold` by EVERY inverse task.

    A class that all models reproduce almost perfectly -- including the worst-
    performing task -- is not evidence of learning; in this dataset it is a
    region whose extent barely changes between test layers. It still counts
    toward pixel accuracy and macro-F1, so results are reported both with and
    without it. The rule is stated rather than hand-picked.
    """
    per_task = defaultdict(dict)
    for task, domain in TASKS:
        if domain != "facies" or task not in all_res:
            continue
        for seed, res in all_res[task].items():
            for c in res.get("pooled_categorical", {}).get("per_class", []):
                if c["f1"] is not None and c["support_px"] > 0:
                    per_task[c["class"]].setdefault(task, []).append(c["f1"])
    facies_tasks = [t for t, d in TASKS if d == "facies" and t in all_res]
    out = []
    for cls, byt in per_task.items():
        if len(byt) == len(facies_tasks) and all(
                np.mean(v) > threshold for v in byt.values()):
            out.append(cls)
    return sorted(out)


def table_restricted(all_res, ev, drop: List[int]) -> List[list]:
    names = [ev.FACIES_LEGEND[c - 1][0] for c in drop]
    rows = [["Task", "Metric",
             "all classes", f"excluding {', '.join(names) if names else '(none)'}",
             "difference", "95% CI (excluded set)"]]
    if not drop:
        return rows
    for task, domain in TASKS:
        if domain != "facies" or task not in all_res:
            continue
        full, restr, per_layer_restr = defaultdict(list), defaultdict(list), defaultdict(dict)
        for seed, res in all_res[task].items():
            cms = res.get("per_layer_confusion", {})
            for layer, cm in cms.items():
                cm = np.array(cm, float)
                r = metrics_from_cm(cm, drop)
                f = metrics_from_cm(cm, [])
                for k in ("pixel_accuracy", "macro_F1", "mIoU"):
                    per_layer_restr[k].setdefault(int(layer), []).append(r[k])
                    full[k].append(f[k])
                    restr[k].append(r[k])
        for k in ("pixel_accuracy", "macro_F1", "mIoU"):
            fm, rm = np.nanmean(full[k]), np.nanmean(restr[k])
            collapsed = {L: float(np.nanmean(v)) for L, v in per_layer_restr[k].items()}
            ci = ev.clustered_bootstrap_ci(collapsed)
            ci_s = (f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
                    if ci.get("ci_low") is not None else "-")
            rows.append([task, k, f"{fm:.4f}", f"{rm:.4f}", f"{rm - fm:+.4f}", ci_s])
    return rows


def table_main(all_res) -> List[list]:
    rows = [["Task", "Target", "n test layers", "Metric",
             "seed 1", "seed 2", "seed 3", "mean ± SD",
             "95% CI (pooled layers)"]]
    for task, domain in TASKS:
        if task not in all_res:
            continue
        seeds = sorted(all_res[task])
        keys = (["pixel_accuracy", "macro_F1", "mIoU"] if domain == "facies"
                else ["MAE", "RMSE", "R2"])
        for key in keys:
            vals, cis = [], []
            for s in (1, 2, 3):
                m = all_res[task].get(s, {}).get("metrics", {}).get(key)
                vals.append(m["point_estimate"] if m else None)
                if m and m.get("ci_low") is not None:
                    cis.append((m["ci_low"], m["ci_high"]))
            n_layers = next(iter(all_res[task].values()))["n_layers"]
            ci = (f"[{np.mean([c[0] for c in cis]):.4f}, "
                  f"{np.mean([c[1] for c in cis]):.4f}]") if cis else "-"
            rows.append([task, domain, n_layers, key,
                         *[f"{v:.4f}" if v is not None else "-" for v in vals],
                         mean_sd(vals), ci])
    return rows


def table_per_class(all_res, ev) -> List[list]:
    rows = [["Task", "Class", "Facies name", "Frequency",
             "Precision", "Recall", "F1", "IoU", "Support (px)"]]
    for task, domain in TASKS:
        if domain != "facies" or task not in all_res:
            continue
        acc = defaultdict(lambda: defaultdict(list))
        for seed, res in all_res[task].items():
            for c in res.get("pooled_categorical", {}).get("per_class", []):
                for k in ("precision", "recall", "f1", "iou", "frequency"):
                    if c.get(k) is not None:
                        acc[c["class"]][k].append(c[k])
                acc[c["class"]]["support"].append(c["support_px"])
        for cls in sorted(acc):
            name = ev.FACIES_LEGEND[cls - 1][0] if cls <= len(ev.FACIES_LEGEND) else "?"
            g = acc[cls]
            rows.append([task, cls, name,
                         f"{np.mean(g['frequency']):.4f}" if g["frequency"] else "-",
                         mean_sd(g["precision"]), mean_sd(g["recall"]),
                         mean_sd(g["f1"]), mean_sd(g["iou"]),
                         int(np.mean(g["support"])) if g["support"] else 0])
    return rows


def table_confusion(all_res, ev) -> List[list]:
    names = [n for n, _ in ev.FACIES_LEGEND]
    rows = [["Task", "Reference \\ Predicted"] + names + ["Recall"]]
    for task, domain in TASKS:
        if domain != "facies" or task not in all_res:
            continue
        mats = [np.array(r["pooled_categorical"]["confusion_matrix"], float)
                for r in all_res[task].values() if "pooled_categorical" in r]
        if not mats:
            continue
        cm = np.mean(mats, axis=0)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        for i, nm in enumerate(names):
            rows.append([task if i == 0 else "", nm]
                        + [f"{x:.3f}" for x in norm[i]]
                        + [f"{norm[i, i]:.3f}"])
    return rows


def table_physical(all_res, ev) -> List[list]:
    rows = [["Task", "Unit", "MAE", "RMSE", "R2",
             "Encoding-floor RMSE", "Model / floor ratio"]]
    for task, domain in TASKS:
        if domain == "facies" or task not in all_res:
            continue
        gather = lambda k: [all_res[task][s]["metrics"][k]["point_estimate"]
                            for s in sorted(all_res[task])
                            if k in all_res[task][s]["metrics"]]
        floor = next(iter(all_res[task].values())).get("encoding_floor", {})
        f_rmse = floor.get("RMSE", float("nan"))
        rmse = np.mean(gather("RMSE")) if gather("RMSE") else float("nan")
        rows.append([task, floor.get("unit", "-"),
                     mean_sd(gather("MAE")), mean_sd(gather("RMSE")),
                     mean_sd(gather("R2")),
                     f"{f_rmse:.6g}",
                     f"{rmse / f_rmse:.1f}x" if f_rmse and np.isfinite(f_rmse) else "-"])
    return rows


def table_comparisons(per_layer, ev) -> List[list]:
    """Paired permutation tests between tasks that share a target domain."""
    rows = [["Task A", "Task B", "Metric", "mean difference",
             "p (paired permutation)", "n paired layers"]]
    groups = defaultdict(list)
    for task, domain in TASKS:
        groups[domain].append(task)
    for domain, tasks in groups.items():
        metric = "macro_F1" if domain == "facies" else "RMSE"
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                a, b = tasks[i], tasks[j]
                if a not in per_layer or b not in per_layer:
                    continue
                # average each layer over seeds first
                def collapse(t):
                    acc = defaultdict(list)
                    for seed, d in per_layer[t].items():
                        for k, v in d.items():
                            acc[k].append(v)
                    return {k: float(np.mean(v)) for k, v in acc.items()}
                pa, pb = collapse(a), collapse(b)
                res = ev.paired_permutation_test(pa, pb)
                if "p_value" not in res:
                    continue
                rows.append([a, b, metric, f"{res['mean_difference']:+.4f}",
                             f"{res['p_value']:.4f}", res["n_layers"]])
    return rows


def table_reproducibility(all_res) -> List[list]:
    rows = [["Item", "Value"]]
    any_task = next(iter(all_res)) if all_res else None
    man = {}
    if any_task:
        seed = sorted(all_res[any_task])[0]
        man = run_manifest(any_task, seed)
    gpu = man.get("gpu", {})
    rows += [
        ["Seeds", "1, 2, 3 (independent runs per task)"],
        ["Checkpoint selection", "best validation score; validation never used as test"],
        ["Split", "blocked by depth, buffer gap, applied BEFORE augmentation"],
        ["Independent test units", "9 original layers (never augmented)"],
        ["Uncertainty", "clustered bootstrap over layers, 10,000 resamples"],
        ["Task comparison", "paired permutation test at layer level, 10,000 permutations"],
        ["TensorFlow", man.get("tensorflow", "-")],
        ["NumPy", man.get("numpy", "-")],
        ["Python", (man.get("python", "-") or "-").split()[0]],
        ["Platform", man.get("platform", "-")],
        ["GPU", ", ".join(gpu.get("nvidia_smi", [])) or "-"],
        ["Determinism flags", json.dumps(man.get("determinism", {}))],
        ["Steps per run", man.get("hyperparameters", {}).get("steps", "-")],
        ["Report generated", datetime.now(timezone.utc).isoformat()],
        ["Report host", platform.platform()],
    ]
    return rows


def table_before_after(all_res) -> List[list]:
    rows = [["Task", "Published PA", "Published metric measured",
             "New metric", "New value (mean ± SD)", "Comparable?"]]
    for task, domain in TASKS:
        if task not in all_res:
            continue
        pub = PUBLISHED.get(task, {})
        if domain == "facies":
            key, label = "macro_F1", "macro F1 over 7 true facies classes"
        else:
            key, label = "RMSE", "RMSE in physical units"
        vals = [all_res[task][s]["metrics"][key]["point_estimate"]
                for s in sorted(all_res[task]) if key in all_res[task][s]["metrics"]]
        rows.append([task, f"{pub.get('PA', float('nan')):.4f}",
                     "argmax over RGB channels, 3 pseudo-classes",
                     label, mean_sd(vals), "NO -- different quantity"])
    return rows


def write_csv(path: str, rows: List[list]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def write_xlsx(path: str, sheets: Dict[str, List[list]]) -> bool:
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        return False
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    hdr_fill = PatternFill("solid", fgColor="1F3864")
    hdr_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    body = Font(name="Arial", size=10)
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name[:31])
        for r in rows:
            ws.append(r)
        for c in range(1, len(rows[0]) + 1):
            cell = ws.cell(row=1, column=c)
            cell.fill, cell.font = hdr_fill, hdr_font
            cell.alignment = Alignment(wrap_text=True, vertical="center",
                                       horizontal="center")
            cell.border = box
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row,
                                max_col=len(rows[0])):
            for cell in row:
                cell.font = body
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                cell.border = box
        widths = [max(12, min(38, max(len(str(r[i])) for r in rows) + 2))
                  for i in range(len(rows[0]))]
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = "A2"
    wb.save(path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["gan", "l1"], default="gan",
                    help="which training arm to score: gan (adversarial, the "
                         "default) or l1 (the non-adversarial baseline)")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    global ARM_SUFFIX, OUT_PREFIX
    ARM_SUFFIX = "" if args.arm == "gan" else "_l1"
    OUT_PREFIX = "" if args.arm == "gan" else "L1_"
    print(f"ARM: {'adversarial (pix2pix)' if args.arm == 'gan' else 'non-GAN baseline (L1 only)'}"
          f"  -> reading runs/*_seed*{ARM_SUFFIX}, writing results/{OUT_PREFIX}*")

    os.makedirs(OUT, exist_ok=True)
    ev = _load("pix2geo_eval", "02_evaluation.py")

    print("Scoring predictions\n" + "-" * 70)
    store = evaluate_all(ev, not args.no_bootstrap)
    all_res, per_layer = store["results"], store["per_layer"]
    if not all_res:
        print("\nNo predictions found. Run 06_predict.py --all first.")
        return 1

    # raw JSON, everything
    raw = {t: {str(s): r for s, r in d.items()} for t, d in all_res.items()}
    with open(os.path.join(OUT, OUT_PREFIX + "raw_results.json"), "w") as fh:
        json.dump(raw, fh, indent=2, default=float)

    trivial = find_trivial_classes(all_res)
    if trivial:
        nm = ", ".join(ev.FACIES_LEGEND[c - 1][0] for c in trivial)
        print(f"\nClasses at F1 > 0.99 in EVERY inverse task: {nm}")
        print("  -> results reported both with and without them (sheet 1b)")

    sheets = {
        "1 Main results":      table_main(all_res),
        "1b Excl trivial":     table_restricted(all_res, ev, trivial),
        "2 Per-class facies":  table_per_class(all_res, ev),
        "3 Confusion":         table_confusion(all_res, ev),
        "4 Physical units":    table_physical(all_res, ev),
        "5 Task comparisons":  table_comparisons(per_layer, ev),
        "6 Reproducibility":   table_reproducibility(all_res),
        "7 Before vs after":   table_before_after(all_res),
    }
    for name, rows in sheets.items():
        write_csv(os.path.join(OUT, OUT_PREFIX + name.replace(" ", "_") + ".csv"), rows)

    xlsx = os.path.join(OUT, OUT_PREFIX + "Pix2Geomodel_V2_Results.xlsx")
    ok = write_xlsx(xlsx, sheets)

    print("\n" + "=" * 70)
    print("MAIN RESULTS")
    print("=" * 70)
    for row in sheets["1 Main results"][1:]:
        print(f"  {row[0]:<24} {row[3]:<16} {row[7]:<22} {row[8]}")

    if trivial:
        print("\n" + "=" * 70)
        print("EXCLUDING CLASSES RECOVERED AT F1 > 0.99 BY EVERY TASK")
        print("=" * 70)
        for row in sheets["1b Excl trivial"][1:]:
            print(f"  {row[0]:<24} {row[1]:<16} {row[2]:>8} -> {row[3]:>8}  "
                  f"({row[4]})  {row[5]}")

    print("\n" + "=" * 70)
    print("TASK COMPARISONS (paired permutation, layer level)")
    print("=" * 70)
    for row in sheets["5 Task comparisons"][1:]:
        sig = "significant" if float(row[4]) < 0.05 else "not significant"
        print(f"  {row[0]:<24} vs {row[1]:<24} diff {row[3]:>9}  p={row[4]}  {sig}")

    print("\n" + "=" * 70)
    print(f"written to {OUT}")
    print(f"  raw_results.json  + 7 CSV tables")
    print(f"  {os.path.basename(xlsx)}" if ok else
          "  (openpyxl not installed -- CSV only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
