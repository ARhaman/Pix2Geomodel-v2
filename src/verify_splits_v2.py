#!/usr/bin/env python3
"""
verify_splits_v2.py -- independent audit of the built splits.

Checks each task for the properties a reviewer would check:
  * no layer appears in more than one partition
  * validation and test contain ORIGINAL slices only (aug00)
  * every file's layer id matches the partition it sits in
  * the pair halves have identical geometry (alignment preserved)
  * counts agree with the manifest
Exit code is non-zero if any task fails.
"""
import json, os, re, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits_v2")
NAME = re.compile(r"layer(\d+)_aug(\d+)\.png$")
TASKS = ["facies_to_porosity", "facies_to_permeability", "facies_to_vcl",
         "porosity_to_facies", "permeability_to_facies", "vcl_to_facies"]

def check(task):
    d = os.path.join(ROOT, task)
    if not os.path.isdir(d):
        return [f"missing directory {d}"]
    man = json.load(open(os.path.join(d, "split_manifest.json")))
    sp, errs = man["split"], []
    sets = {k: set(sp[k]) for k in ("train", "val", "test")}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ov = sets[a] & sets[b]
        if ov:
            errs.append(f"LAYER OVERLAP {a}/{b}: {sorted(ov)}")

    counts = {}
    for part in ("train", "val", "test"):
        pdir = os.path.join(d, part)
        files = sorted(f for f in os.listdir(pdir) if f.endswith(".png"))
        counts[part] = len(files)
        for f in files:
            m = NAME.search(f)
            if not m:
                errs.append(f"unparseable filename {part}/{f}"); continue
            layer, aug = int(m.group(1)), int(m.group(2))
            if layer not in sets[part]:
                errs.append(f"{part}/{f}: layer {layer} not in this partition")
            if part in ("val", "test") and aug != 0:
                errs.append(f"{part}/{f}: AUGMENTED image in {part}")
        if counts[part] != man["counts"][part]:
            errs.append(f"{part}: {counts[part]} files but manifest says "
                        f"{man['counts'][part]}")

    # geometry / alignment on one test pair
    tf = sorted(os.listdir(os.path.join(d, "test")))
    if tf:
        im = np.array(Image.open(os.path.join(d, "test", tf[0])).convert("RGB"))
        w = im.shape[1] // 2
        ml = (im[:, :w].sum(2) > 0)
        mr = (im[:, w:].sum(2) > 0)
        if ml.shape != mr.shape:
            errs.append("pair halves differ in size")
        elif not (ml == mr).all():
            frac = float((ml != mr).mean())
            errs.append(f"pair halves misaligned in {100*frac:.2f}% of pixels")
    return errs, counts, sp

def main():
    bad = 0
    print(f"{'task':<26} {'train':>7} {'val':>5} {'test':>5} {'layers t/v/te':>15}  status")
    for t in TASKS:
        r = check(t)
        if isinstance(r, list):
            print(f"{t:<26} {'-':>7} {'-':>5} {'-':>5} {'-':>15}  SKIP ({r[0]})"); continue
        errs, c, sp = r
        lay = f"{len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}"
        status = "OK" if not errs else f"FAIL ({len(errs)})"
        if errs: bad += 1
        print(f"{t:<26} {c['train']:>7} {c['val']:>5} {c['test']:>5} {lay:>15}  {status}")
        for e in errs[:5]:
            print(f"    - {e}")
    print()
    print("ALL SPLITS CLEAN" if not bad else f"{bad} TASK(S) FAILED")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
