# Data

The reservoir model is proprietary and is not redistributed here.

## Export format

Each property is exported as an RGB PNG under a fixed, variable-specific legend, so the
mapping from colour to physical value is invertible and the same inversion is applied at
evaluation time. Cells outside the reservoir boundary are NODATA black and are excluded
from every metric.

### Legend ranges used in the article

These are the values in `data/legends/*_legend_manifest.json`, which
`02_evaluation.load_legend_manifest()` loads at run time.

| Domain | Type | Encoding | Range | Unit |
|---|---|---|---|---|
| Facies | categorical | seven-class legend | Tight, Very Poor Sand, Poor Sand, Fair Sand, Good Sand, Good Clean Sand, Very Good Sand | — |
| Porosity | continuous | legend-controlled RGB gradient, linear | 0.0 – 0.3 | fraction |
| Permeability | continuous | legend-controlled RGB gradient, log₁₀, reserved zero class at palette index 0 | 0.0001 – 25000 | mD |
| VCL | continuous | legend-controlled RGB gradient, linear | 0.0 – 1.0 | fraction |

> **Load the manifests.** `src/02_evaluation.py` also carries fallback defaults in
> `CONTINUOUS_DOMAINS` — porosity 0–0.225, permeability 0.01–1000 mD, VCL 0–0.50. Those are
> **narrower** than the ranges above and exist only so the module imports without a
> manifest. A run that does not load `data/legends/` encodes and decodes on a different
> scale, and its numbers are not comparable with the article's.

## Grid

The corner-point grid has **364 × 247 × 54 cells**, equivalently 365 × 248 × 55 nodes; the
`cube_shape` field in the legend manifests records the node counts. Of the 54 layers, all 54
are retained. Each layer is a constant-depth map slice, so both image axes are lateral; the
vertical direction is the layer index and is never an image axis. Cells outside the
reservoir boundary occupy 25.78 % of every image.

## Synthetic dataset

`data/synthetic/make_synthetic.py` generates a small synthetic export with the same
*structure* as the real one, so the pipeline can be run and audited without the proprietary
model:

```bash
python data/synthetic/make_synthetic.py        # 18 layers, 256 x 256, ~2 MB
```

It reproduces the four aligned property rasters per layer under the legends above, the same
seven-class facies rule (VCL thresholds at 0.12, 0.20, 0.30, 0.40 with porosity resolving
ties, which again leaves Fair Sand rare at roughly 0.3 % of area), an irregular inactive
boundary of comparable extent, and a depth trend so that contiguous-depth splitting is a
meaningful operation. Encoding goes through `src/02_evaluation.py` itself, and the script
asserts an exact round trip through the published decoder before writing.

It does **not** reproduce the geology. Numbers obtained from the synthetic dataset are not
comparable with the article's and are not intended to be; its purpose is to make the code
runnable and the split logic auditable.

## Running against your own model

Export four aligned property rasters per layer using fixed legends and name them so that
`src/01_layer_level_split.py` can find the layer index — it matches `_layer_<n>_`, for
example `fieldname_layer_07_porosity.png`. Place them under one directory per property, then:

```bash
python src/01_layer_level_split.py --write-default-config cfg.json
# edit cfg.json: source_dir, target_dir, output_dir, pair_half_size
python src/01_layer_level_split.py --config cfg.json
python src/verify_splits_v2.py            # re-checks the split against its manifest
```

The manifests in `splits/` show the expected output, including the split SHA-256 that makes
a partition identifiable.
