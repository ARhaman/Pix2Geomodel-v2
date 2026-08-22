# Data

The reservoir model is proprietary and is not redistributed here.

## Export format

Each property is exported as a palette PNG under a fixed, variable-specific legend, so the
mapping from colour to physical value is invertible and the same inversion is applied at
evaluation time.

| Domain | Type | Encoding | Range |
|---|---|---|---|
| Facies | categorical | seven-class legend | Tight, Very Poor Sand, Poor Sand, Fair Sand, Good Sand, Good Clean Sand, Very Good Sand |
| Porosity | continuous | legend-controlled RGB gradient | 0.00–0.225 |
| Permeability | continuous | log-scaled RGB gradient | 0.01–1000 mD |
| VCL | continuous | legend-controlled RGB gradient | 0.00–0.50 |

Grid 365 × 248 × 55, of which 54 layers are retained. Each layer is a constant-depth map
slice, so both image axes are lateral; the vertical direction is the layer index and is never
an image axis. Cells outside the reservoir boundary occupy 25.78 % of every image and are
excluded from every metric.

## Running against your own model

Export four aligned property rasters per layer using fixed legends, place them under
`export_v2/<property>/`, and run `src/01_layer_level_split.py`. The split manifests in
`splits/` show the expected output.
