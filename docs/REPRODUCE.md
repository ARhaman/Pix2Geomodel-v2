# Reproducing the reported numbers

Without the reservoir model you can still check every published value, because the split
manifests and result tables are committed.

## From the committed results

```bash
python checks/verify_pvalues.py results/per_layer_metrics.csv --all
```

Recomputes the exact paired sign-permutation p-values and prints the attainable grid.

## From decoded predictions

If you have the decoded reference and prediction rasters:

```bash
python checks/reconcile_permeability.py decoded/facies_to_permeability/
python checks/add_missing_stats.py continuous decoded/facies_to_permeability/ --log10
python checks/add_missing_stats.py facies     decoded/vcl_to_facies/
```

## From the raw geomodel

```bash
python src/01_layer_level_split.py --task vcl_to_facies --mode blocked --gap 1 --seed 2025
python src/02_train.py            --task vcl_to_facies --seed 1 --steps 15000
python src/03_predict.py          --task vcl_to_facies --split test
python src/04_metrics.py          --task vcl_to_facies
```

`--mode blocked --gap 1` is what produces the split in `splits/`. Compare the manifest your
run writes against the committed one; they should match field for field apart from the
timestamp and the absolute paths.

## Figures

```bash
python src/make_fig2.py     # leakage-free split schematic
python src/make_fig6.py     # directional variograms, from results/9_Directional_variograms.csv
python src/make_figures.py  # benchmark, positional control, dispersion, information ceiling
```
