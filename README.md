# Pix2Geomodel v2 — leakage-free benchmarking of conditional GANs for facies–property translation

Code, split definitions, evaluation scripts and result files for:

> Al-Fakih, A., Saraih, N., Koeshidayatullah, A., Hanafy, S., Kaka, S. I.
> *Leakage-Free Benchmarking of Pix2Geomodel Against Conventional Property Modelling for
> Facies–Property Translation in a Faulted Cretaceous Fluvial to Marginal-Marine Reservoir.*
> Submitted to **Results in Engineering**, 2026.

This repository supersedes [`ARhaman/GeoPix`](https://github.com/ARhaman/GeoPix), which
describes the earlier 256 × 256 workflow of Al-Fakih et al. (2026, *Geoenergy Science and
Engineering* 258, 214342) and does not implement the evaluation protocol used here.

---

## What is different from v1

| | v1 (GeoPix) | v2 (this repository) |
|---|---|---|
| Split | 70/15/15 at random, **after** augmentation | contiguous depth blocks at **layer level, before** augmentation, with buffer layers |
| Training resolution | 256 × 256 | 2048 × 1024 |
| Metrics | image-space over three colour channels | seven facies classes; continuous targets decoded to physical units |
| Inactive cells | scored as a class | excluded from every metric |
| Baselines | none | 1-D Bayes classifier, class-mean lookup, random forest |
| Controls | none | positional control, encoding-penalty bound |
| Uncertainty | single run | three seeds; clustered bootstrap at layer level |
| Spatial validation | one visual variogram comparison | four directions with bootstrap envelopes, connectivity and transition statistics |

The headline result of v2 is that a one-dimensional Bayes classifier using **no spatial
information** outperforms the conditional GAN on all three inverse tasks.

---

## The split

The leakage-free split is the central methodological claim, so its definition is committed
here rather than described. `splits/` holds one manifest per task, written by
`01_layer_level_split.py`:

```
training    layers 1–36    augmented, 30 realisations per layer  ->  1,116 images
buffer      layer 37       discarded
validation  layers 38–44   not augmented                          ->  7 images
buffer      layer 45       discarded
test        layers 46–54   not augmented                          ->  9 images
```

Every reported number comes from the nine test layers. Nine is small, and it is the binding
constraint on the statistics: an exact two-sided sign-permutation test over 2⁹ = 512 sign
assignments cannot return a p-value below 2/512 = 0.0039.

---

## Layout

```
src/        pipeline: extraction, export, splitting, training, prediction, evaluation, figures
checks/     three diagnostic scripts (see below)
splits/     split manifests, one per translation task — the reproducible split definition
results/    per-run result tables
figures/    published figures, 900 dpi PNG and vector PDF
docs/       REPRODUCE.md, CHANGES_FROM_V1.md
```

---

## Diagnostic scripts

`checks/` holds three scripts written to close specific reviewer points. Each runs
standalone and prints a report; none needs the training pipeline.

**`verify_pvalues.py`** — recomputes the exact paired sign-permutation p-values from the
per-layer differences, and prints the attainable grid. With nine layers the only attainable
two-sided values are 0.0039, 0.0078, 0.0117, … A quoted value off that grid was not produced
by this test.

```bash
python checks/verify_pvalues.py results/per_layer_metrics.csv --all
```

**`reconcile_permeability.py`** — computes the linear-space permeability error both ways,
pooling all cells versus averaging per-layer metrics, and reports which convention
reproduces which published table. MAE is nearly identical under both; RMSE and R² are not,
which is the signature of the discrepancy.

```bash
python checks/reconcile_permeability.py decoded/facies_to_permeability/
```

**`add_missing_stats.py`** — clustered bootstrap confidence intervals at layer level,
normalised RMSE, Pearson and Spearman correlation, confusion matrices and per-class
precision/recall/F1 for all three inverse tasks, and a paired layer-level test of the
network against each baseline.

```bash
python checks/add_missing_stats.py continuous decoded/facies_to_permeability/ --log10
python checks/add_missing_stats.py facies     decoded/vcl_to_facies/
```

Input format for the last two: one `.npz` per held-out layer containing `ref` and `pred`
(and optionally `baseline`), with cells outside the reservoir set to NaN for continuous
targets or to a negative index for facies.

---

## Environment

Training requires TensorFlow 2.10 with CUDA 11.2 and cuDNN 8.1 — the last TensorFlow release
with native Windows GPU support. Later versions change Keras behaviour and will not restore
these checkpoints.

```bash
conda create -n pix2geo python=3.9
conda activate pix2geo
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

The check must print a version starting `2.10` and a non-empty GPU list. The analysis and
figure scripts in `checks/` and `src/` need only NumPy, pandas, SciPy and Matplotlib and run
on CPU.

---

## Data

The reservoir model is proprietary. `data/` documents the export format and the colour
legends, so the pipeline can be run against any equivalent geomodel export. The split
manifests, result tables and figures in this repository are complete, so every number in the
paper can be checked without the raw model.

---

## Citation

See `CITATION.cff`.

## License

Code: MIT. Figures and result tables: CC BY 4.0.
