# Pix2Geomodel v2 

Code, split definitions, evaluation scripts and result files for:

> Al-Fakih, A., Saraih, N., Koeshidayatullah, A., Hanafy, S., Kaka, S. I.
> *Benchmarking Pix2Geomodel Against Conventional Baselines for Bidirectional
> Facies–Property Translation in a Clastic Reservoir.*
> Under revision, **Results in Engineering**, 2026 (RINENG-D-26-10842).

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
src/              pipeline, numbered in execution order:
                    01_layer_level_split.py        the leakage-free split
                    02_evaluation.py               decoders, metrics, statistical tests
                    03_train.py  06_predict.py     training and prediction
                    04_export_common_legend.py     export and colour legends
                    05_run_all.py  07_report.py    orchestration and reporting
                    08–22                          controls, ablations, baselines, figures
                    24–30                          revision-round analyses (see below)
                    verify_splits_v2.py            re-checks a split against its manifest
notebooks/        01 extraction, 02 augmentation, 03 pairing and split (outputs stripped)
configs/          one training configuration per translation task
splits/           split manifests, one per task — the reproducible split definition
data/legends/     the four legend manifests the reported experiments actually used
data/synthetic/   generator and a ready-made synthetic export (see Data)
checks/           three standalone diagnostic scripts (see below)
results/          every table reported in the article, as CSV
figures/          published figures, 900 dpi PNG and vector PDF
docs/             REPRODUCE.md, SETUP.md, CHANGES_FROM_V1.md, DATA_AVAILABILITY.md
```

### Revision-round analyses (`src/24`–`src/30`)

Each of these is self-validating: it recomputes a published quantity first and refuses to
emit anything new unless that reproduction is exact.

| Script | Question it answers |
|---|---|
| `24_l1_forward_metrics.py` | forward metrics for the full and no-adversarial arms, in both aggregation conventions |
| `25_baseline_layerwise.py` | the conventional baselines rescored layerwise, so they are comparable with the network |
| `26_table7_assemble.py` | assembles Table 7 from the validated CSVs rather than by transcription |
| `27_eta2_layerwise.py` | the facies-label information ceiling η², both conventions |
| `28_gan_vs_l1_log10.py` | extends the GAN-vs-L1 paired test to log₁₀ permeability |
| `29_ceiling_test.py` | paired test of the network against the ceiling itself |
| `30_class_imbalance.py` | uniform-prior vs training-prior Bayes: does rebalancing recover rare classes? |

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

The reservoir model is proprietary and is not redistributed. Everything needed to run and
audit the pipeline without it is here:

- **`data/legends/`** — the four legend manifests the reported experiments actually used.
  These matter: `02_evaluation.py` carries fallback defaults (porosity 0–0.225,
  permeability 0.01–1000 mD, VCL 0–0.50) that are **narrower** than the ranges used in the
  paper (0–0.3, 0.0001–25000 mD, 0–1.0). A run that does not load these manifests encodes
  on a different scale and its numbers are not comparable with the article's.
- **`data/synthetic/`** — a generator and a ready-made synthetic export, so the whole
  pipeline can be executed end to end by anyone.
- **`splits/`**, **`results/`**, **`figures/`** — the split definitions, every reported
  table as CSV, and the published figures, so every number in the paper can be checked
  without the raw model.

### Quickstart on synthetic data

```bash
python data/synthetic/make_synthetic.py          # 18 layers, 256 x 256, ~2 MB
python src/01_layer_level_split.py --write-default-config cfg.json
#   point source_dir/target_dir at data/synthetic/export/<property>
#   and set pair_half_size to [256, 256]
python src/01_layer_level_split.py --config cfg.json
```

The generator asserts that a round trip through the published decoder is exact before it
writes anything, and it reproduces the structure the paper's conclusions turn on: the same
seven-class rule (VCL thresholds at 0.12/0.20/0.30/0.40 with porosity breaking ties), an
inactive boundary of comparable extent, a depth trend so contiguous-depth splitting is
meaningful, and a rare Fair Sand class at roughly 0.3 % of area. It does **not** reproduce
the geology, and numbers obtained from it are not comparable with the paper's.

---

## Citation

See `CITATION.cff`.

## License

Code: MIT. Figures and result tables: CC BY 4.0.
