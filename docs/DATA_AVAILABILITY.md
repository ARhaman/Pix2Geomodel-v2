# Data and code availability

This is the statement that appears in the article, and a note on what backs each clause.

## Statement

> The evaluation protocol is the contribution of this work, so it is released in full. The
> repository at https://github.com/ARhaman/Pix2Geomodel-v2 contains the leakage-free split
> logic (`src/01_layer_level_split.py`) together with the layer-level split manifests for
> all six translation tasks, so every training, validation and test partition can be
> reconstructed exactly and re-checked with `src/verify_splits_v2.py`; the augmentation and
> pairing pipeline (`notebooks/02`, `notebooks/03`); the encoding and decoding scripts
> (`src/04_export_common_legend.py`, `src/02_evaluation.py`) together with the four legend
> manifests the reported experiments used; the training and prediction code
> (`src/03_train.py`, `src/06_predict.py`) and the per-task training configurations in
> `configs/`; the analysis, control, ablation and baseline scripts, including the
> self-validating scripts written for this revision; every table reported in this article
> exported as CSV; and the published figures. Because the reservoir model is proprietary and
> cannot be redistributed, `data/synthetic/` provides a generator and a ready-made synthetic
> export with the same structure — the same seven-class facies rule, a comparable inactive
> boundary, a depth trend and a rare Fair Sand class — so that the entire pipeline can be
> executed and audited without it. The synthetic data reproduce the structure of the problem
> and not its geology, and results obtained from them are not comparable with those reported
> here. Additional data and the trained model weights are available from the corresponding
> author on reasonable request, subject to data-sharing restrictions.

## What backs each clause

| Clause | In the repository |
|---|---|
| split logic | `src/01_layer_level_split.py`, `src/verify_splits_v2.py` |
| split manifests, all six tasks | `splits/*_manifest.json`, each carrying a split SHA-256 |
| augmentation pipeline | `notebooks/02_high_resolution_augmentation.ipynb`, `notebooks/03_pair_images_and_split.ipynb` |
| decoding scripts | `src/02_evaluation.py`, `src/04_export_common_legend.py` |
| legends actually used | `data/legends/*_legend_manifest.json` |
| training manifests | `configs/*.json`, one per task |
| training and prediction | `src/03_train.py`, `src/06_predict.py`, `src/05_run_all.py` |
| every reported table | `results/*.csv` |
| figures | `figures/` |
| synthetic data | `data/synthetic/make_synthetic.py` and its generated export |

## Notes

- The legend manifests are load-bearing. `src/02_evaluation.py` carries narrower fallback
  defaults, and a run that does not load the manifests encodes on a different scale. This is
  stated in `data/README.md` and asserted by the synthetic generator.
- The revision-round scripts (`src/24`–`src/30`) are self-validating: each reproduces a
  published quantity before emitting a new one, and exits non-zero if the reproduction
  fails. That property is what makes the released code checkable rather than merely present.
- A Zenodo DOI should be minted from a tagged release before the article is typeset, and the
  statement above extended with it, so the archived snapshot is citable independently of the
  mutable repository.
