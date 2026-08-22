# Data availability — suggested wording

Replace the current statement with the following once the repository is public and the DOI
is minted. Fill in the DOI.

> The evaluation code, split manifests, per-run result tables and the scripts that generate
> every figure in this article are available at
> https://github.com/ARhaman/Pix2Geomodel-v2 and archived at
> https://doi.org/10.5281/zenodo.XXXXXXX. The repository includes the layer-level split
> definitions used for all reported experiments, so the training, validation and test
> partitions can be reconstructed exactly. The reservoir model is proprietary and cannot be
> redistributed; the export format and colour legends are documented so that the workflow
> can be applied to an equivalent geomodel. Additional data are available from the
> corresponding author on reasonable request, subject to data-sharing restrictions.

Why this wording:

- It names what *is* available rather than gesturing at a repository.
- It states the split manifests explicitly, which is what makes the leakage-free claim
  checkable — the point Reviewer 1 comment 4 and Reviewer 5 comment 3 turn on.
- It gives an archived DOI, not only a mutable URL.
- It says plainly why the model itself is withheld, rather than leaving a reviewer to guess.
