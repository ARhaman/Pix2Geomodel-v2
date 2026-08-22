# Changes from GeoPix (v1)

The architecture is unchanged: U-Net generator with eight downsampling and seven upsampling
blocks and skip connections, PatchGAN discriminator, adversarial plus L1 loss with λ = 100,
Adam at 2 × 10⁻⁴, β₁ = 0.5, batch size 1, 15,000 steps.

What changed is the evaluation.

## Splitting
v1 pooled 30 augmented realisations per layer with the 54 originals and split the resulting
1,674 images at random. Transformed copies of a test layer therefore appeared in training.
v2 splits at layer level into contiguous depth blocks before augmentation, discards buffer
layers 37 and 45, and augments the training block only.

## Metrics
v1 scored over three colour channels and counted cells outside the reservoir boundary as a
class. v2 decodes facies to the seven legend classes and continuous targets analytically to
physical units, and excludes the 25.78 % of each image outside the boundary.

## Baselines and controls
v2 adds a one-dimensional Bayes classifier, a class-mean lookup and a random forest; a
positional control that re-runs the trained models on deranged and constant inputs; and a
bound on the accuracy cost of the RGB encoding.

## Uncertainty
v2 trains three seeds per task and resamples uncertainty at layer level.

## Spatial validation
v1 compared one experimental variogram visually. v2 computes directional semivariograms in
four azimuths with bootstrap envelopes, on the grid rather than on rendered images, and adds
connected-component and lateral-transition statistics.

## Consequence
The corrected accuracies are lower than those reported in v1, and the ordering of methods
changes: a one-dimensional histogram classifier using no spatial information outperforms the
conditional GAN on all three inverse tasks.
