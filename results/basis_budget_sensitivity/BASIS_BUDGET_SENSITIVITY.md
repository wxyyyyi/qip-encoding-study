# Basis-STE Training-budget Sensitivity

The formal four-encoding table retains the common fixed budgets (MNIST 10 epochs; Fashion-MNIST 25 epochs). This sensitivity experiment changes only the maximum Basis-STE training budget, using 30 epochs for MNIST and 40 for Fashion-MNIST. Checkpoints are still selected exclusively by validation accuracy before frozen final-holdout evaluation.

## Aggregate result

| Dataset | Fixed/extended epochs | Fixed accuracy (%) | Extended accuracy (%) | Mean paired change (pp) | Extended best epochs (42/123/456) |
|---|---:|---:|---:|---:|---:|
| MNIST | 10/30 | 50.15 +/- 9.25 | 76.25 +/- 13.20 | +26.10 | 16/19/30 |
| FASHION | 25/40 | 59.86 +/- 7.65 | 59.86 +/- 7.65 | +0.00 | 25/11/13 |

## Per-seed result

| Dataset | Seed | Budget | Epochs | Best epoch | Final accuracy (%) | Maximum post-best validation drop (pp) |
|---|---:|---|---:|---:|---:|---:|
| FASHION | 42 | fixed | 25 | 25 | 61.58 | 0.00 |
| FASHION | 42 | extended | 40 | 25 | 61.58 | 47.10 |
| FASHION | 123 | fixed | 25 | 11 | 51.50 | 33.40 |
| FASHION | 123 | extended | 40 | 11 | 51.50 | 33.40 |
| FASHION | 456 | fixed | 25 | 13 | 66.51 | 52.85 |
| FASHION | 456 | extended | 40 | 13 | 66.51 | 52.85 |
| MNIST | 42 | fixed | 10 | 10 | 58.51 | 0.00 |
| MNIST | 42 | extended | 30 | 16 | 79.91 | 49.80 |
| MNIST | 123 | fixed | 10 | 8 | 40.21 | 4.25 |
| MNIST | 123 | extended | 30 | 19 | 87.22 | 52.55 |
| MNIST | 456 | fixed | 10 | 10 | 51.71 | 0.00 |
| MNIST | 456 | extended | 30 | 30 | 61.60 | 0.00 |

## Interpretation

- MNIST was materially undertrained at the original 10-epoch budget. Extending to 30 epochs raises mean final accuracy by about 26 percentage points, although the seed SD remains large.
- Fashion-MNIST gains nothing from extending 25 to 40 epochs: the selected best epochs remain 25, 11, and 13, and the same checkpoints are recovered.
- Validation trajectories are highly non-monotonic. Several runs lose more than 30-50 percentage points after their best epoch and may partially recover later. This supports an optimization-instability conclusion rather than a simple insufficient-epoch explanation.
- MNIST seed 456 reaches a marginally higher validation value at epoch 30 after a major collapse, but its corresponding holdout accuracy is lower than the epoch-15 checkpoint selected under the 20-epoch window. This illustrates validation-selection uncertainty in the unstable STE trajectory and is not evidence of monotonic convergence.
- The extended result remains far below continuous encodings and more seed-sensitive, but the original 10-epoch MNIST accuracy must not be presented as the asymptotic Basis-STE capability.
