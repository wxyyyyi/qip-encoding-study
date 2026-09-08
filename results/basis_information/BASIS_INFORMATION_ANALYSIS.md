# Basis Encoding Information and Gradient Audit

All statistics are computed from the existing trained checkpoints. No model was retrained.

## Test-set information bottleneck

| Dataset | Seed | Unique 16-bit patterns | Ambiguous samples (%) | Pattern-majority ceiling (%) | Mean bit entropy | I(Y;B), bits | Same/different Hamming |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnist | 42 | 71 | 98.10 | 36.35 | 0.323 | 0.796 | 2.04 / 2.45 |
| mnist | 123 | 310 | 89.80 | 48.45 | 0.459 | 1.448 | 2.85 / 3.40 |
| mnist | 456 | 76 | 98.15 | 27.25 | 0.254 | 0.579 | 1.56 / 1.71 |
| fashion | 42 | 91 | 96.35 | 36.65 | 0.291 | 0.911 | 1.65 / 2.03 |
| fashion | 123 | 197 | 91.45 | 51.35 | 0.436 | 1.528 | 2.40 / 3.18 |
| fashion | 456 | 108 | 97.70 | 37.60 | 0.308 | 1.037 | 1.72 / 2.09 |

The pattern-majority ceiling is the accuracy of an oracle that assigns each observed binary pattern to its most frequent class. It quantifies label ambiguity after binarization; it is descriptive and not a population-level theoretical bound.

## Gradient-flow audit

| Parameter group | Gradient tensors present | Gradient L1 norm | Binary tensor requires grad |
|---|---:|---:|---|
| conv | 0/4 | 0.000000 | False |
| fc | 0/2 | 0.000000 | False |
| post_threshold_head | 2/2 | 8.983761 | False |

The hard comparison `(x > 0).float()` produces a tensor without a gradient function. Consequently, convolutional and projection parameters receive no task-loss gradient in the current Basis implementation.

## Initialization drift

| Dataset | Seed | Maximum absolute conv/FC change from seeded initialization |
|---|---:|---:|
| mnist | 42 | 0.0000000000 |
| mnist | 123 | 0.0000000000 |
| mnist | 456 | 0.0000000000 |
| fashion | 42 | 0.0000000000 |
| fashion | 123 | 0.0000000000 |
| fashion | 456 | 0.0000000000 |

A zero drift confirms that the feature extractor remained at initialization for that run. Non-zero drift in a legacy seed-42 checkpoint can also reflect an older initialization protocol, so the direct gradient audit remains the decisive implementation test.

## Manuscript consequence

The existing Basis results cannot isolate encoding information loss from optimization-path failure. They should be labeled as hard-threshold Basis results. A fair end-to-end Basis comparison requires a differentiable estimator such as a straight-through binary function, or a feature extractor pretrained and frozen identically for all encodings.
