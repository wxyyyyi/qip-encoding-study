# Combined Fashion-MNIST Post-training Robustness

IQP-All is the formal all-to-all encoding. IQP-NN is the matched nearest-neighbor topology ablation trained with the same implementation.

## Finite shots

| Encoding | Shots | Accuracy (%) | Seed SD (%) | Within-checkpoint SD (%) | Delta from analytic (pp) |
|---|---:|---:|---:|---:|---:|
| Amplitude | 100 | 89.26 | 0.33 | 0.135 | -0.294 |
| Amplitude | 1000 | 89.54 | 0.31 | 0.102 | -0.011 |
| Angle | 100 | 88.91 | 0.37 | 0.153 | -0.339 |
| Angle | 1000 | 89.17 | 0.22 | 0.095 | -0.078 |
| IQP-All | 100 | 87.98 | 0.40 | 0.318 | -0.433 |
| IQP-All | 1000 | 88.36 | 0.31 | 0.191 | -0.056 |
| Basis-STE | 100 | 59.53 | 7.57 | 0.204 | -0.283 |
| Basis-STE | 1000 | 59.88 | 7.79 | 0.111 | +0.067 |
| IQP-NN | 100 | 88.13 | 0.46 | 0.285 | -0.406 |
| IQP-NN | 1000 | 88.46 | 0.20 | 0.142 | -0.078 |
| Basis-Hard | 100 | 34.37 | 4.72 | 0.000 | +0.000 |
| Basis-Hard | 1000 | 34.37 | 4.72 | 0.000 | +0.000 |

## Local depolarizing noise

| Encoding | p | Accuracy (%) | Seed SD (%) | Delta from analytic subset (pp) |
|---|---:|---:|---:|---:|
| Amplitude | 0.001 | 89.33 | 0.64 | +0.000 |
| Amplitude | 0.005 | 89.27 | 0.58 | -0.067 |
| Amplitude | 0.010 | 89.47 | 0.81 | +0.133 |
| Angle | 0.001 | 90.47 | 0.42 | +0.000 |
| Angle | 0.005 | 90.40 | 0.40 | -0.067 |
| Angle | 0.010 | 90.33 | 0.50 | -0.133 |
| IQP-All | 0.001 | 89.47 | 1.27 | +0.000 |
| IQP-All | 0.005 | 89.20 | 1.04 | -0.267 |
| IQP-All | 0.010 | 89.53 | 0.99 | +0.067 |
| Basis-STE | 0.001 | 61.60 | 8.66 | -0.067 |
| Basis-STE | 0.005 | 61.80 | 8.66 | +0.133 |
| Basis-STE | 0.010 | 62.53 | 9.00 | +0.867 |
| IQP-NN | 0.001 | 89.53 | 1.01 | -0.067 |
| IQP-NN | 0.005 | 89.33 | 1.30 | -0.267 |
| IQP-NN | 0.010 | 89.20 | 1.51 | -0.400 |
| Basis-Hard | 0.001 | 34.27 | 5.30 | -0.400 |
| Basis-Hard | 0.005 | 33.93 | 4.88 | -0.733 |
| Basis-Hard | 0.010 | 33.67 | 5.22 | -1.000 |
