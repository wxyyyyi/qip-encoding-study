# Matched IQP Topology Ablation

IQP-All and IQP-NN use the same initialization sequence, batched QNode implementation, optimizer, data order generator, input scaling, ansatz, validation selection, and final-holdout protocol. Only the IQP interaction pattern differs.

## Final holdout accuracy

| Dataset | IQP-All (%) | IQP-NN (%) | Paired NN-All mean (pp) | Paired-difference SD (pp) |
|---|---:|---:|---:|---:|
| mnist | 97.55 +/- 0.27 | 97.57 +/- 0.04 | +0.017 | 0.245 |
| fashion | 86.96 +/- 0.25 | 87.25 +/- 0.39 | +0.283 | 0.546 |

With only three paired seeds, these small differences are descriptive and do not establish that IQP-NN is statistically superior.

## Circuit resources per branch

| Topology | 1Q gates | CNOT | Total gates | Depth |
|---|---:|---:|---:|---:|
| IQP-All | 30 | 20 | 50 | 28 |
| IQP-NN | 27 | 14 | 41 | 22 |

IQP-All adds 3 one-qubit phase gates, 6 CNOTs, 9 total gates, and 6 depth levels per branch without a consistent predictive gain.

## Finite shots

| Topology | Shots | Delta from analytic (pp) | Within-checkpoint SD (%) |
|---|---:|---:|---:|
| iqp-all | 100 | -0.433 | 0.318 |
| iqp-all | 1000 | -0.056 | 0.191 |
| iqp-nn | 100 | -0.406 | 0.285 |
| iqp-nn | 1000 | -0.078 | 0.142 |

## Local depolarizing noise

| Topology | p | Delta from analytic subset (pp) |
|---|---:|---:|
| iqp-all | 0.001 | +0.000 |
| iqp-all | 0.005 | -0.267 |
| iqp-all | 0.010 | +0.067 |
| iqp-nn | 0.001 | -0.067 |
| iqp-nn | 0.005 | -0.267 |
| iqp-nn | 0.010 | -0.400 |

The 500-sample noise results are non-monotonic at this resolution and do not support a strong topology-robustness ordering.
