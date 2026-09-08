# Basis-Hard to Basis-STE Correction Analysis

Basis-Hard and Basis-STE use the same 16-bit representation type at the quantum-circuit boundary. Basis-STE replaces the non-differentiable BasisState input path with forward-equivalent RX(pi * bit) preparation and a straight-through gradient estimator. Because the legacy Hard and corrected STE trainers are not a strict one-variable pair, their accuracy difference is reported as an observed pipeline recovery rather than a pure causal STE effect.

## Independent 8k final-test result

| Dataset | Basis-Hard accuracy (%) | Basis-STE accuracy (%) | Improvement (pp) | Hard SD (%) | STE SD (%) |
|---|---:|---:|---:|---:|---:|
| MNIST | 29.71 | 50.15 | +20.43 | 2.52 | 9.25 |
| FASHION | 34.90 | 59.86 | +24.96 | 4.46 | 7.65 |

## Mechanism audit on the shared 2k validation partition

| Dataset | Hard/STE feature drift | Hard/STE unique patterns | Hard/STE ambiguous samples (%) | Hard/STE mean bit entropy |
|---|---:|---:|---:|---:|
| MNIST | 0.000 / 0.534 | 152.3 / 378.7 | 95.35 / 56.00 | 0.345 / 0.721 |
| FASHION | 0.000 / 0.641 | 132.0 / 352.0 | 95.17 / 65.40 | 0.345 / 0.748 |

## Interpretation

- The strict zero feature drift of Basis-Hard confirms that its CNN and projection remained at initialization.
- Every Basis-STE run has non-zero Conv and FC gradients and substantial feature-parameter drift, so the optimization-path defect is removed.
- Under the common fixed epoch budgets, the corrected pipeline improves by roughly 20-25 percentage points. Together with the gradient and drift audits, this is evidence that the original failure was partly implementation-induced; it is not treated as a pure STE causal effect.
- The separate budget-sensitivity experiment shows that MNIST improves substantially with more epochs, while Fashion-MNIST does not. Basis-STE remains below continuous encodings and highly seed-sensitive, supporting a difficult discrete surrogate-optimization conclusion rather than a fixed asymptotic capacity claim.
