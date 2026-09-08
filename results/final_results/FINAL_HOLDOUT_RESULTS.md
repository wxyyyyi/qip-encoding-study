# Final Independent Holdout Results

Protocol: `official_train[0:10000]` is the training set, `official_test[0:2000]` is validation, and the previously untouched `official_test[2000:10000]` is the final 8,000-sample test set. The main tables below report only final-test metrics.

## MNIST

| Model | Role | Final test accuracy, mean +/- sample SD (%) | Final test F1, mean +/- sample SD (%) | Validation minus test (pp) |
|---|---|---:|---:|---:|
| amplitude | main_quantum_encoding | 98.26 +/- 0.10 | 98.26 +/- 0.10 | -0.76 |
| angle | main_quantum_encoding | 97.62 +/- 0.24 | 97.62 +/- 0.24 | -1.28 |
| iqp-all | main_quantum_encoding | 97.55 +/- 0.27 | 97.55 +/- 0.28 | -1.00 |
| basis-ste | main_quantum_encoding | 50.15 +/- 9.25 | 46.38 +/- 11.92 | -1.33 |
| iqp-nn | topology_ablation | 97.57 +/- 0.04 | 97.57 +/- 0.04 | -0.88 |
| basis-hard | implementation_ablation | 29.71 +/- 2.52 | 25.41 +/- 3.02 | -3.18 |
| classiccnn | classical_control | 98.03 +/- 0.12 | 98.03 +/- 0.12 | -0.60 |
| mlp-16 | classical_control | 97.95 +/- 0.08 | 97.95 +/- 0.08 | -0.90 |
| mlp-64 | classical_control | 98.12 +/- 0.17 | 98.12 +/- 0.17 | -0.42 |

## FASHION

| Model | Role | Final test accuracy, mean +/- sample SD (%) | Final test F1, mean +/- sample SD (%) | Validation minus test (pp) |
|---|---|---:|---:|---:|
| amplitude | main_quantum_encoding | 88.12 +/- 0.23 | 88.09 +/- 0.20 | +1.43 |
| angle | main_quantum_encoding | 88.17 +/- 0.25 | 88.15 +/- 0.24 | +1.08 |
| iqp-all | main_quantum_encoding | 86.96 +/- 0.25 | 86.84 +/- 0.41 | +1.45 |
| basis-ste | main_quantum_encoding | 59.86 +/- 7.65 | 58.30 +/- 8.14 | -0.05 |
| iqp-nn | topology_ablation | 87.25 +/- 0.39 | 87.21 +/- 0.41 | +1.29 |
| basis-hard | implementation_ablation | 34.90 +/- 4.46 | 31.45 +/- 4.23 | -0.54 |
| classiccnn | classical_control | 88.07 +/- 0.42 | 87.98 +/- 0.58 | +1.84 |
| mlp-16 | classical_control | 87.91 +/- 0.39 | 87.89 +/- 0.34 | +1.45 |
| mlp-64 | classical_control | 88.16 +/- 0.20 | 88.07 +/- 0.15 | +1.20 |

## Interpretation

- Amplitude, Angle, IQP-All, and Basis-STE form the formal four-encoding comparison.
- IQP-NN is the preserved nearest-neighbor topology ablation; it is not used as the standard IQP result.
- Basis-Hard is an implementation ablation because its hard threshold blocks end-to-end feature learning.
- ClassicCNN and the two bottleneck MLPs are classical controls; all formal models now use validation-selected checkpoints.
- Under the common fixed epoch budget, Basis-STE substantially improves over Basis-Hard but remains less accurate and more seed-sensitive than continuous encodings. The separate budget-sensitivity report shows that MNIST was undertrained at 10 epochs while Fashion-MNIST gains nothing from extending 25 to 40 epochs.
- The classical controls remain competitive, so these results do not support a quantum predictive-advantage claim.
