# Main Experiment Result Audit

Generated from the existing metric files and training logs. Standard deviations are sample SD across seeds 42, 123, and 456.

## Aggregated reported metrics

| Dataset | Model | Accuracy, mean +/- SD (%) | F1, mean +/- SD (%) | Best observed test accuracy, mean +/- SD (%) | Final train accuracy, mean +/- SD (%) |
|---|---|---:|---:|---:|---:|
| Fashion-MNIST | Amplitude | 89.20 +/- 0.30 | 89.25 +/- 0.30 | 89.58 +/- 0.42 | 100.00 +/- 0.00 |
| Fashion-MNIST | Angle | 87.92 +/- 0.44 | 88.04 +/- 0.48 | 89.25 +/- 0.26 | 97.92 +/- 0.18 |
| Fashion-MNIST | Basis | 34.13 +/- 4.85 | 31.01 +/- 4.38 | 34.37 +/- 4.72 | 34.92 +/- 4.38 |
| Fashion-MNIST | ClassicCNN | 89.92 +/- 0.51 | 89.92 +/- 0.54 | 89.92 +/- 0.51 | 95.45 +/- 0.32 |
| Fashion-MNIST | IQP | 88.22 +/- 0.89 | 88.25 +/- 0.88 | 88.38 +/- 0.60 | 99.31 +/- 0.03 |
| MNIST | Amplitude | 97.33 +/- 0.16 | 97.33 +/- 0.16 | 97.50 +/- 0.25 | 100.00 +/- 0.01 |
| MNIST | Angle | 96.18 +/- 0.68 | 96.18 +/- 0.68 | 96.35 +/- 0.61 | 99.10 +/- 0.11 |
| MNIST | Basis | 26.03 +/- 4.00 | 21.97 +/- 2.66 | 26.53 +/- 3.86 | 28.97 +/- 3.79 |
| MNIST | ClassicCNN | 97.43 +/- 0.30 | 97.43 +/- 0.31 | 97.43 +/- 0.30 | 99.27 +/- 0.10 |
| MNIST | IQP | 96.58 +/- 0.80 | 96.58 +/- 0.80 | 96.80 +/- 0.61 | 99.59 +/- 0.19 |

## Convergence and generalization diagnostics

The relative convergence epoch is the first epoch reaching 95% of that run's own best observed test accuracy. This permits comparison with the low-accuracy Basis runs without imposing an unreachable absolute threshold.

| Dataset | Model | Mean test-curve accuracy (%) | Epoch to 95% of own best | Final train-test gap (percentage points) |
|---|---|---:|---:|---:|
| Fashion-MNIST | Amplitude | 88.58 +/- 0.41 | 2.00 +/- 0.00 | 10.80 +/- 0.30 |
| Fashion-MNIST | Angle | 85.51 +/- 1.19 | 6.00 +/- 1.00 | 10.00 +/- 0.61 |
| Fashion-MNIST | Basis | 32.73 +/- 4.39 | 6.67 +/- 4.04 | 0.92 +/- 0.32 |
| Fashion-MNIST | ClassicCNN | 87.96 +/- 0.46 | 3.00 +/- 0.00 | 6.47 +/- 0.62 |
| Fashion-MNIST | IQP | 86.13 +/- 0.34 | 3.33 +/- 0.58 | 12.15 +/- 0.43 |
| MNIST | Amplitude | 96.98 +/- 0.18 | 1.00 +/- 0.00 | 2.66 +/- 0.16 |
| MNIST | Angle | 91.57 +/- 0.49 | 3.67 +/- 0.58 | 2.92 +/- 0.60 |
| MNIST | Basis | 24.74 +/- 3.78 | 3.67 +/- 2.08 | 2.94 +/- 1.09 |
| MNIST | ClassicCNN | 96.01 +/- 0.16 | 1.67 +/- 0.58 | 2.09 +/- 0.15 |
| MNIST | IQP | 93.69 +/- 0.75 | 2.33 +/- 0.58 | 3.01 +/- 0.61 |

## Metric consistency

The reported metric files are not based on a uniform checkpoint convention. Some contain final-epoch inference, while others contain inference after reloading a best checkpoint. The table below lists every run where the reported accuracy differs from the final epoch, the best observed test accuracy, or both.

| Dataset | Model | Seed | Reported acc. | Final epoch acc. | Best observed acc. | Best epoch |
|---|---|---:|---:|---:|---:|---:|
| Fashion-MNIST | Amplitude | 42 | 0.8890 | 0.8890 | 0.8910 | 11 |
| Fashion-MNIST | Amplitude | 123 | 0.8950 | 0.8950 | 0.8985 | 23 |
| Fashion-MNIST | Amplitude | 456 | 0.8920 | 0.8920 | 0.8980 | 10 |
| Fashion-MNIST | Angle | 42 | 0.8755 | 0.8755 | 0.8935 | 19 |
| Fashion-MNIST | Angle | 123 | 0.8780 | 0.8780 | 0.8895 | 18 |
| Fashion-MNIST | Angle | 456 | 0.8840 | 0.8840 | 0.8945 | 16 |
| Fashion-MNIST | Basis | 42 | 0.3220 | 0.3220 | 0.3290 | 15 |
| Fashion-MNIST | Basis | 123 | 0.3965 | 0.3925 | 0.3965 | 20 |
| Fashion-MNIST | ClassicCNN | 42 | 0.8960 | 0.8900 | 0.8960 | 21 |
| Fashion-MNIST | ClassicCNN | 123 | 0.8965 | 0.8850 | 0.8965 | 20 |
| Fashion-MNIST | ClassicCNN | 456 | 0.9050 | 0.8945 | 0.9050 | 15 |
| Fashion-MNIST | IQP | 42 | 0.8720 | 0.8720 | 0.8770 | 17 |
| Fashion-MNIST | IQP | 123 | 0.8860 | 0.8675 | 0.8860 | 10 |
| Fashion-MNIST | IQP | 456 | 0.8885 | 0.8755 | 0.8885 | 7 |
| MNIST | Amplitude | 42 | 0.9745 | 0.9745 | 0.9775 | 8 |
| MNIST | Amplitude | 123 | 0.9740 | 0.9740 | 0.9750 | 8 |
| MNIST | Amplitude | 456 | 0.9715 | 0.9715 | 0.9725 | 8 |
| MNIST | Angle | 42 | 0.9555 | 0.9555 | 0.9590 | 9 |
| MNIST | Angle | 123 | 0.9690 | 0.9690 | 0.9705 | 9 |
| MNIST | Basis | 42 | 0.2905 | 0.2905 | 0.2945 | 8 |
| MNIST | Basis | 123 | 0.2755 | 0.2755 | 0.2800 | 9 |
| MNIST | Basis | 456 | 0.2150 | 0.2150 | 0.2215 | 7 |
| MNIST | ClassicCNN | 123 | 0.9740 | 0.9715 | 0.9740 | 7 |
| MNIST | ClassicCNN | 456 | 0.9775 | 0.9725 | 0.9775 | 7 |
| MNIST | IQP | 42 | 0.9605 | 0.9605 | 0.9635 | 6 |
| MNIST | IQP | 456 | 0.9620 | 0.9620 | 0.9655 | 8 |

## Interpretation rule

- `reported_accuracy` is parsed from each existing final metric file and is used for the current descriptive summary.
- `best_observed_test_accuracy` is the maximum test accuracy found in the epoch log. It is diagnostic only because the test set was used during checkpoint selection.
- Before submission, all selected checkpoints should be evaluated under one consistent protocol. A validation split should determine checkpoint selection; the test set should be evaluated once after selection.
- Do not copy the best-observed column into the manuscript as the final test result without fixing the checkpoint-selection protocol.

## Generated data files

- `per_run_results.csv`: all 30 runs and source artifact paths.
- `model_summary.csv`: 10 dataset/model aggregates.
- `metric_consistency_audit.csv`: checkpoint/reporting mismatches.
