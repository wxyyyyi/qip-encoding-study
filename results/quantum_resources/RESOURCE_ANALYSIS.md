# Quantum Circuit Resource Analysis

Generated with PennyLane 0.44.1. Templates were decomposed to RX/RY/RZ/H/X/CNOT gates before counting. Global phases are excluded from physical gate totals.

## Per-branch and four-branch resources

| Encoding | Input features/branch | Single-qubit gates/branch | CNOT/branch | Total gates/branch | Depth/branch | Gates across 4 branches | CNOT across 4 branches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amplitude | 16 | 46 | 36 | 82 | 64 | 328 | 144 |
| Angle | 4 | 20 | 8 | 28 | 13 | 112 | 32 |
| IQP-All | 4 | 30 | 20 | 50 | 28 | 200 | 80 |
| IQP-NN | 4 | 27 | 14 | 41 | 22 | 164 | 56 |
| Basis-STE | 4 | 20 | 8 | 28 | 13 | 112 | 32 |
| Basis-Hard | 4 | 18 | 8 | 26 | 13 | 104 | 32 |

Four branches require 16 qubits for simultaneous execution. Under ideal parallel execution, critical-path depth equals the per-branch depth; on a reused four-qubit device, the branch circuit is executed four times.

Basis-Hard counts use the representative state 0101. Its BasisState decomposition uses one X gate per active bit. Basis-STE uses four fixed RX(pi * bit) gates per branch and is forward-equivalent up to global phase.

Amplitude counts report the reproducible upper-bound decomposition for a generic dense signed real 16-component state. PennyLane can remove phase operations for favorable input sign patterns, so per-sample compiled counts may be lower; the upper bound is used for conservative hardware comparison.

## Trainable parameter counts

| Model | Conv | Feature projection | Quantum | Classifier/surrogate | Total |
|---|---:|---:|---:|---:|---:|
| Amplitude | 4800 | 100416 | 64 | 170 | 105450 |
| Angle | 4800 | 25104 | 64 | 170 | 30138 |
| IQP-All | 4800 | 25104 | 64 | 170 | 30138 |
| IQP-NN | 4800 | 25104 | 64 | 170 | 30138 |
| Basis-STE | 4800 | 25104 | 64 | 170 | 30138 |
| Basis-Hard | 4800 | 25104 | 64 | 170 | 30138 |
| ClassicCNN | 4800 | 0 | 0 | 15690 | 20490 |
| BottleneckMLP-16 | 4800 | 25104 | 0 | 442 | 30346 |
| BottleneckMLP-64 | 4800 | 100416 | 0 | 1210 | 106426 |

## Interpretation

- Angle has the shallowest continuous state preparation and the fewest two-qubit gates among continuous encodings.
- IQP-NN adds six CNOT gates per branch for three nearest-neighbor ZZ feature interactions; IQP-All uses PennyLane's six-pair all-to-all pattern.
- Basis-STE remains shallow at inference and restores end-to-end gradient flow; Basis-Hard is retained only as an implementation ablation.
- Amplitude carries four times as many input features as Angle/IQP/Basis under the same four-qubit branch, but arbitrary state preparation increases gate count and depth.
- The Amplitude hybrid model has substantially more classical projection parameters because it maps the CNN output to 64 rather than 16 features. Accuracy differences must not be attributed to encoding alone.
