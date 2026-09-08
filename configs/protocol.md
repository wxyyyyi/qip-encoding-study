# Fixed protocol

## Model

- Four independent quantum branches.
- Four qubits and two trainable layers per branch.
- Four Pauli-Z expectation values per branch, concatenated before the ten-class linear head.
- Amplitude: 16 projected features per branch and a 64-dimensional projection.
- Angle, IQP-All, IQP-NN and Basis-STE: 4 projected features per branch and a 16-dimensional projection.
- IQP-All: PennyLane `IQPEmbedding`, `n_repeats=1`, `pattern=None`.
- IQP-NN: only the pairs `(0,1)`, `(1,2)` and `(2,3)`.

## Training

- Optimizer: Adam.
- Learning rate: `0.001`.
- Weight decay: `1e-4`.
- Batch size: `64`.
- Fixed epochs: MNIST `10`; Fashion-MNIST `25`.
- Seeds: `42`, `123`, `456`.
- Checkpoint: highest validation accuracy after an epoch.
- Ideal training device: PennyLane `default.qubit` with analytic backpropagation.

## Robustness analyses

- Frozen Fashion-MNIST checkpoints only.
- Finite shots: `100` and `1,000`, with three measurement seeds per checkpoint.
- Depolarizing probabilities: `0.001`, `0.005`, `0.01` on the decomposed circuit.
- Noise subset: 500 validation examples.

## Interpretation boundaries

These settings support an architecture-specific comparison. They do not establish quantum advantage, hardware robustness, a universal encoding ranking, or a causal effect attributable to the straight-through estimator alone.
