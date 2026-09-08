"""Verify feature gradients through AmplitudeEmbedding on the study backend."""

from __future__ import annotations

import pennylane as qml
import torch


def main() -> None:
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(features: torch.Tensor) -> torch.Tensor:
        qml.AmplitudeEmbedding(features, wires=range(4), normalize=True)
        qml.RY(0.37, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    features = torch.linspace(0.1, 1.6, 16, dtype=torch.float64, requires_grad=True)
    output = circuit(features)
    output.backward()
    gradient = features.grad

    print(f"output={output.item():.12f}")
    print(f"gradient_norm={gradient.norm().item():.12e}")
    print(f"nonzero_entries={(gradient.abs() > 1e-12).sum().item()}/{gradient.numel()}")
    print(f"all_finite={bool(torch.isfinite(gradient).all())}")


if __name__ == "__main__":
    main()
