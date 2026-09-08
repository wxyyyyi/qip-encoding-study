import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from _result_store import collect_metrics, read_matching_metrics, run_mode_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "basis_ste"
N_QUBITS = 4
N_PARALLEL = 4
N_LAYERS = 2
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 2_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4


def make_qnode():
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(bits, weights):
        for wire in range(N_QUBITS):
            qml.RX(torch.pi * bits[..., wire], wires=wire)
        for layer in range(N_LAYERS):
            for wire in range(N_QUBITS):
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RX(weights[layer, wire, 1], wires=wire)
            for wire in range(N_QUBITS - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[N_QUBITS - 1, 0])
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    return circuit


class QuantumLayer(nn.Module):
    def __init__(self, qnode):
        super().__init__()
        self.qnode = qnode
        self.weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.1)

    def forward(self, bits):
        measurements = self.qnode(bits, self.weights)
        return torch.stack(list(measurements), dim=-1).float()


class BasisSTEHQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Linear(32 * 7 * 7, N_QUBITS * N_PARALLEL)
        qnode = make_qnode()
        self.quantum_layers = nn.ModuleList(
            [QuantumLayer(qnode) for _ in range(N_PARALLEL)]
        )
        self.output = nn.Linear(N_QUBITS * N_PARALLEL, 10)

    def binary_features(self, inputs):
        continuous = torch.tanh(self.fc(self.conv(inputs.float())))
        hard_bits = (continuous > 0).to(continuous.dtype)
        ste_bits = continuous + (hard_bits - continuous).detach()
        return continuous, hard_bits, ste_bits

    def forward(self, inputs):
        _, _, ste_bits = self.binary_features(inputs)
        branches = ste_bits.split(N_QUBITS, dim=1)
        quantum_outputs = [
            layer(branches[index]) for index, layer in enumerate(self.quantum_layers)
        ]
        return self.output(torch.cat(quantum_outputs, dim=1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(dataset_name, data_root, seed, smoke_test):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset_class = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    official_train = dataset_class(
        root=str(data_root), train=True, download=False, transform=transform
    )
    official_test = dataset_class(
        root=str(data_root), train=False, download=False, transform=transform
    )
    if smoke_test:
        train_indices = list(range(128))
        validation_indices = list(range(64))
        test_indices = list(range(2_000, 2_064))
    else:
        train_indices = list(range(TRAIN_SIZE))
        validation_indices = list(range(VALIDATION_SIZE))
        test_indices = list(range(VALIDATION_SIZE, len(official_test)))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(official_train, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(official_test, validation_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    final_test_loader = DataLoader(
        Subset(official_test, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, final_test_loader


@torch.inference_mode()
def classification_metrics(model, loader, device):
    model.eval()
    labels_all = []
    predictions_all = []
    for inputs, labels in loader:
        predictions = model(inputs.to(device)).argmax(dim=1).cpu().numpy()
        predictions_all.extend(predictions.tolist())
        labels_all.extend(labels.numpy().tolist())
    return {
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision_score(
            labels_all, predictions_all, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels_all, predictions_all, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            labels_all, predictions_all, average="weighted", zero_division=0
        ),
    }


def gradient_group_norm(parameters):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    return float(sum(gradient.detach().abs().sum() for gradient in gradients))


@torch.inference_mode()
def binary_pattern_metrics(model, loader, device):
    model.eval()
    bits_all = []
    labels_all = []
    for inputs, labels in loader:
        _, hard_bits, _ = model.binary_features(inputs.to(device))
        bits_all.append(hard_bits.cpu().numpy().astype(np.uint8))
        labels_all.append(labels.numpy())
    bits = np.concatenate(bits_all)
    labels = np.concatenate(labels_all)
    powers = (1 << np.arange(bits.shape[1], dtype=np.uint32)).reshape(1, -1)
    patterns = (bits.astype(np.uint32) * powers).sum(axis=1)
    unique_patterns, inverse, counts = np.unique(
        patterns, return_inverse=True, return_counts=True
    )
    pattern_class = np.zeros((len(unique_patterns), 10), dtype=np.int64)
    np.add.at(pattern_class, (inverse, labels), 1)
    ambiguous = (pattern_class > 0).sum(axis=1) > 1
    ambiguous_sample_fraction = counts[ambiguous].sum() / len(labels)
    majority_ceiling = pattern_class.max(axis=1).sum() / len(labels)
    activation = bits.mean(axis=0)
    entropy = []
    for probability in activation:
        if probability in (0.0, 1.0):
            entropy.append(0.0)
        else:
            entropy.append(
                -probability * math.log2(probability)
                - (1.0 - probability) * math.log2(1.0 - probability)
            )
    return {
        "unique_patterns": int(len(unique_patterns)),
        "ambiguous_sample_fraction": float(ambiguous_sample_fraction),
        "pattern_majority_ceiling": float(majority_ceiling),
        "mean_bit_activation": float(activation.mean()),
        "mean_bit_entropy": float(np.mean(entropy)),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one(dataset_name, seed, epochs, data_root, output_root, device, smoke_test):
    set_seed(seed)
    train_loader, validation_loader, final_test_loader = make_loaders(
        dataset_name, data_root, seed, smoke_test
    )
    model = BasisSTEHQNN().to(device)
    initial_feature_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith("conv.") or key.startswith("fc.")
    }
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.CrossEntropyLoss()
    output_dir = output_root / dataset_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_validation_model.pth"
    history = []
    best_validation_accuracy = -1.0
    best_epoch = None
    gradient_audit = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            if gradient_audit is None:
                gradient_audit = {
                    "conv_gradient_l1": gradient_group_norm(model.conv.parameters()),
                    "fc_gradient_l1": gradient_group_norm(model.fc.parameters()),
                    "quantum_gradient_l1": gradient_group_norm(model.quantum_layers.parameters()),
                    "output_gradient_l1": gradient_group_norm(model.output.parameters()),
                }
                if gradient_audit["conv_gradient_l1"] <= 0 or gradient_audit["fc_gradient_l1"] <= 0:
                    raise RuntimeError(f"STE gradient audit failed: {gradient_audit}")
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        validation_metrics = classification_metrics(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "loss": total_loss / len(train_loader),
            "train_accuracy": correct / total,
            "validation_accuracy": validation_metrics["accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        print(
            f"{dataset_name:7s} Basis-STE seed={seed:3d} epoch={epoch:2d}/{epochs:2d} "
            f"loss={row['loss']:.4f} train={row['train_accuracy']:.4f} "
            f"val={row['validation_accuracy']:.4f}"
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    final_test_metrics = classification_metrics(model, final_test_loader, device)
    final_state = model.state_dict()
    maximum_feature_drift = max(
        float((final_state[key].detach().cpu() - initial).abs().max())
        for key, initial in initial_feature_state.items()
    )
    validation_pattern_metrics = binary_pattern_metrics(model, validation_loader, device)
    test_pattern_metrics = binary_pattern_metrics(model, final_test_loader, device)
    result = {
        "dataset": dataset_name,
        "model": "Basis-STE",
        "seed": seed,
        "run_mode": "smoke" if smoke_test else "full",
        "epochs": epochs,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "final_test_samples": len(final_test_loader.dataset),
        "validation_indices": "official_test[0:2000]" if not smoke_test else "smoke",
        "final_test_indices": "official_test[2000:10000]" if not smoke_test else "smoke",
        "checkpoint_selection": "best_validation_accuracy",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_test_evaluations": 1,
        "test_accuracy": final_test_metrics["accuracy"],
        "test_precision": final_test_metrics["precision"],
        "test_recall": final_test_metrics["recall"],
        "test_f1": final_test_metrics["f1"],
        "maximum_feature_parameter_drift": maximum_feature_drift,
        **gradient_audit,
        **{f"validation_{key}": value for key, value in validation_pattern_metrics.items()},
        **{f"test_{key}": value for key, value in test_pattern_metrics.items()},
        "checkpoint": checkpoint_path.relative_to(output_root).as_posix(),
        "elapsed_seconds": time.time() - started,
    }
    write_csv(output_dir / "training_log.csv", history)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def sample_sd(values):
    return float(np.std(np.asarray(values), ddof=1)) if len(values) > 1 else 0.0


def write_aggregate(results, output_root):
    write_csv(output_root / "basis_ste_per_run.csv", results)
    summary = []
    for dataset_name in sorted({result["dataset"] for result in results}):
        rows = [result for result in results if result["dataset"] == dataset_name]
        summary.append(
            {
                "dataset": dataset_name,
                "model": "Basis-STE",
                "n_seeds": len(rows),
                "test_accuracy_mean": float(np.mean([row["test_accuracy"] for row in rows])),
                "test_accuracy_sd": sample_sd([row["test_accuracy"] for row in rows]),
                "test_f1_mean": float(np.mean([row["test_f1"] for row in rows])),
                "test_f1_sd": sample_sd([row["test_f1"] for row in rows]),
                "best_validation_accuracy_mean": float(np.mean([row["best_validation_accuracy"] for row in rows])),
                "feature_drift_min": min(row["maximum_feature_parameter_drift"] for row in rows),
                "test_unique_patterns_mean": float(np.mean([row["test_unique_patterns"] for row in rows])),
                "test_ambiguous_sample_fraction_mean": float(np.mean([row["test_ambiguous_sample_fraction"] for row in rows])),
            }
        )
    write_csv(output_root / "basis_ste_summary.csv", summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Train differentiable forward-exact Basis-STE HQNN models.")
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion"], default=["mnist", "fashion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs-mnist", type=int, default=10)
    parser.add_argument("--epochs-fashion", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    active_root = run_mode_root(output_root, args.smoke_test)
    active_root.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print("Protocol: train[0:10000], validation=official_test[0:2000], final_test=official_test[2000:10000]")
    results = []
    for dataset_name in args.datasets:
        if args.smoke_test:
            epochs = 1
        else:
            epochs = (
                args.epochs_mnist
                if dataset_name == "mnist"
                else args.epochs_fashion
            )
        for seed in args.seeds:
            metrics_path = active_root / dataset_name / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists() and not args.force:
                results.append(
                    read_matching_metrics(
                        metrics_path,
                        {
                            "dataset": dataset_name,
                            "model": "Basis-STE",
                            "seed": seed,
                            "run_mode": "smoke" if args.smoke_test else "full",
                        },
                    )
                )
                print(f"Skip existing {dataset_name} seed={seed}")
                continue
            results.append(
                train_one(
                    dataset_name=dataset_name,
                    seed=seed,
                    epochs=epochs,
                    data_root=data_root,
                    output_root=active_root,
                    device=device,
                    smoke_test=args.smoke_test,
                )
            )
    results = collect_metrics(
        active_root,
        required_fields=("dataset", "model", "seed", "run_mode"),
        identity_fields=("dataset", "seed"),
    )
    write_aggregate(results, active_root)
    print(f"Wrote Basis-STE results to {active_root}")


if __name__ == "__main__":
    main()
