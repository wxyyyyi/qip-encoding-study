"""Train the all-to-all, one-repeat PennyLane IQPEmbedding HQNN variant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from _result_store import (
    collect_metrics,
    iqp_pattern_slug,
    read_matching_metrics,
    run_mode_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "iqp"
N_QUBITS = 4
N_PARALLEL = 4
N_LAYERS = 2
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 2_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
IQP_REPEATS = 1


def pattern_pairs(pattern_name: str):
    if pattern_name == "all-to-all":
        return None
    if pattern_name == "nearest-neighbor":
        return tuple((wire, wire + 1) for wire in range(N_QUBITS - 1))
    raise ValueError(f"Unknown IQP pattern: {pattern_name}")


def make_qnode(pattern_name: str):
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # PennyLane 0.44.1 uses every two-wire combination when pattern=None.
        qml.IQPEmbedding(
            features=inputs,
            wires=range(N_QUBITS),
            n_repeats=IQP_REPEATS,
            pattern=pattern_pairs(pattern_name),
        )
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

    def forward(self, inputs):
        measurements = self.qnode(inputs, self.weights)
        return torch.stack(list(measurements), dim=-1).float()


class IQPHQNN(nn.Module):
    def __init__(self, pattern_name: str):
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
        qnode = make_qnode(pattern_name)
        self.qls = nn.ModuleList([QuantumLayer(qnode) for _ in range(N_PARALLEL)])
        self.out = nn.Linear(N_QUBITS * N_PARALLEL, 10)

    def projected_features(self, inputs):
        return torch.pi * torch.tanh(self.fc(self.conv(inputs.float())))

    def forward(self, inputs):
        features = self.projected_features(inputs)
        branches = features.split(N_QUBITS, dim=1)
        quantum_outputs = [
            layer(branches[index]) for index, layer in enumerate(self.qls)
        ]
        return self.out(torch.cat(quantum_outputs, dim=1))


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
        final_test_indices = list(range(2_000, 2_064))
    else:
        train_indices = list(range(TRAIN_SIZE))
        validation_indices = list(range(VALIDATION_SIZE))
        final_test_indices = list(range(VALIDATION_SIZE, len(official_test)))

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
        Subset(official_test, final_test_indices),
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


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_one(
    dataset_name,
    seed,
    epochs,
    data_root,
    output_root,
    device,
    smoke_test,
    pattern_name,
):
    set_seed(seed)
    train_loader, validation_loader, final_test_loader = make_loaders(
        dataset_name, data_root, seed, smoke_test
    )
    model = IQPHQNN(pattern_name).to(device)
    is_all_to_all = pattern_name == "all-to-all"
    model_name = "IQP-All" if is_all_to_all else "IQP-NN-Control"
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
                    "quantum_gradient_l1": gradient_group_norm(model.qls.parameters()),
                    "output_gradient_l1": gradient_group_norm(model.out.parameters()),
                }
                if any(value <= 0 for value in gradient_audit.values()):
                    raise RuntimeError(f"{model_name} gradient audit failed: {gradient_audit}")
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
            f"{dataset_name:7s} {model_name} seed={seed:3d} epoch={epoch:2d}/{epochs:2d} "
            f"loss={row['loss']:.4f} train={row['train_accuracy']:.4f} "
            f"val={row['validation_accuracy']:.4f}",
            flush=True,
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
    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "run_mode": "smoke" if smoke_test else "full",
        "epochs": epochs,
        "iqp_template": "qml.IQPEmbedding",
        "iqp_pattern": "all_to_all" if is_all_to_all else "nearest_neighbor",
        "iqp_pair_count_per_branch": 6 if is_all_to_all else 3,
        "iqp_pairs": (
            "(0,1);(0,2);(0,3);(1,2);(1,3);(2,3)"
            if is_all_to_all
            else "(0,1);(1,2);(2,3)"
        ),
        "iqp_repeats": IQP_REPEATS,
        "input_scaling": "pi*tanh(projected_features)",
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
        "checkpoint": checkpoint_path.relative_to(output_root).as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "pennylane_version": qml.__version__,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.time() - started,
    }
    write_csv(output_dir / "training_log.csv", history)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def sample_sd(values):
    return float(np.std(np.asarray(values), ddof=1)) if len(values) > 1 else 0.0


def write_aggregate(results, output_root, pattern_name):
    stem = "iqp_all" if pattern_name == "all-to-all" else "iqp_nn_control"
    model_name = "IQP-All" if pattern_name == "all-to-all" else "IQP-NN-Control"
    write_csv(output_root / f"{stem}_per_run.csv", results)
    summary = []
    for dataset_name in sorted({result["dataset"] for result in results}):
        rows = [result for result in results if result["dataset"] == dataset_name]
        summary.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "n_seeds": len(rows),
                "test_accuracy_mean": float(np.mean([row["test_accuracy"] for row in rows])),
                "test_accuracy_sd": sample_sd([row["test_accuracy"] for row in rows]),
                "test_precision_mean": float(np.mean([row["test_precision"] for row in rows])),
                "test_precision_sd": sample_sd([row["test_precision"] for row in rows]),
                "test_recall_mean": float(np.mean([row["test_recall"] for row in rows])),
                "test_recall_sd": sample_sd([row["test_recall"] for row in rows]),
                "test_f1_mean": float(np.mean([row["test_f1"] for row in rows])),
                "test_f1_sd": sample_sd([row["test_f1"] for row in rows]),
                "best_validation_accuracy_mean": float(np.mean([row["best_validation_accuracy"] for row in rows])),
                "best_epoch_mean": float(np.mean([row["best_epoch"] for row in rows])),
                "feature_drift_min": min(row["maximum_feature_parameter_drift"] for row in rows),
            }
        )
    write_csv(output_root / f"{stem}_summary.csv", summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one-repeat PennyLane IQPEmbedding HQNN topology variants."
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=["mnist", "fashion"], default=["mnist", "fashion"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs-mnist", type=int, default=10)
    parser.add_argument("--epochs-fashion", type=int, default=25)
    parser.add_argument(
        "--iqp-pattern",
        choices=["all-to-all", "nearest-neighbor"],
        default="all-to-all",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    active_root = (
        run_mode_root(output_root, args.smoke_test)
        / iqp_pattern_slug(args.iqp_pattern)
    )
    active_root.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}", flush=True)
    print(
        f"IQP protocol: qml.IQPEmbedding, pattern={args.iqp_pattern}, "
        "n_repeats=1, input=pi*tanh(z)",
        flush=True,
    )
    print(
        "Data protocol: train[0:10000], validation=official_test[0:2000], "
        "final_test=official_test[2000:10000]",
        flush=True,
    )

    results = []
    for dataset_name in args.datasets:
        epochs = 1 if args.smoke_test else (
            args.epochs_mnist if dataset_name == "mnist" else args.epochs_fashion
        )
        for seed in args.seeds:
            metrics_path = active_root / dataset_name / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists() and not args.force:
                results.append(
                    read_matching_metrics(
                        metrics_path,
                        {
                            "dataset": dataset_name,
                            "seed": seed,
                            "run_mode": "smoke" if args.smoke_test else "full",
                            "iqp_pattern": iqp_pattern_slug(args.iqp_pattern),
                        },
                    )
                )
                print(f"Skip existing {dataset_name} seed={seed}", flush=True)
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
                    pattern_name=args.iqp_pattern,
                )
            )
    results = collect_metrics(
        active_root,
        required_fields=("dataset", "model", "seed", "run_mode", "iqp_pattern"),
        identity_fields=("dataset", "seed"),
    )
    write_aggregate(results, active_root, args.iqp_pattern)
    print(f"Wrote IQP results to {active_root}", flush=True)


if __name__ == "__main__":
    main()
