import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from _result_store import collect_metrics, read_matching_metrics, run_mode_root


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "baselines_validation_selected"
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 2_000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
QUANTUM_OUTPUT_DIM = 16


class BottleneckMLP(nn.Module):
    def __init__(self, bottleneck_dim: int):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, 1, 28, 28)).shape[1]
        self.feature_projection = nn.Linear(conv_out, bottleneck_dim)
        self.classical_surrogate = nn.Linear(bottleneck_dim, QUANTUM_OUTPUT_DIM)
        self.output = nn.Linear(QUANTUM_OUTPUT_DIM, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv(inputs)
        features = torch.tanh(self.feature_projection(features))
        features = torch.tanh(self.classical_surrogate(features))
        return self.output(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(dataset_name: str, data_root: Path, seed: int, sample_limit: int | None):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    dataset_class = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    full_train = dataset_class(
        root=str(data_root), train=True, download=False, transform=transform
    )
    full_test = dataset_class(
        root=str(data_root), train=False, download=False, transform=transform
    )
    train_count = min(TRAIN_SIZE, len(full_train))
    validation_count = min(VALIDATION_SIZE, len(full_test))
    if sample_limit is not None:
        train_count = min(train_count, sample_limit)
        validation_count = min(validation_count, sample_limit)
    train_set = Subset(full_train, list(range(train_count)))
    validation_set = Subset(full_test, list(range(validation_count)))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, validation_loader


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels_all = []
    predictions_all = []
    for inputs, labels in loader:
        outputs = model(inputs.to(device))
        predictions = outputs.argmax(dim=1).cpu().numpy()
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


def train_one(
    dataset_name: str,
    bottleneck_dim: int,
    seed: int,
    epochs: int,
    data_root: Path,
    output_root: Path,
    device: torch.device,
    sample_limit: int | None,
):
    set_seed(seed)
    train_loader, validation_loader = make_loaders(
        dataset_name, data_root, seed, sample_limit
    )
    model = BottleneckMLP(bottleneck_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_function = nn.CrossEntropyLoss()
    output_dir = output_root / dataset_name / f"bottleneck_{bottleneck_dim}" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    checkpoint_path = output_dir / "best_validation_model.pth"
    best_validation_accuracy = -1.0
    best_epoch = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        validation_metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "loss": loss_sum / len(train_loader),
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
            f"{dataset_name:7s} bottleneck={bottleneck_dim:2d} seed={seed:3d} "
            f"epoch={epoch:2d}/{epochs:2d} loss={row['loss']:.4f} "
            f"train={row['train_accuracy']:.4f} val={row['validation_accuracy']:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device).eval()
    selected_metrics = evaluate(model, validation_loader, device)
    result = {
        "dataset": dataset_name,
        "model": f"BottleneckMLP-{bottleneck_dim}",
        "bottleneck_dim": bottleneck_dim,
        "seed": seed,
        "run_mode": "smoke" if sample_limit is not None else "full",
        "epochs": epochs,
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "parameter_count": parameter_count,
        "selection_policy": "best_validation_accuracy",
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "accuracy": selected_metrics["accuracy"],
        "precision": selected_metrics["precision"],
        "recall": selected_metrics["recall"],
        "f1": selected_metrics["f1"],
        "elapsed_seconds": time.time() - started,
        "checkpoint": checkpoint_path.relative_to(output_root).as_posix(),
    }

    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def sample_standard_deviation(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def write_combined_outputs(results: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with (output_root / "baseline_per_run.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    summary = []
    groups = sorted({(row["dataset"], row["bottleneck_dim"]) for row in results})
    for dataset_name, bottleneck_dim in groups:
        rows = [
            row
            for row in results
            if row["dataset"] == dataset_name and row["bottleneck_dim"] == bottleneck_dim
        ]
        summary.append(
            {
                "dataset": dataset_name,
                "model": f"BottleneckMLP-{bottleneck_dim}",
                "n_seeds": len(rows),
                "parameter_count": rows[0]["parameter_count"],
                "accuracy_mean": float(np.mean([row["accuracy"] for row in rows])),
                "accuracy_sd": sample_standard_deviation([row["accuracy"] for row in rows]),
                "f1_mean": float(np.mean([row["f1"] for row in rows])),
                "f1_sd": sample_standard_deviation([row["f1"] for row in rows]),
            }
        )
    with (output_root / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train parameter-matched bottleneck MLP baselines."
    )
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion"], default=["mnist", "fashion"])
    parser.add_argument("--bottlenecks", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    active_root = run_mode_root(args.output_root, args.smoke_test)
    active_root.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    print("Checkpoint selection policy: best validation accuracy")
    results = []
    for dataset_name in args.datasets:
        epochs = 1 if args.smoke_test else (10 if dataset_name == "mnist" else 25)
        sample_limit = 128 if args.smoke_test else None
        for bottleneck_dim in args.bottlenecks:
            for seed in args.seeds:
                metrics_path = (
                    active_root
                    / dataset_name
                    / f"bottleneck_{bottleneck_dim}"
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                if metrics_path.exists() and not args.force:
                    results.append(
                        read_matching_metrics(
                            metrics_path,
                            {
                                "dataset": dataset_name,
                                "bottleneck_dim": bottleneck_dim,
                                "seed": seed,
                                "run_mode": "smoke" if args.smoke_test else "full",
                            },
                        )
                    )
                    print(
                        f"Skip existing {dataset_name} bottleneck={bottleneck_dim} seed={seed}"
                    )
                    continue
                results.append(
                    train_one(
                        dataset_name=dataset_name,
                        bottleneck_dim=bottleneck_dim,
                        seed=seed,
                        epochs=epochs,
                        data_root=args.data_root,
                        output_root=active_root,
                        device=device,
                        sample_limit=sample_limit,
                    )
                )
    results = collect_metrics(
        active_root,
        required_fields=("dataset", "bottleneck_dim", "seed", "run_mode"),
        identity_fields=("dataset", "bottleneck_dim", "seed"),
    )
    write_combined_outputs(results, active_root)
    print(f"Wrote baseline results to {active_root}")


if __name__ == "__main__":
    main()
