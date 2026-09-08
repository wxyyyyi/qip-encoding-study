import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pennylane as qml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "quantum_resources"
N_QUBITS = 4
N_PARALLEL = 4
N_LAYERS = 2
GATE_SET = {
    "CNOT",
    "Hadamard",
    "PauliX",
    "RX",
    "RY",
    "RZ",
    "GlobalPhase",
}


def apply_encoding(name: str, inputs):
    if name == "amplitude":
        qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
    elif name == "angle":
        for wire in range(N_QUBITS):
            qml.RY(inputs[wire], wires=wire)
    elif name == "iqp":
        for wire in range(N_QUBITS):
            qml.Hadamard(wires=wire)
        for wire in range(N_QUBITS):
            qml.RZ(inputs[wire], wires=wire)
        for wire in range(N_QUBITS - 1):
            qml.CNOT(wires=[wire, wire + 1])
            qml.RZ(inputs[wire] * inputs[wire + 1], wires=wire + 1)
            qml.CNOT(wires=[wire, wire + 1])
    elif name == "iqp-all":
        qml.IQPEmbedding(
            features=inputs,
            wires=range(N_QUBITS),
            n_repeats=1,
            pattern=None,
        )
    elif name == "basis":
        qml.BasisState(np.asarray(inputs, dtype=int), wires=range(N_QUBITS))
    elif name == "basis-ste":
        for wire in range(N_QUBITS):
            qml.RX(np.pi * inputs[wire], wires=wire)
    elif name != "none":
        raise ValueError(f"Unknown encoding: {name}")


def apply_ansatz(weights):
    for layer in range(N_LAYERS):
        for wire in range(N_QUBITS):
            qml.RZ(weights[layer, wire, 0], wires=wire)
            qml.RX(weights[layer, wire, 1], wires=wire)
        for wire in range(N_QUBITS - 1):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])


def make_qnode(name: str, include_ansatz: bool):
    device = qml.device("default.qubit", wires=N_QUBITS)

    def circuit(inputs, weights):
        apply_encoding(name, inputs)
        if include_ansatz:
            apply_ansatz(weights)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(N_QUBITS)]

    qnode = qml.QNode(circuit, device, interface=None, diff_method=None)
    return qml.transforms.decompose(qnode, gate_set=GATE_SET)


def logical_depth(operations) -> int:
    wire_depth = {wire: 0 for wire in range(N_QUBITS)}
    max_depth = 0
    for operation in operations:
        wires = [int(wire) for wire in operation.wires]
        if not wires:
            continue
        operation_depth = max(wire_depth[wire] for wire in wires) + 1
        for wire in wires:
            wire_depth[wire] = operation_depth
        max_depth = max(max_depth, operation_depth)
    return max_depth


def inspect_circuit(name: str, inputs, include_ansatz: bool):
    qnode = make_qnode(name, include_ansatz)
    weights = np.linspace(0.071, 1.337, N_LAYERS * N_QUBITS * 2).reshape(
        N_LAYERS, N_QUBITS, 2
    )
    tape = qml.workflow.construct_tape(qnode)(inputs, weights)
    operations = list(tape.operations)
    unsupported = sorted({operation.name for operation in operations} - GATE_SET)
    if unsupported:
        raise RuntimeError(f"Unexpanded operations for {name}: {unsupported}")
    counts = Counter(operation.name for operation in operations)
    physical_operations = [
        operation
        for operation in operations
        if len(operation.wires) > 0 and operation.name != "GlobalPhase"
    ]
    one_qubit = sum(1 for operation in physical_operations if len(operation.wires) == 1)
    two_qubit = sum(1 for operation in physical_operations if len(operation.wires) == 2)
    return {
        "total_gates": len(physical_operations),
        "single_qubit_gates": one_qubit,
        "two_qubit_gates": two_qubit,
        "cnot_gates": counts["CNOT"],
        "depth": logical_depth(physical_operations),
        "gate_types": "; ".join(f"{key}:{counts[key]}" for key in sorted(counts) if key != "GlobalPhase"),
        "global_phases_ignored": counts["GlobalPhase"],
    }


def representative_inputs(name: str):
    if name == "amplitude":
        # A generic dense signed real state reaches the upper-bound decomposition
        # used by PennyLane's four-qubit Mottonen state preparation. Simpler sign
        # patterns can be compiled with fewer phase rotations, so the manuscript
        # reports this reproducible upper-bound profile rather than one favorable
        # input-dependent example.
        values = np.asarray(
            [
                0.04292085450178845,
                -0.1054508707805904,
                -0.38019411075297227,
                0.011689127978976895,
                0.1387267103487034,
                0.18618531234256075,
                0.294657175995066,
                -0.0019948772024763087,
                0.2618036783605149,
                0.4777883056761211,
                -0.4244372806227839,
                -0.11090878120727245,
                -0.3554216091023587,
                -0.2290463640440113,
                0.16696716296766975,
                0.07537526355591857,
            ],
            dtype=float,
        )
        return values / np.linalg.norm(values)
    if name in {"angle", "iqp", "iqp-all"}:
        return np.asarray([0.23, -0.51, 0.79, -1.07], dtype=float)
    if name in {"basis", "basis-ste"}:
        return np.asarray([0, 1, 0, 1], dtype=int)
    return np.zeros(N_QUBITS, dtype=float)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def model_parameter_rows():
    conv_parameters = (1 * 16 * 3 * 3 + 16) + (16 * 32 * 3 * 3 + 32)
    conv_output = 32 * 7 * 7
    classifier_parameters = 16 * 10 + 10
    quantum_parameters = N_PARALLEL * N_LAYERS * N_QUBITS * 2
    rows = []
    for name, feature_dim in (
        ("Amplitude", 64),
        ("Angle", 16),
        ("IQP-All", 16),
        ("IQP-NN", 16),
        ("Basis-STE", 16),
        ("Basis-Hard", 16),
    ):
        projection_parameters = conv_output * feature_dim + feature_dim
        total = conv_parameters + projection_parameters + quantum_parameters + classifier_parameters
        rows.append(
            {
                "model": name,
                "conv_parameters": conv_parameters,
                "feature_projection_parameters": projection_parameters,
                "quantum_trainable_parameters": quantum_parameters,
                "classifier_parameters": classifier_parameters,
                "total_trainable_parameters": total,
                "quantum_input_features": feature_dim,
            }
        )
    rows.extend(
        [
            {
                "model": "ClassicCNN",
                "conv_parameters": conv_parameters,
                "feature_projection_parameters": 0,
                "quantum_trainable_parameters": 0,
                "classifier_parameters": conv_output * 10 + 10,
                "total_trainable_parameters": conv_parameters + conv_output * 10 + 10,
                "quantum_input_features": 0,
            },
            {
                "model": "BottleneckMLP-16",
                "conv_parameters": conv_parameters,
                "feature_projection_parameters": conv_output * 16 + 16,
                "quantum_trainable_parameters": 0,
                "classifier_parameters": (16 * 16 + 16) + classifier_parameters,
                "total_trainable_parameters": 30_346,
                "quantum_input_features": 16,
            },
            {
                "model": "BottleneckMLP-64",
                "conv_parameters": conv_parameters,
                "feature_projection_parameters": conv_output * 64 + 64,
                "quantum_trainable_parameters": 0,
                "classifier_parameters": (64 * 16 + 16) + classifier_parameters,
                "total_trainable_parameters": 106_426,
                "quantum_input_features": 64,
            },
        ]
    )
    return rows


def write_markdown(resource_rows: list[dict], parameter_rows: list[dict], output_root: Path):
    lines = [
        "# Quantum Circuit Resource Analysis",
        "",
        f"Generated with PennyLane {qml.__version__}. Templates were decomposed to RX/RY/RZ/H/X/CNOT gates before counting. Global phases are excluded from physical gate totals.",
        "",
        "## Per-branch and four-branch resources",
        "",
        "| Encoding | Input features/branch | Single-qubit gates/branch | CNOT/branch | Total gates/branch | Depth/branch | Gates across 4 branches | CNOT across 4 branches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in resource_rows:
        lines.append(
            f"| {row['encoding']} | {row['input_features_per_branch']} | "
            f"{row['single_qubit_gates_per_branch']} | {row['cnot_gates_per_branch']} | "
            f"{row['total_gates_per_branch']} | {row['depth_per_branch']} | "
            f"{row['total_gates_four_branches']} | {row['cnot_gates_four_branches']} |"
        )
    lines.extend(
        [
            "",
            "Four branches require 16 qubits for simultaneous execution. Under ideal parallel execution, critical-path depth equals the per-branch depth; on a reused four-qubit device, the branch circuit is executed four times.",
            "",
            "Basis-Hard counts use the representative state 0101. Its BasisState decomposition uses one X gate per active bit. Basis-STE uses four fixed RX(pi * bit) gates per branch and is forward-equivalent up to global phase.",
            "",
            "Amplitude counts report the reproducible upper-bound decomposition for a generic dense signed real 16-component state. PennyLane can remove phase operations for favorable input sign patterns, so per-sample compiled counts may be lower; the upper bound is used for conservative hardware comparison.",
            "",
            "## Trainable parameter counts",
            "",
            "| Model | Conv | Feature projection | Quantum | Classifier/surrogate | Total |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in parameter_rows:
        lines.append(
            f"| {row['model']} | {row['conv_parameters']} | "
            f"{row['feature_projection_parameters']} | {row['quantum_trainable_parameters']} | "
            f"{row['classifier_parameters']} | {row['total_trainable_parameters']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Angle has the shallowest continuous state preparation and the fewest two-qubit gates among continuous encodings.",
            "- IQP-NN adds six CNOT gates per branch for three nearest-neighbor ZZ feature interactions; IQP-All uses PennyLane's six-pair all-to-all pattern.",
            "- Basis-STE remains shallow at inference and restores end-to-end gradient flow; Basis-Hard is retained only as an implementation ablation.",
            "- Amplitude carries four times as many input features as Angle/IQP/Basis under the same four-qubit branch, but arbitrary state preparation increases gate count and depth.",
            "- The Amplitude hybrid model has substantially more classical projection parameters because it maps the CNN output to 64 rather than 16 features. Accuracy differences must not be attributed to encoding alone.",
        ]
    )
    (output_root / "RESOURCE_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Expand and count the four quantum encoding circuits.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ansatz = inspect_circuit("none", representative_inputs("none"), include_ansatz=True)
    resource_rows = []
    detail_rows = []
    for name in ("amplitude", "angle", "iqp-all", "iqp", "basis-ste", "basis"):
        inputs = representative_inputs(name)
        encoding = inspect_circuit(name, inputs, include_ansatz=False)
        full = inspect_circuit(name, inputs, include_ansatz=True)
        input_features = 16 if name == "amplitude" else 4
        resource_rows.append(
            {
                "encoding": {"iqp-all": "IQP-All", "iqp": "IQP-NN", "basis-ste": "Basis-STE", "basis": "Basis-Hard"}.get(name, name.capitalize()),
                "input_features_per_branch": input_features,
                "single_qubit_gates_per_branch": full["single_qubit_gates"],
                "two_qubit_gates_per_branch": full["two_qubit_gates"],
                "cnot_gates_per_branch": full["cnot_gates"],
                "total_gates_per_branch": full["total_gates"],
                "depth_per_branch": full["depth"],
                "total_gates_four_branches": full["total_gates"] * N_PARALLEL,
                "cnot_gates_four_branches": full["cnot_gates"] * N_PARALLEL,
                "parallel_depth_16_qubits": full["depth"],
                "sequential_branch_executions_4_qubits": N_PARALLEL,
                "quantum_trainable_parameters_per_branch": N_LAYERS * N_QUBITS * 2,
                "quantum_trainable_parameters_four_branches": N_PARALLEL * N_LAYERS * N_QUBITS * 2,
                "decomposed_gate_types": full["gate_types"],
                "counting_convention": (
                    "generic_dense_real_upper_bound"
                    if name == "amplitude"
                    else "representative_0101"
                    if name == "basis"
                    else "fixed_circuit_structure"
                ),
            }
        )
        for section, values in (("encoding_only", encoding), ("full_branch", full)):
            detail_rows.append({"encoding": name, "section": section, **values})
    detail_rows.append({"encoding": "shared", "section": "ansatz_only", **ansatz})
    parameter_rows = model_parameter_rows()
    write_csv(output_root / "quantum_resource_summary.csv", resource_rows)
    write_csv(output_root / "quantum_resource_details.csv", detail_rows)
    write_csv(output_root / "model_parameter_counts.csv", parameter_rows)
    write_markdown(resource_rows, parameter_rows, output_root)
    print(f"Wrote resource analysis to {output_root}")


if __name__ == "__main__":
    main()
