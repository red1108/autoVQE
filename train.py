from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from scipy.optimize import differential_evolution, minimize

import prepare

DEFAULT_PROBLEM_PATH = "examples/h2_2q.json"
PROBLEM_PATH = Path(os.environ.get("AUTOVQE_PROBLEM_PATH", DEFAULT_PROBLEM_PATH))
RESULTS_PATH = Path(os.environ.get("AUTOVQE_RESULTS_PATH", "results.tsv"))
MIN_EXPERIMENTS = int(os.environ.get("AUTOVQE_MIN_EXPERIMENTS", "100"))
MAX_EXPERIMENTS = int(os.environ.get("AUTOVQE_MAX_EXPERIMENTS", "0"))
ENERGY_IMPROVEMENT_TOL = 1e-6
ENERGY_EQUIVALENCE_TOL = 5e-4
EXHAUSTION_PATIENCE = int(os.environ.get("AUTOVQE_EXHAUSTION_PATIENCE", "14"))
RUN_MODE = os.environ.get("AUTOVQE_RUN_MODE", "full")
MODEL_CLASS = os.environ.get("AUTOVQE_MODEL_CLASS", "auto")
TARGET_REL_ERROR = float(os.environ.get("AUTOVQE_TARGET_REL_ERROR", "0") or "0")
TARGET_ABS_ERROR = float(os.environ.get("AUTOVQE_TARGET_ABS_ERROR", "0") or "0")
STOP_AT_TARGET = os.environ.get("AUTOVQE_STOP_AT_TARGET", "0") == "1"
TARGET_EXTRA_COMPRESS = int(os.environ.get("AUTOVQE_TARGET_EXTRA_COMPRESS", "0"))


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    layers: int
    optimizer: str
    param_init: str
    learning_rate: float
    seed: int
    edge_mode: str = "full"
    rotation_mode: str = "full"
    reference_state: str = "zero"
    time_budget_seconds: float | None = None
    spsa_steps: int = 24
    restarts: int = 2
    description: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    description: str
    status: str
    energy: float
    overlap: float | None
    metrics: dict[str, int]
    num_params: int
    eval_calls: int
    total_seconds: float

    @property
    def compression_key(self) -> tuple[int, int, int, int]:
        return (
            self.metrics["twoq_count"],
            self.metrics["total_gate_count"],
            self.metrics["depth"],
            self.num_params,
        )


def ensure_results_header() -> None:
    header = "commit\tenergy\tsingleq_count\ttwoq_count\ttotal_gate_count\tnum_params\tstatus\tdescription\n"
    if not RESULTS_PATH.exists() or not RESULTS_PATH.read_text():
        RESULTS_PATH.write_text(header)


def head_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "nogit00"


def code_snapshot() -> str:
    return hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:7]


def experiment_id(index: int, description: str) -> str:
    raw = f"{head_snapshot()}|{code_snapshot()}|{index}|{description}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:7]


def log_result(result: ExperimentResult) -> None:
    row = "\t".join(
        [
            result.run_id,
            f"{result.energy:.6f}",
            str(result.metrics["singleq_count"]),
            str(result.metrics["twoq_count"]),
            str(result.metrics["total_gate_count"]),
            str(result.num_params),
            result.status,
            result.description,
        ]
    )
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        with RESULTS_PATH.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_separator = handle.read(1) != b"\n"
    else:
        needs_separator = False

    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(row + "\n")


def chain_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
    if not problem.coupling_map:
        return [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_control, raw_target in problem.coupling_map:
        if raw_control == raw_target:
            continue
        edge = tuple(sorted((raw_control, raw_target)))
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges or [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]


def support_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    labels = problem.hamiltonian.paulis.to_labels()
    for label in labels:
        active = tuple(label_position_to_qubit(label, index) for index, pauli in enumerate(label) if pauli != "I")
        if len(active) == 2:
            edges.add(tuple(sorted(active)))
    return sorted(edges) or chain_edges(problem)


def label_position_to_qubit(label: str, position: int) -> int:
    return len(label) - position - 1


def label_active_qubits(label: str) -> list[tuple[int, str]]:
    return [
        (label_position_to_qubit(label, position), pauli)
        for position, pauli in enumerate(label)
        if pauli != "I"
    ]


def heisenberg_edge_weights(problem: prepare.Problem) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], list[float]] = {}
    labels = problem.hamiltonian.paulis.to_labels()
    coeffs = problem.hamiltonian.coeffs
    for label, coeff in zip(labels, coeffs, strict=True):
        active = tuple(qubit for qubit, _ in label_active_qubits(label))
        if len(active) != 2:
            continue
        ops = "".join(pauli for _, pauli in label_active_qubits(label))
        if ops not in {"XX", "YY", "ZZ"}:
            continue
        edge = tuple(sorted(active))
        weights.setdefault(edge, []).append(float(np.real(coeff)))

    return {
        edge: float(np.mean(edge_weights))
        for edge, edge_weights in sorted(weights.items())
        if edge_weights
    }


def quick_model_class(problem: prepare.Problem) -> str:
    groups: dict[tuple[int, ...], dict[str, float]] = {}
    labels = problem.hamiltonian.paulis.to_labels()
    coeffs = problem.hamiltonian.coeffs
    locality: list[int] = []
    for label, coeff in zip(labels, coeffs, strict=True):
        active_terms = label_active_qubits(label)
        active = tuple(qubit for qubit, _ in active_terms)
        ops = "".join(pauli for _, pauli in active_terms)
        locality.append(len(active))
        groups.setdefault(active, {})[ops] = float(np.real(coeff))

    all_ops = [op for ops in groups.values() for op in ops if op]
    if all_ops and all(op and set(op) <= {"Z"} for op in all_ops):
        return "classical_ising_or_qubo"

    two_local = [(support, ops) for support, ops in groups.items() if len(support) == 2]
    one_local_ops = {op for support, ops in groups.items() if len(support) == 1 for op in ops}
    zz_edges = sum(1 for _, ops in two_local if set(ops) <= {"ZZ"})
    if zz_edges and "X" in one_local_ops:
        return "transverse_field_ising"

    heisenberg_edges = 0
    for _, ops in two_local:
        if {"XX", "YY", "ZZ"}.issubset(ops):
            coeff_values = [ops["XX"], ops["YY"], ops["ZZ"]]
            if max(coeff_values) - min(coeff_values) <= 1e-9 * max(1.0, max(abs(value) for value in coeff_values)):
                heisenberg_edges += 1
    if two_local and heisenberg_edges / len(two_local) >= 0.6:
        return "weighted_heisenberg_graph"

    if locality and max(locality) > 2:
        return "chemistry_or_general_pauli"
    return "general_two_local_pauli"


def edge_color_groups(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    occupied: list[set[int]] = []
    for edge in edges:
        left, right = edge
        for index, used in enumerate(occupied):
            if left not in used and right not in used:
                groups[index].append(edge)
                used.update(edge)
                break
        else:
            groups.append([edge])
            occupied.append({left, right})
    return groups


def bipartition(num_qubits: int, edges: list[tuple[int, int]]) -> dict[int, int] | None:
    graph: dict[int, list[int]] = {qubit: [] for qubit in range(num_qubits)}
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)

    colors: dict[int, int] = {}
    frontier: list[int] = []
    for start in range(num_qubits):
        if start in colors or not graph[start]:
            continue
        colors[start] = 0
        frontier.append(start)
        while frontier:
            node = frontier.pop()
            for neighbor in graph[node]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    frontier.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return None
    return colors


def prepare_reference_state(circuit: QuantumCircuit, problem: prepare.Problem, spec: ExperimentSpec) -> None:
    if spec.reference_state == "zero":
        return
    if spec.reference_state == "plus":
        for qubit in range(problem.num_qubits):
            circuit.ry(np.pi / 2.0, qubit)
        return
    if spec.reference_state == "minus":
        for qubit in range(problem.num_qubits):
            circuit.ry(-np.pi / 2.0, qubit)
        return
    if spec.reference_state == "hint":
        if not isinstance(problem.initial_state_hint, list):
            raise ValueError("hint reference requested but initial_state_hint is absent")
        if len(problem.initial_state_hint) != problem.num_qubits:
            raise ValueError("initial_state_hint length does not match num_qubits")
        for qubit, bit in enumerate(problem.initial_state_hint):
            if int(bit):
                circuit.x(qubit)
        return
    if spec.reference_state == "neel":
        colors = bipartition(problem.num_qubits, support_edges(problem))
        if colors is None:
            raise ValueError("Neel reference requested for a non-bipartite support graph")
        for qubit, color in colors.items():
            if color == 1:
                circuit.x(qubit)
        return

    if spec.reference_state in {"dimer_even", "dimer_odd"}:
        start = 0 if spec.reference_state == "dimer_even" else 1
        for left in range(start, problem.num_qubits - 1, 2):
            right = left + 1
            circuit.ry(np.pi / 2.0, left)
            circuit.x(right)
            circuit.cx(left, right)
            circuit.rz(np.pi, left)
        return

    raise ValueError(f"unsupported reference state: {spec.reference_state}")


def tfim_reference_state(problem: prepare.Problem) -> str:
    x_field = 0.0
    for label, coeff in zip(problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs, strict=True):
        active = label_active_qubits(label)
        if len(active) == 1 and active[0][1] == "X":
            x_field += float(np.real(coeff))
    return "plus" if x_field < 0.0 else "minus"


def select_edges(
    edges: list[tuple[int, int]],
    mode: str,
    layers: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if mode == "full":
        return [list(edges) for _ in range(layers)]
    if mode == "even":
        subset = [edge for index, edge in enumerate(edges) if index % 2 == 0]
        return [subset for _ in range(layers)]
    if mode == "odd":
        subset = [edge for index, edge in enumerate(edges) if index % 2 == 1]
        return [subset for _ in range(layers)]
    if mode == "alternate":
        return [
            [edge for index, edge in enumerate(edges) if index % 2 == layer % 2]
            for layer in range(layers)
        ]
    if mode == "ends":
        subset = [edges[0]]
        if len(edges) > 1:
            subset.append(edges[-1])
        return [subset for _ in range(layers)]
    if mode == "random_1":
        rng = np.random.default_rng(seed)
        return [[edges[int(rng.integers(len(edges)))]] for _ in range(layers)]
    if mode == "random_2":
        rng = np.random.default_rng(seed)
        per_layer: list[list[tuple[int, int]]] = []
        for _ in range(layers):
            order = rng.permutation(len(edges))
            chosen = sorted(int(index) for index in order[: min(2, len(edges))])
            per_layer.append([edges[index] for index in chosen])
        return per_layer
    raise ValueError(f"unsupported edge mode: {mode}")


def build_hea_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    circuit = QuantumCircuit(problem.num_qubits)

    if spec.rotation_mode == "full":
        params_per_layer = 2 * problem.num_qubits
    elif spec.rotation_mode == "ry_only":
        params_per_layer = problem.num_qubits
    elif spec.rotation_mode == "ry_final":
        params_per_layer = problem.num_qubits
    elif spec.rotation_mode == "shared":
        params_per_layer = 2
    else:
        raise ValueError(f"unsupported rotation mode: {spec.rotation_mode}")

    final_params = problem.num_qubits if spec.rotation_mode == "ry_final" else 0
    theta = ParameterVector("theta", spec.layers * params_per_layer + final_params)
    cursor = 0
    for layer in range(spec.layers):
        if spec.rotation_mode == "shared":
            shared_ry = theta[cursor]
            shared_rz = theta[cursor + 1]
            cursor += 2
            for qubit in range(problem.num_qubits):
                circuit.ry(shared_ry, qubit)
            for qubit in range(problem.num_qubits):
                circuit.rz(shared_rz, qubit)
        elif spec.rotation_mode in {"ry_only", "ry_final"}:
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                cursor += 1
        else:
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                cursor += 1
            for qubit in range(problem.num_qubits):
                circuit.rz(theta[cursor], qubit)
                cursor += 1

        for control, target in edges_by_layer[layer]:
            circuit.cx(control, target)

    if spec.rotation_mode == "ry_final":
        for qubit in range(problem.num_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1

    return circuit, theta


def build_symm_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    width = (problem.num_qubits + 1) // 2
    theta = ParameterVector("theta", 2 * spec.layers * width)
    circuit = QuantumCircuit(problem.num_qubits)
    cursor = 0

    for _ in range(spec.layers):
        for left in range(width):
            right = problem.num_qubits - left - 1
            angle_y = theta[cursor]
            cursor += 1
            circuit.ry(angle_y, left)
            if right != left:
                circuit.ry(angle_y, right)
        for qubit in range(problem.num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for left in range(width):
            right = problem.num_qubits - left - 1
            angle_z = theta[cursor]
            cursor += 1
            circuit.rz(angle_z, left)
            if right != left:
                circuit.rz(angle_z, right)

    return circuit, theta


def build_brick_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", problem.num_qubits * (spec.layers + 1))
    circuit = QuantumCircuit(problem.num_qubits)
    cursor = 0

    for layer in range(spec.layers):
        for qubit in range(problem.num_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1

        if spec.edge_mode == "full":
            parities = [0, 1]
        elif spec.edge_mode == "even":
            parities = [0]
        elif spec.edge_mode == "odd":
            parities = [1]
        else:
            parities = [layer % 2]

        for parity in parities:
            for qubit in range(parity, problem.num_qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)

    for qubit in range(problem.num_qubits):
        circuit.ry(theta[cursor], qubit)
        cursor += 1

    return circuit, theta


def add_zz_block(circuit: QuantumCircuit, control: int, target: int, angle: Any) -> None:
    circuit.cx(control, target)
    circuit.rz(2.0 * angle, target)
    circuit.cx(control, target)


def add_xx_block(circuit: QuantumCircuit, control: int, target: int, angle: Any) -> None:
    circuit.ry(np.pi / 2.0, control)
    circuit.ry(np.pi / 2.0, target)
    add_zz_block(circuit, control, target, angle)
    circuit.ry(-np.pi / 2.0, control)
    circuit.ry(-np.pi / 2.0, target)


def add_yy_block(circuit: QuantumCircuit, control: int, target: int, angle: Any) -> None:
    circuit.rx(-np.pi / 2.0, control)
    circuit.rx(-np.pi / 2.0, target)
    add_zz_block(circuit, control, target, angle)
    circuit.rx(np.pi / 2.0, control)
    circuit.rx(np.pi / 2.0, target)


def add_heisenberg_block(circuit: QuantumCircuit, control: int, target: int, angle: Any) -> None:
    add_xx_block(circuit, control, target, angle)
    add_yy_block(circuit, control, target, angle)
    add_zz_block(circuit, control, target, angle)


def add_pauli_evolution(circuit: QuantumCircuit, label: str, angle: Any) -> None:
    active = label_active_qubits(label)
    if not active:
        return

    for qubit, pauli in active:
        if pauli == "X":
            circuit.ry(np.pi / 2.0, qubit)
        elif pauli == "Y":
            circuit.rx(-np.pi / 2.0, qubit)
        elif pauli != "Z":
            raise ValueError(f"unsupported Pauli in label: {label}")

    target = active[-1][0]
    for qubit, _ in active[:-1]:
        circuit.cx(qubit, target)
    circuit.rz(2.0 * angle, target)
    for qubit, _ in reversed(active[:-1]):
        circuit.cx(qubit, target)

    for qubit, pauli in reversed(active):
        if pauli == "X":
            circuit.ry(-np.pi / 2.0, qubit)
        elif pauli == "Y":
            circuit.rx(np.pi / 2.0, qubit)


def hamiltonian_pauli_terms(problem: prepare.Problem) -> list[tuple[str, float]]:
    terms: list[tuple[str, float]] = []
    labels = problem.hamiltonian.paulis.to_labels()
    coeffs = problem.hamiltonian.coeffs
    for label, coeff in zip(labels, coeffs, strict=True):
        if set(label) == {"I"}:
            continue
        terms.append((label, float(np.real(coeff))))
    return terms


def build_pauli_hva_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    terms = hamiltonian_pauli_terms(problem)
    if not terms:
        return QuantumCircuit(problem.num_qubits), ParameterVector("theta", 0)

    if spec.rotation_mode == "shared":
        params_per_layer = 1
    elif spec.rotation_mode == "term":
        params_per_layer = len(terms)
    else:
        raise ValueError(f"unsupported Pauli HVA rotation mode: {spec.rotation_mode}")

    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        if spec.rotation_mode == "shared":
            shared_time = theta[cursor]
            cursor += 1
            for label, coeff in terms:
                add_pauli_evolution(circuit, label, shared_time * coeff)
        else:
            for label, coeff in terms:
                add_pauli_evolution(circuit, label, theta[cursor] * coeff)
                cursor += 1
    return circuit, theta


def excitation_flip_supports(problem: prepare.Problem) -> list[tuple[int, ...]]:
    supports: set[tuple[int, ...]] = set()
    for label in problem.hamiltonian.paulis.to_labels():
        support = tuple(sorted(qubit for qubit, pauli in label_active_qubits(label) if pauli in {"X", "Y"}))
        if len(support) >= 2:
            supports.add(support)
    return sorted(supports, key=lambda support: (-len(support), support))


def hint_bits(problem: prepare.Problem) -> list[int]:
    if not isinstance(problem.initial_state_hint, list):
        raise ValueError("initial_state_hint is required for a two-state excitation ansatz")
    if len(problem.initial_state_hint) != problem.num_qubits:
        raise ValueError("initial_state_hint length does not match num_qubits")
    return [int(bit) for bit in problem.initial_state_hint]


def flipped_bits(bits: list[int], support: tuple[int, ...]) -> list[int]:
    target = list(bits)
    for qubit in support:
        target[qubit] = 1 - target[qubit]
    return target


def add_two_state_mixer(
    circuit: QuantumCircuit,
    source_bits: list[int],
    target_bits: list[int],
    angle: Any,
) -> None:
    diff = [
        qubit
        for qubit, (source, target) in enumerate(zip(source_bits, target_bits, strict=True))
        if source != target
    ]
    if not diff:
        return

    rotation_qubit = diff[0]
    for qubit in diff[1:]:
        circuit.cx(rotation_qubit, qubit)

    controls = [qubit for qubit in range(len(source_bits)) if qubit != rotation_qubit]
    activated_zero_controls: list[int] = []
    for qubit in controls:
        desired = source_bits[qubit]
        if qubit in diff[1:]:
            desired ^= source_bits[rotation_qubit]
        if desired == 0:
            circuit.x(qubit)
            activated_zero_controls.append(qubit)

    circuit.mcry(angle, controls, rotation_qubit, None, mode="noancilla")

    for qubit in reversed(activated_zero_controls):
        circuit.x(qubit)
    for qubit in reversed(diff[1:]):
        circuit.cx(rotation_qubit, qubit)


def build_two_state_excitation_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    source = hint_bits(problem)
    supports = excitation_flip_supports(problem)
    if not supports:
        raise ValueError("no X/Y excitation supports were found")

    theta = ParameterVector("theta", spec.layers * len(supports))
    circuit = QuantumCircuit(problem.num_qubits)
    for qubit, bit in enumerate(source):
        if bit:
            circuit.x(qubit)

    cursor = 0
    for _ in range(spec.layers):
        for support in supports:
            target = flipped_bits(source, support)
            add_two_state_mixer(circuit, source, target, theta[cursor])
            cursor += 1

    return circuit, theta


def build_heisenberg_hva_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges = support_edges(problem)
    weights = heisenberg_edge_weights(problem)
    if spec.edge_mode == "colored":
        edge_groups = edge_color_groups(edges)
    else:
        edge_groups = select_edges(edges, spec.edge_mode, 1, spec.seed)

    if spec.rotation_mode == "shared":
        params_per_layer = len(edge_groups)
    elif spec.rotation_mode == "edge":
        params_per_layer = sum(len(group) for group in edge_groups)
    else:
        raise ValueError(f"unsupported Heisenberg rotation mode: {spec.rotation_mode}")

    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)

    cursor = 0
    for _ in range(spec.layers):
        for group in edge_groups:
            if spec.rotation_mode == "shared":
                shared_time = theta[cursor]
                cursor += 1
                for control, target in group:
                    add_heisenberg_block(
                        circuit,
                        control,
                        target,
                        shared_time * weights.get((control, target), 1.0),
                    )
            else:
                for control, target in group:
                    add_heisenberg_block(
                        circuit,
                        control,
                        target,
                        theta[cursor] * weights.get((control, target), 1.0),
                    )
                    cursor += 1

    return circuit, theta


def build_tfim_shared_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", 2 * spec.layers)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    cursor = 0

    for layer in range(spec.layers):
        gamma = theta[cursor]
        beta = theta[cursor + 1]
        cursor += 2
        for control, target in edges_by_layer[layer]:
            add_zz_block(circuit, control, target, gamma)
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)

    return circuit, theta


def build_tfim_factorized_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    edge_params = sum(len(layer_edges) for layer_edges in edges_by_layer)
    theta = ParameterVector("theta", edge_params + spec.layers * problem.num_qubits)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0

    for layer in range(spec.layers):
        for control, target in edges_by_layer[layer]:
            add_zz_block(circuit, control, target, theta[cursor])
            cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * theta[cursor], qubit)
            cursor += 1

    return circuit, theta


def build_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    if spec.family == "hea":
        return build_hea_ansatz(problem, spec)
    if spec.family == "symm":
        return build_symm_ansatz(problem, spec)
    if spec.family == "brick":
        return build_brick_ansatz(problem, spec)
    if spec.family == "heisenberg_hva":
        return build_heisenberg_hva_ansatz(problem, spec)
    if spec.family == "pauli_hva":
        return build_pauli_hva_ansatz(problem, spec)
    if spec.family == "two_state_excitation":
        return build_two_state_excitation_ansatz(problem, spec)
    if spec.family == "tfim_shared":
        return build_tfim_shared_ansatz(problem, spec)
    if spec.family == "tfim_factorized":
        return build_tfim_factorized_ansatz(problem, spec)
    raise ValueError(f"unsupported ansatz family: {spec.family}")


def initial_parameters(spec: ExperimentSpec, num_params: int, restart_index: int = 0) -> np.ndarray:
    if spec.param_init == "zeros":
        return np.zeros(num_params, dtype=float)

    rng = np.random.default_rng(spec.seed + 31 * restart_index)
    if spec.param_init == "small_random":
        return rng.uniform(-0.15, 0.15, size=num_params)
    if spec.param_init == "random":
        return rng.uniform(-0.5, 0.5, size=num_params)
    raise ValueError(f"unsupported parameter init mode: {spec.param_init}")


def bind_parameters(circuit: QuantumCircuit, parameters: ParameterVector, values: np.ndarray) -> QuantumCircuit:
    mapping = dict(zip(parameters, values, strict=True))
    return circuit.assign_parameters(mapping, inplace=False)


def training_time_budget(problem: prepare.Problem, spec: ExperimentSpec) -> float:
    if spec.time_budget_seconds is not None:
        return float(spec.time_budget_seconds)
    override = os.environ.get("AUTOVQE_EXPERIMENT_SECONDS")
    if override is not None:
        return float(override)
    return float(2 ** (problem.num_qubits - 2))


def optimize_energy(
    problem: prepare.Problem,
    circuit: QuantumCircuit,
    parameters: ParameterVector,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, float, int]:
    calls = 0
    deadline = time.perf_counter() + training_time_budget(problem, spec)
    max_evals = int(os.environ.get("AUTOVQE_MAX_EVALS", str(prepare.MAX_EVALS)))
    best_values: np.ndarray | None = None
    best_energy = float("inf")

    def objective(values: np.ndarray) -> float:
        nonlocal best_energy, best_values, calls
        if time.perf_counter() >= deadline:
            raise RuntimeError("time budget exhausted")
        if calls >= max_evals:
            raise RuntimeError("evaluation budget exhausted")
        candidate = bind_parameters(circuit, parameters, values)
        calls += 1
        energy = prepare.energy_from_circuit(candidate, problem.hamiltonian)
        if energy < best_energy:
            best_energy = energy
            best_values = np.asarray(values, dtype=float).copy()
        return energy

    num_params = len(parameters)
    values = initial_parameters(spec, num_params)
    if num_params == 0:
        return values, objective(values), calls

    optimizer = spec.optimizer.lower()

    for restart_index in range(spec.restarts):
        if calls >= max_evals or time.perf_counter() >= deadline:
            break

        current = initial_parameters(spec, num_params, restart_index=restart_index)
        current_energy = objective(current)
        if current_energy < best_energy:
            best_energy = current_energy
            best_values = current.copy()

        if optimizer == "spsa":
            rng = np.random.default_rng(spec.seed + 1009 * (restart_index + 1))
            base_scale = 0.18 if spec.param_init == "zeros" else 0.12
            stale = 0
            for step in range(1, spec.spsa_steps + 1):
                if calls + 3 > max_evals or time.perf_counter() >= deadline:
                    break
                ak = spec.learning_rate / (step + 2.0) ** 0.602
                ck = base_scale / step**0.101
                delta = rng.choice((-1.0, 1.0), size=num_params)
                plus = current + ck * delta
                minus = current - ck * delta
                plus_energy = objective(plus)
                minus_energy = objective(minus)
                gradient = ((plus_energy - minus_energy) / (2.0 * ck)) * delta
                proposal = current - ak * gradient
                proposal_energy = objective(proposal)
                if proposal_energy <= current_energy + 1e-7:
                    current = proposal
                    current_energy = proposal_energy
                    stale = 0
                else:
                    stale += 1
                if proposal_energy < best_energy:
                    best_energy = proposal_energy
                    best_values = proposal.copy()
                if stale >= 6:
                    break
            continue

        if optimizer == "coordinate":
            rng = np.random.default_rng(spec.seed + 2027 * (restart_index + 1))
            step_size = spec.learning_rate
            stale = 0
            max_steps = max(12, spec.spsa_steps * 2)
            for _ in range(max_steps):
                if calls + 2 > max_evals or time.perf_counter() >= deadline:
                    break
                index = int(rng.integers(num_params))
                delta = step_size * (0.5 + float(rng.random()))
                plus = current.copy()
                minus = current.copy()
                plus[index] += delta
                minus[index] -= delta
                plus_energy = objective(plus)
                minus_energy = objective(minus)
                candidate = current
                candidate_energy = current_energy
                if plus_energy < candidate_energy:
                    candidate = plus
                    candidate_energy = plus_energy
                if minus_energy < candidate_energy:
                    candidate = minus
                    candidate_energy = minus_energy
                if candidate_energy < current_energy:
                    current = candidate
                    current_energy = candidate_energy
                    stale = 0
                    step_size *= 1.03
                else:
                    stale += 1
                    step_size *= 0.72
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_values = current.copy()
                if stale >= max(10, num_params // 2):
                    break
            continue

        if optimizer == "cobyla":
            remaining = max(1, max_evals - calls)
            try:
                result = minimize(
                    objective,
                    current,
                    method="COBYLA",
                    options={
                        "maxiter": remaining,
                        "rhobeg": max(0.05, float(spec.learning_rate)),
                        "tol": 1e-5,
                        "catol": 1e-5,
                        "disp": False,
                    },
                )
            except RuntimeError:
                continue
            if result.fun < best_energy:
                best_energy = float(result.fun)
                best_values = np.asarray(result.x, dtype=float).copy()
            continue

        if optimizer == "powell":
            remaining = max(1, max_evals - calls)
            try:
                result = minimize(
                    objective,
                    current,
                    method="Powell",
                    options={
                        "maxfev": remaining,
                        "maxiter": remaining,
                        "xtol": 1e-4,
                        "ftol": 1e-5,
                        "disp": False,
                    },
                )
            except RuntimeError:
                continue
            if result.fun < best_energy:
                best_energy = float(result.fun)
                best_values = np.asarray(result.x, dtype=float).copy()
            continue

        if optimizer == "de":
            remaining = max(1, max_evals - calls)
            popsize = 5
            population_evals = max(1, popsize * num_params)
            maxiter = max(1, remaining // population_evals - 1)
            bounds = [(-np.pi, np.pi)] * num_params
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=spec.seed + restart_index,
                    polish=False,
                    tol=1e-3,
                    updating="immediate",
                    workers=1,
                )
            except RuntimeError:
                continue
            if result.fun < best_energy:
                best_energy = float(result.fun)
                best_values = np.asarray(result.x, dtype=float).copy()
            continue

        raise ValueError(f"unsupported optimizer: {spec.optimizer}")

    if best_values is None:
        raise RuntimeError("optimizer never evaluated a candidate")
    return best_values, best_energy, calls


def classify_improvement(
    candidate: ExperimentResult,
    incumbent: ExperimentResult | None,
) -> str | None:
    if incumbent is None:
        return "energy"
    if candidate.energy < incumbent.energy - ENERGY_IMPROVEMENT_TOL:
        return "energy"
    if candidate.energy <= incumbent.energy + ENERGY_EQUIVALENCE_TOL:
        if candidate.compression_key < incumbent.compression_key:
            return "compression"
    return None


def target_threshold(reference_energy: float | None) -> float | None:
    if reference_energy is None:
        return None
    if TARGET_REL_ERROR <= 0.0 and TARGET_ABS_ERROR <= 0.0:
        return None
    return max(TARGET_ABS_ERROR, TARGET_REL_ERROR * abs(reference_energy))


def meets_target(result: ExperimentResult, reference_energy: float | None) -> bool:
    threshold = target_threshold(reference_energy)
    if threshold is None:
        return False
    return abs(result.energy - float(reference_energy)) <= threshold


def summarize_run(index: int, phase: str, result: ExperimentResult) -> None:
    print(
        f"[{index:03d}] {phase:<11} {result.status:<7} "
        f"energy={result.energy:.6f} "
        f"twoq={result.metrics['twoq_count']:<3d} "
        f"total={result.metrics['total_gate_count']:<3d} "
        f"depth={result.metrics['depth']:<3d} "
        f"params={result.num_params:<3d} "
        f"{result.description}"
    )


def crash_result(index: int, description: str, total_seconds: float) -> ExperimentResult:
    return ExperimentResult(
        run_id=experiment_id(index, description),
        description=description,
        status="crash",
        energy=0.0,
        overlap=None,
        metrics={
            "singleq_count": 0,
            "twoq_count": 0,
            "total_gate_count": 0,
            "depth": 0,
        },
        num_params=0,
        eval_calls=0,
        total_seconds=total_seconds,
    )


def run_experiment(
    spec: ExperimentSpec,
    problem: prepare.Problem,
    backend_target: prepare.BackendTarget,
    index: int,
) -> ExperimentResult:
    started = time.perf_counter()
    description = spec.description if RUN_MODE == "full" else f"{spec.description} mode={RUN_MODE}"
    try:
        circuit, parameters = build_ansatz(problem, spec)
        best_values, energy, eval_calls = optimize_energy(problem, circuit, parameters, spec)
        final_circuit = bind_parameters(circuit, parameters, best_values)
        _, compiled = prepare.transpile_and_report(final_circuit, backend_target)
        overlap = prepare.overlap_with_reference(final_circuit, problem.reference_state)
        total_seconds = time.perf_counter() - started
        return ExperimentResult(
            run_id=experiment_id(index, description),
            description=description,
            status="discard",
            energy=energy,
            overlap=overlap,
            metrics=compiled,
            num_params=len(parameters),
            eval_calls=eval_calls,
            total_seconds=total_seconds,
        )
    except Exception as exc:
        total_seconds = time.perf_counter() - started
        crash_description = f"{description} | crash={type(exc).__name__}: {exc}"
        return crash_result(index, crash_description, total_seconds)


def build_spec(
    *,
    family: str,
    layers: int,
    optimizer: str,
    param_init: str,
    learning_rate: float,
    seed: int,
    description: str,
    edge_mode: str = "full",
    rotation_mode: str = "full",
    reference_state: str = "zero",
    spsa_steps: int = 24,
    restarts: int = 2,
) -> ExperimentSpec:
    return ExperimentSpec(
        family=family,
        layers=layers,
        optimizer=optimizer,
        param_init=param_init,
        learning_rate=learning_rate,
        seed=seed,
        edge_mode=edge_mode,
        rotation_mode=rotation_mode,
        reference_state=reference_state,
        spsa_steps=spsa_steps,
        restarts=restarts,
        description=description,
    )


def build_baseline_spec() -> ExperimentSpec:
    return build_spec(
        family="hea",
        layers=2,
        optimizer="spsa",
        param_init="small_random",
        learning_rate=0.18,
        seed=prepare.SEED,
        edge_mode="full",
        rotation_mode="full",
        spsa_steps=28,
        restarts=2,
        description="baseline hea layers=2 optimizer=spsa init=small_random",
    )


def model_family_order() -> list[str]:
    if MODEL_CLASS == "weighted_heisenberg_graph":
        return ["heisenberg_hva", "pauli_hva", "hea", "symm", "brick", "tfim_shared", "tfim_factorized"]
    if MODEL_CLASS == "transverse_field_ising":
        return ["tfim_shared", "tfim_factorized", "hea", "pauli_hva", "brick", "symm"]
    if MODEL_CLASS in {"chemistry_or_general_pauli", "general_two_local_pauli"}:
        return ["two_state_excitation", "pauli_hva", "hea", "brick", "symm", "tfim_factorized", "tfim_shared"]
    if MODEL_CLASS == "classical_ising_or_qubo":
        return ["pauli_hva", "tfim_shared", "hea", "brick", "symm"]
    return ["pauli_hva", "hea", "brick", "symm", "tfim_shared", "tfim_factorized"]


def unique_specs(specs: list[ExperimentSpec]) -> list[ExperimentSpec]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[ExperimentSpec] = []
    for spec in specs:
        key = (
            spec.family,
            spec.layers,
            spec.optimizer,
            spec.param_init,
            spec.learning_rate,
            spec.seed,
            spec.edge_mode,
            spec.rotation_mode,
            spec.reference_state,
            spec.spsa_steps,
            spec.restarts,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def build_diversify_pool(problem: prepare.Problem) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    families = model_family_order()
    hard_tfim = MODEL_CLASS == "transverse_field_ising" and problem.num_qubits >= 8
    for family_index, family in enumerate(families):
        layer_values = [1, 2, 3]
        if hard_tfim and family == "tfim_shared":
            layer_values = [1, 2, 3, 4, 5]
        for layers in layer_values:
            if family == "two_state_excitation":
                if layers == 1:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=1,
                            optimizer="cobyla",
                            param_init="zeros",
                            learning_rate=0.40,
                            seed=prepare.SEED + 5,
                            reference_state="hint",
                            spsa_steps=12,
                            restarts=2,
                            description="diversify ansatz=two_state_excitation layers=1 ref=hint reason=XY-excitation-supports",
                        )
                    )
                continue

            if family == "heisenberg_hva":
                for rotation_mode, reference_state, optimizer, init_mode in [
                    ("shared", "dimer_even", "cobyla", "small_random"),
                    ("shared", "dimer_odd", "cobyla", "small_random"),
                    ("shared", "dimer_even", "powell", "small_random"),
                    ("shared", "dimer_even", "de", "small_random"),
                    ("shared", "neel", "spsa", "zeros"),
                    ("edge", "neel", "spsa", "zeros"),
                    ("shared", "zero", "spsa", "small_random"),
                ]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init=init_mode,
                            learning_rate=0.18 if rotation_mode == "shared" else 0.12,
                            seed=prepare.SEED + 7 * layers + len(specs),
                            edge_mode="colored",
                            rotation_mode=rotation_mode,
                            reference_state=reference_state,
                            spsa_steps=26,
                            restarts=2,
                            description=(
                                "diversify ansatz=heisenberg_hva "
                                f"layers={layers} rot={rotation_mode} ref={reference_state} optimizer={optimizer} "
                                "reason=matched-XXYYZZ"
                            ),
                        )
                    )
                continue

            if family == "pauli_hva":
                for rotation_mode, optimizer, init_mode in [
                    ("shared", "spsa", "zeros"),
                    ("term", "spsa", "small_random"),
                    ("term", "cobyla", "small_random"),
                    ("shared", "coordinate", "small_random"),
                ]:
                    reference_state = "hint" if MODEL_CLASS == "chemistry_or_general_pauli" else "zero"
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init=init_mode,
                            learning_rate=0.18 if rotation_mode == "shared" else 0.20,
                            seed=prepare.SEED + 9 * layers + len(specs),
                            rotation_mode=rotation_mode,
                            reference_state=reference_state,
                            spsa_steps=24 if optimizer == "spsa" else 16,
                            restarts=2,
                            description=(
                                f"diversify ansatz=pauli_hva layers={layers} rot={rotation_mode} "
                                f"ref={reference_state} reason=model_class={MODEL_CLASS}"
                            ),
                        )
                    )
                continue

            if family == "hea":
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="spsa",
                        param_init="small_random",
                        learning_rate=0.18,
                        seed=prepare.SEED + 10 * family_index + layers,
                        edge_mode="alternate" if layers % 2 == 0 else "full",
                        rotation_mode="full",
                        description=f"diversify ansatz=hea layers={layers} rot=full edge={'alternate' if layers % 2 == 0 else 'full'}",
                    )
                )
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="cobyla",
                        param_init="small_random",
                        learning_rate=0.25,
                        seed=prepare.SEED + 150 + 10 * family_index + layers,
                        edge_mode="alternate" if layers > 1 else "full",
                        rotation_mode="ry_only",
                        spsa_steps=16,
                        description=f"diversify ansatz=hea layers={layers} rot=ry_only optimizer=cobyla edge={'alternate' if layers > 1 else 'full'}",
                    )
                )
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="coordinate",
                        param_init="zeros",
                        learning_rate=0.24,
                        seed=prepare.SEED + 100 + 10 * family_index + layers,
                        edge_mode="even" if layers == 1 else "alternate",
                        rotation_mode="ry_only",
                        spsa_steps=16,
                        description=f"diversify ansatz=hea layers={layers} rot=ry_only edge={'even' if layers == 1 else 'alternate'}",
                    )
                )
                if layers >= 2:
                    base_seed = prepare.SEED + 300 + 10 * family_index + layers
                    seed_sweep = [base_seed]
                    if MODEL_CLASS == "transverse_field_ising" and layers == 2:
                        seed_sweep.extend([2, 3, 4])
                    for seed in seed_sweep:
                        specs.append(
                            build_spec(
                                family=family,
                                layers=layers,
                                optimizer="cobyla",
                                param_init="small_random",
                                learning_rate=0.40,
                                seed=seed,
                                edge_mode="full",
                                rotation_mode="ry_final",
                                spsa_steps=16,
                                restarts=2,
                                description=f"diversify ansatz=hea layers={layers} rot=ry_final optimizer=cobyla edge=full seed_sweep",
                            )
                        )
                continue

            if family == "brick":
                for optimizer, edge_mode, init_mode in [
                    ("spsa", "full", "small_random"),
                    ("coordinate", "alternate", "zeros"),
                ]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init=init_mode,
                            learning_rate=0.18 if optimizer == "spsa" else 0.22,
                            seed=prepare.SEED + 20 * family_index + 3 * layers + (0 if optimizer == "spsa" else 1),
                            edge_mode=edge_mode,
                            spsa_steps=22 if optimizer == "spsa" else 16,
                            description=f"diversify ansatz=brick layers={layers} optimizer={optimizer} edge={edge_mode}",
                        )
                    )
                continue

            if family == "symm":
                for optimizer, init_mode in [("spsa", "small_random"), ("coordinate", "zeros")]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init=init_mode,
                            learning_rate=0.16 if optimizer == "spsa" else 0.20,
                            seed=prepare.SEED + 30 * family_index + 3 * layers + (0 if optimizer == "spsa" else 1),
                            spsa_steps=24 if optimizer == "spsa" else 18,
                            description=f"diversify ansatz=symm layers={layers} optimizer={optimizer}",
                        )
                    )
                continue

            edge_modes = ["alternate", "full"] if family == "tfim_shared" else ["ends", "alternate"]
            if family == "tfim_shared" and hard_tfim:
                variants = [
                    ("spsa", "alternate", "small_random"),
                    ("cobyla", "alternate", "small_random"),
                    ("cobyla", "full", "small_random"),
                    ("powell", "full", "small_random"),
                    ("de", "full", "small_random"),
                    ("coordinate", "full", "zeros"),
                ]
            else:
                variants = [
                    ("spsa", edge_modes[0], "small_random"),
                    ("cobyla", edge_modes[0], "small_random"),
                    ("coordinate", edge_modes[1], "zeros"),
                ]
            for optimizer, edge_mode, init_mode in variants:
                reference_state = tfim_reference_state(problem) if MODEL_CLASS == "transverse_field_ising" else "zero"
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer=optimizer,
                        param_init=init_mode,
                        learning_rate=0.14 if optimizer == "spsa" else 0.18,
                        seed=prepare.SEED + 40 * family_index + 3 * layers + (0 if optimizer == "spsa" else 1),
                        edge_mode=edge_mode,
                        reference_state=reference_state,
                        spsa_steps=20 if optimizer == "spsa" else 14,
                        description=f"diversify ansatz={family} layers={layers} optimizer={optimizer} edge={edge_mode} ref={reference_state}",
                    )
                )

    return unique_specs(specs)


def complexify_families(best_family: str) -> list[str]:
    order = [best_family]
    for family in model_family_order():
        if family not in order:
            order.append(family)
    return order[:3]


def build_complexify_pool(best_family: str, problem: prepare.Problem, seed_offset: int = 0) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    hard_tfim = MODEL_CLASS == "transverse_field_ising" and problem.num_qubits >= 8
    for family_index, family in enumerate(complexify_families(best_family)):
        for layers in [4, 5, 6, 7, 8]:
            if family == "heisenberg_hva":
                for rotation_mode, reference_state, optimizer in [
                    ("shared", "dimer_even", "cobyla"),
                    ("shared", "dimer_odd", "cobyla"),
                    ("shared", "dimer_even", "powell"),
                    ("shared", "dimer_even", "de"),
                    ("edge", "neel", "spsa"),
                ]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init="zeros",
                            learning_rate=0.12 if rotation_mode == "edge" else 0.16,
                            seed=prepare.SEED + seed_offset + 35 * family_index + 5 * layers + len(specs),
                            edge_mode="colored",
                            rotation_mode=rotation_mode,
                            reference_state=reference_state,
                            spsa_steps=28,
                            restarts=2,
                            description=f"complexify ansatz=heisenberg_hva layers={layers} rot={rotation_mode} ref={reference_state} optimizer={optimizer} reason=matched-XXYYZZ",
                        )
                    )
                continue

            if family == "pauli_hva":
                for rotation_mode in ["shared", "term"]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="spsa",
                            param_init="small_random",
                            learning_rate=0.10 if rotation_mode == "term" else 0.14,
                            seed=prepare.SEED + seed_offset + 45 * family_index + 5 * layers + len(specs),
                            rotation_mode=rotation_mode,
                            spsa_steps=26,
                            restarts=2,
                            description=f"complexify ansatz=pauli_hva layers={layers} rot={rotation_mode} reason=model_class={MODEL_CLASS}",
                        )
                    )
                continue

            if family == "hea":
                for rotation_mode, edge_mode in [
                    ("full", "full"),
                    ("full", "alternate"),
                    ("ry_only", "alternate"),
                ]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="spsa",
                            param_init="small_random",
                            learning_rate=0.14,
                            seed=prepare.SEED + seed_offset + 50 * family_index + 5 * layers + len(specs),
                            edge_mode=edge_mode,
                            rotation_mode=rotation_mode,
                            spsa_steps=28,
                            restarts=2,
                            description=f"complexify ansatz=hea layers={layers} rot={rotation_mode} edge={edge_mode}",
                        )
                    )
                continue

            if family == "brick":
                for edge_mode in ["full", "alternate", "odd"]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="spsa",
                            param_init="small_random",
                            learning_rate=0.14,
                            seed=prepare.SEED + seed_offset + 70 * family_index + 5 * layers + len(specs),
                            edge_mode=edge_mode,
                            spsa_steps=28,
                            restarts=2,
                            description=f"complexify ansatz=brick layers={layers} edge={edge_mode}",
                        )
                    )
                continue

            if family == "symm":
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="spsa",
                        param_init="small_random",
                        learning_rate=0.12,
                        seed=prepare.SEED + seed_offset + 90 * family_index + 5 * layers,
                        spsa_steps=26,
                        restarts=3,
                        description=f"complexify ansatz=symm layers={layers}",
                    )
                )
                continue

            if family == "tfim_shared" and hard_tfim:
                variants = [
                    ("cobyla", "alternate"),
                    ("cobyla", "full"),
                    ("powell", "full"),
                    ("de", "full"),
                    ("spsa", "full"),
                ]
            else:
                variants = [("spsa", edge_mode) for edge_mode in ["full", "alternate", "random_2"]]
            for optimizer, edge_mode in variants:
                reference_state = tfim_reference_state(problem) if MODEL_CLASS == "transverse_field_ising" else "zero"
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer=optimizer,
                        param_init="small_random",
                        learning_rate=0.12 if optimizer == "spsa" else 0.18,
                        seed=prepare.SEED + seed_offset + 110 * family_index + 5 * layers + len(specs),
                        edge_mode=edge_mode,
                        reference_state=reference_state,
                        spsa_steps=26 if optimizer == "spsa" else 18,
                        restarts=3 if family == "tfim_shared" else 2,
                        description=f"complexify ansatz={family} layers={layers} optimizer={optimizer} edge={edge_mode} ref={reference_state}",
                    )
                )

    return unique_specs(specs)


def compression_family_order(best_family: str) -> list[str]:
    mapping = {
        "heisenberg_hva": ["heisenberg_hva", "pauli_hva", "symm", "brick"],
        "pauli_hva": ["pauli_hva", "hea", "brick", "symm"],
        "hea": ["hea", "brick", "symm"],
        "brick": ["brick", "hea", "symm"],
        "symm": ["symm", "hea", "brick"],
        "tfim_shared": ["tfim_shared", "tfim_factorized", "brick"],
        "tfim_factorized": ["tfim_factorized", "tfim_shared", "brick"],
    }
    return mapping.get(best_family, [best_family, "hea", "brick"])


def build_compression_pool(best_spec: ExperimentSpec, problem: prepare.Problem, round_index: int = 0) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    min_layers = max(1, best_spec.layers - 3)
    max_layers = max(2, best_spec.layers)
    layer_values = list(range(max_layers, min_layers - 1, -1))
    seed_base = prepare.SEED + 1000 + 200 * round_index

    for family_index, family in enumerate(compression_family_order(best_spec.family)):
        for layers in layer_values:
            if family == "heisenberg_hva":
                variants = [
                    ("shared", "dimer_even"),
                    ("shared", "dimer_odd"),
                    ("shared", "neel"),
                    ("shared", "zero"),
                    ("edge", "neel"),
                ]
                for rotation_mode, reference_state in variants:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="coordinate",
                            param_init="small_random",
                            learning_rate=0.16 if rotation_mode == "edge" else 0.20,
                            seed=seed_base + 30 * family_index + 5 * layers + len(specs),
                            edge_mode="colored",
                            rotation_mode=rotation_mode,
                            reference_state=reference_state,
                            spsa_steps=14,
                            restarts=2,
                            description=(
                                "compress ansatz=heisenberg_hva "
                                f"layers={layers} rot={rotation_mode} ref={reference_state} "
                                "reason=matched-XXYYZZ"
                            ),
                        )
                    )
                continue

            if family == "pauli_hva":
                for rotation_mode in ["shared", "term"]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="coordinate",
                            param_init="small_random",
                            learning_rate=0.12 if rotation_mode == "term" else 0.18,
                            seed=seed_base + 35 * family_index + 5 * layers + len(specs),
                            rotation_mode=rotation_mode,
                            spsa_steps=14,
                            restarts=2,
                            description=f"compress ansatz=pauli_hva layers={layers} rot={rotation_mode} reason=model_class={MODEL_CLASS}",
                        )
                    )
                continue

            if family == "hea":
                variants = [
                    ("shared", "random_1"),
                    ("shared", "even"),
                    ("ry_only", "alternate"),
                    ("ry_only", "odd"),
                ]
                for rotation_mode, edge_mode in variants:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="coordinate",
                            param_init="small_random",
                            learning_rate=0.20,
                            seed=seed_base + 40 * family_index + 5 * layers + len(specs),
                            edge_mode=edge_mode,
                            rotation_mode=rotation_mode,
                            spsa_steps=16,
                            restarts=2,
                            description=f"compress ansatz=hea layers={layers} rot={rotation_mode} edge={edge_mode}",
                        )
                    )
                continue

            if family == "brick":
                for edge_mode in ["odd", "even", "alternate"]:
                    specs.append(
                        build_spec(
                            family=family,
                            layers=layers,
                            optimizer="coordinate",
                            param_init="small_random",
                            learning_rate=0.18,
                            seed=seed_base + 60 * family_index + 5 * layers + len(specs),
                            edge_mode=edge_mode,
                            spsa_steps=16,
                            restarts=2,
                            description=f"compress ansatz=brick layers={layers} edge={edge_mode}",
                        )
                    )
                continue

            if family == "symm":
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="coordinate",
                        param_init="small_random",
                        learning_rate=0.16,
                        seed=seed_base + 80 * family_index + 5 * layers,
                        spsa_steps=14,
                        restarts=3,
                        description=f"compress ansatz=symm layers={layers}",
                    )
                )
                continue

            edge_variants = ["ends", "even", "alternate"] if family == "tfim_shared" else ["ends", "random_1", "alternate"]
            for edge_mode in edge_variants:
                reference_state = tfim_reference_state(problem) if MODEL_CLASS == "transverse_field_ising" else "zero"
                specs.append(
                    build_spec(
                        family=family,
                        layers=layers,
                        optimizer="coordinate",
                        param_init="small_random",
                        learning_rate=0.14,
                        seed=seed_base + 100 * family_index + 5 * layers + len(specs),
                        edge_mode=edge_mode,
                        reference_state=reference_state,
                        spsa_steps=14,
                        restarts=3 if family == "tfim_shared" else 2,
                        description=f"compress ansatz={family} layers={layers} edge={edge_mode}",
                    )
                )

    return unique_specs(specs)


def pop_next_untried(
    pool: list[ExperimentSpec],
    tried: set[tuple[Any, ...]],
) -> ExperimentSpec | None:
    while pool:
        spec = pool.pop(0)
        key = (
            spec.family,
            spec.layers,
            spec.optimizer,
            spec.param_init,
            spec.learning_rate,
            spec.seed,
            spec.edge_mode,
            spec.rotation_mode,
            spec.reference_state,
            spec.spsa_steps,
            spec.restarts,
        )
        if key in tried:
            continue
        tried.add(key)
        return spec
    return None


def format_best_summary(best_result: ExperimentResult, reference_energy: float | None) -> str:
    metrics = [
        ("energy", best_result.energy),
        ("reference_energy", reference_energy),
        ("overlap", best_result.overlap),
        ("singleq_count", best_result.metrics["singleq_count"]),
        ("twoq_count", best_result.metrics["twoq_count"]),
        ("total_gate_count", best_result.metrics["total_gate_count"]),
        ("depth", best_result.metrics["depth"]),
        ("num_params", best_result.num_params),
        ("eval_calls", best_result.eval_calls),
        ("total_seconds", best_result.total_seconds),
    ]
    return prepare.format_summary(metrics)


def main() -> None:
    global MODEL_CLASS
    ensure_results_header()
    problem = prepare.load_problem(PROBLEM_PATH)
    backend_target = prepare.build_backend_target(problem)
    if MODEL_CLASS == "auto":
        MODEL_CLASS = quick_model_class(problem)

    best_result: ExperimentResult | None = None
    best_spec: ExperimentSpec | None = None
    tried: set[tuple[Any, ...]] = set()
    diversify_pool = build_diversify_pool(problem)
    complexify_pool: list[ExperimentSpec] = []
    compression_pool: list[ExperimentSpec] = []
    compression_round = 0

    phase = "baseline"
    experiment_count = 0
    runs_since_energy_keep = 0
    runs_since_compression_keep = 0
    families_seen: set[str] = set()
    tried_more_complex = False
    complexify_attempts = 0
    target_reached_at: int | None = None

    while True:
        if phase == "baseline":
            spec = build_baseline_spec()
            phase_label = "baseline"
        elif phase == "diversify":
            spec = pop_next_untried(diversify_pool, tried)
            if spec is None:
                phase = "complexify"
                runs_since_energy_keep = 0
                complexify_attempts = 0
                continue
            phase_label = "diversify"
        elif phase == "complexify":
            if not complexify_pool:
                family = best_spec.family if best_spec is not None else "hea"
                complexify_pool = build_complexify_pool(family, problem)
            spec = pop_next_untried(complexify_pool, tried)
            if spec is None:
                phase = "compress"
                continue
            phase_label = "complexify"
        else:
            if best_spec is None:
                best_spec = build_baseline_spec()
            if not compression_pool:
                compression_pool = build_compression_pool(best_spec, problem, round_index=compression_round)
                compression_round += 1
            spec = pop_next_untried(compression_pool, tried)
            if spec is None:
                compression_pool = []
                continue
            phase_label = "compress"

        experiment_count += 1
        families_seen.add(spec.family)
        previous_best = best_result
        improvement_kind: str | None = None

        result = run_experiment(spec, problem, backend_target, experiment_count)
        if result.status != "crash":
            improvement_kind = classify_improvement(result, previous_best)
            if improvement_kind is not None:
                result = ExperimentResult(**{**result.__dict__, "status": "keep"})
                best_result = result
                best_spec = spec

        if previous_best is None:
            runs_since_energy_keep = 0
            runs_since_compression_keep = 0
        elif improvement_kind == "energy":
            runs_since_energy_keep = 0
            runs_since_compression_keep = 0
        elif improvement_kind == "compression":
            runs_since_energy_keep += 1
            runs_since_compression_keep = 0
        else:
            runs_since_energy_keep += 1
            if phase == "compress":
                runs_since_compression_keep += 1

        log_result(result)
        summarize_run(experiment_count, phase_label, result)

        if STOP_AT_TARGET and best_result is not None and meets_target(best_result, problem.reference_energy):
            if target_reached_at is None:
                target_reached_at = experiment_count
                phase = "compress"
            if experiment_count - target_reached_at >= TARGET_EXTRA_COMPRESS:
                break

        if phase == "baseline":
            phase = "diversify"
            if MAX_EXPERIMENTS and experiment_count >= MAX_EXPERIMENTS:
                break
            continue

        if phase == "diversify":
            if len(families_seen) >= 4 and runs_since_energy_keep >= 10:
                phase = "complexify"
                runs_since_energy_keep = 0
                complexify_attempts = 0
            if MAX_EXPERIMENTS and experiment_count >= MAX_EXPERIMENTS:
                break
            continue

        if phase == "complexify":
            tried_more_complex = True
            complexify_attempts += 1
            if complexify_attempts >= 16 and runs_since_energy_keep >= 12:
                phase = "compress"
            if MAX_EXPERIMENTS and experiment_count >= MAX_EXPERIMENTS:
                break
            continue

        if MAX_EXPERIMENTS and experiment_count >= MAX_EXPERIMENTS:
            break

        if tried_more_complex and experiment_count >= MIN_EXPERIMENTS:
            if runs_since_energy_keep >= EXHAUSTION_PATIENCE and runs_since_compression_keep >= EXHAUSTION_PATIENCE:
                break

    if best_result is None:
        raise RuntimeError("all experiments crashed")

    print(format_best_summary(best_result, problem.reference_energy))


if __name__ == "__main__":
    main()
