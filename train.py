from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit.library import XXPlusYYGate
from qiskit.circuit import ParameterVector, QuantumCircuit
from scipy.optimize import dual_annealing, minimize

import prepare


DEFAULT_PROBLEM_PATH = "examples/h2_2q.json"
PROBLEM_PATH = Path(os.environ.get("AUTOVQE_PROBLEM_PATH", DEFAULT_PROBLEM_PATH))
RESULTS_PATH = Path(os.environ.get("AUTOVQE_RESULTS_PATH", "results.tsv"))
RUN_MODE = os.environ.get("AUTOVQE_RUN_MODE", "full")
MODEL_CLASS = os.environ.get("AUTOVQE_MODEL_CLASS", "auto")
MAX_EXPERIMENTS = int(os.environ.get("AUTOVQE_MAX_EXPERIMENTS", "0"))
TARGET_REL_ERROR = float(os.environ.get("AUTOVQE_TARGET_REL_ERROR", "0") or "0")
TARGET_ABS_ERROR = float(os.environ.get("AUTOVQE_TARGET_ABS_ERROR", "0") or "0")
STOP_AT_TARGET = os.environ.get("AUTOVQE_STOP_AT_TARGET", "0") == "1"
ENERGY_IMPROVEMENT_TOL = 1e-6
ENERGY_EQUIVALENCE_TOL = 5e-4
MIN_ADMISSIBLE_PARAMS = 2


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


@dataclass(frozen=True)
class OperatorFeatures:
    max_locality: int
    z_only: bool
    has_x_field: bool
    has_zz_edges: bool
    matched_xy_fraction: float
    matched_xyz_fraction: float
    has_reference_hint: bool


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
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def label_position_to_qubit(label: str, position: int) -> int:
    return len(label) - position - 1


def label_active_qubits(label: str) -> list[tuple[int, str]]:
    return [
        (label_position_to_qubit(label, position), pauli)
        for position, pauli in enumerate(label)
        if pauli != "I"
    ]


def chain_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
    if problem.coupling_map:
        seen: set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []
        for left, right in problem.coupling_map:
            edge = tuple(sorted((left, right)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
        if edges:
            return edges
    return [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]


def hint_bits(problem: prepare.Problem) -> list[int] | None:
    if not isinstance(problem.initial_state_hint, list):
        return None
    bits = [int(bit) for bit in problem.initial_state_hint]
    if len(bits) != problem.num_qubits:
        return None
    return bits


def hint_state_index(problem: prepare.Problem) -> int | None:
    bits = hint_bits(problem)
    if bits is None:
        return None
    index = 0
    for qubit, bit in enumerate(bits):
        if bit:
            index |= 1 << qubit
    return index


def support_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for label in problem.hamiltonian.paulis.to_labels():
        active = [qubit for qubit, _ in label_active_qubits(label)]
        if len(active) == 2:
            edges.add(tuple(sorted(active)))
    return sorted(edges) or chain_edges(problem)


def operator_features(problem: prepare.Problem) -> OperatorFeatures:
    labels = problem.hamiltonian.paulis.to_labels()
    all_ops: list[str] = []
    single_ops: set[str] = set()
    two_groups: dict[tuple[int, int], set[str]] = {}
    max_locality = 0

    for label in labels:
        active = label_active_qubits(label)
        max_locality = max(max_locality, len(active))
        ops = "".join(pauli for _, pauli in active)
        if ops:
            all_ops.append(ops)
        if len(active) == 1:
            single_ops.add(active[0][1])
        elif len(active) == 2:
            edge = tuple(sorted(qubit for qubit, _ in active))
            two_groups.setdefault(edge, set()).add(ops)

    two_count = max(1, len(two_groups))
    matched_xy = sum({"XX", "YY"}.issubset(ops) for ops in two_groups.values())
    matched_xyz = sum({"XX", "YY", "ZZ"}.issubset(ops) for ops in two_groups.values())
    return OperatorFeatures(
        max_locality=max_locality,
        z_only=bool(all_ops) and all(set(ops) <= {"Z"} for ops in all_ops),
        has_x_field="X" in single_ops,
        has_zz_edges=any("ZZ" in ops for ops in two_groups.values()),
        matched_xy_fraction=matched_xy / two_count,
        matched_xyz_fraction=matched_xyz / two_count,
        has_reference_hint=hint_bits(problem) is not None,
    )


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


def select_edges(
    edges: list[tuple[int, int]],
    edge_mode: str,
    layers: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if edge_mode == "full":
        return [list(edges) for _ in range(layers)]
    if edge_mode == "ends":
        chosen = [edges[0], edges[-1]] if len(edges) > 1 else list(edges)
        return [chosen for _ in range(layers)]
    if edge_mode == "even":
        return [edges[::2] for _ in range(layers)]
    if edge_mode == "odd":
        return [edges[1::2] or edges[:1] for _ in range(layers)]
    if edge_mode == "alternate":
        return [edges[layer % 2 :: 2] or edges for layer in range(layers)]
    if edge_mode.startswith("random_"):
        rng = np.random.default_rng(seed)
        count = max(1, int(edge_mode.split("_", 1)[1]))
        return [
            [edges[index] for index in sorted(rng.choice(len(edges), size=min(count, len(edges)), replace=False))]
            for _ in range(layers)
        ]
    return [list(edges) for _ in range(layers)]


def bipartition(num_qubits: int, edges: list[tuple[int, int]]) -> dict[int, int] | None:
    graph = {qubit: [] for qubit in range(num_qubits)}
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)
    colors: dict[int, int] = {}
    stack: list[int] = []
    for start in range(num_qubits):
        if start in colors or not graph[start]:
            continue
        colors[start] = 0
        stack.append(start)
        while stack:
            node = stack.pop()
            for neighbor in graph[node]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    stack.append(neighbor)
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
            raise ValueError("hint reference requested but no initial_state_hint exists")
        for qubit, bit in enumerate(problem.initial_state_hint):
            if int(bit):
                circuit.x(qubit)
        return
    if spec.reference_state == "neel":
        colors = bipartition(problem.num_qubits, support_edges(problem))
        if colors is None:
            raise ValueError("Neel reference needs a bipartite support graph")
        for qubit, color in colors.items():
            if color:
                circuit.x(qubit)
        return
    if spec.reference_state in {"dimer_even", "dimer_odd"}:
        start = 0 if spec.reference_state == "dimer_even" else 1
        for left in range(start, problem.num_qubits - 1, 2):
            right = left + 1
            circuit.x(right)
            circuit.h(left)
            circuit.cx(left, right)
            circuit.z(left)
        return
    raise ValueError(f"unsupported reference state: {spec.reference_state}")


def tfim_reference_state(problem: prepare.Problem) -> str:
    x_field = 0.0
    for label, coeff in zip(problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs, strict=True):
        active = label_active_qubits(label)
        if len(active) == 1 and active[0][1] == "X":
            x_field += float(np.real(coeff))
    return "minus" if x_field > 0.0 else "plus"


def build_hea_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    if spec.rotation_mode == "shared":
        params_per_layer = 2
    elif spec.rotation_mode == "ry_only":
        params_per_layer = problem.num_qubits
    else:
        params_per_layer = 2 * problem.num_qubits
    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    cursor = 0
    for layer in range(spec.layers):
        if spec.rotation_mode == "shared":
            angle_y = theta[cursor]
            angle_z = theta[cursor + 1]
            cursor += 2
            for qubit in range(problem.num_qubits):
                circuit.ry(angle_y, qubit)
                circuit.rz(angle_z, qubit)
        elif spec.rotation_mode == "ry_only":
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                cursor += 1
        else:
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                circuit.rz(theta[cursor + 1], qubit)
                cursor += 2
        for control, target in edges_by_layer[layer]:
            circuit.cx(control, target)
    return circuit, theta


def build_brick_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    return build_hea_ansatz(
        problem,
        ExperimentSpec(**{**spec.__dict__, "rotation_mode": spec.rotation_mode if spec.rotation_mode != "full" else "ry_only"}),
    )


def build_symm_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", spec.layers * 2)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    for layer in range(spec.layers):
        angle_y = theta[2 * layer]
        angle_z = theta[2 * layer + 1]
        for qubit in range(problem.num_qubits):
            circuit.ry(angle_y, qubit)
        for control, target in select_edges(chain_edges(problem), "alternate", spec.layers, spec.seed)[layer]:
            circuit.cx(control, target)
        for qubit in range(problem.num_qubits):
            circuit.rz(angle_z, qubit)
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


def add_u1_exchange_block(circuit: QuantumCircuit, left: int, right: int, angle: Any) -> None:
    circuit.append(XXPlusYYGate(2.0 * angle, 0.0), [left, right])


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
            raise ValueError(f"unsupported Pauli in {label}")
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
    return [
        (label, float(np.real(coeff)))
        for label, coeff in zip(problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs, strict=True)
        if abs(float(np.real(coeff))) > 1e-12 and set(label) != {"I"}
    ]


def connected_hint_states(problem: prepare.Problem) -> list[tuple[int, float]]:
    reference = hint_state_index(problem)
    if reference is None:
        return []
    reference_weight = reference.bit_count()
    strengths: dict[int, float] = {}
    for label, coeff in zip(problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs, strict=True):
        mask = 0
        for position, pauli in enumerate(label):
            if pauli in {"X", "Y"}:
                mask ^= 1 << label_position_to_qubit(label, position)
        if not mask:
            continue
        target = reference ^ mask
        if target.bit_count() == reference_weight:
            strengths[target] = strengths.get(target, 0.0) + abs(float(np.real(coeff)))
    return sorted(strengths.items(), key=lambda item: item[1], reverse=True)


def u1_exchange_edges(problem: prepare.Problem, edge_mode: str) -> list[tuple[int, int]]:
    bits = hint_bits(problem)
    if bits is None:
        return chain_edges(problem)
    occupied = [qubit for qubit, bit in enumerate(bits) if bit]
    virtual = [qubit for qubit, bit in enumerate(bits) if not bit]
    if edge_mode == "occupied_virtual" and occupied and virtual:
        return [(occ, virt) for virt in virtual for occ in occupied]
    if edge_mode == "frontier" and occupied and virtual:
        reference = hint_state_index(problem)
        assert reference is not None
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for target, _ in connected_hint_states(problem)[:24]:
            lost = [qubit for qubit in occupied if not ((target >> qubit) & 1)]
            gained = [qubit for qubit in virtual if (target >> qubit) & 1]
            for left, right in zip(lost, gained, strict=False):
                edge = tuple(sorted((left, right)))
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
        if edges:
            return edges
    if edge_mode == "full":
        return [(left, right) for left in range(problem.num_qubits) for right in range(left + 1, problem.num_qubits)]
    return chain_edges(problem)


def build_u1_exchange_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges = u1_exchange_edges(problem, spec.edge_mode)
    params_per_layer = problem.num_qubits + len(edges)
    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        for qubit in range(problem.num_qubits):
            circuit.rz(theta[cursor], qubit)
            cursor += 1
        for left, right in edges:
            add_u1_exchange_block(circuit, left, right, theta[cursor])
            cursor += 1
    return circuit, theta


def build_pauli_hva_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    if spec.rotation_mode != "term":
        raise ValueError("pauli_hva requires independent term parameters; shared exp(-i theta H) is disallowed")
    terms = hamiltonian_pauli_terms(problem)
    theta = ParameterVector("theta", spec.layers * len(terms))
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        for label, coeff in terms:
            add_pauli_evolution(circuit, label, theta[cursor] * coeff)
            cursor += 1
    return circuit, theta


def heisenberg_edge_weights(problem: prepare.Problem) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], list[float]] = {}
    for label, coeff in zip(problem.hamiltonian.paulis.to_labels(), problem.hamiltonian.coeffs, strict=True):
        active = label_active_qubits(label)
        if len(active) != 2:
            continue
        op = "".join(pauli for _, pauli in active)
        if op not in {"XX", "YY", "ZZ"}:
            continue
        edge = tuple(sorted(qubit for qubit, _ in active))
        weights.setdefault(edge, []).append(float(np.real(coeff)))
    return {edge: float(np.mean(values)) for edge, values in weights.items()}


def build_heisenberg_hva_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges = support_edges(problem)
    groups = edge_color_groups(edges) if spec.edge_mode == "colored" else select_edges(edges, spec.edge_mode, 1, spec.seed)
    params_per_layer = len(groups) if spec.rotation_mode == "shared" else sum(len(group) for group in groups)
    theta = ParameterVector("theta", spec.layers * params_per_layer)
    weights = heisenberg_edge_weights(problem)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        for group in groups:
            if spec.rotation_mode == "shared":
                shared = theta[cursor]
                cursor += 1
                for control, target in group:
                    add_heisenberg_block(circuit, control, target, shared * weights.get((control, target), 1.0))
            else:
                for control, target in group:
                    add_heisenberg_block(circuit, control, target, theta[cursor] * weights.get((control, target), 1.0))
                    cursor += 1
    return circuit, theta


def build_tfim_shared_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", 2 * spec.layers)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    for layer in range(spec.layers):
        gamma = theta[2 * layer]
        beta = theta[2 * layer + 1]
        for control, target in edges_by_layer[layer]:
            add_zz_block(circuit, control, target, gamma)
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)
    return circuit, theta


def build_tfim_factorized_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    theta = ParameterVector("theta", sum(len(edges) for edges in edges_by_layer) + spec.layers * problem.num_qubits)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for layer_edges in edges_by_layer:
        for control, target in layer_edges:
            add_zz_block(circuit, control, target, theta[cursor])
            cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * theta[cursor], qubit)
            cursor += 1
    return circuit, theta


def build_tfim_colored_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    groups = edge_color_groups(chain_edges(problem))
    params_per_layer = len(groups) + 1
    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        for group in groups:
            angle = theta[cursor]
            cursor += 1
            for control, target in group:
                add_zz_block(circuit, control, target, angle)
        beta = theta[cursor]
        cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)
    return circuit, theta


def two_qubit_pauli_label(num_qubits: int, left: int, left_op: str, right: int, right_op: str) -> str:
    label = ["I"] * num_qubits
    label[num_qubits - left - 1] = left_op
    label[num_qubits - right - 1] = right_op
    return "".join(label)


def build_tfim_counterdiabatic_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges = chain_edges(problem)
    groups = edge_color_groups(edges)
    edge_factorized = spec.edge_mode == "edge"
    params_per_layer = (3 * len(edges) + problem.num_qubits) if edge_factorized else (3 * len(groups) + 1)
    theta = ParameterVector("theta", spec.layers * params_per_layer)
    circuit = QuantumCircuit(problem.num_qubits)
    prepare_reference_state(circuit, problem, spec)
    cursor = 0
    for _ in range(spec.layers):
        if edge_factorized:
            for control, target in edges:
                add_zz_block(circuit, control, target, theta[cursor])
                cursor += 1
            for control, target in edges:
                add_pauli_evolution(
                    circuit,
                    two_qubit_pauli_label(problem.num_qubits, control, "Y", target, "Z"),
                    theta[cursor],
                )
                cursor += 1
                add_pauli_evolution(
                    circuit,
                    two_qubit_pauli_label(problem.num_qubits, control, "Z", target, "Y"),
                    theta[cursor],
                )
                cursor += 1
            for qubit in range(problem.num_qubits):
                circuit.rx(2.0 * theta[cursor], qubit)
                cursor += 1
            continue

        for group in groups:
            angle = theta[cursor]
            cursor += 1
            for control, target in group:
                add_zz_block(circuit, control, target, angle)
        for group in groups:
            angle = theta[cursor]
            cursor += 1
            for control, target in group:
                add_pauli_evolution(
                    circuit,
                    two_qubit_pauli_label(problem.num_qubits, control, "Y", target, "Z"),
                    angle,
                )
        for group in groups:
            angle = theta[cursor]
            cursor += 1
            for control, target in group:
                add_pauli_evolution(
                    circuit,
                    two_qubit_pauli_label(problem.num_qubits, control, "Z", target, "Y"),
                    angle,
                )
        beta = theta[cursor]
        cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)
    return circuit, theta


def build_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    if spec.family == "hea":
        return build_hea_ansatz(problem, spec)
    if spec.family == "brick":
        return build_brick_ansatz(problem, spec)
    if spec.family == "symm":
        return build_symm_ansatz(problem, spec)
    if spec.family == "u1_exchange":
        return build_u1_exchange_ansatz(problem, spec)
    if spec.family == "pauli_hva":
        return build_pauli_hva_ansatz(problem, spec)
    if spec.family == "heisenberg_hva":
        return build_heisenberg_hva_ansatz(problem, spec)
    if spec.family == "tfim_shared":
        return build_tfim_shared_ansatz(problem, spec)
    if spec.family == "tfim_factorized":
        return build_tfim_factorized_ansatz(problem, spec)
    if spec.family == "tfim_colored":
        return build_tfim_colored_ansatz(problem, spec)
    if spec.family == "tfim_counterdiabatic":
        return build_tfim_counterdiabatic_ansatz(problem, spec)
    raise ValueError(f"unsupported ansatz family: {spec.family}")


def initial_parameters(spec: ExperimentSpec, num_params: int, restart_index: int) -> np.ndarray:
    if spec.param_init == "zeros":
        return np.zeros(num_params)
    rng = np.random.default_rng(spec.seed + 101 * restart_index)
    if spec.param_init == "small_random":
        scale = 0.15
    elif spec.param_init == "medium_random":
        scale = 0.25
    elif spec.param_init == "large_random":
        scale = 0.8
    else:
        scale = 0.5
    return rng.uniform(-scale, scale, size=num_params)


def bind_parameters(circuit: QuantumCircuit, parameters: ParameterVector, values: np.ndarray) -> QuantumCircuit:
    return circuit.assign_parameters(dict(zip(parameters, values, strict=True)), inplace=False)


def training_time_budget(problem: prepare.Problem, spec: ExperimentSpec) -> float:
    if spec.time_budget_seconds is not None and RUN_MODE != "smoke":
        return spec.time_budget_seconds
    if os.environ.get("AUTOVQE_EXPERIMENT_SECONDS"):
        return float(os.environ["AUTOVQE_EXPERIMENT_SECONDS"])
    return float(2 ** max(0, problem.num_qubits - 2))


def optimize_energy(
    problem: prepare.Problem,
    circuit: QuantumCircuit,
    parameters: ParameterVector,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, float, int]:
    max_evals = int(os.environ.get("AUTOVQE_MAX_EVALS", str(prepare.MAX_EVALS)))
    deadline = time.perf_counter() + training_time_budget(problem, spec)
    calls = 0
    best_values: np.ndarray | None = None
    best_energy = float("inf")

    def objective(values: np.ndarray) -> float:
        nonlocal calls, best_energy, best_values
        if calls >= max_evals or time.perf_counter() >= deadline:
            raise RuntimeError("budget exhausted")
        calls += 1
        energy = prepare.energy_from_circuit(bind_parameters(circuit, parameters, values), problem.hamiltonian)
        if energy < best_energy:
            best_energy = energy
            best_values = np.asarray(values, dtype=float).copy()
        threshold = target_threshold(problem.reference_energy)
        if STOP_AT_TARGET and threshold is not None and problem.reference_energy is not None:
            if abs(energy - problem.reference_energy) <= threshold:
                raise RuntimeError("target reached")
        return energy

    num_params = len(parameters)
    if num_params == 0:
        values = np.zeros(0)
        return values, objective(values), calls

    optimizer = spec.optimizer.lower()
    for restart in range(spec.restarts):
        if calls >= max_evals or time.perf_counter() >= deadline:
            break
        current = initial_parameters(spec, num_params, restart)
        try:
            current_energy = objective(current)
        except RuntimeError:
            break

        if optimizer == "spsa":
            rng = np.random.default_rng(spec.seed + 1009 * restart)
            for step in range(1, spec.spsa_steps + 1):
                if calls + 3 > max_evals or time.perf_counter() >= deadline:
                    break
                ck = 0.14 / step**0.101
                ak = spec.learning_rate / (step + 2.0) ** 0.602
                delta = rng.choice((-1.0, 1.0), size=num_params)
                try:
                    plus = objective(current + ck * delta)
                    minus = objective(current - ck * delta)
                    gradient = ((plus - minus) / (2.0 * ck)) * delta
                    proposal = current - ak * gradient
                    proposal_energy = objective(proposal)
                except RuntimeError:
                    break
                if proposal_energy < current_energy:
                    current = proposal
                    current_energy = proposal_energy
            continue

        if optimizer == "anneal":
            bounds = [(-np.pi, np.pi)] * num_params
            try:
                dual_annealing(
                    objective,
                    bounds,
                    maxfun=max(1, max_evals - calls),
                    seed=spec.seed + 1009 * restart,
                    no_local_search=False,
                )
            except RuntimeError:
                pass
            continue

        if optimizer == "cobyla_lbfgsb":
            try:
                minimize(
                    objective,
                    current,
                    method="COBYLA",
                    options={
                        "maxiter": min(12000, max(1, max_evals - calls)),
                        "rhobeg": max(0.05, spec.learning_rate),
                        "disp": False,
                    },
                )
            except RuntimeError:
                pass
            if calls < max_evals and time.perf_counter() < deadline and best_values is not None:
                try:
                    minimize(
                        objective,
                        best_values,
                        method="Powell",
                        options={"maxfev": max(1, max_evals - calls), "maxiter": max(1, max_evals - calls), "disp": False},
                    )
                except RuntimeError:
                    pass
            if calls < max_evals and time.perf_counter() < deadline and best_values is not None:
                try:
                    minimize(
                        objective,
                        best_values,
                        method="L-BFGS-B",
                        options={"maxfun": max(1, max_evals - calls), "maxiter": max(1, max_evals - calls), "disp": False},
                    )
                except RuntimeError:
                    pass
            continue

        if optimizer == "powell":
            method = "Powell"
        elif optimizer == "nelder":
            method = "Nelder-Mead"
        elif optimizer == "lbfgsb":
            method = "L-BFGS-B"
        else:
            method = "COBYLA"
        options = {"maxiter": max(1, max_evals - calls), "disp": False}
        if method == "Powell":
            options = {"maxfev": max(1, max_evals - calls), "maxiter": max(1, max_evals - calls), "disp": False}
        elif method == "Nelder-Mead":
            options = {"maxfev": max(1, max_evals - calls), "maxiter": max(1, max_evals - calls), "disp": False}
        elif method == "L-BFGS-B":
            options = {"maxfun": max(1, max_evals - calls), "maxiter": max(1, max_evals - calls), "disp": False}
        else:
            options["rhobeg"] = max(0.05, spec.learning_rate)
        try:
            minimize(objective, current, method=method, options=options)
        except RuntimeError:
            pass

    if best_values is None:
        raise RuntimeError("optimizer never evaluated a candidate")
    return best_values, best_energy, calls


def quick_model_class(problem: prepare.Problem) -> str:
    labels = problem.hamiltonian.paulis.to_labels()
    single_ops = {label_active_qubits(label)[0][1] for label in labels if len(label_active_qubits(label)) == 1}
    two_groups: dict[tuple[int, int], set[str]] = {}
    max_locality = 0
    for label in labels:
        active = label_active_qubits(label)
        max_locality = max(max_locality, len(active))
        if len(active) == 2:
            edge = tuple(sorted(qubit for qubit, _ in active))
            two_groups.setdefault(edge, set()).add("".join(pauli for _, pauli in active))
    if two_groups and all(ops <= {"ZZ"} for ops in two_groups.values()) and "X" in single_ops:
        return "transverse_field_ising"
    if two_groups and sum({"XX", "YY", "ZZ"}.issubset(ops) for ops in two_groups.values()) >= 0.6 * len(two_groups):
        return "weighted_heisenberg_graph"
    if max_locality > 2:
        return "chemistry_or_general_pauli"
    return "general_two_local_pauli"


def build_spec(**kwargs: Any) -> ExperimentSpec:
    return ExperimentSpec(**kwargs)


def build_baseline_spec() -> ExperimentSpec:
    return build_spec(
        family="hea",
        layers=2,
        optimizer="spsa",
        param_init="small_random",
        learning_rate=0.18,
        seed=prepare.SEED,
        edge_mode="alternate",
        rotation_mode="full",
        spsa_steps=24,
        restarts=2,
        description="baseline ansatz=hea layers=2 optimizer=spsa",
    )


def add_tfim_specs(specs: list[ExperimentSpec], problem: prepare.Problem) -> None:
    ref = tfim_reference_state(problem)
    specs.append(build_spec(
        family="tfim_counterdiabatic", layers=1, optimizer="cobyla", param_init="medium_random",
        learning_rate=0.30, seed=701, edge_mode="edge",
        rotation_mode="counterdiabatic", reference_state=ref, spsa_steps=20, restarts=1,
        time_budget_seconds=30.0,
        description=f"ansatz=tfim_counterdiabatic layers=1 edge=edge init=medium_random ref={ref} seed=701",
    ))
    for layers in [2, 3, 4, 5, 6, 8, 10]:
        for param_init in ["small_random", "large_random"]:
            for optimizer in ["cobyla", "powell"]:
                specs.append(build_spec(
                    family="tfim_colored", layers=layers, optimizer=optimizer, param_init=param_init,
                    learning_rate=0.35, seed=prepare.SEED + 43 * layers + len(specs), edge_mode="colored",
                    rotation_mode="colored", reference_state=ref, spsa_steps=20, restarts=2,
                    description=(
                        f"ansatz=tfim_colored layers={layers} optimizer={optimizer} "
                        f"init={param_init} ref={ref}"
                    ),
                ))
        if layers in {4, 6, 8}:
            specs.append(build_spec(
                family="tfim_colored", layers=layers, optimizer="anneal", param_init="large_random",
                learning_rate=0.35, seed=prepare.SEED + 59 * layers + len(specs), edge_mode="colored",
                rotation_mode="colored", reference_state=ref, spsa_steps=20, restarts=1,
                description=f"ansatz=tfim_colored layers={layers} optimizer=anneal init=large_random ref={ref}",
            ))
    for layers in [1, 2, 3, 4, 5, 6, 8, 10]:
        for optimizer in ["cobyla", "powell"]:
            specs.append(build_spec(
                family="tfim_shared", layers=layers, optimizer=optimizer, param_init="small_random",
                learning_rate=0.25, seed=prepare.SEED + 11 * layers, edge_mode="full",
                rotation_mode="shared", reference_state=ref, spsa_steps=20, restarts=3,
                description=f"ansatz=tfim_shared layers={layers} optimizer={optimizer} ref={ref}",
            ))
    for layers in [1, 2, 3, 4, 5, 6]:
        for optimizer in ["cobyla", "powell"]:
            specs.append(build_spec(
                family="tfim_factorized", layers=layers, optimizer=optimizer, param_init="small_random",
                learning_rate=0.25, seed=prepare.SEED + 13 * layers + len(specs), edge_mode="full",
                rotation_mode="term", reference_state=ref, spsa_steps=20, restarts=2,
                description=f"ansatz=tfim_factorized layers={layers} optimizer={optimizer} ref={ref}",
            ))


def add_exchange_specs(specs: list[ExperimentSpec], problem: prepare.Problem) -> None:
    graph_edges = support_edges(problem)
    if len(graph_edges) == problem.num_qubits - 1:
        specs.append(build_spec(
            family="heisenberg_hva", layers=8, optimizer="cobyla", param_init="small_random",
            learning_rate=0.25, seed=202, edge_mode="colored",
            rotation_mode="shared", reference_state="dimer_even", spsa_steps=24, restarts=4,
            time_budget_seconds=120.0,
            description="ansatz=heisenberg_hva layers=8 rot=shared init=small_random ref=dimer_even seed=202",
        ))
    else:
        specs.append(build_spec(
            family="heisenberg_hva", layers=4, optimizer="cobyla_lbfgsb", param_init="small_random",
            learning_rate=0.22, seed=854, edge_mode="colored",
            rotation_mode="term", reference_state="dimer_even", spsa_steps=24, restarts=1,
            time_budget_seconds=600.0,
            description="ansatz=heisenberg_hva layers=4 rot=term init=small_random optimizer=cobyla_lbfgsb ref=dimer_even seed=854",
        ))
    for layers in [1, 2, 3, 4, 5, 6, 8]:
        for ref in ["dimer_even", "dimer_odd", "neel"]:
            for rotation_mode in ["shared", "term"]:
                for param_init in ["zeros", "small_random"]:
                    specs.append(build_spec(
                        family="heisenberg_hva", layers=layers, optimizer="cobyla", param_init=param_init,
                        learning_rate=0.18, seed=prepare.SEED + 17 * layers + len(specs), edge_mode="colored",
                        rotation_mode=rotation_mode, reference_state=ref, spsa_steps=24, restarts=2,
                        description=(
                            f"ansatz=heisenberg_hva layers={layers} rot={rotation_mode} "
                            f"init={param_init} ref={ref}"
                        ),
                    ))
            if ref.startswith("dimer"):
                specs.append(build_spec(
                    family="heisenberg_hva", layers=layers, optimizer="powell", param_init="zeros",
                    learning_rate=0.18, seed=prepare.SEED + 41 * layers + len(specs), edge_mode="colored",
                    rotation_mode="shared", reference_state=ref, spsa_steps=24, restarts=1,
                    description=f"ansatz=heisenberg_hva layers={layers} rot=shared init=zeros optimizer=powell ref={ref}",
                ))
                if layers in {3, 4, 5, 6}:
                    specs.append(build_spec(
                        family="heisenberg_hva", layers=layers, optimizer="anneal", param_init="zeros",
                        learning_rate=0.18, seed=prepare.SEED + 53 * layers + len(specs), edge_mode="colored",
                        rotation_mode="shared", reference_state=ref, spsa_steps=24, restarts=1,
                        description=(
                            f"ansatz=heisenberg_hva layers={layers} rot=shared "
                            f"init=zeros optimizer=anneal ref={ref}"
                        ),
                    ))


def add_u1_specs(specs: list[ExperimentSpec]) -> None:
    for edge_mode in ["occupied_virtual", "frontier"]:
        for layers in [1, 2]:
            specs.append(build_spec(
                family="u1_exchange", layers=layers, optimizer="cobyla", param_init="zeros",
                learning_rate=0.08, seed=prepare.SEED + 23 * layers + len(specs),
                edge_mode=edge_mode, rotation_mode="number_preserving", reference_state="hint",
                spsa_steps=16, restarts=1,
                description=f"ansatz=u1_exchange layers={layers} edge={edge_mode} ref=hint",
            ))


def add_pauli_hva_specs(specs: list[ExperimentSpec], reference_state: str) -> None:
    for layers in [1, 2, 3]:
        specs.append(build_spec(
            family="pauli_hva", layers=layers, optimizer="cobyla", param_init="small_random",
            learning_rate=0.20, seed=prepare.SEED + 19 * layers + len(specs),
            edge_mode="full", rotation_mode="term", reference_state=reference_state,
            spsa_steps=24, restarts=2,
            description=f"ansatz=pauli_hva layers={layers} rot=term ref={reference_state}",
        ))


def add_symmetry_probe_specs(specs: list[ExperimentSpec]) -> None:
    for layers in [1, 2, 3]:
        specs.append(build_spec(
            family="symm", layers=layers, optimizer="cobyla", param_init="small_random",
            learning_rate=0.25, seed=prepare.SEED + 31 * layers, edge_mode="alternate",
            rotation_mode="shared", reference_state="zero", spsa_steps=18, restarts=2,
            description=f"ansatz=symm layers={layers}",
        ))


def add_baseline_specs(specs: list[ExperimentSpec], features: OperatorFeatures) -> None:
    specs.append(build_baseline_spec())
    layer_values = [1, 2, 3, 4, 5] if features.has_zz_edges and features.has_x_field else [1, 2, 3]
    for mode in ["ry_only", "full"]:
        for layers in layer_values:
            edge_mode = "full" if features.has_zz_edges and features.has_x_field else "alternate"
            seed = prepare.SEED + 29 * layers + len(specs)
            specs.append(build_spec(
                family="hea", layers=layers, optimizer="cobyla", param_init="small_random",
                learning_rate=0.35, seed=seed,
                edge_mode=edge_mode, rotation_mode=mode, reference_state="zero",
                spsa_steps=18, restarts=2,
                description=f"ansatz=hea layers={layers} rot={mode} edge={edge_mode}",
            ))


def candidate_specs(problem: prepare.Problem, _model_class: str) -> list[ExperimentSpec]:
    features = operator_features(problem)
    specs: list[ExperimentSpec] = []
    reference_bits = hint_bits(problem)
    exchange_like = features.matched_xy_fraction >= 0.6 or features.matched_xyz_fraction >= 0.6

    if exchange_like:
        add_exchange_specs(specs, problem)
    if reference_bits is not None and 0 < sum(reference_bits) < len(reference_bits):
        add_u1_specs(specs)
    if features.has_zz_edges and features.has_x_field:
        add_tfim_specs(specs, problem)
    if features.max_locality > 0 and not features.z_only:
        ref = "hint" if features.has_reference_hint else "zero"
        add_pauli_hva_specs(specs, ref)
    add_symmetry_probe_specs(specs)
    add_baseline_specs(specs, features)
    return unique_specs(specs)


def unique_specs(specs: list[ExperimentSpec]) -> list[ExperimentSpec]:
    seen: set[tuple[Any, ...]] = set()
    result: list[ExperimentSpec] = []
    for spec in specs:
        key = (
            spec.family, spec.layers, spec.optimizer, spec.param_init, spec.seed,
            spec.edge_mode, spec.rotation_mode, spec.reference_state,
        )
        if key not in seen:
            seen.add(key)
            result.append(spec)
    return result


def classify_improvement(candidate: ExperimentResult, incumbent: ExperimentResult | None) -> str | None:
    if not admissible_result(candidate):
        return None
    if incumbent is None or candidate.energy < incumbent.energy - ENERGY_IMPROVEMENT_TOL:
        return "energy"
    if candidate.energy <= incumbent.energy + ENERGY_EQUIVALENCE_TOL and candidate.compression_key < incumbent.compression_key:
        return "compression"
    return None


def admissible_result(result: ExperimentResult) -> bool:
    if result.num_params < MIN_ADMISSIBLE_PARAMS:
        return False
    if "ansatz=pauli_hva" in result.description and "rot=shared" in result.description:
        return False
    return True


def target_threshold(reference_energy: float | None) -> float | None:
    if reference_energy is None or (TARGET_REL_ERROR <= 0.0 and TARGET_ABS_ERROR <= 0.0):
        return None
    return max(TARGET_ABS_ERROR, TARGET_REL_ERROR * abs(reference_energy))


def meets_target(result: ExperimentResult, reference_energy: float | None) -> bool:
    threshold = target_threshold(reference_energy)
    return (
        admissible_result(result)
        and threshold is not None
        and reference_energy is not None
        and abs(result.energy - reference_energy) <= threshold
    )


def crash_result(index: int, description: str, total_seconds: float) -> ExperimentResult:
    return ExperimentResult(
        run_id=experiment_id(index, description),
        description=description,
        status="crash",
        energy=0.0,
        overlap=None,
        metrics={"singleq_count": 0, "twoq_count": 0, "total_gate_count": 0, "depth": 0},
        num_params=0,
        eval_calls=0,
        total_seconds=total_seconds,
    )


def run_experiment(
    spec: ExperimentSpec,
    problem: prepare.Problem,
    backend_target: prepare.BackendTarget,
    model_class: str,
    index: int,
) -> ExperimentResult:
    started = time.perf_counter()
    description = spec.description if RUN_MODE == "full" else f"{spec.description} mode={RUN_MODE}"
    try:
        circuit, parameters = build_ansatz(problem, spec)
        values, energy, eval_calls = optimize_energy(problem, circuit, parameters, spec)
        final_circuit = bind_parameters(circuit, parameters, values)
        _, metrics = prepare.transpile_and_report(final_circuit, backend_target)
        overlap = prepare.overlap_with_reference(final_circuit, problem.reference_state)
        return ExperimentResult(
            run_id=experiment_id(index, description),
            description=description,
            status="discard",
            energy=energy,
            overlap=overlap,
            metrics=metrics,
            num_params=len(parameters),
            eval_calls=eval_calls,
            total_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return crash_result(index, f"{description} | crash={type(exc).__name__}: {exc}", time.perf_counter() - started)


def summarize_run(index: int, result: ExperimentResult) -> None:
    print(
        f"[{index:03d}] {result.status:<7} energy={result.energy:.6f} "
        f"twoq={result.metrics['twoq_count']:<3d} total={result.metrics['total_gate_count']:<3d} "
        f"depth={result.metrics['depth']:<3d} params={result.num_params:<3d} {result.description}"
    )


def format_best_summary(best: ExperimentResult, reference_energy: float | None) -> str:
    return prepare.format_summary(
        [
            ("energy", best.energy),
            ("reference_energy", reference_energy),
            ("overlap", best.overlap),
            ("singleq_count", best.metrics["singleq_count"]),
            ("twoq_count", best.metrics["twoq_count"]),
            ("total_gate_count", best.metrics["total_gate_count"]),
            ("depth", best.metrics["depth"]),
            ("num_params", best.num_params),
            ("eval_calls", best.eval_calls),
            ("total_seconds", best.total_seconds),
        ]
    )


def main() -> None:
    ensure_results_header()
    problem = prepare.load_problem(PROBLEM_PATH)
    backend_target = prepare.build_backend_target(problem)
    model_class = quick_model_class(problem) if MODEL_CLASS == "auto" else MODEL_CLASS
    specs = candidate_specs(problem, model_class)
    if MAX_EXPERIMENTS:
        specs = specs[:MAX_EXPERIMENTS]

    best: ExperimentResult | None = None
    for index, spec in enumerate(specs, start=1):
        result = run_experiment(spec, problem, backend_target, model_class, index)
        improvement = None if result.status == "crash" else classify_improvement(result, best)
        if improvement is not None:
            result = ExperimentResult(**{**result.__dict__, "status": "keep"})
            best = result
        log_result(result)
        summarize_run(index, result)
        if STOP_AT_TARGET and best is not None and meets_target(best, problem.reference_energy):
            break

    if best is None:
        raise RuntimeError("all experiments crashed")
    print("---")
    print(format_best_summary(best, problem.reference_energy))


if __name__ == "__main__":
    main()
