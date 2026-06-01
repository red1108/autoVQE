from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap

TIME_BUDGET = 30.0
MAX_EVALS = 150
SEED = 7
EXACT_REFERENCE_MAX_QUBITS = 10
DEFAULT_BASIS_GATES = ["rx", "ry", "rz", "cx"]
DEFAULT_PROBLEM_PATH = Path("examples/h2_2q.json")
TWO_QUBIT_GATES = {
    "cx",
    "cz",
    "swap",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "iswap",
}


@dataclass(frozen=True)
class BackendTarget:
    basis_gates: list[str]
    coupling_map: list[list[int]] | None


@dataclass(frozen=True)
class Problem:
    name: str
    num_qubits: int
    pauli_terms: list[dict[str, Any]]
    hamiltonian: SparsePauliOp
    reference_energy: float | None
    reference_state: np.ndarray | None
    symmetry: Any | None
    basis_gates: list[str]
    coupling_map: list[list[int]] | None
    initial_state_hint: Any | None


def load_problem(path: str | Path = DEFAULT_PROBLEM_PATH) -> Problem:
    path = Path(path)
    raw = json.loads(path.read_text())
    pauli_terms = raw.get("pauli_terms")
    if not isinstance(pauli_terms, list) or not pauli_terms:
        raise ValueError(f"{path} must define a non-empty pauli_terms list")

    first_label = None
    op_terms: list[tuple[str, complex]] = []
    for index, term in enumerate(pauli_terms):
        if not isinstance(term, dict):
            raise ValueError(f"pauli_terms[{index}] must be an object")
        label = term.get("pauli")
        coeff = term.get("coeff")
        if not isinstance(label, str) or not label:
            raise ValueError(f"pauli_terms[{index}].pauli must be a non-empty string")
        if first_label is None:
            first_label = label
        if len(label) != len(first_label):
            raise ValueError("all pauli labels must have the same length")
        if not set(label).issubset({"I", "X", "Y", "Z"}):
            raise ValueError(f"invalid Pauli label: {label}")
        try:
            op_terms.append((label, complex(coeff)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid coeff for pauli_terms[{index}]") from exc

    assert first_label is not None
    hamiltonian = SparsePauliOp.from_list(op_terms).simplify()
    num_qubits = len(first_label)

    reference_energy = raw.get("reference_energy")
    if reference_energy is not None:
        reference_energy = float(reference_energy)
        reference_state = None
    else:
        reference_energy, reference_state = exact_reference(hamiltonian, num_qubits)

    basis_gates = raw.get("basis_gates") or DEFAULT_BASIS_GATES
    if not isinstance(basis_gates, list) or not all(isinstance(gate, str) for gate in basis_gates):
        raise ValueError("basis_gates must be a list of strings")

    coupling_map = raw.get("coupling_map")
    if coupling_map is not None:
        if not isinstance(coupling_map, list):
            raise ValueError("coupling_map must be a list of [control, target] pairs")
        validated_edges: list[list[int]] = []
        for edge in coupling_map:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(qubit, int) for qubit in edge)
            ):
                raise ValueError("each coupling_map entry must be a two-item integer list")
            validated_edges.append(edge)
        coupling_map = validated_edges

    return Problem(
        name=str(raw.get("name") or "unnamed_problem"),
        num_qubits=num_qubits,
        pauli_terms=pauli_terms,
        hamiltonian=hamiltonian,
        reference_energy=reference_energy,
        reference_state=reference_state,
        symmetry=raw.get("symmetry"),
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        initial_state_hint=raw.get("initial_state_hint"),
    )


def exact_reference(
    hamiltonian: SparsePauliOp, num_qubits: int
) -> tuple[float | None, np.ndarray | None]:
    if num_qubits > EXACT_REFERENCE_MAX_QUBITS:
        return None, None

    matrix = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    index = int(np.argmin(eigenvalues))
    return float(np.real(eigenvalues[index])), np.asarray(eigenvectors[:, index])


def build_backend_target(problem: Problem) -> BackendTarget:
    return BackendTarget(
        basis_gates=list(problem.basis_gates),
        coupling_map=[list(edge) for edge in problem.coupling_map] if problem.coupling_map else None,
    )


def transpile_circuit(
    circuit: QuantumCircuit,
    backend_target: BackendTarget,
    optimization_level: int = 1,
) -> QuantumCircuit:
    coupling = None
    if backend_target.coupling_map:
        coupling = CouplingMap(backend_target.coupling_map)

    return transpile(
        circuit,
        basis_gates=backend_target.basis_gates,
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=SEED,
    )


def compiled_metrics(compiled: QuantumCircuit) -> dict[str, int]:
    singleq_count = 0
    twoq_count = 0
    for instruction in compiled.data:
        if instruction.operation.num_qubits == 1:
            singleq_count += 1
        elif instruction.operation.num_qubits == 2:
            twoq_count += 1

    return {
        "singleq_count": singleq_count,
        "twoq_count": twoq_count,
        "total_gate_count": len(compiled.data),
        "depth": int(compiled.depth() or 0),
    }


def transpile_and_report(
    circuit: QuantumCircuit,
    backend_target: BackendTarget,
    optimization_level: int = 1,
) -> tuple[QuantumCircuit, dict[str, int]]:
    compiled = transpile_circuit(circuit, backend_target, optimization_level=optimization_level)
    return compiled, compiled_metrics(compiled)


def energy_from_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(hamiltonian)
    return float(np.real(value))


def overlap_with_reference(circuit: QuantumCircuit, reference_state: np.ndarray | None) -> float | None:
    if reference_state is None:
        return None
    state = Statevector.from_instruction(circuit).data
    overlap = np.vdot(reference_state, state)
    return float(np.abs(overlap) ** 2)


def format_summary(metrics: list[tuple[str, Any]]) -> str:
    lines = ["---"]
    for key, value in metrics:
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        lines.append(f"{key + ':':<18}{rendered}")
    return "\n".join(lines)


def problem_summary(problem: Problem, backend_target: BackendTarget) -> str:
    coupling_edges = 0 if backend_target.coupling_map is None else len(backend_target.coupling_map)
    metrics = [
        ("name", problem.name),
        ("num_qubits", problem.num_qubits),
        ("num_terms", len(problem.hamiltonian.paulis)),
        ("basis_gates", ",".join(backend_target.basis_gates)),
        ("coupling_edges", coupling_edges),
        ("reference_energy", problem.reference_energy),
    ]
    return format_summary(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an AutoVQE problem file")
    parser.add_argument("problem", nargs="?", default=str(DEFAULT_PROBLEM_PATH))
    args = parser.parse_args()

    problem = load_problem(args.problem)
    backend_target = build_backend_target(problem)
    print(problem_summary(problem, backend_target))


if __name__ == "__main__":
    main()
