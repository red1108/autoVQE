"""Circuit transpilation and evaluator-owned hardware metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .contracts import PublicProblem


TRANSPILER_SEED = 7
CANONICAL_BASIS_GATES = ("rz", "sx", "x", "cx")


@dataclass(frozen=True)
class BackendTarget:
    basis_gates: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...] | None = None

    def __init__(
        self,
        basis_gates: Sequence[str],
        coupling_map: Sequence[Sequence[int]] | None = None,
    ) -> None:
        gates = tuple(str(gate) for gate in basis_gates)
        if not gates or any(not gate for gate in gates):
            raise ValueError("backend target requires non-empty basis gate names")
        edges = None
        if coupling_map:
            if any(len(edge) != 2 for edge in coupling_map):
                raise ValueError("coupling-map entries must contain two qubits")
            edges = tuple((int(edge[0]), int(edge[1])) for edge in coupling_map)
        object.__setattr__(self, "basis_gates", gates)
        object.__setattr__(self, "coupling_map", edges)


def backend_target_from_problem(problem: PublicProblem) -> BackendTarget | None:
    """Convert optional problem constraints into a transpiler target."""

    if not problem.backend.basis_gates and not problem.backend.coupling_map:
        return None
    return BackendTarget(
        basis_gates=problem.backend.basis_gates or CANONICAL_BASIS_GATES,
        coupling_map=problem.backend.coupling_map or None,
    )


def canonical_backend_target() -> BackendTarget:
    """Return the fixed all-to-all target used for comparable resource counts."""

    return BackendTarget(basis_gates=CANONICAL_BASIS_GATES)


def transpile_circuit(
    circuit: QuantumCircuit,
    backend_target: BackendTarget,
    optimization_level: int = 1,
) -> QuantumCircuit:
    coupling = (
        None
        if backend_target.coupling_map is None
        else CouplingMap(list(backend_target.coupling_map))
    )
    return transpile(
        circuit,
        basis_gates=list(backend_target.basis_gates),
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=TRANSPILER_SEED,
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
    compiled = transpile_circuit(
        circuit,
        backend_target,
        optimization_level=optimization_level,
    )
    return compiled, compiled_metrics(compiled)


__all__ = [
    "BackendTarget",
    "CANONICAL_BASIS_GATES",
    "TRANSPILER_SEED",
    "backend_target_from_problem",
    "canonical_backend_target",
    "compiled_metrics",
    "transpile_and_report",
    "transpile_circuit",
]
