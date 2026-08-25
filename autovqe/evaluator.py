"""Evaluator-owned optimization, energy, and circuit resource measurements.

This module deliberately keeps one narrow trust boundary: candidates supply a
typed circuit shape, while this evaluator supplies every numeric parameter,
optimizer setting, energy, and resource count.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from scipy.optimize import minimize

from .ansatz import (
    AnsatzIRValidationError,
    AnsatzSpec,
    CompiledAnsatz,
    compile_ansatz,
    operation_paulis,
)
from .problem import BackendSpec, PublicProblem, hamiltonian_from_problem
from .probes import (
    _expectation_energy,
    initial_state_circuit,
    validate_hamiltonian_observable,
)


CANONICAL_BASIS_GATES = ("rz", "sx", "x", "cx")
EVALUATOR_SEED = 7


class EvaluationError(RuntimeError):
    """Raised when trusted evaluator inputs or settings are invalid."""


class _CallLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationProtocol:
    max_evals: int = 80
    restarts: int = 2
    seed: int = EVALUATOR_SEED
    initial_scale: float = 0.15
    transpile_optimization_level: int = 1
    audit_binding_scale: float = 0.271828
    audit_binding_count: int = 3

    def validate(self) -> None:
        if isinstance(self.max_evals, bool) or not isinstance(self.max_evals, int) or self.max_evals <= 0:
            raise EvaluationError("max_evals must be a positive integer")
        if isinstance(self.restarts, bool) or not isinstance(self.restarts, int) or self.restarts <= 0:
            raise EvaluationError("restarts must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvaluationError("seed must be an integer")
        if not math.isfinite(self.initial_scale) or self.initial_scale < 0:
            raise EvaluationError("initial_scale must be finite and non-negative")
        if (
            isinstance(self.transpile_optimization_level, bool)
            or not isinstance(self.transpile_optimization_level, int)
            or self.transpile_optimization_level not in {0, 1, 2, 3}
        ):
            raise EvaluationError("transpile_optimization_level must be 0, 1, 2, or 3")
        if not math.isfinite(self.audit_binding_scale) or self.audit_binding_scale <= 0:
            raise EvaluationError("audit_binding_scale must be finite and positive")
        if (
            isinstance(self.audit_binding_count, bool)
            or not isinstance(self.audit_binding_count, int)
            or self.audit_binding_count < 3
        ):
            raise EvaluationError("audit_binding_count must be an integer of at least 3")


@dataclass(frozen=True)
class ResourceAudit:
    valid: bool
    audit: dict[str, Any]
    resources: dict[str, int]
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    valid: bool
    best_energy: float | None
    baseline_energy: float | None
    trace_summary: tuple[tuple[int, float], ...]
    objective_calls: int
    optimized_parameter_binding: dict[str, float] | None
    audit: dict[str, Any]
    resources: dict[str, int]
    violations: tuple[str, ...] = ()
    objective_activity_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _AuditRun:
    result: ResourceAudit
    compiled: CompiledAnsatz | None = None
    coordinates: tuple["_ParameterCoordinate", ...] = ()


@dataclass(frozen=True)
class _ParameterCoordinate:
    """One submitted parameter expressed in a canonical optimizer coordinate."""

    name: str
    pivot: float


@dataclass(frozen=True)
class _LinearOperation:
    """One generator and its raw-label linear coefficients."""

    qubits: tuple[int, ...]
    terms: tuple[tuple[str, float], ...]
    pauli: str

def _expanded_operations(spec: AnsatzSpec) -> tuple[_LinearOperation, ...]:
    expanded: list[_LinearOperation] = []
    for operation in spec.operations:
        for word, scale in operation_paulis(operation):
            pairs = sorted(zip(operation.qubits, word, strict=True))
            expanded.append(
                _LinearOperation(
                    tuple(qubit for qubit, _ in pairs),
                    ((operation.parameter, scale),),
                    "".join(letter for _, letter in pairs),
                )
            )
    return tuple(expanded)

def _operation_key(operation: _LinearOperation) -> tuple[Any, ...]:
    shape = tuple(sorted(value for _, value in operation.terms))
    return operation.qubits, operation.pauli, shape

def _commute(left: _LinearOperation, right: _LinearOperation) -> bool:
    if set(left.qubits).isdisjoint(right.qubits):
        return True
    left_letters = dict(zip(
        left.qubits, left.pauli, strict=True
    ))
    right_letters = dict(zip(
        right.qubits, right.pauli, strict=True
    ))
    disagreements = sum(
        left_letters[qubit] != right_letters[qubit]
        for qubit in set(left_letters) & set(right_letters)
    )
    return disagreements % 2 == 0

def _canonical_order(operations: tuple[_LinearOperation, ...]) -> tuple[_LinearOperation, ...]:
    successors: list[list[int]] = [[] for _ in operations]
    indegree = [0] * len(operations)
    for left in range(len(operations)):
        for right in range(left + 1, len(operations)):
            if not _commute(operations[left], operations[right]):
                successors[left].append(right)
                indegree[right] += 1
    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered: list[_LinearOperation] = []
    while ready:
        chosen = min(ready, key=lambda index: (_operation_key(operations[index]), index))
        ready.remove(chosen)
        ordered.append(operations[chosen])
        for successor in successors[chosen]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(ordered) != len(operations):
        raise EvaluationError("operation dependency graph contains a cycle")
    return tuple(ordered)

def _same_generator(left: _LinearOperation, right: _LinearOperation) -> bool:
    return left.qubits == right.qubits and left.pauli == right.pauli

def _sum_rotations(left: _LinearOperation, right: _LinearOperation) -> _LinearOperation | None:
    coefficients: dict[str, float] = {}
    for name, value in left.terms + right.terms:
        coefficients[name] = coefficients.get(name, 0.0) + value
    terms = tuple(sorted(
        (name, value) for name, value in coefficients.items() if value != 0.0
    ))
    if not terms:
        return None
    return _LinearOperation(left.qubits, terms, left.pauli)

def _canonical_operations(spec: AnsatzSpec) -> tuple[_LinearOperation, ...]:
    combined: list[_LinearOperation] = []
    for operation in _canonical_order(_expanded_operations(spec)):
        if combined and _same_generator(combined[-1], operation):
            previous = combined.pop()
            merged = _sum_rotations(previous, operation)
            if merged is not None:
                combined.append(merged)
        else:
            combined.append(operation)
    return tuple(combined)

def _parameter_vectors(spec: AnsatzSpec, operations: tuple[_LinearOperation, ...]
                       ) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    names = spec.parameter_names
    vectors = {name: [0.0] * len(operations) for name in names}
    for index, operation in enumerate(operations):
        for name, coefficient in operation.terms:
            if name not in vectors:
                raise EvaluationError(f"unknown raw parameter in canonical circuit: {name}")
            vectors[name][index] = coefficient
    return names, tuple(tuple(vectors[name]) for name in names)

def _canonical_row_space(rows: Sequence[Sequence[float]], tolerance: float = 1e-12
                         ) -> tuple[tuple[float, ...], ...]:
    matrix = [list(map(float, row)) for row in rows]
    if not matrix:
        return ()
    width = len(matrix[0])
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, len(matrix))
                      if abs(matrix[row][column]) > tolerance), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if abs(factor) > tolerance:
                matrix[row] = [value - factor * pivot_value for value, pivot_value
                               in zip(matrix[row], matrix[pivot_row], strict=True)]
        pivot_row += 1
        if pivot_row == len(matrix):
            break
    result: list[tuple[float, ...]] = []
    for row in matrix:
        cleaned = tuple(0.0 if abs(value) <= tolerance else round(float(value), 12)
                        for value in row)
        if any(value != 0.0 for value in cleaned):
            result.append(cleaned)
    return tuple(result)

def _canonical_analysis(spec: AnsatzSpec) -> tuple[
    tuple[_LinearOperation, ...],
    tuple[str, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    operations = _canonical_operations(spec)
    names, vectors = _parameter_vectors(spec, operations)
    return operations, names, vectors, _canonical_row_space(vectors)


def _identity(spec: AnsatzSpec, analysis: tuple[Any, ...]) -> str:
    operations, _, _, row_space = analysis
    canonical = {
        "semantic_identity_version": 6,
        "version": spec.version,
        "num_qubits": spec.num_qubits,
        "parameter_count": len(row_space),
        "parameter_subspace": row_space,
        "operations": [
            {
                "qubits": list(operation.qubits),
                "pauli": operation.pauli,
            }
            for operation in operations
        ],
    }
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def candidate_identity(spec: AnsatzSpec | Mapping[str, Any]) -> str:
    """Canonical physical-family identity used to reject optimizer retries."""

    parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
    return _identity(parsed, _canonical_analysis(parsed))

def _metrics(circuit: QuantumCircuit) -> dict[str, int]:
    two = sum(instruction.operation.num_qubits == 2 for instruction in circuit.data)
    return {
        "twoq_count": int(two),
        "total_gate_count": len(circuit.data),
        "depth": int(circuit.depth() or 0),
    }

def _transpiled_metrics(
    circuit: QuantumCircuit,
    backend: BackendSpec,
    optimization_level: int,
) -> dict[str, int]:
    if not backend.basis_gates and not backend.coupling_map:
        return _metrics(circuit)
    coupling = CouplingMap(list(backend.coupling_map)) if backend.coupling_map else None
    compiled = transpile(
        circuit,
        basis_gates=list(backend.basis_gates or CANONICAL_BASIS_GATES),
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=7,
    )
    return _metrics(compiled)

def _audit_rng(identity: str, seed: int) -> np.random.Generator:
    material = f"autovqe-resource-audit-v1:{seed}:{identity}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(material))

def _worst_metrics(samples: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not samples:
        raise EvaluationError("resource audit requires at least one binding")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise EvaluationError("resource metric keys changed across audit bindings")
    return {key: max(int(sample[key]) for sample in samples) for key in sorted(keys)}

def _parameter_coordinates(
    compiled: CompiledAnsatz, analysis: tuple[Any, ...]
) -> tuple[_ParameterCoordinate, ...]:
    audit = compiled.audit
    raw = audit.get("trainable_parameter_names")
    if raw is None:
        raw = tuple(compiled.parameters)
    audited = tuple(str(name) for name in raw)
    _, names, vectors, row_space = analysis
    if set(audited) != set(compiled.parameters) or set(names) != set(audited):
        raise EvaluationError("compiled parameters do not match the audit")
    if len(row_space) != len(names):
        raise EvaluationError("trainable parameters are redundant after gate canonicalization")

    coordinates: list[tuple[tuple[float, ...], _ParameterCoordinate]] = []
    for name, vector in zip(names, vectors, strict=True):
        active = [value for value in vector if abs(value) > 1e-12]
        if not active:
            raise EvaluationError(f"parameter {name!r} has no canonical generator coordinate")
        largest = max(abs(value) for value in active)
        pivot = next(value for value in active if abs(value) == largest)
        signature = tuple(
            0.0 if abs(value) <= 1e-12 else round(float(value / pivot), 12)
            for value in vector
        )
        coordinates.append((signature, _ParameterCoordinate(name, float(pivot))))
    return tuple(
        coordinate
        for _, coordinate in sorted(
            coordinates, key=lambda item: (item[0], item[1].name)
        )
    )


def _submitted_values(
    coordinates: tuple[_ParameterCoordinate, ...], values: np.ndarray
) -> np.ndarray:
    if values.shape != (len(coordinates),):
        raise EvaluationError("parameter vector has the wrong shape")
    return np.asarray(
        [
            float(value) / coordinate.pivot
            for coordinate, value in zip(coordinates, values, strict=True)
        ],
        dtype=float,
    )


def _bind(
    compiled: CompiledAnsatz,
    coordinates: tuple[_ParameterCoordinate, ...],
    values: np.ndarray,
) -> QuantumCircuit:
    submitted = _submitted_values(coordinates, values)
    mapping = {
        compiled.parameters[coordinate.name]: float(value)
        for coordinate, value in zip(coordinates, submitted, strict=True)
    }
    return compiled.circuit.assign_parameters(mapping, inplace=False)

def _audit_candidate(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol,
) -> _AuditRun:
    try:
        parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
        compiled = compile_ansatz(parsed)
        if compiled.circuit.num_qubits != problem.num_qubits:
            raise EvaluationError("AnsatzSpec num_qubits does not match the problem")
        circuit = initial_state_circuit(problem)
        circuit.compose(compiled.circuit, inplace=True)
        compiled = CompiledAnsatz(circuit, compiled.parameters, compiled.audit)
        analysis = _canonical_analysis(parsed)
        coordinates = _parameter_coordinates(compiled, analysis)
        canonical = BackendSpec(CANONICAL_BASIS_GATES)
        samples = [
            _transpiled_metrics(
                compiled.circuit, problem.backend, protocol.transpile_optimization_level
            ),
            _transpiled_metrics(
                compiled.circuit, canonical, protocol.transpile_optimization_level
            ),
        ]
        rng = _audit_rng(_identity(parsed, analysis), protocol.seed)
        for _ in range(protocol.audit_binding_count):
            values = rng.uniform(
                -protocol.audit_binding_scale,
                protocol.audit_binding_scale,
                size=len(coordinates),
            )
            bound = _bind(compiled, coordinates, values)
            samples.append(
                _transpiled_metrics(
                    bound, problem.backend, protocol.transpile_optimization_level
                )
            )
            samples.append(
                _transpiled_metrics(
                    bound, canonical, protocol.transpile_optimization_level
                )
            )
        resources = _worst_metrics(samples)
        resources["parameters"] = len(coordinates)
        return _AuditRun(
            ResourceAudit(True, compiled.audit, resources),
            compiled,
            coordinates,
        )
    except (AnsatzIRValidationError, EvaluationError, ValueError, KeyError) as exc:
        return _AuditRun(ResourceAudit(False, {}, {}, (f"{type(exc).__name__}: {exc}",)))

def audit_public_candidate(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> ResourceAudit:
    selected = protocol or EvaluationProtocol()
    selected.validate()
    return _audit_candidate(
        problem,
        spec,
        protocol=selected,
    ).result

def _trace_summary(values: Sequence[float]) -> tuple[tuple[int, float], ...]:
    if not values:
        return ()
    selected = {1, len(values)}
    index = 2
    while index < len(values):
        selected.add(index)
        index *= 2
    return tuple((index, float(values[index - 1])) for index in sorted(selected))


def _optimize(
    compiled: CompiledAnsatz,
    hamiltonian: SparsePauliOp,
    protocol: EvaluationProtocol,
    coordinates: tuple[_ParameterCoordinate, ...],
) -> tuple[np.ndarray, float, int, tuple[tuple[int, float], ...], float]:
    calls = 0
    best = float("inf")
    best_values = np.zeros(len(coordinates), dtype=float)
    trace: list[float] = []
    observed: list[float] = []

    def objective(values: np.ndarray) -> float:
        nonlocal calls, best, best_values
        if calls >= protocol.max_evals:
            raise _CallLimit
        values = np.asarray(values, dtype=float)
        energy = _expectation_energy(_bind(compiled, coordinates, values), hamiltonian)
        calls += 1
        observed.append(energy)
        if energy < best:
            best, best_values = energy, values.copy()
        trace.append(best)
        return energy

    if not coordinates:
        objective(np.zeros(0, dtype=float))
    else:
        for restart in range(protocol.restarts):
            if calls >= protocol.max_evals:
                break
            rng = np.random.default_rng(protocol.seed + 1009 * restart)
            initial = rng.uniform(
                -protocol.initial_scale,
                protocol.initial_scale,
                len(coordinates),
            )
            try:
                minimize(
                    objective,
                    initial,
                    method="COBYLA",
                    options={
                        "maxiter": protocol.max_evals - calls,
                        "rhobeg": max(0.05, protocol.initial_scale),
                        "tol": 1e-8,
                    },
                )
            except _CallLimit:
                pass
    if not observed:
        raise EvaluationError("optimizer did not evaluate the objective")
    return best_values, best, calls, _trace_summary(trace), max(observed) - min(observed)


def _active_norm(hamiltonian: SparsePauliOp) -> float:
    values = [
        abs(complex(coefficient))
        for label, coefficient in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs, strict=True)
        if set(label) != {"I"}
    ]
    return float(np.linalg.norm(values)) if values else 0.0


def evaluate_public_problem(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> EvaluationResult:
    selected = protocol or EvaluationProtocol()
    selected.validate()
    try:
        operator = validate_hamiltonian_observable(hamiltonian_from_problem(problem))
        audit = _audit_candidate(problem, spec, protocol=selected)
        if not audit.result.valid or audit.compiled is None:
            raise EvaluationError(audit.result.violations[0])
        baseline = _expectation_energy(initial_state_circuit(problem), operator)
        values, energy, calls, trace, span = _optimize(
            audit.compiled,
            operator,
            selected,
            audit.coordinates,
        )
        submitted_values = _submitted_values(audit.coordinates, values)
        active_norm = _active_norm(operator)
        return EvaluationResult(
            valid=True,
            best_energy=energy,
            baseline_energy=baseline,
            trace_summary=trace,
            objective_calls=calls,
            optimized_parameter_binding={
                coordinate.name: float(value)
                for coordinate, value in zip(
                    audit.coordinates, submitted_values, strict=True
                )
            },
            audit=audit.result.audit,
            resources=audit.result.resources,
            objective_activity_fraction=None if active_norm == 0 else span / active_norm,
        )
    except Exception as exc:
        return EvaluationResult(
            valid=False,
            best_energy=None,
            baseline_energy=None,
            trace_summary=(),
            objective_calls=0,
            optimized_parameter_binding=None,
            audit={},
            resources={},
            violations=(f"{type(exc).__name__}: {exc}",),
        )
