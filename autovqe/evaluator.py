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
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap
from scipy.optimize import minimize

from .ansatz import AnsatzIRValidationError, AnsatzSpec, CompiledAnsatz, compile_ansatz
from .problem import PublicProblem, hamiltonian_from_problem


CANONICAL_BASIS_GATES = ("rz", "sx", "x", "cx")
EVALUATOR_SEED = 7
_HERMITICITY_TOLERANCE = 1e-12
_IMAGINARY_TOLERANCE = 1e-10


class EvaluationError(RuntimeError):
    """Raised when trusted evaluator inputs or settings are invalid."""


class _CallLimit(RuntimeError):
    pass


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
    metrics: dict[str, int]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    valid: bool
    best_energy: float | None
    trace_summary: tuple[tuple[int, float], ...]
    objective_calls: int
    optimizer: str
    seed: int
    optimized_parameter_binding: dict[str, float] | None
    audit: dict[str, Any]
    metrics: dict[str, int]
    violations: tuple[str, ...] = ()
    objective_energy_span: float | None = None
    hamiltonian_active_norm: float | None = None
    objective_activity_fraction: float | None = None
    constant_hamiltonian: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRun:
    result: EvaluationResult
    best_values: tuple[float, ...]
    final_circuit: QuantumCircuit | None


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
    """One generator with affine parameter arguments used only for identity."""

    macro: str
    qubits: tuple[int, ...]
    arguments: tuple[tuple[str, float, tuple[tuple[str, float], ...]], ...]
    options_json: str

def _linear_operation(
    macro: str, qubits: Sequence[int], parameters: Mapping[str, Any], options: Mapping[str, Any]
) -> _LinearOperation:
    arguments = tuple(
        (
            str(argument),
            float(expression.constant),
            tuple(sorted((term.parameter.name, float(term.coefficient)) for term in expression.terms)),
        )
        for argument, expression in sorted(parameters.items())
    )
    encoded = json.dumps(
        dict(options), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return _LinearOperation(str(macro), tuple(map(int, qubits)), arguments, encoded)

def _expanded_operations(spec: AnsatzSpec) -> tuple[_LinearOperation, ...]:
    expanded: list[_LinearOperation] = []
    for operation in spec.operations:
        if operation.macro == "PauliRotation":
            pairs = sorted(zip(
                operation.qubits, str(operation.options["pauli"]), strict=True
            ))
            expanded.append(
                _linear_operation(
                    "PauliRotation",
                    tuple(qubit for qubit, _ in pairs),
                    operation.parameters,
                    {"pauli": "".join(letter for _, letter in pairs)},
                )
            )
            continue
        if operation.macro in {"XYExchange", "IsotropicExchange"}:
            paulis = (("XX", "YY", "ZZ") if operation.macro == "IsotropicExchange"
                       else ("XX", "YY"))
            expanded.extend(
                _linear_operation(
                    "PauliRotation", sorted(operation.qubits),
                    operation.parameters, {"pauli": pauli},
                )
                for pauli in paulis
            )
            continue
        expanded.append(_linear_operation(
            operation.macro, operation.qubits,
            operation.parameters, operation.options,
        ))
    return tuple(expanded)

def _operation_key(operation: _LinearOperation) -> tuple[Any, ...]:
    shape = tuple(
        (argument, tuple(sorted(value for _, value in terms)), constant)
        for argument, constant, terms in operation.arguments
    )
    return operation.macro, operation.qubits, operation.options_json, shape

def _commute(left: _LinearOperation, right: _LinearOperation) -> bool:
    if set(left.qubits).isdisjoint(right.qubits):
        return True
    if left.macro != "PauliRotation" or right.macro != "PauliRotation":
        return False
    left_letters = dict(zip(
        left.qubits, json.loads(left.options_json)["pauli"], strict=True
    ))
    right_letters = dict(zip(
        right.qubits, json.loads(right.options_json)["pauli"], strict=True
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
    return (
        left.macro == right.macro == "PauliRotation"
        and left.qubits == right.qubits
        and left.options_json == right.options_json
        and tuple(item[0] for item in left.arguments) == ("angle",)
        and tuple(item[0] for item in right.arguments) == ("angle",)
    )

def _sum_rotations(left: _LinearOperation, right: _LinearOperation) -> _LinearOperation | None:
    constant = left.arguments[0][1] + right.arguments[0][1]
    coefficients: dict[str, float] = {}
    for name, value in left.arguments[0][2] + right.arguments[0][2]:
        coefficients[name] = coefficients.get(name, 0.0) + value
    terms = tuple(sorted(
        (name, value) for name, value in coefficients.items() if value != 0.0
    ))
    if not terms and constant == 0.0:
        return None
    return _LinearOperation(
        left.macro, left.qubits, (("angle", constant, terms),), left.options_json
    )

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
    names = tuple(parameter.name for parameter in spec.parameters)
    coordinates = [item for operation in operations for item in operation.arguments]
    vectors = {name: [0.0] * len(coordinates) for name in names}
    for index, (_, _, terms) in enumerate(coordinates):
        for name, coefficient in terms:
            if name not in vectors:
                raise EvaluationError(f"undeclared parameter in canonical circuit: {name}")
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

def _canonical_spec(spec: AnsatzSpec) -> dict[str, Any]:
    operations = _canonical_operations(spec)
    _, vectors = _parameter_vectors(spec, operations)
    row_space = _canonical_row_space(vectors)
    return {
        "semantic_identity_version": 5,
        "version": spec.version,
        "num_qubits": spec.num_qubits,
        "parameter_count": len(row_space),
        "parameter_subspace": row_space,
        "operations": [
            {
                "macro": operation.macro,
                "qubits": list(operation.qubits),
                "parameter_arguments": [{"name": name, "constant": constant}
                                        for name, constant, _ in operation.arguments],
                "options": json.loads(operation.options_json),
            }
            for operation in operations
        ],
    }


def candidate_identity(spec: AnsatzSpec | Mapping[str, Any]) -> str:
    """Canonical physical-family identity used to reject optimizer retries."""

    try:
        parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
        value: Any = _canonical_spec(parsed)
        prefix = "valid:"
    except (TypeError, ValueError, KeyError):
        value = dict(spec) if isinstance(spec, Mapping) else spec.to_dict()
        prefix = "invalid:"
    return prefix + json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def _backend_target(problem: PublicProblem) -> BackendTarget | None:
    if not problem.backend.basis_gates and not problem.backend.coupling_map:
        return None
    return BackendTarget(
        problem.backend.basis_gates or CANONICAL_BASIS_GATES,
        problem.backend.coupling_map or None,
    )

def _metrics(circuit: QuantumCircuit) -> dict[str, int]:
    single = sum(instruction.operation.num_qubits == 1 for instruction in circuit.data)
    two = sum(instruction.operation.num_qubits == 2 for instruction in circuit.data)
    return {
        "singleq_count": int(single),
        "twoq_count": int(two),
        "total_gate_count": len(circuit.data),
        "depth": int(circuit.depth() or 0),
    }

def _transpiled_metrics(
    circuit: QuantumCircuit,
    target: BackendTarget | None,
    optimization_level: int,
) -> dict[str, int]:
    if target is None:
        return _metrics(circuit)
    coupling = None if target.coupling_map is None else CouplingMap(list(target.coupling_map))
    compiled = transpile(
        circuit,
        basis_gates=list(target.basis_gates),
        coupling_map=coupling,
        optimization_level=optimization_level,
        seed_transpiler=7,
    )
    return _metrics(compiled)

def _prefix(prefix: str, values: Mapping[str, int]) -> dict[str, int]:
    return {f"{prefix}_{key}": int(value) for key, value in values.items()}

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

def _validate_hamiltonian(operator: SparsePauliOp) -> SparsePauliOp:
    if not isinstance(operator, SparsePauliOp):
        raise EvaluationError("Hamiltonian must be a SparsePauliOp")
    coefficients = np.asarray(operator.coeffs, dtype=complex)
    if not np.all(np.isfinite(coefficients.real)) or not np.all(np.isfinite(coefficients.imag)):
        raise EvaluationError("Hamiltonian coefficients must be finite")
    simplified = operator.simplify(atol=1e-14)
    if np.any(np.abs(np.asarray(simplified.coeffs).imag) > _HERMITICITY_TOLERANCE):
        raise EvaluationError("Hamiltonian must be Hermitian with real Pauli coefficients")
    return simplified

def _with_initial_state(compiled: CompiledAnsatz, occupation: Sequence[int] | None) -> CompiledAnsatz:
    if occupation is None:
        return compiled
    bits = tuple(occupation)
    if len(bits) != compiled.circuit.num_qubits or any(
        isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1)
        for bit in bits
    ):
        raise EvaluationError("initial occupation must contain one integer 0/1 bit per qubit")
    if not any(bits):
        return compiled
    circuit = QuantumCircuit(compiled.circuit.num_qubits, name=compiled.circuit.name)
    for qubit, bit in enumerate(bits):
        if bit:
            circuit.x(qubit)
    circuit.compose(compiled.circuit, inplace=True)
    return CompiledAnsatz(circuit=circuit, parameters=compiled.parameters, audit=compiled.audit)

def _audit_dict(compiled: CompiledAnsatz) -> dict[str, Any]:
    audit = compiled.audit
    return audit if isinstance(audit, dict) else audit.to_dict()

def _parameter_coordinates(
    compiled: CompiledAnsatz, spec: AnsatzSpec
) -> tuple[_ParameterCoordinate, ...]:
    audit = _audit_dict(compiled)
    raw = audit.get("trainable_parameter_names")
    if raw is None:
        raw = tuple(compiled.parameters)
    audited = tuple(str(name) for name in raw)
    operations = _canonical_operations(spec)
    names, vectors = _parameter_vectors(spec, operations)
    if set(audited) != set(compiled.parameters) or set(names) != set(audited):
        raise EvaluationError("compiled parameters do not match the audit")
    if len(_canonical_row_space(vectors)) != len(names):
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
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    num_qubits: int,
    occupation: Sequence[int] | None,
    target: BackendTarget | None,
    protocol: EvaluationProtocol,
) -> _AuditRun:
    try:
        parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
        compiled = compile_ansatz(parsed)
        if compiled.circuit.num_qubits != num_qubits:
            raise EvaluationError("AnsatzSpec num_qubits does not match the problem")
        compiled = _with_initial_state(compiled, occupation)
        coordinates = _parameter_coordinates(compiled, parsed)
        canonical = BackendTarget(CANONICAL_BASIS_GATES)
        template = _transpiled_metrics(compiled.circuit, target, protocol.transpile_optimization_level)
        canonical_template = _transpiled_metrics(compiled.circuit, canonical, protocol.transpile_optimization_level)
        rng = _audit_rng(candidate_identity(parsed), protocol.seed)
        samples: list[Mapping[str, int]] = []
        canonical_samples: list[Mapping[str, int]] = []
        for _ in range(protocol.audit_binding_count):
            values = rng.uniform(
                -protocol.audit_binding_scale,
                protocol.audit_binding_scale,
                size=len(coordinates),
            )
            bound = _bind(compiled, coordinates, values)
            samples.append(
                _transpiled_metrics(bound, target, protocol.transpile_optimization_level)
            )
            canonical_samples.append(
                _transpiled_metrics(bound, canonical, protocol.transpile_optimization_level)
            )
        metrics = {
            **_prefix("template", template),
            **_prefix("audit_worst", _worst_metrics(samples)),
            **_prefix("canonical_template", canonical_template),
            **_prefix("canonical_audit_worst", _worst_metrics(canonical_samples)),
            "audit_binding_count": protocol.audit_binding_count,
        }
        return _AuditRun(
            ResourceAudit(True, _audit_dict(compiled), metrics),
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
        spec,
        num_qubits=problem.num_qubits,
        occupation=problem.initial_state.occupation,
        target=_backend_target(problem),
        protocol=selected,
    ).result

def _expectation(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.parameters:
        raise EvaluationError("trusted evaluator received an unbound circuit")
    value = complex(Statevector.from_instruction(circuit).expectation_value(hamiltonian))
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise EvaluationError("expectation value must be finite")
    if abs(value.imag) > _IMAGINARY_TOLERANCE * max(1.0, abs(value.real)):
        raise EvaluationError("Hermitian Hamiltonian produced a complex expectation")
    return float(value.real)

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
        energy = _expectation(_bind(compiled, coordinates, values), hamiltonian)
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


def evaluate_ansatz(
    hamiltonian: SparsePauliOp,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    backend_target: BackendTarget | None = None,
    protocol: EvaluationProtocol | None = None,
    initial_occupation: Sequence[int] | None = None,
) -> EvaluationRun:
    selected = protocol or EvaluationProtocol()
    selected.validate()
    try:
        operator = _validate_hamiltonian(hamiltonian)
        audit = _audit_candidate(
            spec,
            num_qubits=operator.num_qubits,
            occupation=initial_occupation,
            target=backend_target,
            protocol=selected,
        )
        if not audit.result.valid or audit.compiled is None:
            raise EvaluationError(audit.result.violations[0])
        values, energy, calls, trace, span = _optimize(
            audit.compiled,
            operator,
            selected,
            audit.coordinates,
        )
        submitted_values = _submitted_values(audit.coordinates, values)
        final = _bind(audit.compiled, audit.coordinates, values)
        metrics = {
            **audit.result.metrics,
            **_prefix("final", _transpiled_metrics(final, backend_target, selected.transpile_optimization_level)),
            **_prefix("canonical_final", _transpiled_metrics(final, BackendTarget(CANONICAL_BASIS_GATES), selected.transpile_optimization_level)),
        }
        active_norm = _active_norm(operator)
        result = EvaluationResult(
            valid=True,
            best_energy=energy,
            trace_summary=trace,
            objective_calls=calls,
            optimizer="cobyla",
            seed=selected.seed,
            optimized_parameter_binding={
                coordinate.name: float(value)
                for coordinate, value in zip(
                    audit.coordinates, submitted_values, strict=True
                )
            },
            audit=audit.result.audit,
            metrics=metrics,
            objective_energy_span=span,
            hamiltonian_active_norm=active_norm,
            objective_activity_fraction=None if active_norm == 0 else span / active_norm,
            constant_hamiltonian=active_norm == 0,
        )
        return EvaluationRun(
            result,
            tuple(float(value) for value in submitted_values),
            final,
        )
    except Exception as exc:
        result = EvaluationResult(
            valid=False,
            best_energy=None,
            trace_summary=(),
            objective_calls=0,
            optimizer="cobyla",
            seed=selected.seed,
            optimized_parameter_binding=None,
            audit={},
            metrics={},
            violations=(f"{type(exc).__name__}: {exc}",),
        )
        return EvaluationRun(result, (), None)


def evaluate_public_problem(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> EvaluationRun:
    return evaluate_ansatz(
        hamiltonian_from_problem(problem),
        spec,
        backend_target=_backend_target(problem),
        protocol=protocol,
        initial_occupation=problem.initial_state.occupation,
    )
