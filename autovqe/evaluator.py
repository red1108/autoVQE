"""Trusted compilation, resource accounting, and VQE evaluation.

Candidate submissions contain only an :class:`AnsatzSpec`. Initial-state
preparation, optimization settings, parameter values, and every reported
metric are evaluator-owned.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from .ansatz_ir import AnsatzIRValidationError, AnsatzSpec, OperationSpec
from .backend import (
    BackendTarget,
    backend_target_from_problem,
    canonical_backend_target,
    compiled_metrics,
    transpile_and_report,
)
from .compiler import CompiledAnsatz, compile_ansatz
from .contracts import PublicProblem
from .problem import hamiltonian_from_problem


EVALUATOR_SEED = 7
HAMILTONIAN_HERMITICITY_TOLERANCE = 1e-12
EXPECTATION_IMAGINARY_TOLERANCE = 1e-10
_COMBINABLE_ROTATION_MACROS = frozenset(
    {"PauliRotation", "XYExchange", "IsotropicExchange"}
)


class EvaluationError(RuntimeError):
    """Raised when evaluator-owned settings or inputs are invalid."""


class _BudgetExhausted(RuntimeError):
    pass


class _CandidateAuditError(EvaluationError):
    """A candidate-specific audit failure, not an evaluator failure."""


@dataclass(frozen=True)
class EvaluationProtocol:
    """Evaluator-owned, fixed-COBYLA protocol and resource-audit settings."""

    max_evals: int = 80
    restarts: int = 2
    seed: int = EVALUATOR_SEED
    initial_scale: float = 0.15
    transpile_optimization_level: int = 1
    audit_binding_scale: float = 0.271828
    audit_binding_count: int = 3

    def validate(self) -> None:
        if isinstance(self.max_evals, bool) or not isinstance(self.max_evals, int):
            raise EvaluationError("max_evals must be an integer")
        if self.max_evals <= 0:
            raise EvaluationError("max_evals must be positive")
        if isinstance(self.restarts, bool) or not isinstance(self.restarts, int):
            raise EvaluationError("restarts must be an integer")
        if self.restarts <= 0:
            raise EvaluationError("restarts must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EvaluationError("seed must be an integer")
        if not np.isfinite(self.initial_scale) or self.initial_scale < 0:
            raise EvaluationError("initial_scale must be finite and non-negative")
        if (
            isinstance(self.transpile_optimization_level, bool)
            or not isinstance(self.transpile_optimization_level, int)
            or self.transpile_optimization_level not in {0, 1, 2, 3}
        ):
            raise EvaluationError("transpile_optimization_level must be 0, 1, 2, or 3")
        if not np.isfinite(self.audit_binding_scale) or self.audit_binding_scale <= 0:
            raise EvaluationError("audit_binding_scale must be finite and positive")
        if (
            isinstance(self.audit_binding_count, bool)
            or not isinstance(self.audit_binding_count, int)
            or self.audit_binding_count < 3
        ):
            raise EvaluationError("audit_binding_count must be an integer of at least 3")


@dataclass(frozen=True)
class ResourceAudit:
    """Compile-only evidence available before any objective evaluation."""

    valid: bool
    audit: dict[str, Any]
    metrics: dict[str, int]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluator-owned numerical result safe to serialize as evidence.

    ``trace_summary`` contains ``(objective_call, best_energy_so_far)`` points
    at exponentially spaced calls plus the last call. This preserves useful
    convergence evidence without placing every optimizer sample in run state.
    """

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
    """Evaluation result plus non-serialized evaluator working state."""

    result: EvaluationResult
    best_values: tuple[float, ...]
    final_circuit: QuantumCircuit | None


@dataclass(frozen=True)
class _ResourceAuditRun:
    result: ResourceAudit
    compiled: CompiledAnsatz | None = None
    parameter_names: tuple[str, ...] = ()


def _parameter_incidence(
    parsed: AnsatzSpec,
) -> dict[str, list[tuple[int, str, float]]]:
    incidence: dict[str, list[tuple[int, str, float]]] = {
        parameter.name: [] for parameter in parsed.parameters
    }
    for operation_index, operation in enumerate(parsed.operations):
        for argument_name, expression in sorted(operation.parameters.items()):
            for term in expression.terms:
                incidence[term.parameter.name].append(
                    (operation_index, argument_name, float(term.coefficient))
                )
    return incidence


def _same_rotation_generator(left: Any, right: Any) -> bool:
    return bool(
        left.macro in _COMBINABLE_ROTATION_MACROS
        and right.macro == left.macro
        and right.qubits == left.qubits
        and right.options == left.options
        and set(left.parameters) == {"angle"}
        and set(right.parameters) == {"angle"}
    )


def _operation_order_key(operation: Any) -> tuple[Any, ...]:
    parameter_shape = tuple(
        (
            argument,
            tuple(sorted(float(term.coefficient) for term in expression.terms)),
            float(expression.constant),
        )
        for argument, expression in sorted(operation.parameters.items())
    )
    return (
        operation.macro,
        tuple(operation.qubits),
        json.dumps(operation.to_dict()["options"], sort_keys=True, separators=(",", ":")),
        parameter_shape,
    )


def _identity_generator_operations(operations: Sequence[Any]) -> tuple[Any, ...]:
    """Expand trusted shorthand into canonical Pauli-generator rotations."""

    expanded: list[Any] = []
    for operation in operations:
        if operation.macro == "PauliRotation":
            pairs = sorted(
                zip(
                    operation.qubits,
                    str(operation.options["pauli"]),
                    strict=True,
                )
            )
            expanded.append(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=tuple(qubit for qubit, _ in pairs),
                    parameters={"angle": operation.parameters["angle"]},
                    options={"pauli": "".join(letter for _, letter in pairs)},
                )
            )
            continue
        if operation.macro in {"XYExchange", "IsotropicExchange"}:
            qubits = tuple(sorted(operation.qubits))
            paulis = ("XX", "YY", "ZZ") if operation.macro == "IsotropicExchange" else ("XX", "YY")
            expanded.extend(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=qubits,
                    parameters={"angle": operation.parameters["angle"]},
                    options={"pauli": pauli},
                )
                for pauli in paulis
            )
            continue
        expanded.append(operation)
    return tuple(expanded)


def _operations_provably_commute(left: Any, right: Any) -> bool:
    if set(left.qubits).isdisjoint(right.qubits):
        return True
    if left.macro != "PauliRotation" or right.macro != "PauliRotation":
        return False
    left_letters = dict(zip(left.qubits, left.options["pauli"], strict=True))
    right_letters = dict(zip(right.qubits, right.options["pauli"], strict=True))
    anticommuting_factors = sum(
        left_letters[qubit] != right_letters[qubit]
        for qubit in set(left_letters) & set(right_letters)
    )
    return anticommuting_factors % 2 == 0


def _canonical_disjoint_order(operations: tuple[Any, ...]) -> tuple[Any, ...]:
    """Choose one order while preserving every noncommuting dependency."""

    count = len(operations)
    successors: list[list[int]] = [[] for _ in range(count)]
    indegree = [0] * count
    for left in range(count):
        for right in range(left + 1, count):
            if not _operations_provably_commute(operations[left], operations[right]):
                successors[left].append(right)
                indegree[right] += 1

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    ordered: list[Any] = []
    while ready:
        chosen = min(ready, key=lambda index: (_operation_order_key(operations[index]), index))
        ready.remove(chosen)
        ordered.append(operations[chosen])
        for successor in successors[chosen]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(ordered) != count:
        raise EvaluationError("operation dependency graph contains a cycle")
    return tuple(ordered)


def _coalesced_rotations(parsed: AnsatzSpec) -> tuple[Any, ...]:
    """Canonicalize disjoint commutations and combine identical generators."""

    combined: list[Any] = []
    identity_operations = _identity_generator_operations(parsed.operations)
    for operation in _canonical_disjoint_order(identity_operations):
        if combined and _same_rotation_generator(combined[-1], operation):
            previous = combined.pop()
            angle = previous.parameters["angle"].plus(operation.parameters["angle"])
            if angle.terms or angle.constant != 0.0:
                combined.append(
                    type(operation)(
                        macro=operation.macro,
                        qubits=operation.qubits,
                        parameters={"angle": angle},
                        options=operation.options,
                    )
                )
            continue
        combined.append(operation)
    return tuple(combined)


def _normalized_incidence_signature(
    entries: Sequence[tuple[int, str, float]],
) -> tuple[tuple[int, str, float], ...]:
    if not entries:
        return ()
    pivot = float(entries[0][2])
    return tuple(
        (operation_index, argument_name, float(coefficient) / pivot)
        for operation_index, argument_name, coefficient in entries
    )


def _semantic_parameter_order(parsed: AnsatzSpec) -> tuple[str, ...]:
    incidence = _parameter_incidence(parsed)
    return tuple(
        sorted(
            incidence,
            key=lambda name: (
                _normalized_incidence_signature(incidence[name]),
                name,
            ),
        )
    )


def _canonical_row_space(
    rows: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-12,
) -> tuple[tuple[float, ...], ...]:
    """Return deterministic reduced-row-echelon coordinates for one span."""

    matrix = [list(map(float, row)) for row in rows]
    if not matrix:
        return ()
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise EvaluationError("parameter incidence rows have inconsistent width")
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(matrix))
                if abs(matrix[row][column]) > tolerance
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row == pivot_row:
                continue
            factor = matrix[row][column]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[pivot_row], strict=True
                )
            ]
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    canonical: list[tuple[float, ...]] = []
    for row in matrix:
        cleaned = tuple(
            0.0 if abs(value) <= tolerance else round(float(value), 12)
            for value in row
        )
        if any(value != 0.0 for value in cleaned):
            canonical.append(cleaned)
    return tuple(canonical)


def _canonical_spec(spec: AnsatzSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Return an operation and parameter-subspace identity for one family."""

    parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
    operations = _coalesced_rotations(parsed)
    coordinates = [
        (operation_index, argument_name, expression)
        for operation_index, operation in enumerate(operations)
        for argument_name, expression in sorted(operation.parameters.items())
    ]
    parameter_vectors: dict[str, list[float]] = {
        parameter.name: [0.0] * len(coordinates) for parameter in parsed.parameters
    }
    for coordinate_index, (_, _, expression) in enumerate(coordinates):
        for term in expression.terms:
            parameter_vectors[term.parameter.name][coordinate_index] = float(
                term.coefficient
            )
    parameter_subspace = _canonical_row_space(tuple(parameter_vectors.values()))

    normalized_operations: list[dict[str, Any]] = []
    for operation in operations:
        normalized_operations.append(
            {
                "macro": operation.macro,
                "qubits": list(operation.qubits),
                "parameter_arguments": [
                    {
                        "name": argument_name,
                        "constant": float(expression.constant),
                    }
                    for argument_name, expression in sorted(
                        operation.parameters.items()
                    )
                ],
                "options": operation.to_dict()["options"],
            }
        )

    return {
        "semantic_identity_version": 5,
        "version": parsed.version,
        "num_qubits": parsed.num_qubits,
        "parameter_count": len(parameter_subspace),
        "parameter_subspace": parameter_subspace,
        "operations": normalized_operations,
    }


def candidate_identity(spec: AnsatzSpec | Mapping[str, Any]) -> str:
    """Return canonical text for semantic duplicate detection and audit seeds."""

    try:
        canonical = _canonical_spec(spec)
        prefix = "valid:"
    except (TypeError, ValueError, KeyError):
        canonical = dict(spec) if isinstance(spec, Mapping) else spec.to_dict()
        prefix = "invalid:"
    return prefix + json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _validate_hamiltonian(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    if not isinstance(hamiltonian, SparsePauliOp):
        raise EvaluationError("Hamiltonian must be a SparsePauliOp")
    coefficients = np.asarray(hamiltonian.coeffs, dtype=complex)
    if not np.all(np.isfinite(coefficients.real)) or not np.all(
        np.isfinite(coefficients.imag)
    ):
        raise EvaluationError("Hamiltonian coefficients must be finite")
    simplified = hamiltonian.simplify(atol=1e-14)
    canonical_coefficients = np.asarray(simplified.coeffs, dtype=complex)
    if np.any(
        np.abs(canonical_coefficients.imag) > HAMILTONIAN_HERMITICITY_TOLERANCE
    ):
        raise EvaluationError("Hamiltonian must be Hermitian with real Pauli coefficients")
    return simplified


def _validated_occupation(
    occupation: Sequence[int] | None,
    *,
    num_qubits: int,
) -> tuple[int, ...] | None:
    if occupation is None:
        return None
    checked = tuple(occupation)
    if len(checked) != num_qubits or any(
        isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1)
        for bit in checked
    ):
        raise EvaluationError(
            "initial occupation must contain one integer 0/1 bit per qubit"
        )
    return checked


def _prepend_initial_state(
    compiled: CompiledAnsatz,
    occupation: Sequence[int] | None,
) -> CompiledAnsatz:
    checked = _validated_occupation(occupation, num_qubits=compiled.circuit.num_qubits)
    if checked is None or not any(checked):
        return compiled
    circuit = QuantumCircuit(compiled.circuit.num_qubits, name=compiled.circuit.name)
    for qubit, bit in enumerate(checked):
        if bit:
            circuit.x(qubit)
    circuit.compose(compiled.circuit, inplace=True)
    return CompiledAnsatz(
        circuit=circuit,
        parameters=compiled.parameters,
        audit=compiled.audit,
    )


def _physical_metrics(
    circuit: QuantumCircuit,
    backend_target: BackendTarget | None,
    *,
    optimization_level: int,
) -> dict[str, int]:
    if backend_target is None:
        return compiled_metrics(circuit)
    _, metrics = transpile_and_report(
        circuit,
        backend_target,
        optimization_level=optimization_level,
    )
    return metrics


def _parameter_order(
    compiled: CompiledAnsatz,
    parsed: AnsatzSpec | None = None,
) -> tuple[str, ...]:
    if parsed is None:
        return tuple(compiled.audit.trainable_parameter_names)
    ordered = _semantic_parameter_order(parsed)
    if set(ordered) != set(compiled.parameters):
        raise EvaluationError("semantic parameter order does not match compiled parameters")
    return ordered


def _bind(
    compiled: CompiledAnsatz,
    names: tuple[str, ...],
    values: np.ndarray,
) -> QuantumCircuit:
    if values.shape != (len(names),):
        raise EvaluationError("parameter vector has the wrong shape")
    mapping = {
        compiled.parameters[name]: float(value)
        for name, value in zip(names, values, strict=True)
    }
    return compiled.circuit.assign_parameters(mapping, inplace=False)


def _expectation(state: Statevector, hamiltonian: SparsePauliOp) -> float:
    if state.num_qubits != hamiltonian.num_qubits:
        raise EvaluationError("ansatz and Hamiltonian qubit counts differ")
    value = complex(state.expectation_value(hamiltonian))
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise EvaluationError("Hamiltonian expectation value must be finite")
    tolerance = EXPECTATION_IMAGINARY_TOLERANCE * max(1.0, abs(value.real))
    if abs(value.imag) > tolerance:
        raise EvaluationError("Hermitian Hamiltonian produced a non-real expectation value")
    return float(value.real)


def _energy(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.parameters:
        raise EvaluationError("trusted evaluator received an unbound circuit")
    state = Statevector.from_instruction(circuit)
    return _expectation(state, hamiltonian)


def _summarize_trace(best_trace: Sequence[float]) -> tuple[tuple[int, float], ...]:
    if not best_trace:
        return ()
    selected = {1, len(best_trace)}
    call = 2
    while call < len(best_trace):
        selected.add(call)
        call *= 2
    return tuple((index, float(best_trace[index - 1])) for index in sorted(selected))


def _optimize(
    compiled: CompiledAnsatz,
    hamiltonian: SparsePauliOp,
    protocol: EvaluationProtocol,
    names: tuple[str, ...],
) -> tuple[
    np.ndarray,
    float,
    int,
    tuple[tuple[int, float], ...],
    float,
]:
    num_params = len(names)
    calls = 0
    best_trace: list[float] = []
    best_energy = float("inf")
    best_values = np.zeros(num_params, dtype=float)
    minimum_observed = float("inf")
    maximum_observed = float("-inf")

    def objective(values: np.ndarray) -> float:
        nonlocal calls, best_energy, best_values
        nonlocal minimum_observed, maximum_observed
        if calls >= protocol.max_evals:
            raise _BudgetExhausted("objective-call budget exhausted")
        values = np.asarray(values, dtype=float)
        energy = _energy(_bind(compiled, names, values), hamiltonian)
        calls += 1
        minimum_observed = min(minimum_observed, energy)
        maximum_observed = max(maximum_observed, energy)
        if energy < best_energy:
            best_energy = energy
            best_values = values.copy()
        best_trace.append(best_energy)
        return energy

    if num_params == 0:
        objective(np.zeros(0, dtype=float))
        return (
            best_values,
            best_energy,
            calls,
            _summarize_trace(best_trace),
            maximum_observed - minimum_observed,
        )

    for restart in range(protocol.restarts):
        if calls >= protocol.max_evals:
            break
        rng = np.random.default_rng(protocol.seed + 1009 * restart)
        initial = rng.uniform(
            -protocol.initial_scale,
            protocol.initial_scale,
            size=num_params,
        )
        remaining = protocol.max_evals - calls
        try:
            minimize(
                objective,
                initial,
                method="COBYLA",
                options={
                    "maxiter": remaining,
                    "rhobeg": max(0.05, protocol.initial_scale),
                    "tol": 1e-8,
                },
            )
        except _BudgetExhausted:
            pass

    if calls == 0:
        raise EvaluationError("optimizer did not evaluate the objective")
    return (
        best_values,
        best_energy,
        calls,
        _summarize_trace(best_trace),
        maximum_observed - minimum_observed,
    )


def _hamiltonian_active_norm(hamiltonian: SparsePauliOp) -> float:
    coefficients = [
        abs(complex(coefficient))
        for label, coefficient in zip(
            hamiltonian.paulis.to_labels(),
            hamiltonian.coeffs,
            strict=True,
        )
        if any(factor != "I" for factor in label)
    ]
    return float(np.linalg.norm(coefficients)) if coefficients else 0.0


def _prefix_metrics(prefix: str, metrics: Mapping[str, int]) -> dict[str, int]:
    return {f"{prefix}_{key}": int(value) for key, value in metrics.items()}


def _audit_rng(identity: str, seed: int) -> np.random.Generator:
    material = f"autovqe-resource-audit-v1:{seed}:{identity}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(material))


def _worst_metrics(samples: list[Mapping[str, int]]) -> dict[str, int]:
    if not samples:
        raise EvaluationError("resource audit requires at least one binding")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise EvaluationError("resource metric keys changed across audit bindings")
    return {key: max(int(sample[key]) for sample in samples) for key in sorted(keys)}


def _audit_candidate_resources(
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    expected_num_qubits: int,
    backend_target: BackendTarget | None,
    protocol: EvaluationProtocol,
    initial_occupation: Sequence[int] | None,
) -> _ResourceAuditRun:
    try:
        identity = candidate_identity(spec)
        parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
        compiled = compile_ansatz(parsed)
        if compiled.circuit.num_qubits != expected_num_qubits:
            raise _CandidateAuditError(
                "AnsatzSpec num_qubits does not match the problem"
            )
        compiled = _prepend_initial_state(compiled, initial_occupation)
        names = _parameter_order(compiled, parsed)
        canonical_target = canonical_backend_target()
        template_metrics = _physical_metrics(
            compiled.circuit,
            backend_target,
            optimization_level=protocol.transpile_optimization_level,
        )
        canonical_template_metrics = _physical_metrics(
            compiled.circuit,
            canonical_target,
            optimization_level=protocol.transpile_optimization_level,
        )

        rng = _audit_rng(identity, protocol.seed)
        audit_samples: list[Mapping[str, int]] = []
        canonical_audit_samples: list[Mapping[str, int]] = []
        for _ in range(protocol.audit_binding_count):
            values = rng.uniform(
                -protocol.audit_binding_scale,
                protocol.audit_binding_scale,
                size=len(names),
            )
            bound = _bind(compiled, names, values)
            audit_samples.append(
                _physical_metrics(
                    bound,
                    backend_target,
                    optimization_level=protocol.transpile_optimization_level,
                )
            )
            canonical_audit_samples.append(
                _physical_metrics(
                    bound,
                    canonical_target,
                    optimization_level=protocol.transpile_optimization_level,
                )
            )

        metrics = {
            **_prefix_metrics("template", template_metrics),
            **_prefix_metrics("audit_worst", _worst_metrics(audit_samples)),
            **_prefix_metrics("canonical_template", canonical_template_metrics),
            **_prefix_metrics(
                "canonical_audit_worst",
                _worst_metrics(canonical_audit_samples),
            ),
            "audit_binding_count": protocol.audit_binding_count,
        }
        result = ResourceAudit(
            valid=True,
            audit=compiled.audit.to_dict(),
            metrics=metrics,
        )
        return _ResourceAuditRun(
            result=result,
            compiled=compiled,
            parameter_names=names,
        )
    except (AnsatzIRValidationError, _CandidateAuditError) as exc:
        return _ResourceAuditRun(
            result=ResourceAudit(
                valid=False,
                audit={},
                metrics={},
                violations=(f"{type(exc).__name__}: {exc}",),
            )
        )


def audit_public_candidate(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> ResourceAudit:
    """Compile and conservatively account for a candidate without optimizing."""

    selected_protocol = protocol or EvaluationProtocol()
    selected_protocol.validate()
    run = _audit_candidate_resources(
        spec,
        expected_num_qubits=problem.num_qubits,
        backend_target=backend_target_from_problem(problem),
        protocol=selected_protocol,
        initial_occupation=problem.initial_state.occupation,
    )
    return run.result


def evaluate_ansatz(
    hamiltonian: SparsePauliOp,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    backend_target: BackendTarget | None = None,
    protocol: EvaluationProtocol | None = None,
    initial_occupation: Sequence[int] | None = None,
) -> EvaluationRun:
    """Compile and optimize an ansatz without trusting candidate-owned values."""

    selected_protocol = protocol or EvaluationProtocol()
    selected_protocol.validate()
    try:
        canonical_hamiltonian = _validate_hamiltonian(hamiltonian)
        audit_run = _audit_candidate_resources(
            spec,
            expected_num_qubits=canonical_hamiltonian.num_qubits,
            backend_target=backend_target,
            protocol=selected_protocol,
            initial_occupation=initial_occupation,
        )
        if not audit_run.result.valid:
            raise EvaluationError(audit_run.result.violations[0])
        assert audit_run.compiled is not None
        (
            best_values,
            best_energy,
            calls,
            trace_summary,
            objective_energy_span,
        ) = _optimize(
            audit_run.compiled,
            canonical_hamiltonian,
            selected_protocol,
            audit_run.parameter_names,
        )
        hamiltonian_active_norm = _hamiltonian_active_norm(canonical_hamiltonian)
        constant_hamiltonian = hamiltonian_active_norm == 0.0
        objective_activity_fraction = (
            None
            if constant_hamiltonian
            else objective_energy_span / hamiltonian_active_norm
        )
        final_circuit = _bind(
            audit_run.compiled,
            audit_run.parameter_names,
            best_values,
        )
        final_metrics = _physical_metrics(
            final_circuit,
            backend_target,
            optimization_level=selected_protocol.transpile_optimization_level,
        )
        canonical_final_metrics = _physical_metrics(
            final_circuit,
            canonical_backend_target(),
            optimization_level=selected_protocol.transpile_optimization_level,
        )
        metrics = {
            **audit_run.result.metrics,
            **_prefix_metrics("final", final_metrics),
            **_prefix_metrics("canonical_final", canonical_final_metrics),
        }
        result = EvaluationResult(
            valid=True,
            best_energy=best_energy,
            trace_summary=trace_summary,
            objective_calls=calls,
            optimizer="cobyla",
            seed=selected_protocol.seed,
            optimized_parameter_binding={
                name: float(value)
                for name, value in zip(
                    audit_run.parameter_names,
                    best_values,
                    strict=True,
                )
            },
            audit=audit_run.result.audit,
            metrics=metrics,
            objective_energy_span=objective_energy_span,
            hamiltonian_active_norm=hamiltonian_active_norm,
            objective_activity_fraction=objective_activity_fraction,
            constant_hamiltonian=constant_hamiltonian,
        )
        return EvaluationRun(
            result=result,
            best_values=tuple(float(value) for value in best_values),
            final_circuit=final_circuit,
        )
    except Exception as exc:
        result = EvaluationResult(
            valid=False,
            best_energy=None,
            trace_summary=(),
            objective_calls=0,
            optimizer="cobyla",
            seed=selected_protocol.seed,
            optimized_parameter_binding=None,
            audit={},
            metrics={},
            violations=(f"{type(exc).__name__}: {exc}",),
        )
        return EvaluationRun(result=result, best_values=(), final_circuit=None)


def evaluate_public_problem(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> EvaluationRun:
    """Evaluate a candidate with the problem's initial state prepended internally."""

    return evaluate_ansatz(
        hamiltonian_from_problem(problem),
        spec,
        backend_target=backend_target_from_problem(problem),
        protocol=protocol,
        initial_occupation=problem.initial_state.occupation,
    )


__all__ = [
    "EVALUATOR_SEED",
    "EvaluationError",
    "EvaluationProtocol",
    "EvaluationResult",
    "EvaluationRun",
    "ResourceAudit",
    "audit_public_candidate",
    "candidate_identity",
    "evaluate_ansatz",
    "evaluate_public_problem",
]
