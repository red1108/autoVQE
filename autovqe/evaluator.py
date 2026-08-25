from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from . import prepare
from .ansatz_ir import AnsatzSpec
from .compiler import CompiledAnsatz, compile_ansatz
from .contracts import PublicProblem


CANONICAL_BASIS_GATES = ("rz", "sx", "x", "cx")
HAMILTONIAN_HERMITICITY_TOLERANCE = 1e-12
EXPECTATION_IMAGINARY_TOLERANCE = 1e-10
_COMBINABLE_ROTATION_MACROS = frozenset(
    {"PauliRotation", "XYExchange", "IsotropicExchange"}
)


class EvaluationError(RuntimeError):
    """Raised when the trusted evaluator cannot evaluate a submission."""


class _BudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationProtocol:
    """Evaluator-owned optimization settings.

    A scored candidate never supplies this object. Keeping it separate from
    ``AnsatzSpec`` prevents optimizer or seed changes from masquerading as an
    ansatz improvement.
    """

    optimizer: str = "cobyla"
    max_evals: int = 80
    restarts: int = 2
    seed: int = prepare.SEED
    initial_scale: float = 0.15
    learning_rate: float = 0.18
    spsa_steps: int = 24
    transpile_optimization_level: int = 1
    generic_binding_scale: float = 0.271828
    generic_binding_count: int = 3

    def validate(self) -> None:
        if self.optimizer not in {"cobyla", "spsa"}:
            raise EvaluationError(f"unsupported evaluator optimizer: {self.optimizer}")
        if self.max_evals <= 0:
            raise EvaluationError("max_evals must be positive")
        if self.restarts <= 0:
            raise EvaluationError("restarts must be positive")
        if self.initial_scale < 0:
            raise EvaluationError("initial_scale must be non-negative")
        if not np.isfinite(self.generic_binding_scale) or self.generic_binding_scale <= 0:
            raise EvaluationError("generic_binding_scale must be finite and positive")
        if (
            isinstance(self.generic_binding_count, bool)
            or not isinstance(self.generic_binding_count, int)
            or self.generic_binding_count < 3
        ):
            raise EvaluationError("generic_binding_count must be an integer of at least 3")


@dataclass(frozen=True)
class EvaluationReceipt:
    valid: bool
    best_energy: float | None
    energy_trace: tuple[float, ...]
    best_energy_trace: tuple[float, ...]
    objective_calls: int
    optimizer: str
    seed: int
    optimized_parameter_binding: dict[str, float] | None
    audit: dict[str, Any]
    metrics: dict[str, int]
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrivateEvaluationResult:
    """Evaluator-internal circuit state accompanying measured results.

    The optimized scalar binding is deliberately copied into the result so a
    terminal result can be reproduced without trusting agent-authored values.
    The compiled circuit remains evaluator-internal.
    """

    receipt: EvaluationReceipt
    best_values: tuple[float, ...]
    final_circuit: QuantumCircuit | None


def _parameter_incidence(
    parsed: AnsatzSpec,
) -> tuple[tuple[Any, ...], dict[str, list[tuple[int, str, float]]]]:
    operations = tuple(parsed.iter_operations())
    incidence: dict[str, list[tuple[int, str, float]]] = {
        parameter.name: [] for parameter in parsed.parameters
    }
    for operation_index, operation in enumerate(operations):
        for argument_name, expression in sorted(operation.parameters.items()):
            for term in expression.terms:
                incidence[term.parameter.name].append(
                    (operation_index, argument_name, float(term.coefficient))
                )
    return operations, incidence


def _same_rotation_generator(left: Any, right: Any) -> bool:
    return bool(
        left.macro in _COMBINABLE_ROTATION_MACROS
        and right.macro == left.macro
        and right.qubits == left.qubits
        and right.options == left.options
        and set(left.parameters) == {"angle"}
        and set(right.parameters) == {"angle"}
    )


def _coalesced_rotations(parsed: AnsatzSpec) -> tuple[Any, ...]:
    """Collapse adjacent exponentials of the same trusted generator.

    Layer boundaries are already presentation-only in the semantic identity.  For
    the three trusted one-parameter exponential macros, consecutive equal
    generators also satisfy ``exp(-iaG) exp(-ibG) = exp(-i(a+b)G)``.  Folding
    them prevents split/cancel submissions from obtaining a fresh evaluator
    seed while deliberately leaving non-adjacent/noncommuting algebra alone.
    """

    combined: list[Any] = []
    for operation in parsed.iter_operations():
        if combined and _same_rotation_generator(combined[-1], operation):
            previous = combined.pop()
            angle = previous.parameters["angle"].plus(
                operation.parameters["angle"]
            )
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


def _incidence_for_operations(
    operations: tuple[Any, ...],
) -> dict[str, list[tuple[int, str, float]]]:
    incidence: dict[str, list[tuple[int, str, float]]] = {}
    for operation_index, operation in enumerate(operations):
        for argument_name, expression in sorted(operation.parameters.items()):
            for term in expression.terms:
                incidence.setdefault(term.parameter.name, []).append(
                    (operation_index, argument_name, float(term.coefficient))
                )
    return incidence


def _semantic_parameter_order(parsed: AnsatzSpec) -> tuple[str, ...]:
    _, incidence = _parameter_incidence(parsed)
    return tuple(
        sorted(incidence, key=lambda name: (tuple(incidence[name]), name))
    )


def _canonical_spec(spec: AnsatzSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Return an alpha-normalized, presentation-independent ansatz family.

    Candidate names, layer labels/boundaries, parameter spelling, and
    declaration order do not change the physical variational family.  Including
    those fields would let an agent obtain fresh evaluator generic bindings by
    resubmitting the same circuit with cosmetic edits.
    """

    parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
    operations = _coalesced_rotations(parsed)
    incidence = _incidence_for_operations(operations)

    # Incidence signatures are invariant to alpha-renaming.  Sorting by the
    # old name only breaks ties between parameters with identical incidence;
    # such parameters occur together with identical coefficients, so the
    # serialized multiset of assigned canonical names is still invariant.
    ordered_names = sorted(
        incidence,
        key=lambda name: (tuple(incidence[name]), name),
    )
    normalized_names = {
        name: f"p{index}" for index, name in enumerate(ordered_names)
    }

    normalized_operations: list[dict[str, Any]] = []
    for operation in operations:
        normalized_parameters: dict[str, Any] = {}
        for argument_name, expression in sorted(operation.parameters.items()):
            normalized_parameters[argument_name] = {
                "terms": sorted(
                    (
                        {
                            "parameter": normalized_names[term.parameter.name],
                            "coefficient": float(term.coefficient),
                        }
                        for term in expression.terms
                    ),
                    key=lambda term: (term["parameter"], term["coefficient"]),
                ),
                "constant": float(expression.constant),
            }
        normalized_operations.append(
            {
                "macro": operation.macro,
                "qubits": list(operation.qubits),
                "parameters": normalized_parameters,
                "options": operation.to_dict()["options"],
            }
        )

    return {
        "semantic_identity_version": 1,
        "version": parsed.version,
        "num_qubits": parsed.num_qubits,
        "parameter_count": len(incidence),
        "reference": None if parsed.reference is None else parsed.reference.to_dict(),
        "operations": normalized_operations,
    }


def candidate_identity(spec: AnsatzSpec | Mapping[str, Any]) -> str:
    """Return canonical text for semantic duplicate detection and sampling."""

    try:
        canonical = _canonical_spec(spec)
        prefix = "valid:"
    except (TypeError, ValueError, KeyError):
        # Invalid submissions still need a stable internal identity. They
        # cannot be evaluated or deduplicated as a legal physical family, so
        # keep them in a separate namespace from compiled candidates.
        canonical = dict(spec) if isinstance(spec, Mapping) else spec.to_dict()
        prefix = "invalid:"
    return prefix + json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def hamiltonian_from_public(problem: PublicProblem) -> SparsePauliOp:
    return SparsePauliOp.from_list(
        [
            (term.pauli, complex(term.real, term.imag))
            for term in problem.pauli_terms
        ]
    ).simplify(atol=1e-14)


def _validate_hamiltonian(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    """Return a canonical finite Hermitian Hamiltonian.

    ``PublicProblem`` already enforces this contract, but ``evaluate_ansatz``
    is also a public low-level entry point.  It must not silently turn a
    non-Hermitian objective into a different problem by taking only the real
    part of its expectation value.
    """

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
        np.abs(canonical_coefficients.imag)
        > HAMILTONIAN_HERMITICITY_TOLERANCE
    ):
        raise EvaluationError(
            "Hamiltonian must be Hermitian with real Pauli coefficients"
        )
    return simplified


def _validate_public_reference(
    parsed: AnsatzSpec,
    expected_qubits: tuple[int, ...],
) -> None:
    """Enforce the evaluator-owned computational-basis preparation."""

    if expected_qubits:
        if (
            parsed.reference is None
            or parsed.reference.macro != "X"
            or parsed.reference.qubits != expected_qubits
        ):
            raise EvaluationError(
                "candidate reference must exactly match the evaluator-owned "
                "public computational-basis reference"
            )
    elif parsed.reference is not None:
        raise EvaluationError(
            "candidate cannot introduce a reference preparation when the "
            "public problem declares none"
        )


def backend_target_from_public(
    problem: PublicProblem,
) -> prepare.BackendTarget | None:
    """Return the declared lowering target, or no target when unspecified.

    Backend basis names are used only after trusted IR compilation for physical
    lowering and accounting.  They never add entries to the macro allowlist.
    """

    coupling_map = (
        [list(edge) for edge in problem.backend.coupling_map]
        if problem.backend.coupling_map
        else None
    )
    # Connectivity without an instruction basis is not a complete transpiler
    # target; treat it as informational and fall back to logical metrics.
    if not problem.backend.basis_gates:
        return None
    return prepare.BackendTarget(
        basis_gates=list(problem.backend.basis_gates),
        coupling_map=coupling_map,
    )


def canonical_backend_target() -> prepare.BackendTarget:
    """Return the fixed candidate-independent resource-accounting target."""

    return prepare.BackendTarget(
        basis_gates=list(CANONICAL_BASIS_GATES),
        # No coupling map means all-to-all connectivity.  This removes device
        # routing differences from the cross-candidate canonical comparison.
        coupling_map=None,
    )


def _logical_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    singleq = 0
    twoq = 0
    for instruction in circuit.data:
        if instruction.operation.num_qubits == 1:
            singleq += 1
        elif instruction.operation.num_qubits == 2:
            twoq += 1
    return {
        "singleq_count": singleq,
        "twoq_count": twoq,
        "total_gate_count": len(circuit.data),
        "depth": int(circuit.depth() or 0),
    }


def _physical_metrics(
    circuit: QuantumCircuit,
    backend_target: prepare.BackendTarget | None,
    *,
    optimization_level: int,
) -> dict[str, int]:
    if backend_target is None:
        return _logical_metrics(circuit)
    _, metrics = prepare.transpile_and_report(
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


def _energy(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.num_qubits != hamiltonian.num_qubits:
        raise EvaluationError("ansatz and Hamiltonian qubit counts differ")
    if circuit.parameters:
        raise EvaluationError("trusted evaluator received an unbound circuit")
    value = complex(Statevector.from_instruction(circuit).expectation_value(hamiltonian))
    if not np.isfinite(value.real) or not np.isfinite(value.imag):
        raise EvaluationError("Hamiltonian expectation value must be finite")
    tolerance = EXPECTATION_IMAGINARY_TOLERANCE * max(1.0, abs(value.real))
    if abs(value.imag) > tolerance:
        raise EvaluationError(
            "Hermitian Hamiltonian produced a non-real expectation value"
        )
    return float(value.real)


def _optimize(
    compiled: CompiledAnsatz,
    hamiltonian: SparsePauliOp,
    protocol: EvaluationProtocol,
    names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    names = _parameter_order(compiled) if names is None else names
    num_params = len(names)
    calls = 0
    raw_trace: list[float] = []
    best_trace: list[float] = []
    best_energy = float("inf")
    best_values = np.zeros(num_params, dtype=float)

    def objective(values: np.ndarray) -> float:
        nonlocal calls, best_energy, best_values
        if calls >= protocol.max_evals:
            raise _BudgetExhausted("objective-call budget exhausted")
        values = np.asarray(values, dtype=float)
        energy = _energy(_bind(compiled, names, values), hamiltonian)
        calls += 1
        raw_trace.append(energy)
        if energy < best_energy:
            best_energy = energy
            best_values = values.copy()
        best_trace.append(best_energy)
        return energy

    if num_params == 0:
        objective(np.zeros(0, dtype=float))
        return best_values, raw_trace, best_trace

    for restart in range(protocol.restarts):
        if calls >= protocol.max_evals:
            break
        rng = np.random.default_rng(protocol.seed + 1009 * restart)
        current = rng.uniform(-protocol.initial_scale, protocol.initial_scale, size=num_params)

        if protocol.optimizer == "spsa":
            try:
                current_energy = objective(current)
            except _BudgetExhausted:
                break
            for step in range(1, protocol.spsa_steps + 1):
                if calls + 3 > protocol.max_evals:
                    break
                ck = 0.14 / step**0.101
                ak = protocol.learning_rate / (step + 2.0) ** 0.602
                delta = rng.choice((-1.0, 1.0), size=num_params)
                try:
                    plus = objective(current + ck * delta)
                    minus = objective(current - ck * delta)
                    gradient = ((plus - minus) / (2.0 * ck)) * delta
                    proposal = current - ak * gradient
                    proposal_energy = objective(proposal)
                except _BudgetExhausted:
                    break
                if proposal_energy < current_energy:
                    current = proposal
                    current_energy = proposal_energy
            continue

        remaining = protocol.max_evals - calls
        if remaining <= 0:
            break
        try:
            minimize(
                objective,
                current,
                method="COBYLA",
                options={
                    "maxiter": remaining,
                    "rhobeg": max(0.05, protocol.initial_scale),
                    "tol": 1e-8,
                },
            )
        except _BudgetExhausted:
            pass

    if not raw_trace:
        raise EvaluationError("optimizer did not evaluate the objective")
    return best_values, raw_trace, best_trace


def _prefix_metrics(prefix: str, metrics: Mapping[str, int]) -> dict[str, int]:
    return {f"{prefix}_{key}": int(value) for key, value in metrics.items()}


def _generic_rng(candidate_identity_value: str, evaluator_seed: int) -> np.random.Generator:
    """Return a deterministic, candidate-specific RNG for resource probes."""

    material = f"autovqe-generic-bindings-v1:{evaluator_seed}:{candidate_identity_value}".encode(
        "ascii"
    )
    padding = (-len(material)) % 4
    entropy = np.frombuffer(material + (b"\0" * padding), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _worst_metrics(samples: list[Mapping[str, int]]) -> dict[str, int]:
    if not samples:
        raise EvaluationError("generic resource audit requires at least one binding")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise EvaluationError("generic resource metric keys changed across bindings")
    return {key: max(int(sample[key]) for sample in samples) for key in sorted(keys)}


def evaluate_ansatz(
    hamiltonian: SparsePauliOp,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    backend_target: prepare.BackendTarget | None = None,
    protocol: EvaluationProtocol | None = None,
    _trusted_reference_qubits: tuple[int, ...] | None = None,
) -> PrivateEvaluationResult:
    """Compile and evaluate an ansatz without trusting candidate-reported data."""

    selected_protocol = protocol or EvaluationProtocol()
    selected_protocol.validate()
    identity = candidate_identity(spec)
    try:
        hamiltonian = _validate_hamiltonian(hamiltonian)
        # Compilation consults only the trusted macro registry.  Both the
        # declared backend and the canonical backend below are lowering/count
        # targets; neither can authorize a candidate-supplied operation.
        parsed = spec if isinstance(spec, AnsatzSpec) else AnsatzSpec.from_dict(spec)
        if _trusted_reference_qubits is not None:
            _validate_public_reference(parsed, _trusted_reference_qubits)
        compiled = compile_ansatz(parsed)
        if compiled.circuit.num_qubits != hamiltonian.num_qubits:
            raise EvaluationError("AnsatzSpec num_qubits does not match the Hamiltonian")

        names = _parameter_order(compiled, parsed)
        canonical_target = canonical_backend_target()
        template_metrics = _physical_metrics(
            compiled.circuit,
            backend_target,
            optimization_level=selected_protocol.transpile_optimization_level,
        )
        canonical_template_metrics = _physical_metrics(
            compiled.circuit,
            canonical_target,
            optimization_level=selected_protocol.transpile_optimization_level,
        )

        generic_rng = _generic_rng(identity, selected_protocol.seed)
        generic_samples = []
        canonical_generic_samples = []
        for _ in range(selected_protocol.generic_binding_count):
            generic_values = generic_rng.uniform(
                -selected_protocol.generic_binding_scale,
                selected_protocol.generic_binding_scale,
                size=len(names),
            )
            generic_circuit = _bind(compiled, names, generic_values)
            generic_samples.append(
                _physical_metrics(
                    generic_circuit,
                    backend_target,
                    optimization_level=selected_protocol.transpile_optimization_level,
                )
            )
            canonical_generic_samples.append(
                _physical_metrics(
                    generic_circuit,
                    canonical_target,
                    optimization_level=selected_protocol.transpile_optimization_level,
                )
            )
        generic_worst_metrics = _worst_metrics(generic_samples)
        canonical_generic_worst_metrics = _worst_metrics(canonical_generic_samples)

        best_values, raw_trace, best_trace = _optimize(
            compiled,
            hamiltonian,
            selected_protocol,
            names,
        )
        final_circuit = _bind(compiled, names, best_values)
        final_metrics = _physical_metrics(
            final_circuit,
            backend_target,
            optimization_level=selected_protocol.transpile_optimization_level,
        )
        canonical_final_metrics = _physical_metrics(
            final_circuit,
            canonical_target,
            optimization_level=selected_protocol.transpile_optimization_level,
        )
        metrics = {
            **_prefix_metrics("template", template_metrics),
            **_prefix_metrics("generic_worst", generic_worst_metrics),
            # Backward-compatible aliases.  These now mean the worst value
            # across evaluator-owned generic bindings, never one public point.
            **_prefix_metrics("generic", generic_worst_metrics),
            **_prefix_metrics("final", final_metrics),
            **_prefix_metrics("canonical_template", canonical_template_metrics),
            **_prefix_metrics(
                "canonical_generic_worst", canonical_generic_worst_metrics
            ),
            **_prefix_metrics("canonical_final", canonical_final_metrics),
            "generic_binding_count": selected_protocol.generic_binding_count,
        }
        receipt = EvaluationReceipt(
            valid=True,
            best_energy=min(raw_trace),
            energy_trace=tuple(raw_trace),
            best_energy_trace=tuple(best_trace),
            objective_calls=len(raw_trace),
            optimizer=selected_protocol.optimizer,
            seed=selected_protocol.seed,
            optimized_parameter_binding={
                name: float(value)
                for name, value in zip(names, best_values, strict=True)
            },
            audit=compiled.audit.to_dict(),
            metrics=metrics,
        )
        return PrivateEvaluationResult(
            receipt=receipt,
            best_values=tuple(float(value) for value in best_values),
            final_circuit=final_circuit,
        )
    except Exception as exc:
        receipt = EvaluationReceipt(
            valid=False,
            best_energy=None,
            energy_trace=(),
            best_energy_trace=(),
            objective_calls=0,
            optimizer=selected_protocol.optimizer,
            seed=selected_protocol.seed,
            optimized_parameter_binding=None,
            audit={},
            metrics={},
            violations=(f"{type(exc).__name__}: {exc}",),
        )
        return PrivateEvaluationResult(receipt=receipt, best_values=(), final_circuit=None)


def evaluate_public_problem(
    problem: PublicProblem,
    spec: AnsatzSpec | Mapping[str, Any],
    *,
    protocol: EvaluationProtocol | None = None,
) -> PrivateEvaluationResult:
    """Trusted convenience entry point for an agent-safe public problem."""

    return evaluate_ansatz(
        hamiltonian_from_public(problem),
        spec,
        backend_target=backend_target_from_public(problem),
        protocol=protocol,
        _trusted_reference_qubits=tuple(
            index
            for index, bit in enumerate(problem.reference.occupation or ())
            if bit
        ),
    )
