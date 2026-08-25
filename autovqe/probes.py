"""Lean evaluator-owned probes for structure discovery and gate validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .ansatz import OperationSpec
from .problem import PublicProblem, hamiltonian_from_problem

EXACT_SYMMETRY_TOLERANCE = 1e-10
MIN_SPECIAL_CHARGE_FRACTION = 1e-3
GENERATOR_HERMITICITY_TOLERANCE = 1e-12
MIN_GENERATOR_NORM = 1e-8
MAX_GENERATOR_TERMS = 256
MAX_GENERATOR_COEFFICIENT = 1e6
MAX_COMMUTATOR_TERM_PRODUCTS = 65_536
PROBE_COST_TERM_BLOCK = 4_096


class ProbeValidationError(ValueError):
    """Raised when a probe is malformed or scientifically vacuous."""


@dataclass(frozen=True)
class ProbeResult:
    probe_type: str
    metrics: dict[str, float | int | bool | str | list[float]]
    cost_units: float
    valid: bool = True
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def initial_state_circuit(problem: PublicProblem) -> QuantumCircuit:
    circuit = QuantumCircuit(problem.num_qubits)
    for qubit, bit in enumerate(problem.initial_state.occupation or ()):
        if bit:
            circuit.x(qubit)
    return circuit


def _coefficient_norm(operator: SparsePauliOp) -> float:
    return float(np.linalg.norm(np.asarray(operator.simplify(atol=1e-14).coeffs, complex)))


def _center_operator(operator: SparsePauliOp) -> SparsePauliOp:
    identity = "I" * operator.num_qubits
    terms = [
        (label, complex(coeff))
        for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs, strict=True)
        if label != identity
    ]
    return SparsePauliOp.from_list(terms or [(identity, 0.0)]).simplify(atol=1e-14)


def _finite_hermitian(operator: SparsePauliOp, label: str) -> SparsePauliOp:
    if not isinstance(operator, SparsePauliOp):
        raise ProbeValidationError(f"{label} must be a SparsePauliOp")
    coefficients = np.asarray(operator.coeffs, dtype=complex)
    if not np.all(np.isfinite(coefficients.real)) or not np.all(np.isfinite(coefficients.imag)):
        raise ProbeValidationError(f"{label} coefficients must be finite")
    simplified = operator.simplify(atol=1e-14)
    if np.any(np.abs(np.asarray(simplified.coeffs).imag) > GENERATOR_HERMITICITY_TOLERANCE):
        raise ProbeValidationError(f"{label} must be Hermitian with real Pauli coefficients")
    return simplified


def validate_hamiltonian_observable(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    return _finite_hermitian(hamiltonian, "Hamiltonian")


def validate_generator_observable(
    generator: SparsePauliOp, *, min_norm: float = MIN_GENERATOR_NORM
) -> float:
    if not math.isfinite(float(min_norm)) or min_norm <= 0.0:
        raise ValueError("min_norm must be finite and positive")
    active_norm = _coefficient_norm(_center_operator(_finite_hermitian(generator, "generator")))
    if not math.isfinite(active_norm):
        raise ProbeValidationError("generator norm must be finite")
    if active_norm < min_norm:
        raise ProbeValidationError(
            f"generator is identity-only, zero, or below the minimum active norm {min_norm:g}"
        )
    return active_norm


def _check_term_product_cap(hamiltonian: SparsePauliOp, generator: SparsePauliOp) -> int:
    products = len(hamiltonian.paulis) * len(generator.simplify(atol=1e-14).paulis)
    if products > MAX_COMMUTATOR_TERM_PRODUCTS:
        raise ProbeValidationError(
            f"commutator probe exceeds the {MAX_COMMUTATOR_TERM_PRODUCTS}-term-product cap"
        )
    return products


def normalized_commutator(hamiltonian: SparsePauliOp, generator: SparsePauliOp) -> float:
    """Scale-invariant Pauli-coefficient norm of ``[H, Q]``."""
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    if hamiltonian.num_qubits != generator.num_qubits:
        raise ProbeValidationError("Hamiltonian and generator qubit counts differ")
    q_norm = validate_generator_observable(generator)
    h_norm = _coefficient_norm(_center_operator(hamiltonian))
    if h_norm <= 1e-14:
        raise ProbeValidationError("Hamiltonian has no non-identity component")
    generator = generator.simplify(atol=1e-14)
    _check_term_product_cap(hamiltonian, generator)
    commutator = (
        hamiltonian.compose(generator) - generator.compose(hamiltonian)
    ).simplify(atol=1e-14)
    return _coefficient_norm(commutator) / (2.0 * h_norm * q_norm)


def distance_from_hamiltonian_span(
    hamiltonian: SparsePauliOp, generator: SparsePauliOp
) -> float:
    """Relative distance of ``Q`` from the span of ``I`` and ``H``."""
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    validate_generator_observable(generator)
    if hamiltonian.num_qubits != generator.num_qubits:
        raise ProbeValidationError("Hamiltonian and generator qubit counts differ")
    centered_h = _center_operator(hamiltonian)
    centered_q = _center_operator(generator.simplify(atol=1e-14))
    h_map = dict(zip(centered_h.paulis.to_labels(), centered_h.coeffs, strict=True))
    q_map = dict(zip(centered_q.paulis.to_labels(), centered_q.coeffs, strict=True))
    labels = sorted(set(h_map) | set(q_map))
    h_vector = np.asarray([h_map.get(label, 0.0) for label in labels], complex)
    q_vector = np.asarray([q_map.get(label, 0.0) for label in labels], complex)
    h_squared = float(np.real(np.vdot(h_vector, h_vector)))
    if h_squared <= 1e-28:
        return 1.0
    projection = np.vdot(h_vector, q_vector) / h_squared
    return float(np.linalg.norm(q_vector - projection * h_vector) / np.linalg.norm(q_vector))


def validate_symmetry_generator(
    hamiltonian: SparsePauliOp,
    generator: SparsePauliOp,
    *,
    min_hamiltonian_span_distance: float = 1e-6,
) -> None:
    if distance_from_hamiltonian_span(hamiltonian, generator) < min_hamiltonian_span_distance:
        raise ProbeValidationError("generator is a trivial copy of the Hamiltonian")


def _explicit_pauli_sum(num_qubits: int, recipe: Mapping[str, Any]) -> SparsePauliOp:
    if set(recipe) != {"type", "terms"}:
        raise ProbeValidationError("pauli_sum fields must be exactly type and terms")
    raw_terms = recipe.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ProbeValidationError("pauli_sum requires a non-empty terms list")
    if len(raw_terms) > MAX_GENERATOR_TERMS:
        raise ProbeValidationError(f"pauli_sum exceeds the {MAX_GENERATOR_TERMS}-term cap")
    terms: list[tuple[str, float]] = []
    labels: set[str] = set()
    for index, raw in enumerate(raw_terms):
        if not isinstance(raw, Mapping):
            raise ProbeValidationError(f"terms[{index}] must be an object")
        if not {"pauli"} <= set(raw) <= {"pauli", "coeff"}:
            raise ProbeValidationError(f"terms[{index}] fields must be pauli and optional coeff")
        pauli = raw.get("pauli")
        if not isinstance(pauli, str) or len(pauli) != num_qubits or set(pauli) - set("IXYZ"):
            raise ProbeValidationError(f"terms[{index}].pauli is invalid")
        if pauli in labels:
            raise ProbeValidationError(f"terms[{index}].pauli is duplicated")
        raw_coeff = raw.get("coeff", 1.0)
        if isinstance(raw_coeff, bool) or not isinstance(raw_coeff, (int, float)):
            raise ProbeValidationError(f"terms[{index}].coeff must be a real JSON number")
        coeff = float(raw_coeff)
        if not math.isfinite(coeff):
            raise ProbeValidationError(f"terms[{index}].coeff must be finite")
        if abs(coeff) > MAX_GENERATOR_COEFFICIENT:
            raise ProbeValidationError(
                f"terms[{index}].coeff exceeds {MAX_GENERATOR_COEFFICIENT:g}"
            )
        labels.add(pauli)
        terms.append((pauli, coeff))
    return SparsePauliOp.from_list(terms).simplify(atol=1e-14)


def _global_pauli_sum(num_qubits: int, recipe: Mapping[str, Any]) -> SparsePauliOp:
    if not {"type", "pauli"} <= set(recipe) <= {"type", "pauli", "selector"}:
        raise ProbeValidationError(
            "global_pauli_sum fields must be type, pauli, and optional selector"
        )
    if num_qubits > MAX_GENERATOR_TERMS:
        raise ProbeValidationError(f"global Pauli sum exceeds the {MAX_GENERATOR_TERMS}-term cap")
    pauli = recipe.get("pauli")
    if pauli not in {"X", "Y", "Z"}:
        raise ProbeValidationError("global Pauli sum requires pauli X, Y, or Z")
    if recipe.get("selector", "all_sites") != "all_sites":
        raise ProbeValidationError("global Pauli sum only supports selector=all_sites")
    terms = []
    for qubit in range(num_qubits):
        label = ["I"] * num_qubits
        label[num_qubits - qubit - 1] = str(pauli)
        terms.append(("".join(label), 1.0))
    return SparsePauliOp.from_list(terms).simplify(atol=1e-14)


def generator_from_recipe(num_qubits: int, recipe: Mapping[str, Any]) -> SparsePauliOp:
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ProbeValidationError("num_qubits must be a positive integer")
    if recipe.get("type") == "pauli_sum":
        return _explicit_pauli_sum(num_qubits, recipe)
    if recipe.get("type") == "global_pauli_sum":
        return _global_pauli_sum(num_qubits, recipe)
    raise ProbeValidationError(f"unsupported generator recipe type: {recipe.get('type')!r}")


def _probe_inputs_and_cost(
    hamiltonian: SparsePauliOp, request: Mapping[str, Any]
) -> tuple[SparsePauliOp, SparsePauliOp, float]:
    if set(request) != {"type", "generator"}:
        raise ProbeValidationError("probe fields must be exactly type and generator")
    if request.get("type") != "normalized_commutator":
        raise ProbeValidationError(f"unsupported probe type: {request.get('type')!r}")
    recipe = request.get("generator")
    if not isinstance(recipe, Mapping):
        raise ProbeValidationError("probe requires a generator recipe")
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    generator = generator_from_recipe(hamiltonian.num_qubits, recipe)
    validate_generator_observable(generator)
    products = _check_term_product_cap(hamiltonian, generator)
    return hamiltonian, generator, round(0.25 * max(1, math.ceil(products / PROBE_COST_TERM_BLOCK)), 10)


def algebraic_probe_cost_units(
    hamiltonian: SparsePauliOp, request: Mapping[str, Any]
) -> float:
    return _probe_inputs_and_cost(hamiltonian, request)[2]


def run_public_probe(problem: PublicProblem, request: Mapping[str, Any]) -> ProbeResult:
    """Run the sole public probe: an evaluator-owned normalized commutator."""
    hamiltonian, generator, cost = _probe_inputs_and_cost(
        hamiltonian_from_problem(problem), request
    )
    validate_symmetry_generator(hamiltonian, generator)
    residual = normalized_commutator(hamiltonian, generator)
    return ProbeResult(
        "normalized_commutator",
        {
            "residual": residual,
            "exact": residual <= EXACT_SYMMETRY_TOLERANCE,
            "hamiltonian_span_distance": distance_from_hamiltonian_span(
                hamiltonian, generator
            ),
        },
        cost,
    )


def _statevector(initial_state: QuantumCircuit | Statevector | np.ndarray) -> Statevector:
    if isinstance(initial_state, Statevector):
        return initial_state
    if isinstance(initial_state, QuantumCircuit):
        if initial_state.parameters:
            raise ProbeValidationError("initial-state circuit must be fully bound")
        return Statevector.from_instruction(initial_state)
    return Statevector(np.asarray(initial_state, dtype=complex))


def initial_state_moments(
    initial_state: QuantumCircuit | Statevector | np.ndarray, generator: SparsePauliOp
) -> tuple[float, float]:
    """Measure internal sector evidence for a trusted charge."""
    state = _statevector(initial_state)
    if state.num_qubits != generator.num_qubits:
        raise ProbeValidationError("initial state and generator qubit counts differ")
    active_norm = validate_generator_observable(generator)
    mean = complex(state.expectation_value(generator))
    squared = generator.compose(generator).simplify(atol=1e-14)
    variance = float(np.real(complex(state.expectation_value(squared)) - mean * mean))
    return float(np.real(mean)), max(0.0, variance) / (active_norm * active_norm)


def _full_pauli_label(num_qubits: int, qubits: Sequence[int], pauli: str) -> str:
    if len(qubits) != len(pauli):
        raise ProbeValidationError("Pauli word and support lengths differ")
    label = ["I"] * num_qubits
    for qubit, letter in zip(qubits, pauli, strict=True):
        if not 0 <= qubit < num_qubits:
            raise ProbeValidationError("operation support is outside the register")
        label[num_qubits - qubit - 1] = letter
    return "".join(label)


def operation_generator(num_qubits: int, operation: OperationSpec) -> SparsePauliOp:
    if operation.macro == "PauliRotation":
        word = _full_pauli_label(num_qubits, operation.qubits, str(operation.options["pauli"]))
        return SparsePauliOp.from_list([(word, 1.0)])
    if operation.macro in {"XYExchange", "IsotropicExchange"}:
        terms = [
            (_full_pauli_label(num_qubits, operation.qubits, "XX"), 1.0),
            (_full_pauli_label(num_qubits, operation.qubits, "YY"), 1.0),
        ]
        if operation.macro == "IsotropicExchange":
            terms.append((_full_pauli_label(num_qubits, operation.qubits, "ZZ"), 1.0))
        return SparsePauliOp.from_list(terms).simplify(atol=1e-14)
    raise ProbeValidationError(f"unsupported trusted operation macro: {operation.macro}")


def operation_symmetry_residuals(
    num_qubits: int, operations: Sequence[OperationSpec], charge: SparsePauliOp
) -> list[float]:
    return [
        normalized_commutator(operation_generator(num_qubits, operation), charge)
        for operation in operations
    ]


def _nonnegative_metric(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ProbeValidationError(f"{name} must be finite and non-negative")
    return float(value)


def validate_special_operation_relevance(
    num_qubits: int,
    operation: OperationSpec,
    charge: SparsePauliOp,
    *,
    symmetry_residual: float,
    sector_variance: float,
    tolerance: float = EXACT_SYMMETRY_TOLERANCE,
) -> tuple[float, float, float, float, float]:
    """Reject spectator-only or diluted charges used to unlock special gates."""
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ProbeValidationError("num_qubits must be a positive integer")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ProbeValidationError("relevance tolerance must be a finite positive number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ProbeValidationError("relevance tolerance must be a finite positive number")
    symmetry_residual = _nonnegative_metric(symmetry_residual, "symmetry residual")
    sector_variance = _nonnegative_metric(sector_variance, "sector variance")
    if charge.num_qubits != num_qubits:
        raise ProbeValidationError("operation and charge qubit counts differ")
    if any(qubit < 0 or qubit >= num_qubits for qubit in operation.qubits):
        raise ProbeValidationError("operation support is outside the register")

    full_norm = validate_generator_observable(charge)
    centered = _center_operator(charge.simplify(atol=1e-14))
    touching_terms = [
        (label, complex(coeff))
        for label, coeff in zip(centered.paulis.to_labels(), centered.coeffs, strict=True)
        if any(label[num_qubits - qubit - 1] != "I" for qubit in operation.qubits)
    ]
    if not touching_terms:
        raise ProbeValidationError(
            "claimed symmetry has no nontrivial charge on the special operation support"
        )
    touching_charge = SparsePauliOp.from_list(touching_terms).simplify(atol=1e-14)
    touching_norm = _coefficient_norm(touching_charge)
    fraction = touching_norm / full_norm
    if fraction < MIN_SPECIAL_CHARGE_FRACTION:
        raise ProbeValidationError(
            "claimed symmetry charge on the special operation support is too small "
            f"relative to the full charge: fraction={fraction:.3e}, "
            f"minimum={MIN_SPECIAL_CHARGE_FRACTION:.3e}"
        )
    residual = normalized_commutator(operation_generator(num_qubits, operation), touching_charge)
    if residual > tolerance:
        raise ProbeValidationError(
            "special operation does not preserve the overlapping symmetry charge: "
            f"residual={residual:.3e}"
        )
    conditioned_residual = symmetry_residual / fraction
    conditioned_variance = sector_variance / (fraction * fraction)
    if conditioned_residual > tolerance:
        raise ProbeValidationError(
            "claimed conservation is too weak on the special operation support: "
            f"conditioned_residual={conditioned_residual:.3e}"
        )
    if conditioned_variance > tolerance:
        raise ProbeValidationError(
            "initial sector evidence is too weak on the special operation support: "
            f"conditioned_variance={conditioned_variance:.3e}"
        )
    return touching_norm, fraction, residual, conditioned_residual, conditioned_variance


def energy_from_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.parameters:
        raise ProbeValidationError("energy probe requires a fully bound circuit")
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    if circuit.num_qubits != hamiltonian.num_qubits:
        raise ProbeValidationError("circuit and Hamiltonian qubit counts differ")
    state = Statevector.from_instruction(circuit)
    return float(np.real(state.expectation_value(hamiltonian)))
