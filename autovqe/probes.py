from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from .contracts import PublicProblem
from .ansatz_ir import OperationSpec
from .problem import hamiltonian_from_problem


EXACT_SYMMETRY_TOLERANCE = 1e-10
MIN_SPECIAL_CHARGE_FRACTION = 1e-3
MAX_DENSE_PROBE_QUBITS = 8
GENERATOR_HERMITICITY_TOLERANCE = 1e-12
MIN_GENERATOR_NORM = 1e-8
MAX_GENERATOR_TERMS = 256
MAX_GENERATOR_COEFFICIENT = 1e6
MAX_COMMUTATOR_TERM_PRODUCTS = 65_536
PROBE_COST_TERM_BLOCK = 4_096
MOMENT_COST_TERM_BLOCK = 64


class ProbeValidationError(ValueError):
    """Raised when a probe request is malformed or scientifically vacuous."""


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
    """Build the evaluator-owned computational-basis initial state."""

    occupation = problem.initial_state.occupation
    circuit = QuantumCircuit(problem.num_qubits)
    if occupation is not None:
        for qubit, bit in enumerate(occupation):
            if bit:
                circuit.x(qubit)
    return circuit


def _coefficient_norm(operator: SparsePauliOp) -> float:
    simplified = operator.simplify(atol=1e-14)
    return float(np.linalg.norm(np.asarray(simplified.coeffs, dtype=complex)))


def _center_operator(operator: SparsePauliOp) -> SparsePauliOp:
    labels = operator.paulis.to_labels()
    identity = "I" * operator.num_qubits
    terms = [
        (label, complex(coeff))
        for label, coeff in zip(labels, operator.coeffs, strict=True)
        if label != identity
    ]
    if not terms:
        return SparsePauliOp.from_list([(identity, 0.0)])
    return SparsePauliOp.from_list(terms).simplify(atol=1e-14)


def validate_generator_observable(
    generator: SparsePauliOp,
    *,
    min_norm: float = MIN_GENERATOR_NORM,
) -> float:
    """Validate a physical Hermitian generator and return its active norm.

    The returned norm excludes the identity component.  It is also the scale
    used to make initial-state variance tests invariant to an agent-chosen
    overall coefficient.
    """

    if not isinstance(generator, SparsePauliOp):
        raise ProbeValidationError("generator must be a SparsePauliOp")
    if not math.isfinite(float(min_norm)) or min_norm <= 0.0:
        raise ValueError("min_norm must be finite and positive")

    coefficients = np.asarray(generator.coeffs, dtype=complex)
    if not np.all(np.isfinite(coefficients.real)) or not np.all(
        np.isfinite(coefficients.imag)
    ):
        raise ProbeValidationError("generator coefficients must be finite")

    simplified = generator.simplify(atol=1e-14)
    simplified_coefficients = np.asarray(simplified.coeffs, dtype=complex)
    if np.any(np.abs(simplified_coefficients.imag) > GENERATOR_HERMITICITY_TOLERANCE):
        raise ProbeValidationError("generator must be Hermitian with real Pauli coefficients")

    active_norm = _coefficient_norm(_center_operator(simplified))
    if not math.isfinite(active_norm):
        raise ProbeValidationError("generator norm must be finite")
    if active_norm < min_norm:
        raise ProbeValidationError(
            "generator is identity-only, zero, or below the minimum active norm "
            f"{min_norm:g}"
        )
    return active_norm


def validate_hamiltonian_observable(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    """Return a finite Hermitian Pauli Hamiltonian for trusted probes."""

    if not isinstance(hamiltonian, SparsePauliOp):
        raise ProbeValidationError("Hamiltonian must be a SparsePauliOp")
    coefficients = np.asarray(hamiltonian.coeffs, dtype=complex)
    if not np.all(np.isfinite(coefficients.real)) or not np.all(
        np.isfinite(coefficients.imag)
    ):
        raise ProbeValidationError("Hamiltonian coefficients must be finite")
    simplified = hamiltonian.simplify(atol=1e-14)
    canonical_coefficients = np.asarray(simplified.coeffs, dtype=complex)
    if np.any(
        np.abs(canonical_coefficients.imag)
        > GENERATOR_HERMITICITY_TOLERANCE
    ):
        raise ProbeValidationError(
            "Hamiltonian must be Hermitian with real Pauli coefficients"
        )
    return simplified


def normalized_commutator(
    hamiltonian: SparsePauliOp,
    generator: SparsePauliOp,
) -> float:
    """Return a scale-invariant Pauli-coefficient norm of ``[H, Q]``.

    Pauli strings are orthogonal in the Hilbert--Schmidt inner product, so the
    omitted common ``sqrt(2**n)`` factor cancels in this normalization.
    """

    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    if hamiltonian.num_qubits != generator.num_qubits:
        raise ProbeValidationError("Hamiltonian and generator qubit counts differ")
    q_norm = validate_generator_observable(generator)
    centered_h = _center_operator(hamiltonian)
    h_norm = _coefficient_norm(centered_h)
    if h_norm <= 1e-14:
        raise ProbeValidationError("Hamiltonian has no non-identity component")
    term_products = len(hamiltonian.paulis) * len(generator.simplify(atol=1e-14).paulis)
    if term_products > MAX_COMMUTATOR_TERM_PRODUCTS:
        raise ProbeValidationError(
            "commutator probe exceeds the "
            f"{MAX_COMMUTATOR_TERM_PRODUCTS}-term-product cap"
        )
    commutator = (hamiltonian.compose(generator) - generator.compose(hamiltonian)).simplify(atol=1e-14)
    return _coefficient_norm(commutator) / (2.0 * h_norm * q_norm)


def distance_from_hamiltonian_span(
    hamiltonian: SparsePauliOp,
    generator: SparsePauliOp,
) -> float:
    """Distance of ``Q`` from the span of ``I`` and ``H`` in Pauli space."""

    if hamiltonian.num_qubits != generator.num_qubits:
        raise ProbeValidationError("Hamiltonian and generator qubit counts differ")
    centered_h = _center_operator(hamiltonian)
    centered_q = _center_operator(generator)
    h_labels = centered_h.paulis.to_labels()
    q_labels = centered_q.paulis.to_labels()
    labels = sorted(set(h_labels) | set(q_labels))
    h_map = dict(zip(h_labels, centered_h.coeffs, strict=True))
    q_map = dict(zip(q_labels, centered_q.coeffs, strict=True))
    h_vector = np.asarray([h_map.get(label, 0.0) for label in labels], dtype=complex)
    q_vector = np.asarray([q_map.get(label, 0.0) for label in labels], dtype=complex)
    denominator = float(np.linalg.norm(q_vector))
    if denominator <= 1e-14:
        raise ProbeValidationError("generator is zero or proportional to identity")
    h_squared = float(np.real(np.vdot(h_vector, h_vector)))
    if h_squared <= 1e-28:
        return 1.0
    coefficient = np.vdot(h_vector, q_vector) / h_squared
    residual = q_vector - coefficient * h_vector
    return float(np.linalg.norm(residual) / denominator)


def validate_symmetry_generator(
    hamiltonian: SparsePauliOp,
    generator: SparsePauliOp,
    *,
    min_hamiltonian_span_distance: float = 1e-6,
) -> None:
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    validate_generator_observable(generator)
    if distance_from_hamiltonian_span(hamiltonian, generator) < min_hamiltonian_span_distance:
        raise ProbeValidationError("generator is a trivial copy of the Hamiltonian")


def generator_from_recipe(num_qubits: int, recipe: Mapping[str, Any]) -> SparsePauliOp:
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ProbeValidationError("num_qubits must be a positive integer")
    recipe_type = str(recipe.get("type", ""))
    if recipe_type == "pauli_sum":
        if set(recipe) != {"type", "terms"}:
            raise ProbeValidationError("pauli_sum fields must be exactly type and terms")
        raw_terms = recipe.get("terms")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ProbeValidationError("pauli_sum requires a non-empty terms list")
        if len(raw_terms) > MAX_GENERATOR_TERMS:
            raise ProbeValidationError(
                f"pauli_sum exceeds the {MAX_GENERATOR_TERMS}-term cap"
            )
        terms: list[tuple[str, complex]] = []
        labels: set[str] = set()
        for index, raw in enumerate(raw_terms):
            if not isinstance(raw, Mapping):
                raise ProbeValidationError(f"terms[{index}] must be an object")
            if not {"pauli"} <= set(raw) <= {"pauli", "coeff"}:
                raise ProbeValidationError(
                    f"terms[{index}] fields must be pauli and optional coeff"
                )
            label = raw.get("pauli")
            if not isinstance(label, str) or len(label) != num_qubits:
                raise ProbeValidationError(f"terms[{index}].pauli has the wrong width")
            if not set(label).issubset({"I", "X", "Y", "Z"}):
                raise ProbeValidationError(f"terms[{index}].pauli is invalid")
            if label in labels:
                raise ProbeValidationError(f"terms[{index}].pauli is duplicated")
            raw_coefficient = raw.get("coeff", 1.0)
            if isinstance(raw_coefficient, bool) or not isinstance(
                raw_coefficient, (int, float)
            ):
                raise ProbeValidationError(
                    f"terms[{index}].coeff must be a real JSON number"
                )
            coefficient = float(raw_coefficient)
            if not math.isfinite(coefficient):
                raise ProbeValidationError(f"terms[{index}].coeff must be finite")
            if abs(coefficient) > MAX_GENERATOR_COEFFICIENT:
                raise ProbeValidationError(
                    f"terms[{index}].coeff exceeds {MAX_GENERATOR_COEFFICIENT:g}"
                )
            labels.add(label)
            terms.append((label, coefficient))
        return SparsePauliOp.from_list(terms).simplify(atol=1e-14)

    if recipe_type in {"global_pauli_sum", "orbit_pauli_sum"}:
        pauli_field = "pauli" if recipe_type == "global_pauli_sum" else "seed"
        if not {"type", pauli_field} <= set(recipe) <= {
            "type",
            pauli_field,
            "selector",
        }:
            raise ProbeValidationError(
                f"{recipe_type} fields must be type, {pauli_field}, and optional selector"
            )
        if num_qubits > MAX_GENERATOR_TERMS:
            raise ProbeValidationError(
                f"global Pauli sum exceeds the {MAX_GENERATOR_TERMS}-term cap"
            )
        pauli = recipe.get(pauli_field)
        selector = recipe.get("selector", "all_sites")
        if pauli not in {"X", "Y", "Z"}:
            raise ProbeValidationError("global Pauli sum requires pauli X, Y, or Z")
        if selector != "all_sites":
            raise ProbeValidationError("MVP global Pauli sum only supports selector=all_sites")
        terms = []
        for qubit in range(num_qubits):
            label = ["I"] * num_qubits
            label[num_qubits - qubit - 1] = str(pauli)
            terms.append(("".join(label), 1.0))
        return SparsePauliOp.from_list(terms).simplify(atol=1e-14)

    raise ProbeValidationError(f"unsupported generator recipe type: {recipe_type!r}")


def _probe_inputs_and_cost(
    hamiltonian: SparsePauliOp,
    request: Mapping[str, Any],
) -> tuple[str, SparsePauliOp, float]:
    if set(request) != {"type", "generator"}:
        raise ProbeValidationError("algebraic probe fields must be exactly type and generator")
    probe_type = str(request.get("type", ""))
    recipe = request.get("generator")
    if not isinstance(recipe, Mapping):
        raise ProbeValidationError("probe requires a generator recipe")

    canonical_hamiltonian = validate_hamiltonian_observable(hamiltonian)
    generator = generator_from_recipe(canonical_hamiltonian.num_qubits, recipe)
    validate_generator_observable(generator)
    generator_terms = len(generator.simplify(atol=1e-14).paulis)

    if probe_type == "normalized_commutator":
        hamiltonian_terms = len(canonical_hamiltonian.paulis)
        term_products = hamiltonian_terms * generator_terms
        if term_products > MAX_COMMUTATOR_TERM_PRODUCTS:
            raise ProbeValidationError(
                "commutator probe exceeds the "
                f"{MAX_COMMUTATOR_TERM_PRODUCTS}-term-product cap"
            )
        blocks = max(1, math.ceil(term_products / PROBE_COST_TERM_BLOCK))
        return probe_type, generator, round(0.25 * blocks, 10)

    if probe_type == "initial_state_moments":
        blocks = max(1, math.ceil(generator_terms / MOMENT_COST_TERM_BLOCK))
        return probe_type, generator, round(0.25 * blocks, 10)

    raise ProbeValidationError(f"unsupported algebraic probe type: {probe_type!r}")


def algebraic_probe_cost_units(
    hamiltonian: SparsePauliOp,
    request: Mapping[str, Any],
) -> float:
    """Preflight an algebraic probe and return its complexity-derived charge."""

    _, _, cost = _probe_inputs_and_cost(hamiltonian, request)
    return cost


def _full_pauli_label(
    num_qubits: int,
    qubits: Sequence[int],
    pauli: str,
) -> str:
    if len(qubits) != len(pauli):
        raise ProbeValidationError("Pauli word and support lengths differ")
    label = ["I"] * num_qubits
    for qubit, letter in zip(qubits, pauli, strict=True):
        if not 0 <= qubit < num_qubits:
            raise ProbeValidationError("operation support is outside the register")
        label[num_qubits - qubit - 1] = letter
    return "".join(label)


def operation_generator(
    num_qubits: int,
    operation: OperationSpec,
) -> SparsePauliOp:
    """Return the trusted Hermitian generator of one logical macro."""

    if operation.macro == "PauliRotation":
        return SparsePauliOp.from_list(
            [
                (
                    _full_pauli_label(
                        num_qubits,
                        operation.qubits,
                        str(operation.options["pauli"]),
                    ),
                    1.0,
                )
            ]
        )
    if operation.macro in {"XYExchange", "IsotropicExchange"}:
        terms = [
            (_full_pauli_label(num_qubits, operation.qubits, "XX"), 1.0),
            (_full_pauli_label(num_qubits, operation.qubits, "YY"), 1.0),
        ]
        if operation.macro == "IsotropicExchange":
            terms.append(
                (_full_pauli_label(num_qubits, operation.qubits, "ZZ"), 1.0)
            )
        return SparsePauliOp.from_list(terms).simplify(atol=1e-14)
    raise ProbeValidationError(f"unsupported trusted operation macro: {operation.macro}")


def validate_special_operation_relevance(
    num_qubits: int,
    operation: OperationSpec,
    charge: SparsePauliOp,
    *,
    symmetry_residual: float,
    sector_variance: float,
    tolerance: float = EXACT_SYMMETRY_TOLERANCE,
) -> tuple[float, float, float, float, float]:
    """Validate that a symmetry is nontrivially relevant to an operation.

    Commutation with a charge on a disjoint spectator is automatic and does
    not justify a conservation-specialized gate.  This check therefore forms
    ``Q_touch`` from the centered charge terms whose Pauli support intersects
    the operation support.  It returns ``(||Q_touch||, residual)`` only when
    that charge is a material fraction of the full active charge and the
    trusted operation generator preserves it.
    """

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ProbeValidationError("num_qubits must be a positive integer")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ProbeValidationError("relevance tolerance must be a finite positive number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ProbeValidationError("relevance tolerance must be a finite positive number")
    for value, name in (
        (symmetry_residual, "symmetry residual"),
        (sector_variance, "sector variance"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ProbeValidationError(f"{name} must be finite and non-negative")
    if charge.num_qubits != num_qubits:
        raise ProbeValidationError("operation and charge qubit counts differ")
    if any(qubit < 0 or qubit >= num_qubits for qubit in operation.qubits):
        raise ProbeValidationError("operation support is outside the register")

    full_active_norm = validate_generator_observable(charge)
    centered_charge = _center_operator(charge.simplify(atol=1e-14))
    labels = centered_charge.paulis.to_labels()
    touching_terms = [
        (label, complex(coefficient))
        for label, coefficient in zip(
            labels, centered_charge.coeffs, strict=True
        )
        if any(label[num_qubits - qubit - 1] != "I" for qubit in operation.qubits)
    ]
    if not touching_terms:
        raise ProbeValidationError(
            "claimed symmetry has no nontrivial charge on the special operation support"
        )

    touching_charge = SparsePauliOp.from_list(touching_terms).simplify(atol=1e-14)
    touching_norm = _coefficient_norm(touching_charge)
    relevant_fraction = touching_norm / full_active_norm
    if relevant_fraction < MIN_SPECIAL_CHARGE_FRACTION:
        raise ProbeValidationError(
            "claimed symmetry charge on the special operation support is too small "
            f"relative to the full charge: fraction={relevant_fraction:.3e}, "
            f"minimum={MIN_SPECIAL_CHARGE_FRACTION:.3e}"
        )

    residual = normalized_commutator(
        operation_generator(num_qubits, operation), touching_charge
    )
    if residual > tolerance:
        raise ProbeValidationError(
            "special operation does not preserve the overlapping symmetry charge: "
            f"residual={residual:.3e}"
        )
    conditioned_symmetry_residual = float(symmetry_residual) / relevant_fraction
    conditioned_sector_variance = float(sector_variance) / (
        relevant_fraction * relevant_fraction
    )
    if conditioned_symmetry_residual > tolerance:
        raise ProbeValidationError(
            "claimed conservation is too weak on the special operation support: "
            f"conditioned_residual={conditioned_symmetry_residual:.3e}"
        )
    if conditioned_sector_variance > tolerance:
        raise ProbeValidationError(
            "initial sector evidence is too weak on the special operation support: "
            f"conditioned_variance={conditioned_sector_variance:.3e}"
        )
    return (
        touching_norm,
        relevant_fraction,
        residual,
        conditioned_symmetry_residual,
        conditioned_sector_variance,
    )


def operation_symmetry_residuals(
    num_qubits: int,
    operations: Sequence[OperationSpec],
    charge: SparsePauliOp,
) -> list[float]:
    return [
        normalized_commutator(operation_generator(num_qubits, operation), charge)
        for operation in operations
    ]


def _statevector(initial_state: QuantumCircuit | Statevector | np.ndarray) -> Statevector:
    if isinstance(initial_state, Statevector):
        return initial_state
    if isinstance(initial_state, QuantumCircuit):
        if initial_state.parameters:
            raise ProbeValidationError("initial-state circuit must be fully bound")
        return Statevector.from_instruction(initial_state)
    vector = np.asarray(initial_state, dtype=complex)
    return Statevector(vector)


def initial_state_moments(
    initial_state: QuantumCircuit | Statevector | np.ndarray,
    generator: SparsePauliOp,
) -> tuple[float, float]:
    state = _statevector(initial_state)
    if state.num_qubits != generator.num_qubits:
        raise ProbeValidationError("initial state and generator qubit counts differ")
    active_norm = validate_generator_observable(generator)
    mean = complex(state.expectation_value(generator))
    squared = generator.compose(generator).simplify(atol=1e-14)
    second = complex(state.expectation_value(squared))
    raw_variance = float(np.real(second - mean * mean))
    # Variance scales quadratically with Q.  Divide by the squared active norm
    # so an agent cannot make a non-eigenstate look exact merely by submitting
    # epsilon * Q.  Tiny negative values are round-off for Hermitian Q.
    variance = max(0.0, raw_variance) / (active_norm * active_norm)
    return float(np.real(mean)), variance


def unitary_commutation_residual(
    circuit: QuantumCircuit,
    generator: SparsePauliOp,
    *,
    parameter_values: Mapping[Parameter, float] | None = None,
) -> float:
    if circuit.num_qubits != generator.num_qubits:
        raise ProbeValidationError("circuit and generator qubit counts differ")
    if circuit.num_qubits > MAX_DENSE_PROBE_QUBITS:
        raise ProbeValidationError(
            f"dense macro probe is capped at {MAX_DENSE_PROBE_QUBITS} qubits"
        )
    bound = circuit
    if circuit.parameters:
        if parameter_values is None:
            raise ProbeValidationError("parameterized circuit requires explicit audit bindings")
        bound = circuit.assign_parameters(parameter_values, inplace=False)
    unitary = Operator(bound).data
    charge = generator.to_matrix()
    residual = unitary @ charge - charge @ unitary
    denominator = max(2.0 * np.linalg.norm(charge), 1e-15)
    return float(np.linalg.norm(residual) / denominator)


def energy_from_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.parameters:
        raise ProbeValidationError("energy probe requires a fully bound circuit")
    state = Statevector.from_instruction(circuit)
    return float(np.real(state.expectation_value(hamiltonian)))


def gradient_snapshot(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    *,
    values: Sequence[float] | None = None,
    epsilon: float = 1e-4,
) -> tuple[list[float], int]:
    parameters = tuple(sorted(circuit.parameters, key=lambda item: item.name))
    if values is None:
        center = np.zeros(len(parameters), dtype=float)
    else:
        center = np.asarray(values, dtype=float)
    if center.shape != (len(parameters),):
        raise ProbeValidationError("gradient values have the wrong length")
    if epsilon <= 0:
        raise ProbeValidationError("gradient epsilon must be positive")

    gradients: list[float] = []
    calls = 0
    for index, _parameter in enumerate(parameters):
        plus = center.copy()
        minus = center.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_circuit = circuit.assign_parameters(dict(zip(parameters, plus, strict=True)), inplace=False)
        minus_circuit = circuit.assign_parameters(dict(zip(parameters, minus, strict=True)), inplace=False)
        plus_energy = energy_from_circuit(plus_circuit, hamiltonian)
        minus_energy = energy_from_circuit(minus_circuit, hamiltonian)
        gradients.append((plus_energy - minus_energy) / (2.0 * epsilon))
        calls += 2
    return gradients, calls


def run_algebraic_probe(
    hamiltonian: SparsePauliOp,
    request: Mapping[str, Any],
    *,
    initial_state: QuantumCircuit | Statevector | np.ndarray | None = None,
) -> ProbeResult:
    probe_type, generator, cost_units = _probe_inputs_and_cost(
        hamiltonian, request
    )

    if probe_type == "normalized_commutator":
        validate_symmetry_generator(hamiltonian, generator)
        residual = normalized_commutator(hamiltonian, generator)
        return ProbeResult(
            probe_type=probe_type,
            metrics={
                "residual": residual,
                "exact": residual <= EXACT_SYMMETRY_TOLERANCE,
                "hamiltonian_span_distance": distance_from_hamiltonian_span(hamiltonian, generator),
            },
            cost_units=cost_units,
        )

    if probe_type == "initial_state_moments":
        if initial_state is None:
            raise ProbeValidationError(
                "initial_state_moments requires an initial state"
            )
        mean, variance = initial_state_moments(initial_state, generator)
        return ProbeResult(
            probe_type=probe_type,
            metrics={"mean": mean, "variance": variance},
            cost_units=cost_units,
        )

    raise AssertionError("validated probe type was not dispatched")


def run_public_probe(
    problem: PublicProblem,
    request: Mapping[str, Any],
) -> ProbeResult:
    """Run an evaluator-owned algebraic probe from an agent-safe problem."""

    return run_algebraic_probe(
        hamiltonian_from_problem(problem),
        request,
        initial_state=initial_state_circuit(problem),
    )
