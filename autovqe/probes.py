"""Evaluator-owned symmetry, sector, gate, and energy measurements."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from .ansatz import OperationSpec, operation_paulis, pauli_label
from .problem import PublicProblem, hamiltonian_from_problem

EXACT_SYMMETRY_TOLERANCE = 1e-10
MIN_SPECIAL_CHARGE_FRACTION, MIN_GENERATOR_NORM = 1e-3, 1e-8
MAX_GENERATOR_TERMS, MAX_GENERATOR_COEFFICIENT = 256, 1e6
MAX_COMMUTATOR_TERM_PRODUCTS, PROBE_COST_TERM_BLOCK = 65_536, 4_096

class ProbeValidationError(ValueError):
    pass

@dataclass(frozen=True)
class ProbeResult:
    probe_type: str
    metrics: dict[str, Any]
    cost_units: float
    valid: bool = True
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def initial_state_circuit(problem: PublicProblem) -> QuantumCircuit:
    circuit = QuantumCircuit(problem.num_qubits)
    for qubit, bit in enumerate(problem.initial_occupation or ()):
        if bit: circuit.x(qubit)
    return circuit

def _observable(operator: SparsePauliOp, label: str) -> SparsePauliOp:
    if not isinstance(operator, SparsePauliOp): raise ProbeValidationError(f"{label} must be a SparsePauliOp")
    coefficients = np.asarray(operator.coeffs, complex)
    if not np.all(np.isfinite(coefficients)): raise ProbeValidationError(f"{label} coefficients must be finite")
    operator = operator.simplify(atol=1e-14)
    if np.any(np.abs(np.asarray(operator.coeffs).imag) > 1e-12): raise ProbeValidationError(f"{label} must be Hermitian with real Pauli coefficients")
    return operator

def _center(operator: SparsePauliOp) -> SparsePauliOp:
    identity = "I" * operator.num_qubits
    terms = [(label, complex(coeff)) for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs, strict=True) if label != identity]
    return SparsePauliOp.from_list(terms or [(identity, 0.0)]).simplify(atol=1e-14)

def _norm(operator: SparsePauliOp) -> float:
    return float(np.linalg.norm(np.asarray(operator.coeffs, complex)))

def validate_hamiltonian_observable(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    return _observable(hamiltonian, "Hamiltonian")

def _generator(generator: SparsePauliOp, minimum: float = MIN_GENERATOR_NORM) -> tuple[SparsePauliOp, float]:
    if not math.isfinite(float(minimum)) or minimum <= 0: raise ValueError("min_norm must be finite and positive")
    generator = _observable(generator, "generator"); active = _norm(_center(generator))
    if not math.isfinite(active) or active < minimum: raise ProbeValidationError(f"generator is identity-only, zero, or below the minimum active norm {minimum:g}")
    return generator, active

def validate_generator_observable(generator: SparsePauliOp, *, min_norm: float = MIN_GENERATOR_NORM) -> float:
    return _generator(generator, min_norm)[1]

def _commutator(hamiltonian: SparsePauliOp, generator: SparsePauliOp) -> tuple[float, int, SparsePauliOp, SparsePauliOp]:
    hamiltonian, (generator, q_norm) = _observable(hamiltonian, "Hamiltonian"), _generator(generator)
    if hamiltonian.num_qubits != generator.num_qubits: raise ProbeValidationError("Hamiltonian and generator qubit counts differ")
    h_norm = _norm(_center(hamiltonian))
    if h_norm <= 1e-14: raise ProbeValidationError("Hamiltonian has no non-identity component")
    products = len(hamiltonian.paulis) * len(generator.paulis)
    if products > MAX_COMMUTATOR_TERM_PRODUCTS: raise ProbeValidationError(f"commutator probe exceeds the {MAX_COMMUTATOR_TERM_PRODUCTS}-term-product cap")
    commutator = (hamiltonian.compose(generator) - generator.compose(hamiltonian)).simplify(atol=1e-14)
    return _norm(commutator) / (2 * h_norm * q_norm), products, hamiltonian, generator

def normalized_commutator(hamiltonian: SparsePauliOp, generator: SparsePauliOp) -> float:
    return _commutator(hamiltonian, generator)[0]

def _span_distance(hamiltonian: SparsePauliOp, generator: SparsePauliOp) -> float:
    h, q = _center(hamiltonian), _center(generator)
    hm, qm = dict(zip(h.paulis.to_labels(), h.coeffs, strict=True)), dict(zip(q.paulis.to_labels(), q.coeffs, strict=True))
    labels = sorted(set(hm) | set(qm))
    hv, qv = np.asarray([hm.get(label, 0) for label in labels]), np.asarray([qm.get(label, 0) for label in labels])
    squared = float(np.real(np.vdot(hv, hv)))
    if squared <= 1e-28: return 1.0
    return float(np.linalg.norm(qv - np.vdot(hv, qv) / squared * hv) / np.linalg.norm(qv))

def generator_from_recipe(num_qubits: int, recipe: Mapping[str, Any]) -> SparsePauliOp:
    if type(num_qubits) is not int or num_qubits <= 0:
        raise ProbeValidationError("num_qubits must be a positive integer")
    if not isinstance(recipe, Mapping):
        raise ProbeValidationError("generator recipe must be an object")
    kind = recipe.get("type")
    if kind == "global_pauli_sum":
        if not {"type", "pauli"} <= set(recipe) <= {"type", "pauli", "selector"}:
            raise ProbeValidationError(
                "global_pauli_sum fields must be type, pauli, and optional selector"
            )
        pauli = recipe.get("pauli")
        if pauli not in {"X", "Y", "Z"} or recipe.get("selector", "all_sites") != "all_sites":
            raise ProbeValidationError("global Pauli sum requires X, Y, or Z on all_sites")
        if num_qubits > MAX_GENERATOR_TERMS:
            raise ProbeValidationError(
                f"global Pauli sum exceeds the {MAX_GENERATOR_TERMS}-term cap"
            )
        terms = [(pauli_label(num_qubits, (qubit,), pauli), 1.0) for qubit in range(num_qubits)]
    elif kind == "pauli_sum":
        raw_terms = recipe.get("terms")
        if (
            set(recipe) != {"type", "terms"}
            or not isinstance(raw_terms, list)
            or not raw_terms
            or len(raw_terms) > MAX_GENERATOR_TERMS
        ):
            raise ProbeValidationError("pauli_sum requires only a non-empty bounded terms list")
        terms, seen = [], set()
        for index, raw in enumerate(raw_terms):
            if not isinstance(raw, Mapping) or not {"pauli"} <= set(raw) <= {"pauli", "coeff"}:
                raise ProbeValidationError(f"terms[{index}] fields are invalid")
            pauli, coeff = raw.get("pauli"), raw.get("coeff", 1.0)
            if not isinstance(pauli, str) or len(pauli) != num_qubits or set(pauli) - set("IXYZ") or pauli in seen: raise ProbeValidationError(f"terms[{index}].pauli is invalid or duplicated")
            if isinstance(coeff, bool) or not isinstance(coeff, (int, float)) or not math.isfinite(coeff) or abs(coeff) > MAX_GENERATOR_COEFFICIENT: raise ProbeValidationError(f"terms[{index}].coeff is invalid")
            seen.add(pauli)
            terms.append((pauli, float(coeff)))
    else:
        raise ProbeValidationError(f"unsupported generator recipe type: {kind!r}")
    return SparsePauliOp.from_list(terms).simplify(atol=1e-14)

def run_public_probe(problem: PublicProblem, request: Mapping[str, Any]) -> ProbeResult:
    if set(request) != {"type", "generator"} or request.get("type") != "normalized_commutator":
        raise ProbeValidationError(f"unsupported probe type: {request.get('type')!r}")
    if not isinstance(request.get("generator"), Mapping):
        raise ProbeValidationError("probe requires a generator recipe")
    generator = generator_from_recipe(problem.num_qubits, request["generator"])
    residual, products, hamiltonian, generator = _commutator(
        hamiltonian_from_problem(problem), generator
    )
    distance = _span_distance(hamiltonian, generator)
    if distance < 1e-6: raise ProbeValidationError("generator is a trivial copy of the Hamiltonian")
    metrics = {"residual": residual, "exact": residual <= EXACT_SYMMETRY_TOLERANCE, "hamiltonian_span_distance": distance}
    return ProbeResult("normalized_commutator", metrics, round(0.25 * max(1, math.ceil(products / PROBE_COST_TERM_BLOCK)), 10))

def initial_state_moments(initial_state: QuantumCircuit, generator: SparsePauliOp) -> tuple[float, float]:
    if initial_state.parameters: raise ProbeValidationError("initial-state circuit must be fully bound")
    state = Statevector.from_instruction(initial_state)
    if state.num_qubits != generator.num_qubits: raise ProbeValidationError("initial state and generator qubit counts differ")
    active = validate_generator_observable(generator); mean = complex(state.expectation_value(generator))
    squared = generator.compose(generator).simplify(atol=1e-14)
    variance = float(np.real(complex(state.expectation_value(squared)) - mean * mean))
    return float(mean.real), max(0.0, variance) / active**2

def operation_generator(num_qubits: int, operation: OperationSpec) -> SparsePauliOp:
    terms = [
        (pauli_label(num_qubits, operation.qubits, word), coeff)
        for word, coeff in operation_paulis(operation)
    ]
    try:
        return SparsePauliOp.from_list(terms).simplify(atol=1e-14)
    except ValueError as exc:
        raise ProbeValidationError(str(exc)) from exc

def operation_symmetry_residuals(num_qubits: int, operations: Sequence[OperationSpec], charge: SparsePauliOp) -> list[float]:
    return [normalized_commutator(operation_generator(num_qubits, operation), charge) for operation in operations]

def _metric(value: float, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or positive and value == 0: raise ProbeValidationError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return float(value)

def validate_special_operation_relevance(num_qubits: int, operation: OperationSpec, charge: SparsePauliOp, *, symmetry_residual: float, sector_variance: float, tolerance: float = EXACT_SYMMETRY_TOLERANCE) -> tuple[float, float, float, float, float]:
    if type(num_qubits) is not int or num_qubits <= 0: raise ProbeValidationError("num_qubits must be a positive integer")
    tolerance, symmetry_residual, sector_variance = _metric(tolerance, "relevance tolerance", True), _metric(symmetry_residual, "symmetry residual"), _metric(sector_variance, "sector variance")
    if charge.num_qubits != num_qubits or any(not 0 <= qubit < num_qubits for qubit in operation.qubits): raise ProbeValidationError("operation and charge supports differ")
    full_norm, centered = validate_generator_observable(charge), _center(charge.simplify(atol=1e-14))
    touching = [(label, complex(coeff)) for label, coeff in zip(centered.paulis.to_labels(), centered.coeffs, strict=True) if any(label[num_qubits - qubit - 1] != "I" for qubit in operation.qubits)]
    if not touching: raise ProbeValidationError("claimed symmetry has no nontrivial charge on the special operation support")
    touching_charge = SparsePauliOp.from_list(touching).simplify(atol=1e-14)
    touching_norm = _norm(touching_charge); fraction = touching_norm / full_norm
    if fraction < MIN_SPECIAL_CHARGE_FRACTION: raise ProbeValidationError(f"claimed symmetry charge on the special operation support is too small: {fraction:.3e}")
    residual = normalized_commutator(operation_generator(num_qubits, operation), touching_charge)
    conditioned_residual, conditioned_variance = symmetry_residual / fraction, sector_variance / fraction**2
    if residual > tolerance: raise ProbeValidationError(f"special operation does not preserve the overlapping symmetry charge: residual={residual:.3e}")
    if conditioned_residual > tolerance: raise ProbeValidationError(f"claimed conservation is too weak on the special operation support: conditioned_residual={conditioned_residual:.3e}")
    if conditioned_variance > tolerance: raise ProbeValidationError(f"initial sector evidence is too weak on the special operation support: conditioned_variance={conditioned_variance:.3e}")
    return touching_norm, fraction, residual, conditioned_residual, conditioned_variance

def _expectation_energy(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    value = complex(Statevector.from_instruction(circuit).expectation_value(hamiltonian))
    if not math.isfinite(value.real) or not math.isfinite(value.imag): raise ProbeValidationError("expectation value must be finite")
    if abs(value.imag) > 1e-10 * max(1.0, abs(value.real)): raise ProbeValidationError("Hermitian Hamiltonian produced a complex expectation")
    return float(value.real)

def energy_from_circuit(circuit: QuantumCircuit, hamiltonian: SparsePauliOp) -> float:
    if circuit.parameters: raise ProbeValidationError("energy probe requires a fully bound circuit")
    hamiltonian = validate_hamiltonian_observable(hamiltonian)
    if circuit.num_qubits != hamiltonian.num_qubits: raise ProbeValidationError("circuit and Hamiltonian qubit counts differ")
    return _expectation_energy(circuit, hamiltonian)
