"""Build agent-safe and evaluator-private views of a prepared problem."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PrivateEvaluationContext,
    PublicProblem,
    ReferenceSpec,
    SCHEMA_VERSION,
    SectorSpec,
    assert_agent_safe,
    canonical_json,
)

if TYPE_CHECKING:
    from .prepare import Problem


def _active_qubits(label: str) -> tuple[int, ...]:
    return tuple(
        len(label) - position - 1
        for position, pauli in enumerate(label)
        if pauli != "I"
    )


def _public_pauli_terms(problem: "Problem") -> tuple[PauliTerm, ...]:
    combined: dict[str, complex] = {}
    for label, coefficient in zip(
        problem.hamiltonian.paulis.to_labels(),
        problem.hamiltonian.coeffs,
        strict=True,
    ):
        combined[label] = combined.get(label, 0.0j) + complex(coefficient)
    return tuple(
        PauliTerm(pauli=label, real=value.real, imag=value.imag)
        for label, value in sorted(combined.items())
    )


def _symmetry_mapping(problem: "Problem") -> Mapping[str, Any]:
    return problem.symmetry if isinstance(problem.symmetry, Mapping) else {}


def _positive_int(metadata: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def encoding_from_prepare(problem: "Problem") -> EncodingSpec:
    """Extract allowlisted encoding facts without forwarding arbitrary metadata."""

    metadata = _symmetry_mapping(problem)
    mapping = None
    for key in ("mapping", "fermion_mapping", "qubit_mapping"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            mapping = value
            break

    attributes: list[tuple[str, str | int | float | bool]] = []
    for key in ("basis", "orbital_order", "spin_order"):
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            attributes.append((key, value))

    return EncodingSpec(
        name="qubit_pauli",
        mapping=mapping,
        qubit_order="qiskit_little_endian",
        spin_orbitals=_positive_int(metadata, "spin_orbitals"),
        active_orbitals=_positive_int(metadata, "active_orbitals"),
        attributes=tuple(attributes),
    )


def sector_from_prepare(problem: "Problem") -> SectorSpec:
    """Extract only explicit, scalar sector declarations from problem metadata."""

    metadata = _symmetry_mapping(problem)
    key_map = (
        ("active_electrons", "particle_number"),
        ("particle_number", "particle_number"),
        ("magnetization", "magnetization"),
        ("spin_projection", "spin_projection"),
        ("total_spin", "total_spin"),
        ("parity", "parity"),
    )
    values: dict[str, str | int | float | bool] = {}
    for source_key, public_key in key_map:
        value = metadata.get(source_key)
        if isinstance(value, (str, int, float, bool)) and not isinstance(value, complex):
            values.setdefault(public_key, value)
    return SectorSpec(
        symmetries=tuple(values),
        values=tuple(values.items()),
    )


def reference_from_prepare(problem: "Problem") -> ReferenceSpec:
    """Convert an allowed preparation hint, never an exact reference vector."""

    hint = problem.initial_state_hint
    if isinstance(hint, list) and len(hint) == problem.num_qubits:
        try:
            occupation = tuple(int(bit) for bit in hint)
        except (TypeError, ValueError):
            occupation = ()
        if occupation and all(bit in (0, 1) for bit in occupation):
            return ReferenceSpec(
                kind="computational_basis",
                occupation=occupation,
                source="initial_state_hint",
            )

    dominant = _symmetry_mapping(problem).get("dominant_occupation")
    if (
        isinstance(dominant, str)
        and len(dominant) == problem.num_qubits
        and set(dominant).issubset({"0", "1"})
    ):
        return ReferenceSpec(
            kind="computational_basis",
            occupation=tuple(int(bit) for bit in dominant),
            source="declared_occupation_hint",
        )
    return ReferenceSpec()


def backend_from_prepare(problem: "Problem") -> BackendSpec:
    coupling_map = ()
    if problem.coupling_map is not None:
        coupling_map = tuple((int(left), int(right)) for left, right in problem.coupling_map)
    return BackendSpec(
        basis_gates=tuple(problem.basis_gates),
        coupling_map=coupling_map,
    )


def public_problem_from_prepare(
    problem: "Problem",
    *,
    encoding: EncodingSpec | None = None,
    sector: SectorSpec | None = None,
    reference: ReferenceSpec | None = None,
    backend: BackendSpec | None = None,
) -> PublicProblem:
    """Create the public view of ``prepare.Problem``."""

    return PublicProblem.create(
        problem_id=str(problem.name).strip() or "unnamed_problem",
        num_qubits=problem.num_qubits,
        pauli_terms=_public_pauli_terms(problem),
        encoding=encoding or encoding_from_prepare(problem),
        sector=sector or sector_from_prepare(problem),
        reference=reference or reference_from_prepare(problem),
        backend=backend or backend_from_prepare(problem),
    )


def private_context_from_prepare(
    problem: "Problem",
    *,
    public_problem: PublicProblem | None = None,
) -> PrivateEvaluationContext:
    """Create the evaluator-only view, including exact reference data."""

    public_problem = public_problem or public_problem_from_prepare(problem)
    state = None
    if problem.reference_state is not None:
        state = tuple(complex(amplitude) for amplitude in problem.reference_state)
    return PrivateEvaluationContext(
        public_problem=public_problem,
        source_name=str(problem.name),
        reference_energy=problem.reference_energy,
        reference_state=state,
    )


@dataclass(frozen=True)
class StructuralObservation:
    """Mechanically derived facts, with no ansatz classification or advice."""

    term_count: int
    identity_term_count: int
    max_locality: int
    locality_counts: tuple[tuple[int, int], ...]
    pauli_counts: tuple[tuple[str, int], ...]
    support_graph_edges: tuple[tuple[int, int], ...]
    has_complex_coefficients: bool

    def __post_init__(self) -> None:
        if self.term_count <= 0:
            raise ValueError("term_count must be positive")
        if not 0 <= self.identity_term_count <= self.term_count:
            raise ValueError("identity_term_count is inconsistent")
        if not 0 <= self.max_locality:
            raise ValueError("max_locality must be non-negative")
        object.__setattr__(self, "locality_counts", tuple(sorted(self.locality_counts)))
        object.__setattr__(self, "pauli_counts", tuple(sorted(self.pauli_counts)))
        object.__setattr__(self, "support_graph_edges", tuple(sorted(set(self.support_graph_edges))))


HamiltonianObservation = StructuralObservation


def structural_observation(public_problem: PublicProblem) -> StructuralObservation:
    locality_counts: Counter[int] = Counter()
    pauli_counts: Counter[str] = Counter()
    support_edges: set[tuple[int, int]] = set()
    identity_terms = 0
    has_complex = False

    for term in public_problem.pauli_terms:
        support = _active_qubits(term.pauli)
        locality_counts[len(support)] += 1
        if not support:
            identity_terms += 1
        if len(support) == 2:
            support_edges.add(tuple(sorted(support)))
        for pauli in term.pauli:
            if pauli != "I":
                pauli_counts[pauli] += 1
        has_complex = has_complex or term.imag != 0.0

    return StructuralObservation(
        term_count=len(public_problem.pauli_terms),
        identity_term_count=identity_terms,
        max_locality=max(locality_counts, default=0),
        locality_counts=tuple(locality_counts.items()),
        pauli_counts=tuple(pauli_counts.items()),
        support_graph_edges=tuple(support_edges),
        has_complex_coefficients=has_complex,
    )


@dataclass(frozen=True)
class ObservationBundle:
    """The sole bundle intended to cross from evaluator to agent."""

    public_problem: PublicProblem
    structure: StructuralObservation
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        assert_agent_safe(self)

    @property
    def problem(self) -> PublicProblem:
        """Short alias useful to consumers; it does not add serialized data."""

        return self.public_problem

    def to_canonical_json(self) -> str:
        assert_agent_safe(self)
        return canonical_json(self)

def observation_bundle_from_prepare(
    problem: "Problem",
    *,
    public_problem: PublicProblem | None = None,
) -> ObservationBundle:
    public_problem = public_problem or public_problem_from_prepare(problem)
    return ObservationBundle(
        public_problem=public_problem,
        structure=structural_observation(public_problem),
    )


@dataclass(frozen=True)
class ProblemViews:
    """Atomic result of splitting one prepared problem at the trust boundary."""

    public_problem: PublicProblem
    private_context: PrivateEvaluationContext
    observation_bundle: ObservationBundle

    def __post_init__(self) -> None:
        if self.private_context.public_problem != self.public_problem:
            raise ValueError("private context references a different public problem")
        if self.observation_bundle.public_problem != self.public_problem:
            raise ValueError("observation bundle references a different public problem")

    @property
    def public(self) -> PublicProblem:
        return self.public_problem

    @property
    def private(self) -> PrivateEvaluationContext:
        return self.private_context

    @property
    def safe(self) -> ObservationBundle:
        return self.observation_bundle


def adapt_prepare_problem(
    problem: "Problem",
    *,
    encoding: EncodingSpec | None = None,
    sector: SectorSpec | None = None,
    reference: ReferenceSpec | None = None,
    backend: BackendSpec | None = None,
) -> ProblemViews:
    """Split ``prepare.Problem`` into public, private, and agent-safe views."""

    public_problem = public_problem_from_prepare(
        problem,
        encoding=encoding,
        sector=sector,
        reference=reference,
        backend=backend,
    )
    return ProblemViews(
        public_problem=public_problem,
        private_context=private_context_from_prepare(problem, public_problem=public_problem),
        observation_bundle=observation_bundle_from_prepare(problem, public_problem=public_problem),
    )


# Concise aliases for callers that already know the source is prepare.Problem.
adapt_problem = adapt_prepare_problem
from_prepare_problem = adapt_prepare_problem


__all__ = [
    "HamiltonianObservation",
    "ObservationBundle",
    "ProblemViews",
    "StructuralObservation",
    "adapt_prepare_problem",
    "adapt_problem",
    "backend_from_prepare",
    "encoding_from_prepare",
    "from_prepare_problem",
    "observation_bundle_from_prepare",
    "private_context_from_prepare",
    "public_problem_from_prepare",
    "reference_from_prepare",
    "sector_from_prepare",
    "structural_observation",
]
