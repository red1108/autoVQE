"""Compact, mechanically derived observations for an AutoVQE problem."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .contracts import (
    BackendSpec,
    EncodingSpec,
    InitialStateSpec,
    PublicProblem,
    SCHEMA_VERSION,
    SectorSpec,
)


MAX_EXACT_SUPPORT_GRAPH_EDGES = 64


def _active_qubits(label: str) -> tuple[int, ...]:
    return tuple(
        len(label) - position - 1
        for position, pauli in enumerate(label)
        if pauli != "I"
    )


@dataclass(frozen=True)
class StructuralObservation:
    """Hamiltonian structure computed without classifying an ansatz."""

    term_count: int
    identity_term_count: int
    max_locality: int
    locality_counts: tuple[tuple[int, int], ...]
    pauli_counts: tuple[tuple[str, int], ...]
    support_graph_edge_count: int
    support_graph_degrees: tuple[tuple[int, int], ...]
    support_graph_components: tuple[tuple[int, ...], ...]
    support_graph_edges_complete: bool
    support_graph_edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.term_count <= 0:
            raise ValueError("term_count must be positive")
        if not 0 <= self.identity_term_count <= self.term_count:
            raise ValueError("identity_term_count is inconsistent")
        if self.max_locality < 0:
            raise ValueError("max_locality must be non-negative")
        if self.support_graph_edge_count < 0:
            raise ValueError("support_graph_edge_count must be non-negative")
        object.__setattr__(self, "locality_counts", tuple(sorted(self.locality_counts)))
        object.__setattr__(self, "pauli_counts", tuple(sorted(self.pauli_counts)))

        degrees = tuple(sorted(self.support_graph_degrees))
        if len({qubit for qubit, _ in degrees}) != len(degrees):
            raise ValueError("support_graph_degrees contains duplicate qubits")
        if any(qubit < 0 or degree < 0 for qubit, degree in degrees):
            raise ValueError("support graph degrees must be non-negative")
        if sum(degree for _, degree in degrees) != 2 * self.support_graph_edge_count:
            raise ValueError("support graph degrees do not match edge count")
        object.__setattr__(self, "support_graph_degrees", degrees)

        components = tuple(
            sorted(
                (tuple(sorted(component)) for component in self.support_graph_components),
                key=lambda component: component[0] if component else -1,
            )
        )
        component_qubits = [qubit for component in components for qubit in component]
        if any(not component for component in components):
            raise ValueError("support graph components must be non-empty")
        if len(component_qubits) != len(set(component_qubits)):
            raise ValueError("support graph components overlap")
        if set(component_qubits) != {qubit for qubit, _ in degrees}:
            raise ValueError("support graph components must cover every qubit")
        object.__setattr__(self, "support_graph_components", components)

        edges = tuple(sorted(set(self.support_graph_edges)))
        if self.support_graph_edges_complete:
            if len(edges) != self.support_graph_edge_count:
                raise ValueError("complete support graph edges do not match edge count")
        elif edges:
            raise ValueError("incomplete support graph must not expose partial edges")
        object.__setattr__(
            self,
            "support_graph_edges",
            edges,
        )


@dataclass(frozen=True)
class ProblemObservation:
    """Compact agent-facing facts; Pauli terms remain in the supplied input file."""

    problem_id: str
    num_qubits: int
    encoding: EncodingSpec
    sector: SectorSpec
    initial_state: InitialStateSpec
    backend: BackendSpec
    structure: StructuralObservation
    schema_version: str = SCHEMA_VERSION


def structural_observation(problem: PublicProblem) -> StructuralObservation:
    locality_counts: Counter[int] = Counter()
    pauli_counts: Counter[str] = Counter()
    support_edges: set[tuple[int, int]] = set()
    identity_terms = 0

    for term in problem.pauli_terms:
        support = _active_qubits(term.pauli)
        locality_counts[len(support)] += 1
        if not support:
            identity_terms += 1
        if len(support) == 2:
            support_edges.add(tuple(sorted(support)))
        for pauli in term.pauli:
            if pauli != "I":
                pauli_counts[pauli] += 1

    neighbors = {qubit: set() for qubit in range(problem.num_qubits)}
    for left, right in support_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)

    components: list[tuple[int, ...]] = []
    unseen = set(neighbors)
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[int] = []
        unseen.remove(start)
        while stack:
            qubit = stack.pop()
            component.append(qubit)
            discovered = sorted(neighbors[qubit] & unseen, reverse=True)
            for neighbor in discovered:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))

    edges_complete = len(support_edges) <= MAX_EXACT_SUPPORT_GRAPH_EDGES

    return StructuralObservation(
        term_count=len(problem.pauli_terms),
        identity_term_count=identity_terms,
        max_locality=max(locality_counts, default=0),
        locality_counts=tuple(locality_counts.items()),
        pauli_counts=tuple(pauli_counts.items()),
        support_graph_edge_count=len(support_edges),
        support_graph_degrees=tuple(
            (qubit, len(neighbors[qubit])) for qubit in range(problem.num_qubits)
        ),
        support_graph_components=tuple(components),
        support_graph_edges_complete=edges_complete,
        support_graph_edges=(tuple(support_edges) if edges_complete else ()),
    )


def observe_problem(problem: PublicProblem) -> ProblemObservation:
    """Return the compact facts presented by ``harness inspect`` and run init."""

    return ProblemObservation(
        problem_id=problem.problem_id,
        num_qubits=problem.num_qubits,
        encoding=problem.encoding,
        sector=problem.sector,
        initial_state=problem.initial_state,
        backend=problem.backend,
        structure=structural_observation(problem),
    )


__all__ = [
    "MAX_EXACT_SUPPORT_GRAPH_EDGES",
    "ProblemObservation",
    "StructuralObservation",
    "observe_problem",
    "structural_observation",
]
