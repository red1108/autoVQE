"""Lean public-problem model, loader, and mechanical observations."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from qiskit.quantum_info import SparsePauliOp

SCHEMA_VERSION = "1"
DEFAULT_PROBLEM_PATH = Path("user_problem/hamiltonian.json")
DEFAULT_BASIS_GATES = ("rx", "ry", "rz", "cx")
MAX_EXACT_SUPPORT_GRAPH_EDGES = 64
_TOP_FIELDS = set(
    "name pauli_terms basis_gates coupling_map initial_state_hint source_note symmetry".split()
)
_SYMMETRY_FIELDS = set(
    "mapping basis orbital_order spin_order spin_orbitals active_orbitals "
    "active_electrons particle_number magnetization spin_projection total_spin parity".split()
)

def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result

@dataclass(frozen=True)
class PauliTerm:
    pauli: str
    real: float
    imag: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.pauli, str) or not self.pauli or set(self.pauli) - set("IXYZ"):
            raise ValueError(f"invalid Pauli label: {self.pauli!r}")
        object.__setattr__(self, "real", _finite(self.real, "real"))
        object.__setattr__(self, "imag", _finite(self.imag, "imag"))

@dataclass(frozen=True)
class BackendSpec:
    basis_gates: tuple[str, ...] = ()
    coupling_map: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        gates = tuple(self.basis_gates)
        if any(not isinstance(gate, str) or not gate for gate in gates):
            raise ValueError("basis gate names must be non-empty strings")
        edges = tuple(tuple(edge) for edge in self.coupling_map)
        if any(
            len(edge) != 2
            or any(isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in edge)
            or edge[0] == edge[1]
            for edge in edges
        ):
            raise ValueError("coupling-map edges must join distinct non-negative qubits")
        object.__setattr__(self, "basis_gates", gates)
        object.__setattr__(self, "coupling_map", edges)

@dataclass(frozen=True)
class PublicProblem:
    problem_id: str
    num_qubits: int
    pauli_terms: tuple[PauliTerm, ...]
    symmetry: tuple[tuple[str, Any], ...] = ()
    initial_occupation: tuple[int, ...] | None = None
    backend: BackendSpec = BackendSpec()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        terms = tuple(self.pauli_terms)
        if not isinstance(self.problem_id, str) or not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int) or self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if not terms or any(len(term.pauli) != self.num_qubits for term in terms):
            raise ValueError("non-empty Pauli terms must match num_qubits")
        if any(term.imag != 0.0 for term in terms):
            raise ValueError("Hamiltonian Pauli coefficients must be real")
        occupation = self.initial_occupation
        if occupation is not None and (
            len(occupation) != self.num_qubits
            or any(
                isinstance(bit, bool)
                or not isinstance(bit, int)
                or bit not in (0, 1)
                for bit in occupation
            )
        ):
            raise ValueError("initial_occupation must contain one integer 0/1 bit per qubit")
        if any(max(edge) >= self.num_qubits for edge in self.backend.coupling_map):
            raise ValueError("coupling-map qubit is outside the problem")
        object.__setattr__(self, "pauli_terms", terms)
        object.__setattr__(self, "symmetry", tuple(sorted(self.symmetry)))
        object.__setattr__(self, "initial_occupation", None if occupation is None else tuple(occupation))

    @classmethod
    def create(
        cls,
        *,
        num_qubits: int,
        pauli_terms: Sequence[PauliTerm],
        problem_id: str = "problem",
        symmetry: Mapping[str, Any] | None = None,
        initial_occupation: Sequence[int] | None = None,
        backend: BackendSpec | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> PublicProblem:
        return cls(
            problem_id,
            num_qubits,
            tuple(pauli_terms),
            tuple((symmetry or {}).items()),
            None if initial_occupation is None else tuple(initial_occupation),
            backend or BackendSpec(),
            schema_version,
        )

def canonical_data(value: Any) -> Any:
    """Convert contracts and standard containers to finite JSON data."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: canonical_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("canonical JSON mappings require string keys")
        return {key: canonical_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    if isinstance(value, float):
        return _finite(value, "float")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")

def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def decode_json_object(text: str, source: str | Path = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is not allowed: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return value


def _read(path: Path) -> dict[str, Any]:
    try:
        return decode_json_object(path.read_text(encoding="utf-8-sig"), path)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read problem {path}: {exc}") from exc

def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value

def _integer(value: Any, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value

def _symmetry(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("symmetry", {})
    if not isinstance(value, dict):
        raise ValueError("symmetry must be an object")
    extra = set(value) - _SYMMETRY_FIELDS
    if extra:
        raise ValueError(f"invalid symmetry fields: missing=[] extra={sorted(extra)}")
    for field in ("mapping", "basis", "orbital_order", "spin_order"):
        if field in value:
            _text(value[field], f"symmetry.{field}")
    for field in ("spin_orbitals", "active_orbitals"):
        if field in value:
            _integer(value[field], f"symmetry.{field}", 1)
    for field in ("active_electrons", "particle_number"):
        if field in value:
            _integer(value[field], f"symmetry.{field}", 0)
    for field in ("magnetization", "spin_projection"):
        if field in value:
            _finite(value[field], f"symmetry.{field}")
    if "total_spin" in value and _finite(value["total_spin"], "symmetry.total_spin") < 0:
        raise ValueError("symmetry.total_spin must be non-negative")
    if "parity" in value:
        parity = value["parity"]
        _text(parity, "symmetry.parity") if isinstance(parity, str) else _finite(parity, "symmetry.parity")
    if value.get("active_electrons") != value.get("particle_number") and all(
        field in value for field in ("active_electrons", "particle_number")
    ):
        raise ValueError("symmetry.active_electrons and symmetry.particle_number must agree")
    return dict(value)

def _terms(raw: Mapping[str, Any], path: Path) -> tuple[int, tuple[PauliTerm, ...]]:
    entries = raw.get("pauli_terms")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must define a non-empty pauli_terms list")
    width: int | None = None
    combined: dict[str, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"pauli", "coeff"}:
            raise ValueError(f"pauli_terms[{index}] must contain exactly pauli and coeff")
        label = entry["pauli"]
        if not isinstance(label, str) or not label or set(label) - set("IXYZ"):
            raise ValueError(f"invalid pauli_terms[{index}].pauli")
        if width is None:
            width = len(label)
        elif len(label) != width:
            raise ValueError("all Pauli labels must have the same length")
        combined[label] = combined.get(label, 0.0) + _finite(entry["coeff"], f"pauli_terms[{index}].coeff")
    terms = tuple(PauliTerm(label, coeff) for label, coeff in sorted(combined.items()) if coeff != 0.0)
    if not terms:
        raise ValueError("Hamiltonian cannot simplify to zero")
    if all(set(term.pauli) == {"I"} for term in terms):
        raise ValueError("constant-only Hamiltonian has no variational ansatz problem")
    assert width is not None
    return width, terms

def load_problem_document(path: str | Path = DEFAULT_PROBLEM_PATH) -> tuple[PublicProblem, dict[str, Any]]:
    source = Path(path)
    raw = _read(source)
    missing, extra = {"name", "pauli_terms"} - set(raw), set(raw) - _TOP_FIELDS
    if missing or extra:
        raise ValueError(f"invalid problem document {source} fields: missing={sorted(missing)} extra={sorted(extra)}")
    name = _text(raw["name"], "name").strip()
    if "source_note" in raw:
        _text(raw["source_note"], "source_note")
    metadata = _symmetry(raw)
    width, terms = _terms(raw, source)

    basis = raw.get("basis_gates", list(DEFAULT_BASIS_GATES))
    if not isinstance(basis, list) or any(not isinstance(gate, str) or not gate for gate in basis):
        raise ValueError("basis_gates must be a list of non-empty strings")
    coupling = raw.get("coupling_map", [])
    if not isinstance(coupling, list) or any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(isinstance(q, bool) or not isinstance(q, int) for q in edge)
        for edge in coupling
    ):
        raise ValueError("coupling_map must be a list of two-item integer lists")
    has_hint = "initial_state_hint" in raw
    hint = raw.get("initial_state_hint")
    if has_hint and (
        not isinstance(hint, list)
        or len(hint) != width
        or any(isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1) for bit in hint)
    ):
        raise ValueError(f"initial_state_hint must be a {width}-item integer 0/1 list")

    problem = PublicProblem.create(
        problem_id=name,
        num_qubits=width,
        pauli_terms=terms,
        symmetry=metadata,
        initial_occupation=tuple(hint) if has_hint else None,
        backend=BackendSpec(tuple(basis), tuple(tuple(edge) for edge in coupling)),
    )
    document = canonical_data(raw)
    assert isinstance(document, dict)
    return problem, document

def load_problem(path: str | Path = DEFAULT_PROBLEM_PATH) -> PublicProblem:
    return load_problem_document(path)[0]

def hamiltonian_from_problem(problem: PublicProblem) -> SparsePauliOp:
    return SparsePauliOp.from_list([(term.pauli, complex(term.real, term.imag)) for term in problem.pauli_terms])

@dataclass(frozen=True)
class StructuralObservation:
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

@dataclass(frozen=True)
class ProblemObservation:
    problem_id: str
    num_qubits: int
    symmetry: tuple[tuple[str, Any], ...]
    initial_occupation: tuple[int, ...] | None
    backend: BackendSpec
    structure: StructuralObservation
    schema_version: str = SCHEMA_VERSION

def observe_problem(problem: PublicProblem) -> ProblemObservation:
    locality: Counter[int] = Counter()
    paulis: Counter[str] = Counter()
    edges: set[tuple[int, int]] = set()
    identity_count = 0
    for term in problem.pauli_terms:
        support = tuple(len(term.pauli) - index - 1 for index, value in enumerate(term.pauli) if value != "I")
        locality[len(support)] += 1
        identity_count += not support
        if len(support) == 2:
            edges.add(tuple(sorted(support)))
        paulis.update(value for value in term.pauli if value != "I")

    neighbors = {qubit: set() for qubit in range(problem.num_qubits)}
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    unseen, components = set(neighbors), []
    while unseen:
        stack, component = [min(unseen)], []
        unseen.remove(stack[0])
        while stack:
            qubit = stack.pop()
            component.append(qubit)
            for neighbor in sorted(neighbors[qubit] & unseen, reverse=True):
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))

    complete = len(edges) <= MAX_EXACT_SUPPORT_GRAPH_EDGES
    structure = StructuralObservation(
        len(problem.pauli_terms),
        identity_count,
        max(locality, default=0),
        tuple(sorted(locality.items())),
        tuple(sorted(paulis.items())),
        len(edges),
        tuple((qubit, len(neighbors[qubit])) for qubit in range(problem.num_qubits)),
        tuple(components),
        complete,
        tuple(sorted(edges)) if complete else (),
    )
    return ProblemObservation(
        problem.problem_id,
        problem.num_qubits,
        problem.symmetry,
        problem.initial_occupation,
        problem.backend,
        structure,
    )
