"""Public Hamiltonian input and compact structural inspection."""
from __future__ import annotations
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from qiskit.quantum_info import SparsePauliOp

SCHEMA_VERSION = "1"
DEFAULT_PROBLEM_PATH = Path("user_problem/hamiltonian.json")
DEFAULT_BASIS_GATES = ("rx", "ry", "rz", "cx")
TOP_FIELDS = set("name pauli_terms basis_gates coupling_map initial_state_hint source_note symmetry".split())
SYMMETRY_FIELDS = set(
    "mapping basis orbital_order spin_order spin_orbitals active_orbitals active_electrons "
    "particle_number magnetization spin_projection total_spin parity".split()
)

def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return 0.0 if value == 0.0 else value

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
        gates, edges = tuple(self.basis_gates), tuple(tuple(edge) for edge in self.coupling_map)
        if any(not isinstance(gate, str) or not gate for gate in gates):
            raise ValueError("basis gate names must be non-empty strings")
        invalid_edge = any(
            len(edge) != 2
            or edge[0] == edge[1]
            or any(isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in edge)
            for edge in edges
        )
        if invalid_edge:
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
        terms, occupation = tuple(self.pauli_terms), self.initial_occupation
        if not isinstance(self.problem_id, str) or not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int) or self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if not terms or any(len(term.pauli) != self.num_qubits for term in terms):
            raise ValueError("non-empty Pauli terms must match num_qubits")
        if any(term.imag for term in terms):
            raise ValueError("Hamiltonian Pauli coefficients must be real")
        if any(not isinstance(key, str) for key, _ in self.symmetry):
            raise ValueError("symmetry keys must be strings")
        if occupation is not None and (
            len(occupation) != self.num_qubits
            or any(type(bit) is not int or bit not in (0, 1) for bit in occupation)
        ):
            raise ValueError("initial_occupation must contain one integer 0/1 bit per qubit")
        if any(max(edge) >= self.num_qubits for edge in self.backend.coupling_map):
            raise ValueError("coupling-map qubit is outside the problem")
        object.__setattr__(self, "pauli_terms", terms)
        object.__setattr__(self, "symmetry", tuple(sorted(self.symmetry, key=lambda item: item[0])))
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
        occupation = None if initial_occupation is None else tuple(initial_occupation)
        return cls(
            problem_id,
            num_qubits,
            tuple(pauli_terms),
            tuple((symmetry or {}).items()),
            occupation,
            backend or BackendSpec(),
            schema_version,
        )

def canonical_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
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

def decode_json_object(text: str, source: str | Path = "JSON") -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {value}")

    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return value

def _symmetry(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = raw.get("symmetry", {})
    if not isinstance(value, dict):
        raise ValueError("symmetry must be an object")
    if set(value) - SYMMETRY_FIELDS:
        raise ValueError(f"invalid symmetry fields: {sorted(set(value) - SYMMETRY_FIELDS)}")
    for field in ("mapping", "basis", "orbital_order", "spin_order"):
        if field in value and (not isinstance(value[field], str) or not value[field].strip()):
            raise ValueError(f"symmetry.{field} must be a non-empty string")
    for field, minimum in {"spin_orbitals": 1, "active_orbitals": 1, "active_electrons": 0, "particle_number": 0}.items():
        if field in value and (type(value[field]) is not int or value[field] < minimum):
            raise ValueError(f"symmetry.{field} must be an integer >= {minimum}")
    for field in ("magnetization", "spin_projection", "total_spin"):
        if field in value and _finite(value[field], f"symmetry.{field}") < (0 if field == "total_spin" else -math.inf):
            raise ValueError("symmetry.total_spin must be non-negative")
    if "parity" in value:
        parity = value["parity"]
        if isinstance(parity, str):
            if not parity.strip():
                raise ValueError("symmetry.parity must be non-empty")
        else:
            _finite(parity, "symmetry.parity")
    if all(field in value for field in ("active_electrons", "particle_number")) and value["active_electrons"] != value["particle_number"]:
        raise ValueError("symmetry.active_electrons and symmetry.particle_number must agree")
    return value

def load_problem_document(path: str | Path = DEFAULT_PROBLEM_PATH) -> tuple[PublicProblem, dict[str, Any]]:
    source = Path(path)
    try:
        raw = decode_json_object(source.read_text(encoding="utf-8-sig"), source)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read problem {source}: {exc}") from exc
    missing, extra = {"name", "pauli_terms"} - set(raw), set(raw) - TOP_FIELDS
    if missing or extra:
        raise ValueError(f"invalid problem document {source} fields: missing={sorted(missing)} extra={sorted(extra)}")
    name, entries = raw["name"], raw["pauli_terms"]
    if not isinstance(name, str) or not name.strip() or not isinstance(entries, list) or not entries:
        raise ValueError("name and a non-empty pauli_terms list are required")
    if "source_note" in raw and (not isinstance(raw["source_note"], str) or not raw["source_note"].strip()):
        raise ValueError("source_note must be a non-empty string")
    width, combined = 0, {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"pauli", "coeff"}:
            raise ValueError(f"pauli_terms[{index}] must contain exactly pauli and coeff")
        label = entry["pauli"]
        if not isinstance(label, str) or not label or set(label) - set("IXYZ") or (width and len(label) != width):
            raise ValueError(f"invalid or inconsistent pauli_terms[{index}].pauli")
        width = width or len(label)
        combined[label] = combined.get(label, 0.0) + _finite(entry["coeff"], f"pauli_terms[{index}].coeff")
    terms = tuple(PauliTerm(label, coeff) for label, coeff in sorted(combined.items()) if coeff)
    if not terms:
        raise ValueError("Hamiltonian cannot simplify to zero")
    if all(set(term.pauli) == {"I"} for term in terms):
        raise ValueError("constant-only Hamiltonian has no variational ansatz problem")
    basis = raw.get("basis_gates", list(DEFAULT_BASIS_GATES))
    coupling = raw.get("coupling_map", [])
    hint = raw.get("initial_state_hint")
    if not isinstance(basis, list) or not isinstance(coupling, list) or any(not isinstance(edge, list) for edge in coupling):
        raise ValueError("basis_gates and coupling_map must be JSON lists")
    if "initial_state_hint" in raw and (
        not isinstance(hint, list)
        or len(hint) != width
        or any(type(bit) is not int or bit not in (0, 1) for bit in hint)
    ):
        raise ValueError(f"initial_state_hint must be a {width}-item integer 0/1 list")
    problem = PublicProblem.create(
        num_qubits=width,
        pauli_terms=terms,
        problem_id=name.strip(),
        symmetry=_symmetry(raw),
        initial_occupation=hint if "initial_state_hint" in raw else None,
        backend=BackendSpec(tuple(basis), tuple(tuple(edge) for edge in coupling)),
    )
    document = canonical_data(raw)
    assert isinstance(document, dict)
    return problem, document

def load_problem(path: str | Path = DEFAULT_PROBLEM_PATH) -> PublicProblem:
    return load_problem_document(path)[0]

def hamiltonian_from_problem(problem: PublicProblem) -> SparsePauliOp:
    return SparsePauliOp.from_list([(term.pauli, complex(term.real, term.imag)) for term in problem.pauli_terms])

def observe_problem(problem: PublicProblem) -> dict[str, Any]:
    locality, paulis, edges = Counter(), Counter(), set()
    for term in problem.pauli_terms:
        support = tuple(len(term.pauli) - index - 1 for index, value in enumerate(term.pauli) if value != "I")
        locality[len(support)] += 1
        paulis.update(value for value in term.pauli if value != "I")
        if len(support) == 2:
            edges.add(tuple(sorted(support)))
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
    complete = len(edges) <= 64
    structure = {
        "term_count": len(problem.pauli_terms), "identity_term_count": locality.get(0, 0),
        "max_locality": max(locality, default=0), "locality_counts": tuple(sorted(locality.items())),
        "pauli_counts": tuple(sorted(paulis.items())), "support_graph_edge_count": len(edges),
        "support_graph_degrees": tuple((q, len(neighbors[q])) for q in range(problem.num_qubits)),
        "support_graph_components": tuple(components), "support_graph_edges_complete": complete,
        "support_graph_edges": tuple(sorted(edges)) if complete else (),
    }
    return {
        "problem_id": problem.problem_id,
        "num_qubits": problem.num_qubits,
        "symmetry": problem.symmetry,
        "initial_occupation": problem.initial_occupation,
        "backend": problem.backend,
        "structure": structure,
        "schema_version": problem.schema_version,
    }
