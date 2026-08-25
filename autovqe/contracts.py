"""Immutable data contracts for the agent/evaluator boundary.

This module intentionally depends only on the Python standard library.  Public
contracts contain the Hamiltonian and declared execution constraints, while
exact reference data lives in :class:`PrivateEvaluationContext`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, Union


JsonScalar = Union[None, bool, int, float, str]
SectorValue = Union[bool, int, float, str]

SCHEMA_VERSION = "1"
FORBIDDEN_AGENT_KEYS = frozenset(
    {
        "candidate",
        "candidates",
        "exact_energy",
        "exact_state",
        "model_class",
        "recommendation",
        "recommendations",
        "reference_energy",
        "reference_state",
        "reference_vector",
    }
)


def _finite_float(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if value == 0.0 else value


def _pairs(
    values: Sequence[tuple[str, SectorValue]],
    field_name: str,
) -> tuple[tuple[str, SectorValue], ...]:
    result: list[tuple[str, SectorValue]] = []
    seen: set[str] = set()
    for key, value in values:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        if key in seen:
            raise ValueError(f"duplicate {field_name} key: {key}")
        if not isinstance(value, (bool, int, float, str)):
            raise TypeError(f"{field_name}[{key!r}] has a non-scalar value")
        if isinstance(value, float):
            value = _finite_float(value, f"{field_name}[{key!r}]")
        seen.add(key)
        result.append((key, value))
    return tuple(sorted(result, key=lambda item: item[0]))


@dataclass(frozen=True)
class PauliTerm:
    """One canonical Pauli term in a public Hamiltonian.

    Labels use Qiskit's little-endian display convention: the rightmost
    character acts on qubit 0.
    """

    pauli: str
    real: float
    imag: float = 0.0

    def __post_init__(self) -> None:
        if not self.pauli or not set(self.pauli).issubset({"I", "X", "Y", "Z"}):
            raise ValueError(f"invalid Pauli label: {self.pauli!r}")
        object.__setattr__(self, "real", _finite_float(self.real, "real"))
        object.__setattr__(self, "imag", _finite_float(self.imag, "imag"))


@dataclass(frozen=True)
class EncodingSpec:
    """Public metadata describing how the physical problem is encoded."""

    name: str = "qubit_pauli"
    mapping: str | None = None
    qubit_order: str = "qiskit_little_endian"
    spin_orbitals: int | None = None
    active_orbitals: int | None = None
    attributes: tuple[tuple[str, SectorValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("encoding name must be non-empty")
        if self.qubit_order != "qiskit_little_endian":
            raise ValueError(
                "the MVP evaluator only supports qubit_order='qiskit_little_endian'"
            )
        for field_name in ("spin_orbitals", "active_orbitals"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive")
        object.__setattr__(self, "attributes", _pairs(self.attributes, "encoding attributes"))


@dataclass(frozen=True)
class SectorSpec:
    """Publicly declared conserved charges and requested sector values."""

    symmetries: tuple[str, ...] = ()
    values: tuple[tuple[str, SectorValue], ...] = ()

    def __post_init__(self) -> None:
        symmetries = tuple(sorted(set(self.symmetries)))
        if any(not isinstance(item, str) or not item for item in symmetries):
            raise ValueError("sector symmetries must be non-empty strings")
        object.__setattr__(self, "symmetries", symmetries)
        object.__setattr__(self, "values", _pairs(self.values, "sector values"))


@dataclass(frozen=True)
class ReferenceSpec:
    """Agent-visible preparation hint, never an exact eigenstate reference.

    ``occupation`` is a computational-basis preparation hint supplied as part
    of the problem.  Tuple index ``q`` is the bit prepared on qubit ``q``.
    Exact evaluator energies and vectors are deliberately not fields of this
    contract.
    """

    kind: str = "none"
    occupation: tuple[int, ...] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("reference kind must be non-empty")
        if self.occupation is not None:
            occupation = tuple(int(bit) for bit in self.occupation)
            if any(bit not in (0, 1) for bit in occupation):
                raise ValueError("reference occupation must contain only 0/1 bits")
            object.__setattr__(self, "occupation", occupation)


@dataclass(frozen=True)
class BackendSpec:
    """Agent-visible native gate and connectivity constraints."""

    basis_gates: tuple[str, ...] = ()
    coupling_map: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        basis_gates = tuple(str(gate) for gate in self.basis_gates)
        if any(not gate for gate in basis_gates):
            raise ValueError("basis gate names must be non-empty")
        edges = tuple((int(left), int(right)) for left, right in self.coupling_map)
        if any(left < 0 or right < 0 or left == right for left, right in edges):
            raise ValueError("coupling-map edges must join distinct non-negative qubits")
        object.__setattr__(self, "basis_gates", basis_gates)
        object.__setattr__(self, "coupling_map", edges)


@dataclass(frozen=True)
class PublicProblem:
    """The complete problem view that an ansatz-producing agent may inspect."""

    problem_id: str
    num_qubits: int
    pauli_terms: tuple[PauliTerm, ...]
    encoding: EncodingSpec
    sector: SectorSpec
    reference: ReferenceSpec
    backend: BackendSpec
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        terms = tuple(self.pauli_terms)
        if not terms:
            raise ValueError("pauli_terms must be non-empty")
        if any(len(term.pauli) != self.num_qubits for term in terms):
            raise ValueError("all Pauli labels must match num_qubits")
        # Every Pauli word is Hermitian, so a qubit Hamiltonian written in the
        # Pauli basis is Hermitian exactly when its canonical coefficients are
        # real.  Do not let downstream evaluators silently discard an
        # imaginary expectation value.
        if any(term.imag != 0.0 for term in terms):
            raise ValueError("public Hamiltonian Pauli coefficients must be real")
        if self.reference.occupation is not None and len(self.reference.occupation) != self.num_qubits:
            raise ValueError("reference occupation must match num_qubits")
        for left, right in self.backend.coupling_map:
            if left >= self.num_qubits or right >= self.num_qubits:
                raise ValueError("coupling-map qubit is outside the public problem")
        object.__setattr__(self, "pauli_terms", terms)
        assert_agent_safe(self)

    @classmethod
    def create(
        cls,
        *,
        problem_id: str = "problem",
        num_qubits: int,
        pauli_terms: Sequence[PauliTerm],
        encoding: EncodingSpec,
        sector: SectorSpec,
        reference: ReferenceSpec,
        backend: BackendSpec,
        schema_version: str = SCHEMA_VERSION,
    ) -> "PublicProblem":
        """Create a public problem with a caller-supplied descriptive identifier."""

        return cls(
            problem_id=problem_id,
            num_qubits=num_qubits,
            pauli_terms=tuple(pauli_terms),
            encoding=encoding,
            sector=sector,
            reference=reference,
            backend=backend,
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class PrivateEvaluationContext:
    """Evaluator-only data.  Instances must never be passed to the agent."""

    public_problem: PublicProblem
    source_name: str
    reference_energy: float | None = None
    reference_state: tuple[complex, ...] | None = None

    def __post_init__(self) -> None:
        if self.reference_energy is not None:
            object.__setattr__(
                self,
                "reference_energy",
                _finite_float(self.reference_energy, "reference_energy"),
            )
        if self.reference_state is not None:
            state = tuple(complex(amplitude) for amplitude in self.reference_state)
            expected = 1 << self.public_problem.num_qubits
            if len(state) != expected:
                raise ValueError(f"reference_state must contain {expected} amplitudes")
            if any(not math.isfinite(value.real) or not math.isfinite(value.imag) for value in state):
                raise ValueError("reference_state amplitudes must be finite")
            object.__setattr__(self, "reference_state", state)


# Descriptive aliases keep call sites readable without creating parallel types.
EncodingMetadata = EncodingSpec
SectorMetadata = SectorSpec
ReferenceMetadata = ReferenceSpec
BackendConstraints = BackendSpec


def _canonical_data(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical_data(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return _canonical_data(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings require string keys")
            result[key] = _canonical_data(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_data(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, complex):
        return {
            "imag": _finite_float(value.imag, "complex.imag"),
            "real": _finite_float(value.real, "complex.real"),
        }
    if isinstance(value, float):
        return _finite_float(value, "float")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_data(value: Any) -> Any:
    """Convert supported contracts and stdlib containers to JSON data."""

    return _canonical_data(value)


def canonical_json(value: Any) -> str:
    """Serialize a supported value as stable, portable JSON."""

    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


T = TypeVar("T")


def assert_agent_safe(value: T) -> T:
    """Reject private or policy-derived fields in agent-facing data."""

    def walk(item: Any, path: str) -> None:
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                lowered = field.name.lower()
                if lowered in FORBIDDEN_AGENT_KEYS:
                    raise ValueError(f"agent-facing data contains forbidden field {path}{field.name}")
                walk(getattr(item, field.name), f"{path}{field.name}.")
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise TypeError("agent-facing mappings require string keys")
                lowered = key.lower()
                if lowered in FORBIDDEN_AGENT_KEYS:
                    raise ValueError(f"agent-facing data contains forbidden key {path}{key}")
                walk(nested, f"{path}{key}.")
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}{index}.")

    walk(value, "")
    return value


__all__ = [
    "BackendConstraints",
    "BackendSpec",
    "EncodingMetadata",
    "EncodingSpec",
    "FORBIDDEN_AGENT_KEYS",
    "PauliTerm",
    "PrivateEvaluationContext",
    "PublicProblem",
    "ReferenceMetadata",
    "ReferenceSpec",
    "SCHEMA_VERSION",
    "SectorMetadata",
    "SectorSpec",
    "assert_agent_safe",
    "canonical_data",
    "canonical_json",
]
