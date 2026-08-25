"""Validated, serializable data models used throughout AutoVQE."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, Union


JsonScalar = Union[None, bool, int, float, str]
SectorValue = Union[bool, int, float, str]
SCHEMA_VERSION = "1"


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
    """One Pauli word and its coefficient."""

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
    """How the physical problem was mapped to qubits."""

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
            raise ValueError("qubit_order must be 'qiskit_little_endian'")
        for field_name in ("spin_orbitals", "active_orbitals"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive")
        object.__setattr__(self, "attributes", _pairs(self.attributes, "encoding attributes"))


@dataclass(frozen=True)
class SectorSpec:
    """Declared conserved quantities and requested sector values."""

    symmetries: tuple[str, ...] = ()
    values: tuple[tuple[str, SectorValue], ...] = ()

    def __post_init__(self) -> None:
        symmetries = tuple(sorted(set(self.symmetries)))
        if any(not isinstance(item, str) or not item for item in symmetries):
            raise ValueError("sector symmetries must be non-empty strings")
        object.__setattr__(self, "symmetries", symmetries)
        object.__setattr__(self, "values", _pairs(self.values, "sector values"))


@dataclass(frozen=True)
class InitialStateSpec:
    """Optional computational-basis preparation supplied with the problem."""

    kind: str = "none"
    occupation: tuple[int, ...] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("initial-state kind must be non-empty")
        if self.occupation is None:
            return
        occupation = tuple(int(bit) for bit in self.occupation)
        if any(bit not in (0, 1) for bit in occupation):
            raise ValueError("initial-state occupation must contain only 0/1 bits")
        object.__setattr__(self, "occupation", occupation)


@dataclass(frozen=True)
class BackendSpec:
    """Declared native-gate and connectivity constraints."""

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
    """The single validated representation of an AutoVQE input problem."""

    problem_id: str
    num_qubits: int
    pauli_terms: tuple[PauliTerm, ...]
    encoding: EncodingSpec
    sector: SectorSpec
    initial_state: InitialStateSpec
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
        if any(term.imag != 0.0 for term in terms):
            raise ValueError("Hamiltonian Pauli coefficients must be real")
        occupation = self.initial_state.occupation
        if occupation is not None and len(occupation) != self.num_qubits:
            raise ValueError("initial-state occupation must match num_qubits")
        for left, right in self.backend.coupling_map:
            if left >= self.num_qubits or right >= self.num_qubits:
                raise ValueError("coupling-map qubit is outside the problem")
        object.__setattr__(self, "pauli_terms", terms)

    @classmethod
    def create(
        cls,
        *,
        problem_id: str = "problem",
        num_qubits: int,
        pauli_terms: Sequence[PauliTerm],
        encoding: EncodingSpec | None = None,
        sector: SectorSpec | None = None,
        initial_state: InitialStateSpec | None = None,
        backend: BackendSpec | None = None,
        schema_version: str = SCHEMA_VERSION,
    ) -> "PublicProblem":
        return cls(
            problem_id=problem_id,
            num_qubits=num_qubits,
            pauli_terms=tuple(pauli_terms),
            encoding=encoding or EncodingSpec(),
            sector=sector or SectorSpec(),
            initial_state=initial_state or InitialStateSpec(),
            backend=backend or BackendSpec(),
            schema_version=schema_version,
        )


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
            "real": _finite_float(value.real, "complex.real"),
            "imag": _finite_float(value.imag, "complex.imag"),
        }
    if isinstance(value, float):
        return _finite_float(value, "float")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_data(value: Any) -> Any:
    """Convert supported contracts and standard containers to JSON data."""

    return _canonical_data(value)


def canonical_json(value: Any) -> str:
    """Serialize a supported value as stable JSON."""

    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "BackendSpec",
    "EncodingSpec",
    "InitialStateSpec",
    "PauliTerm",
    "PublicProblem",
    "SCHEMA_VERSION",
    "SectorSpec",
    "canonical_data",
    "canonical_json",
]
