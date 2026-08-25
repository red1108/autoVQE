"""Load a Hamiltonian JSON file into AutoVQE's single problem model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from qiskit.quantum_info import SparsePauliOp

from .contracts import (
    BackendSpec,
    EncodingSpec,
    InitialStateSpec,
    PauliTerm,
    PublicProblem,
    SectorSpec,
    canonical_data,
)


DEFAULT_PROBLEM_PATH = Path("user_problem/hamiltonian.json")
DEFAULT_BASIS_GATES = ("rx", "ry", "rz", "cx")
_TOP_LEVEL_FIELDS = {
    "name",
    "pauli_terms",
    "basis_gates",
    "coupling_map",
    "initial_state_hint",
    "source_note",
    "symmetry",
}
_REQUIRED_TOP_LEVEL_FIELDS = {"name", "pauli_terms"}
_SYMMETRY_FIELDS = {
    "mapping",
    "basis",
    "orbital_order",
    "spin_order",
    "spin_orbitals",
    "active_orbitals",
    "active_electrons",
    "particle_number",
    "magnetization",
    "spin_projection",
    "total_spin",
    "parity",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read problem {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return raw


def _field_error(scope: str, *, missing: set[str], extra: set[str]) -> ValueError:
    return ValueError(
        f"invalid {scope} fields: missing={sorted(missing)} extra={sorted(extra)}"
    )


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _real(value: Any, field: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if non_negative and result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _validated_symmetry(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "symmetry" not in raw:
        return {}
    value = raw["symmetry"]
    if not isinstance(value, dict):
        raise ValueError("symmetry must be an object")
    extra = set(value) - _SYMMETRY_FIELDS
    if extra:
        raise _field_error("symmetry", missing=set(), extra=extra)

    for field in ("mapping", "basis", "orbital_order", "spin_order"):
        if field in value:
            _nonempty_text(value[field], f"symmetry.{field}")
    for field in ("spin_orbitals", "active_orbitals"):
        if field in value:
            _integer(value[field], f"symmetry.{field}", minimum=1)
    for field in ("active_electrons", "particle_number"):
        if field in value:
            _integer(value[field], f"symmetry.{field}", minimum=0)
    for field in ("magnetization", "spin_projection"):
        if field in value:
            _real(value[field], f"symmetry.{field}")
    if "total_spin" in value:
        _real(value["total_spin"], "symmetry.total_spin", non_negative=True)
    if "parity" in value:
        parity = value["parity"]
        if isinstance(parity, str):
            _nonempty_text(parity, "symmetry.parity")
        elif isinstance(parity, bool) or not isinstance(parity, (int, float)):
            raise ValueError("symmetry.parity must be a finite number or non-empty string")
        else:
            _real(parity, "symmetry.parity")

    if (
        "active_electrons" in value
        and "particle_number" in value
        and value["active_electrons"] != value["particle_number"]
    ):
        raise ValueError(
            "symmetry.active_electrons and symmetry.particle_number must agree"
        )
    return dict(value)


def _validate_document_schema(raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    missing = _REQUIRED_TOP_LEVEL_FIELDS - set(raw)
    extra = set(raw) - _TOP_LEVEL_FIELDS
    if missing or extra:
        raise _field_error(f"problem document {path}", missing=missing, extra=extra)
    _nonempty_text(raw["name"], "name")
    if "source_note" in raw:
        _nonempty_text(raw["source_note"], "source_note")
    return _validated_symmetry(raw)


def _encoding(metadata: Mapping[str, Any]) -> EncodingSpec:
    mapping = metadata.get("mapping")
    attributes = tuple(
        (key, value)
        for key in ("basis", "orbital_order", "spin_order")
        if (value := metadata.get(key)) is not None
    )
    return EncodingSpec(
        mapping=mapping,
        spin_orbitals=metadata.get("spin_orbitals"),
        active_orbitals=metadata.get("active_orbitals"),
        attributes=attributes,
    )


def _sector(metadata: Mapping[str, Any]) -> SectorSpec:
    key_map = (
        ("active_electrons", "particle_number"),
        ("particle_number", "particle_number"),
        ("magnetization", "magnetization"),
        ("spin_projection", "spin_projection"),
        ("total_spin", "total_spin"),
        ("parity", "parity"),
    )
    values: dict[str, str | int | float | bool] = {}
    for source, target in key_map:
        if source in metadata:
            values.setdefault(target, metadata[source])
    return SectorSpec(symmetries=tuple(values), values=tuple(values.items()))


def _occupation(value: Any, *, num_qubits: int, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != num_qubits:
        raise ValueError(f"{field} must be a {num_qubits}-item 0/1 list")
    if any(isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1) for bit in value):
        raise ValueError(f"{field} must contain only integer 0/1 bits")
    return tuple(value)


def _initial_state(
    raw: Mapping[str, Any],
    *,
    num_qubits: int,
) -> InitialStateSpec:
    if "initial_state_hint" in raw:
        hint = raw["initial_state_hint"]
        return InitialStateSpec(
            kind="computational_basis",
            occupation=_occupation(
                hint,
                num_qubits=num_qubits,
                field="initial_state_hint",
            ),
            source="initial_state_hint",
        )
    return InitialStateSpec()


def _backend(raw: Mapping[str, Any]) -> BackendSpec:
    basis_gates = raw.get("basis_gates", list(DEFAULT_BASIS_GATES))
    if not isinstance(basis_gates, list) or not all(
        isinstance(gate, str) and gate for gate in basis_gates
    ):
        raise ValueError("basis_gates must be a list of non-empty strings")

    coupling_map = raw.get("coupling_map", [])
    if not isinstance(coupling_map, list):
        raise ValueError("coupling_map must be a list of [control, target] pairs")
    edges: list[tuple[int, int]] = []
    for index, edge in enumerate(coupling_map):
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in edge)
        ):
            raise ValueError(
                f"coupling_map[{index}] must be a two-item integer list"
            )
        edges.append((edge[0], edge[1]))
    return BackendSpec(basis_gates=tuple(basis_gates), coupling_map=tuple(edges))


def _pauli_terms(raw: Mapping[str, Any], path: Path) -> tuple[int, tuple[PauliTerm, ...]]:
    entries = raw.get("pauli_terms")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must define a non-empty pauli_terms list")

    width: int | None = None
    combined: dict[str, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"pauli", "coeff"}:
            raise ValueError(
                f"pauli_terms[{index}] must contain exactly pauli and coeff"
            )
        label = entry["pauli"]
        if not isinstance(label, str) or not label or not set(label) <= {"I", "X", "Y", "Z"}:
            raise ValueError(f"invalid pauli_terms[{index}].pauli")
        if width is None:
            width = len(label)
        elif len(label) != width:
            raise ValueError("all Pauli labels must have the same length")

        coefficient = entry["coeff"]
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ValueError(f"pauli_terms[{index}].coeff must be a real number")
        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise ValueError(f"pauli_terms[{index}].coeff must be finite")
        combined[label] = combined.get(label, 0.0) + coefficient

    terms = tuple(
        PauliTerm(pauli=label, real=coefficient)
        for label, coefficient in sorted(combined.items())
        if coefficient != 0.0
    )
    if not terms:
        raise ValueError("Hamiltonian cannot simplify to zero")
    assert width is not None
    return width, terms


def load_problem_document(
    path: str | Path = DEFAULT_PROBLEM_PATH,
) -> tuple[PublicProblem, dict[str, Any]]:
    """Return the model and canonical validated v1 input document."""

    source = Path(path)
    raw = _read_object(source)
    metadata = _validate_document_schema(raw, source)
    num_qubits, terms = _pauli_terms(raw, source)
    problem = PublicProblem.create(
        problem_id=str(raw["name"]).strip(),
        num_qubits=num_qubits,
        pauli_terms=terms,
        encoding=_encoding(metadata),
        sector=_sector(metadata),
        initial_state=_initial_state(raw, num_qubits=num_qubits),
        backend=_backend(raw),
    )
    document = canonical_data(raw)
    if not isinstance(document, dict):
        raise AssertionError("canonical problem document must be an object")
    return problem, document


def load_problem(path: str | Path = DEFAULT_PROBLEM_PATH) -> PublicProblem:
    """Read and validate one AutoVQE problem without deriving a solution."""

    return load_problem_document(path)[0]


def hamiltonian_from_problem(problem: PublicProblem) -> SparsePauliOp:
    """Construct the Qiskit operator used by probes and evaluation."""

    return SparsePauliOp.from_list(
        [
            (term.pauli, complex(term.real, term.imag))
            for term in problem.pauli_terms
        ]
    )


__all__ = [
    "DEFAULT_BASIS_GATES",
    "DEFAULT_PROBLEM_PATH",
    "hamiltonian_from_problem",
    "load_problem",
    "load_problem_document",
]
