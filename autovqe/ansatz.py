"""Closed compiler for the small, typed AutoVQE circuit language."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


ALLOWED_GATES = frozenset({"PauliRotation", "XYExchange", "IsotropicExchange"})
ALLOWED_SCALES = frozenset({-2.0, -1.0, -0.5, 0.5, 1.0, 2.0})
_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class AnsatzIRValidationError(ValueError):
    """Raised when an ansatz specification is malformed."""


class AnsatzCompilerError(AnsatzIRValidationError):
    """Raised when a well-formed specification violates compiler policy."""


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnsatzIRValidationError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise AnsatzIRValidationError(f"{context} keys must be strings")
    return value


def _array(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnsatzIRValidationError(f"{context} must be an array")
    return value


def _strict(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
    *,
    required: set[str] = frozenset(),
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise AnsatzIRValidationError(
            f"{context} contains unsupported fields: {sorted(unknown)}"
        )
    if missing:
        raise AnsatzIRValidationError(f"{context} is missing fields: {sorted(missing)}")


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnsatzIRValidationError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnsatzIRValidationError(f"{context} must be a finite number")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnsatzIRValidationError(f"{context} must be an integer")
    return value


def _parameter_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _PARAMETER_NAME.fullmatch(value):
        raise AnsatzIRValidationError(
            f"{context} must match [A-Za-z_][A-Za-z0-9_.-]{{0,127}}"
        )
    return value


@dataclass(frozen=True)
class OperationSpec:
    """One trusted gate driven by one zero-origin parameter."""

    gate: str
    qubits: tuple[int, ...]
    parameter: str
    scale: float = 1.0
    pauli: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.gate, str) or not self.gate:
            raise AnsatzIRValidationError("operation.gate must be a non-empty string")
        qubits = tuple(self.qubits)
        for index, qubit in enumerate(qubits):
            _integer(qubit, f"operation.qubits[{index}]")
        if len(qubits) != len(set(qubits)):
            raise AnsatzIRValidationError("operation qubits must be unique")
        scale = _finite(self.scale, "operation.scale")
        if scale == 0.0:
            raise AnsatzIRValidationError("operation.scale cannot be zero")
        if self.pauli is not None and not isinstance(self.pauli, str):
            raise AnsatzIRValidationError("operation.pauli must be a string")
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "parameter", _parameter_name(self.parameter, "operation.parameter"))
        object.__setattr__(self, "scale", scale)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "gate": self.gate,
            "qubits": list(self.qubits),
            "parameter": self.parameter,
        }
        if self.scale != 1.0:
            result["scale"] = self.scale
        if self.pauli is not None:
            result["pauli"] = self.pauli
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationSpec":
        payload = _object(value, "operation")
        _strict(
            payload,
            {"gate", "qubits", "parameter", "scale", "pauli"},
            "operation",
            required={"gate", "qubits", "parameter"},
        )
        return cls(
            gate=payload["gate"],
            qubits=tuple(_array(payload["qubits"], "operation.qubits")),
            parameter=payload["parameter"],
            scale=payload.get("scale", 1.0),
            pauli=payload.get("pauli"),
        )


@dataclass(frozen=True)
class AnsatzSpec:
    num_qubits: int
    operations: tuple[OperationSpec, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        num_qubits = _integer(self.num_qubits, "ansatz.num_qubits")
        if num_qubits <= 0:
            raise AnsatzIRValidationError("ansatz.num_qubits must be positive")
        version = _integer(self.version, "ansatz.version")
        if version != 1:
            raise AnsatzIRValidationError(f"unsupported ansatz IR version: {version}")
        operations = tuple(self.operations)
        if not all(isinstance(item, OperationSpec) for item in operations):
            raise AnsatzIRValidationError("ansatz.operations must contain OperationSpec objects")
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "version", version)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Unique raw labels, ordered by first use in the submitted circuit."""

        return tuple(dict.fromkeys(operation.parameter for operation in self.operations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "num_qubits": self.num_qubits,
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnsatzSpec":
        payload = _object(value, "ansatz")
        _strict(
            payload,
            {"version", "num_qubits", "operations"},
            "ansatz",
            required={"num_qubits"},
        )
        operations = _array(payload.get("operations", ()), "ansatz.operations")
        return cls(
            num_qubits=payload["num_qubits"],
            operations=tuple(OperationSpec.from_dict(item) for item in operations),
            version=payload.get("version", 1),
        )


@dataclass(frozen=True)
class CompiledAnsatz:
    """A trusted circuit, its Qiskit parameters, and compiler-derived audit."""

    circuit: QuantumCircuit
    parameters: Mapping[str, Parameter]
    audit: dict[str, Any]


def _coerce_spec(value: AnsatzSpec | Mapping[str, Any]) -> AnsatzSpec:
    if isinstance(value, AnsatzSpec):
        return value
    if isinstance(value, Mapping):
        return AnsatzSpec.from_dict(value)
    raise AnsatzCompilerError(
        "compiler input must be an AnsatzSpec or its serialized dictionary"
    )


def _validate_operation(operation: OperationSpec, *, num_qubits: int, path: str) -> None:
    if operation.gate not in ALLOWED_GATES:
        raise AnsatzCompilerError(f"{path}: unknown or untrusted gate {operation.gate!r}")
    for index, qubit in enumerate(operation.qubits):
        if not 0 <= qubit < num_qubits:
            raise AnsatzCompilerError(
                f"{path}/qubits/{index}={qubit} is outside [0, {num_qubits})"
            )
    if operation.scale not in ALLOWED_SCALES:
        raise AnsatzCompilerError(
            f"{path}/scale must be one of {sorted(ALLOWED_SCALES)}"
        )
    if operation.gate == "PauliRotation":
        pauli = operation.pauli
        if pauli is None:
            raise AnsatzCompilerError(f"{path}: PauliRotation requires pauli")
        if not pauli or any(letter not in "XYZ" for letter in pauli):
            raise AnsatzCompilerError(f"{path}: pauli must contain active X, Y, or Z only")
        if len(operation.qubits) != len(pauli):
            raise AnsatzCompilerError(f"{path}: pauli length must equal qubit locality")
    else:
        if operation.pauli is not None:
            raise AnsatzCompilerError(f"{path}: {operation.gate} does not take pauli")
        if len(operation.qubits) != 2:
            raise AnsatzCompilerError(f"{path}: {operation.gate} requires two qubits")


def operation_paulis(operation: OperationSpec) -> tuple[tuple[str, float], ...]:
    """Return the trusted local Pauli generator terms for one operation."""

    if operation.gate == "PauliRotation":
        if operation.pauli is None:
            raise AnsatzCompilerError("PauliRotation requires pauli")
        words = (operation.pauli,)
    elif operation.gate == "XYExchange":
        words = ("XX", "YY")
    elif operation.gate == "IsotropicExchange":
        words = ("XX", "YY", "ZZ")
    else:
        raise AnsatzCompilerError(f"unknown or untrusted gate {operation.gate!r}")
    return tuple((word, operation.scale) for word in words)


def pauli_label(num_qubits: int, qubits: Sequence[int], word: str) -> str:
    """Embed a local Pauli word using Qiskit's little-endian label order."""

    if len(qubits) != len(word) or any(not 0 <= qubit < num_qubits for qubit in qubits):
        raise AnsatzCompilerError("Pauli word support is invalid")
    label = ["I"] * num_qubits
    for qubit, letter in zip(qubits, word, strict=True):
        label[num_qubits - qubit - 1] = letter
    return "".join(label)


def _emit(circuit: QuantumCircuit, operation: OperationSpec, angle: Any) -> None:
    qubits = operation.qubits
    if operation.gate == "PauliRotation":
        assert operation.pauli is not None
        pauli = operation.pauli
        for qubit, letter in zip(qubits, pauli, strict=True):
            if letter == "X":
                circuit.h(qubit)
            elif letter == "Y":
                circuit.sdg(qubit)
                circuit.h(qubit)
        for control, target in zip(qubits[:-1], qubits[1:]):
            circuit.cx(control, target)
        circuit.rz(2.0 * angle, qubits[-1])
        for control, target in reversed(tuple(zip(qubits[:-1], qubits[1:]))):
            circuit.cx(control, target)
        for qubit, letter in reversed(tuple(zip(qubits, pauli, strict=True))):
            if letter == "X":
                circuit.h(qubit)
            elif letter == "Y":
                circuit.h(qubit)
                circuit.s(qubit)
    elif operation.gate == "XYExchange":
        circuit.append(XXPlusYYGate(4.0 * angle, 0.0), list(qubits))
    else:
        left, right = qubits
        circuit.rxx(2.0 * angle, left, right)
        circuit.ryy(2.0 * angle, left, right)
        circuit.rzz(2.0 * angle, left, right)


def compile_ansatz(value: AnsatzSpec | Mapping[str, Any]) -> CompiledAnsatz:
    """Validate and compile v1 JSON using the closed three-gate implementation."""

    spec = _coerce_spec(value)
    occurrences: Counter[str] = Counter()
    for index, operation in enumerate(spec.operations):
        _validate_operation(
            operation,
            num_qubits=spec.num_qubits,
            path=f"/operations/{index}",
        )
        occurrences[operation.parameter] += 1

    names = spec.parameter_names
    qiskit_parameters = {name: Parameter(name) for name in names}
    circuit = QuantumCircuit(spec.num_qubits, name="ansatz")
    for operation in spec.operations:
        _emit(
            circuit,
            operation,
            operation.scale * qiskit_parameters[operation.parameter],
        )

    audit = {
        "unique_trainable_params": len(names),
        "trainable_parameter_names": list(names),
        "parameter_occurrences": {name: occurrences[name] for name in names},
        "operations": len(spec.operations),
    }
    return CompiledAnsatz(circuit, qiskit_parameters, audit)


__all__ = [
    "ALLOWED_GATES",
    "ALLOWED_SCALES",
    "AnsatzCompilerError",
    "AnsatzIRValidationError",
    "AnsatzSpec",
    "CompiledAnsatz",
    "OperationSpec",
    "compile_ansatz",
]
