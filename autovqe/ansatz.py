"""Closed compiler for AutoVQE's three-gate circuit language."""
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
PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")

class AnsatzIRValidationError(ValueError):
    pass


class AnsatzCompilerError(AnsatzIRValidationError):
    pass

def _fields(value: Any, allowed: set[str], required: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AnsatzIRValidationError(f"{label} must be an object with string keys")
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown:
        raise AnsatzIRValidationError(f"{label} contains unsupported fields: {sorted(unknown)}")
    if missing:
        raise AnsatzIRValidationError(f"{label} is missing fields: {sorted(missing)}")
    return value

def _array(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnsatzIRValidationError(f"{label} must be an array")
    return value

@dataclass(frozen=True)
class OperationSpec:
    gate: str
    qubits: tuple[int, ...]
    parameter: str
    scale: float = 1.0
    pauli: str | None = None

    def __post_init__(self) -> None:
        qubits = tuple(self.qubits)
        if not isinstance(self.gate, str) or not self.gate:
            raise AnsatzIRValidationError("operation.gate must be a non-empty string")
        if any(type(qubit) is not int for qubit in qubits) or len(qubits) != len(set(qubits)):
            raise AnsatzIRValidationError("operation qubits must be unique integers")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or not self.scale
        ):
            raise AnsatzIRValidationError("operation.scale must be finite and nonzero")
        if not isinstance(self.parameter, str) or not PARAMETER_NAME.fullmatch(self.parameter):
            raise AnsatzIRValidationError(
                "operation.parameter must match [A-Za-z_][A-Za-z0-9_.-]{0,127}"
            )
        if self.pauli is not None and not isinstance(self.pauli, str):
            raise AnsatzIRValidationError("operation.pauli must be a string")
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "scale", float(self.scale))

    def to_dict(self) -> dict[str, Any]:
        result = {"gate": self.gate, "qubits": list(self.qubits), "parameter": self.parameter}
        if self.scale != 1.0:
            result["scale"] = self.scale
        if self.pauli is not None:
            result["pauli"] = self.pauli
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperationSpec:
        value = _fields(
            value,
            {"gate", "qubits", "parameter", "scale", "pauli"},
            {"gate", "qubits", "parameter"},
            "operation",
        )
        return cls(
            value["gate"],
            tuple(_array(value["qubits"], "operation.qubits")),
            value["parameter"],
            value.get("scale", 1.0),
            value.get("pauli"),
        )

@dataclass(frozen=True)
class AnsatzSpec:
    num_qubits: int
    operations: tuple[OperationSpec, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if type(self.num_qubits) is not int or self.num_qubits <= 0:
            raise AnsatzIRValidationError("ansatz.num_qubits must be positive")
        if type(self.version) is not int or self.version != 1:
            raise AnsatzIRValidationError(f"unsupported ansatz IR version: {self.version}")
        if any(not isinstance(item, OperationSpec) for item in operations):
            raise AnsatzIRValidationError("ansatz.operations must contain OperationSpec objects")
        object.__setattr__(self, "operations", operations)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(operation.parameter for operation in self.operations))

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "num_qubits": self.num_qubits, "operations": [item.to_dict() for item in self.operations]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnsatzSpec:
        value = _fields(value, {"version", "num_qubits", "operations"}, {"num_qubits"}, "ansatz")
        operations = _array(value.get("operations", ()), "ansatz.operations")
        return cls(
            value["num_qubits"],
            tuple(OperationSpec.from_dict(item) for item in operations),
            value.get("version", 1),
        )

@dataclass(frozen=True)
class CompiledAnsatz:
    circuit: QuantumCircuit
    parameters: Mapping[str, Parameter]
    audit: dict[str, Any]

def _validate(operation: OperationSpec, num_qubits: int, path: str) -> None:
    if operation.gate not in ALLOWED_GATES:
        raise AnsatzCompilerError(f"{path}: unknown or untrusted gate {operation.gate!r}")
    if any(not 0 <= qubit < num_qubits for qubit in operation.qubits):
        raise AnsatzCompilerError(f"{path}: qubit is outside [0, {num_qubits})")
    if operation.scale not in ALLOWED_SCALES:
        raise AnsatzCompilerError(f"{path}/scale must be one of {sorted(ALLOWED_SCALES)}")
    if operation.gate == "PauliRotation":
        invalid_pauli = (
            not operation.pauli
            or set(operation.pauli) - set("XYZ")
            or len(operation.pauli) != len(operation.qubits)
        )
        if invalid_pauli:
            raise AnsatzCompilerError(
                f"{path}: PauliRotation requires one active pauli letter per qubit"
            )
    elif operation.pauli is not None or len(operation.qubits) != 2:
        raise AnsatzCompilerError(f"{path}: {operation.gate} requires two qubits and no pauli")

def operation_paulis(operation: OperationSpec) -> tuple[tuple[str, float], ...]:
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
        for qubit, letter in zip(qubits, operation.pauli, strict=True):
            if letter == "X":
                circuit.h(qubit)
            elif letter == "Y":
                circuit.sdg(qubit)
                circuit.h(qubit)
        for left, right in zip(qubits[:-1], qubits[1:]):
            circuit.cx(left, right)
        circuit.rz(2 * angle, qubits[-1])
        for left, right in reversed(tuple(zip(qubits[:-1], qubits[1:]))):
            circuit.cx(left, right)
        for qubit, letter in reversed(tuple(zip(qubits, operation.pauli, strict=True))):
            if letter == "X":
                circuit.h(qubit)
            elif letter == "Y":
                circuit.h(qubit)
                circuit.s(qubit)
    elif operation.gate == "XYExchange":
        circuit.append(XXPlusYYGate(4 * angle, 0), list(qubits))
    else:
        circuit.rxx(2 * angle, *qubits)
        circuit.ryy(2 * angle, *qubits)
        circuit.rzz(2 * angle, *qubits)

def compile_ansatz(value: AnsatzSpec | Mapping[str, Any]) -> CompiledAnsatz:
    if not isinstance(value, AnsatzSpec):
        if not isinstance(value, Mapping):
            raise AnsatzCompilerError("compiler input must be an AnsatzSpec or dictionary")
        value = AnsatzSpec.from_dict(value)
    occurrences: Counter[str] = Counter()
    for index, operation in enumerate(value.operations):
        _validate(operation, value.num_qubits, f"/operations/{index}")
        occurrences[operation.parameter] += 1
    names = value.parameter_names
    parameters = {name: Parameter(name) for name in names}
    circuit = QuantumCircuit(value.num_qubits, name="ansatz")
    for operation in value.operations:
        _emit(circuit, operation, operation.scale * parameters[operation.parameter])
    audit = {"unique_trainable_params": len(names), "trainable_parameter_names": list(names), "parameter_occurrences": {name: occurrences[name] for name in names}, "operations": len(value.operations)}
    return CompiledAnsatz(circuit, parameters, audit)
