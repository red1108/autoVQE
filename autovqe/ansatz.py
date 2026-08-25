"""Lean, closed compiler for AutoVQE ansatz specifications.

Candidates provide JSON data, never Qiskit objects or gate implementations.  The
only accepted operations are the three gates below, and every gate angle is a
zero-offset multiple of exactly one declared parameter.  This keeps parameter
sharing while making the circuit and its audit entirely evaluator-derived.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


ALLOWED_GATES = frozenset({"PauliRotation", "XYExchange", "IsotropicExchange"})
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


def _freeze_json(value: Any, context: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite(value, context)
    if isinstance(value, Mapping):
        checked = _object(value, context)
        return MappingProxyType(
            {key: _freeze_json(child, f"{context}.{key}") for key, child in checked.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(child, f"{context}[]") for child in value)
    raise AnsatzIRValidationError(f"{context} must contain JSON values only")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True)
class ParameterRef:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PARAMETER_NAME.fullmatch(self.name):
            raise AnsatzIRValidationError(
                "parameter name must match [A-Za-z_][A-Za-z0-9_.-]{0,127}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterRef":
        payload = _object(value, "parameter")
        _strict(payload, {"name"}, "parameter", required={"name"})
        return cls(payload["name"])


@dataclass(frozen=True)
class ParameterTerm:
    parameter: ParameterRef
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, ParameterRef):
            raise AnsatzIRValidationError("term.parameter must be a ParameterRef")
        coefficient = _finite(self.coefficient, "term.coefficient")
        if coefficient == 0.0:
            raise AnsatzIRValidationError("term.coefficient cannot be zero")
        object.__setattr__(self, "coefficient", coefficient)

    def to_dict(self) -> dict[str, Any]:
        return {"parameter": self.parameter.name, "coefficient": self.coefficient}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterTerm":
        payload = _object(value, "expression term")
        _strict(
            payload,
            {"parameter", "coefficient"},
            "expression term",
            required={"parameter"},
        )
        name = payload["parameter"]
        if not isinstance(name, str):
            raise AnsatzIRValidationError("expression term.parameter must be a string")
        return cls(ParameterRef(name), payload.get("coefficient", 1.0))


@dataclass(frozen=True)
class ParameterExpression:
    """Serializable affine expression; compilation accepts one zero-offset term."""

    terms: tuple[ParameterTerm, ...] = ()
    constant: float = 0.0

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not all(isinstance(term, ParameterTerm) for term in terms):
            raise AnsatzIRValidationError("expression.terms must contain ParameterTerm objects")
        names = [term.parameter.name for term in terms]
        if len(names) != len(set(names)):
            raise AnsatzIRValidationError("an expression cannot repeat a parameter")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "constant", _finite(self.constant, "expression.constant"))

    @classmethod
    def parameter(
        cls, parameter: str | ParameterRef, coefficient: float = 1.0
    ) -> "ParameterExpression":
        reference = parameter if isinstance(parameter, ParameterRef) else ParameterRef(parameter)
        return cls((ParameterTerm(reference, coefficient),), 0.0)

    @classmethod
    def literal(cls, value: float) -> "ParameterExpression":
        return cls((), value)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(term.parameter.name for term in self.terms)

    @property
    def is_trainable(self) -> bool:
        return bool(self.terms)

    @property
    def is_zero_at_origin(self) -> bool:
        return self.constant == 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [term.to_dict() for term in self.terms],
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterExpression":
        payload = _object(value, "parameter expression")
        if "parameter" in payload:
            _strict(payload, {"parameter", "coefficient"}, "parameter expression")
            name = payload["parameter"]
            if not isinstance(name, str):
                raise AnsatzIRValidationError(
                    "parameter expression.parameter must be a string"
                )
            return cls.parameter(name, payload.get("coefficient", 1.0))
        if "literal" in payload:
            _strict(payload, {"literal"}, "parameter expression", required={"literal"})
            return cls.literal(payload["literal"])
        _strict(payload, {"terms", "constant"}, "parameter expression")
        terms = _array(payload.get("terms", ()), "parameter expression.terms")
        return cls(
            tuple(ParameterTerm.from_dict(term) for term in terms),
            payload.get("constant", 0.0),
        )


AngleExpression = ParameterExpression


@dataclass(frozen=True)
class OperationSpec:
    macro: str
    qubits: tuple[int, ...]
    parameters: Mapping[str, ParameterExpression]
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.macro, str) or not self.macro:
            raise AnsatzIRValidationError("operation.macro must be a non-empty string")
        qubits = tuple(self.qubits)
        for index, qubit in enumerate(qubits):
            _integer(qubit, f"operation.qubits[{index}]")
        if len(qubits) != len(set(qubits)):
            raise AnsatzIRValidationError("operation qubits must be unique")
        parameters = _object(self.parameters, "operation.parameters")
        if not all(
            name and isinstance(expression, ParameterExpression)
            for name, expression in parameters.items()
        ):
            raise AnsatzIRValidationError(
                "operation parameters must map non-empty names to ParameterExpression objects"
            )
        options = _object(self.options, "operation.options")
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "parameters", MappingProxyType(dict(parameters)))
        object.__setattr__(
            self,
            "options",
            MappingProxyType(
                {key: _freeze_json(child, f"operation.options.{key}") for key, child in options.items()}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro": self.macro,
            "qubits": list(self.qubits),
            "parameters": {
                name: expression.to_dict() for name, expression in self.parameters.items()
            },
            "options": _thaw_json(self.options),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationSpec":
        payload = _object(value, "operation")
        _strict(
            payload,
            {"macro", "qubits", "parameters", "options"},
            "operation",
            required={"macro"},
        )
        parameters = _object(payload.get("parameters", {}), "operation.parameters")
        return cls(
            macro=payload["macro"],
            qubits=tuple(_array(payload.get("qubits", ()), "operation.qubits")),
            parameters={
                name: ParameterExpression.from_dict(expression)
                for name, expression in parameters.items()
            },
            options=dict(_object(payload.get("options", {}), "operation.options")),
        )


@dataclass(frozen=True)
class AnsatzSpec:
    num_qubits: int
    parameters: tuple[ParameterRef, ...] = ()
    operations: tuple[OperationSpec, ...] = ()
    name: str = "ansatz"
    version: int = 1

    def __post_init__(self) -> None:
        num_qubits = _integer(self.num_qubits, "ansatz.num_qubits")
        if num_qubits <= 0:
            raise AnsatzIRValidationError("ansatz.num_qubits must be positive")
        if not isinstance(self.name, str) or not self.name:
            raise AnsatzIRValidationError("ansatz.name must be a non-empty string")
        version = _integer(self.version, "ansatz.version")
        if version != 1:
            raise AnsatzIRValidationError(f"unsupported ansatz IR version: {version}")
        parameters = tuple(
            ParameterRef(item) if isinstance(item, str) else item for item in self.parameters
        )
        if not all(isinstance(item, ParameterRef) for item in parameters):
            raise AnsatzIRValidationError("ansatz.parameters must contain ParameterRef objects")
        names = [item.name for item in parameters]
        if len(names) != len(set(names)):
            raise AnsatzIRValidationError("declared parameter names must be unique")
        operations = tuple(self.operations)
        if not all(isinstance(item, OperationSpec) for item in operations):
            raise AnsatzIRValidationError("ansatz.operations must contain OperationSpec objects")
        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "version", version)

    def iter_operations(self) -> Iterator[OperationSpec]:
        yield from self.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "num_qubits": self.num_qubits,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnsatzSpec":
        payload = _object(value, "ansatz")
        _strict(
            payload,
            {"version", "name", "num_qubits", "parameters", "operations"},
            "ansatz",
            required={"num_qubits"},
        )
        parameters = []
        for item in _array(payload.get("parameters", ()), "ansatz.parameters"):
            parameters.append(ParameterRef(item) if isinstance(item, str) else ParameterRef.from_dict(item))
        operations = _array(payload.get("operations", ()), "ansatz.operations")
        return cls(
            num_qubits=payload["num_qubits"],
            parameters=tuple(parameters),
            operations=tuple(OperationSpec.from_dict(item) for item in operations),
            name=payload.get("name", "ansatz"),
            version=payload.get("version", 1),
        )


@dataclass(frozen=True)
class CompiledAnsatz:
    """A trusted circuit, its Qiskit parameters, and a compiler-derived dict audit."""

    circuit: QuantumCircuit
    parameters: Mapping[str, Parameter]
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("compiled circuit must be a QuantumCircuit")
        if not isinstance(self.parameters, Mapping) or not all(
            isinstance(name, str) and isinstance(parameter, Parameter)
            for name, parameter in self.parameters.items()
        ):
            raise TypeError("compiled parameters must map names to Qiskit Parameters")
        if not isinstance(self.audit, dict):
            raise TypeError("compiled audit must be a dict")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "audit", dict(self.audit))


def _coerce_spec(value: AnsatzSpec | Mapping[str, Any]) -> AnsatzSpec:
    if isinstance(value, AnsatzSpec):
        return value
    if isinstance(value, Mapping):
        return AnsatzSpec.from_dict(value)
    raise AnsatzCompilerError(
        "compiler input must be an AnsatzSpec or its serialized dictionary"
    )


def _validate_operation(
    operation: OperationSpec,
    *,
    declared: set[str],
    num_qubits: int,
    path: str,
) -> ParameterTerm:
    if operation.macro not in ALLOWED_GATES:
        raise AnsatzCompilerError(f"{path}: unknown or untrusted gate {operation.macro!r}")
    for index, qubit in enumerate(operation.qubits):
        if not 0 <= qubit < num_qubits:
            raise AnsatzCompilerError(
                f"{path}/qubits/{index}={qubit} is outside [0, {num_qubits})"
            )
    if set(operation.parameters) != {"angle"}:
        raise AnsatzCompilerError(f"{path}: gate parameters must be ['angle']")
    if operation.macro == "PauliRotation":
        if set(operation.options) != {"pauli"}:
            raise AnsatzCompilerError(f"{path}: PauliRotation options must be ['pauli']")
        pauli = operation.options["pauli"]
        if not isinstance(pauli, str) or not pauli or any(letter not in "XYZ" for letter in pauli):
            raise AnsatzCompilerError(f"{path}: pauli must contain active X, Y, or Z only")
        if len(operation.qubits) != len(pauli):
            raise AnsatzCompilerError(f"{path}: pauli length must equal qubit locality")
    else:
        if operation.options:
            raise AnsatzCompilerError(f"{path}: {operation.macro} takes no options")
        if len(operation.qubits) != 2:
            raise AnsatzCompilerError(f"{path}: {operation.macro} requires two qubits")
    angle = operation.parameters["angle"]
    if len(angle.terms) != 1:
        raise AnsatzCompilerError(f"{path}/parameters/angle must contain exactly one term")
    if angle.constant != 0.0:
        raise AnsatzCompilerError(f"{path}/parameters/angle must have zero constant")
    term = angle.terms[0]
    if term.parameter.name not in declared:
        raise AnsatzCompilerError(
            f"{path}/parameters/angle references undeclared parameter {term.parameter.name!r}"
        )
    return term


def _emit(circuit: QuantumCircuit, operation: OperationSpec, angle: Any) -> None:
    qubits = operation.qubits
    if operation.macro == "PauliRotation":
        pauli = operation.options["pauli"]
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
    elif operation.macro == "XYExchange":
        circuit.append(XXPlusYYGate(4.0 * angle, 0.0), list(qubits))
    else:
        left, right = qubits
        circuit.rxx(2.0 * angle, left, right)
        circuit.ryy(2.0 * angle, left, right)
        circuit.rzz(2.0 * angle, left, right)


def compile_ansatz(value: AnsatzSpec | Mapping[str, Any]) -> CompiledAnsatz:
    """Validate and compile v1 JSON using the closed three-gate implementation."""

    spec = _coerce_spec(value)
    declared_names = tuple(parameter.name for parameter in spec.parameters)
    declared = set(declared_names)
    occurrences: Counter[str] = Counter()
    macros: Counter[str] = Counter()
    terms: list[ParameterTerm] = []
    for index, operation in enumerate(spec.operations):
        term = _validate_operation(
            operation,
            declared=declared,
            num_qubits=spec.num_qubits,
            path=f"/operations/{index}",
        )
        terms.append(term)
        occurrences[term.parameter.name] += 1
        macros[operation.macro] += 1
    unused = [name for name in declared_names if occurrences[name] == 0]
    if unused:
        raise AnsatzCompilerError(f"declared parameters must be used: {unused}")

    qiskit_parameters = {name: Parameter(name) for name in declared_names}
    circuit = QuantumCircuit(spec.num_qubits, name=spec.name)
    for operation, term in zip(spec.operations, terms, strict=True):
        _emit(circuit, operation, term.coefficient * qiskit_parameters[term.parameter.name])

    fixed_literals = [
        {
            "path": f"/operations/{index}/parameters/angle/terms/0/coefficient",
            "value": term.coefficient,
            "role": "scale",
        }
        for index, term in enumerate(terms)
        if term.coefficient != 1.0
    ]
    gate_counts = {name: int(count) for name, count in circuit.count_ops().items()}
    audit = {
        "unique_trainable_params": len(declared_names),
        "trainable_parameter_names": list(declared_names),
        "parameter_occurrences": {
            name: occurrences[name] for name in declared_names
        },
        "unused_parameters": [],
        "operations": len(spec.operations),
        "logical_macros": dict(macros),
        "spec_nodes": 1 + len(declared_names) + 3 * len(spec.operations),
        "fixed_literals": fixed_literals,
        "gates": sum(gate_counts.values()),
        "gate_counts": gate_counts,
    }
    return CompiledAnsatz(circuit, qiskit_parameters, audit)


def validate_ansatz(value: AnsatzSpec | Mapping[str, Any]) -> dict[str, Any]:
    """Return the same evaluator-owned audit used by compilation."""

    return compile_ansatz(value).audit


def parameter_expression(
    parameter: str | ParameterRef, coefficient: float = 1.0
) -> ParameterExpression:
    return ParameterExpression.parameter(parameter, coefficient)


__all__ = [
    "ALLOWED_GATES",
    "AngleExpression",
    "AnsatzCompilerError",
    "AnsatzIRValidationError",
    "AnsatzSpec",
    "CompiledAnsatz",
    "OperationSpec",
    "ParameterExpression",
    "ParameterRef",
    "ParameterTerm",
    "compile_ansatz",
    "parameter_expression",
    "validate_ansatz",
]
