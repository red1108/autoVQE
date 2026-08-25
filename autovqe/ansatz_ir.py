"""Serializable, typed intermediate representation for AutoVQE ansatzes.

The IR deliberately contains no Qiskit instructions, callables, matrices, or
opaque payloads.  It describes a circuit only through names that a trusted
compiler can resolve and affine expressions over declared trainable
parameters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence


_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")


class AnsatzIRValidationError(ValueError):
    """Raised when an ansatz IR object is malformed or unsafe."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnsatzIRValidationError(f"{context} must be an object")
    return value


def _strict_keys(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise AnsatzIRValidationError(f"{context} contains unsupported fields: {rendered}")


def _finite_number(value: Any, context: str) -> float:
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


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnsatzIRValidationError(f"{context} must be an array")
    return value


def _freeze_json(value: Any, context: str) -> Any:
    """Validate a JSON-compatible structural value and make it immutable."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_number(value, context)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise AnsatzIRValidationError(f"{context} keys must be strings")
            frozen[key] = _freeze_json(child, f"{context}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(child, f"{context}[]") for child in value)
    raise AnsatzIRValidationError(
        f"{context} must contain JSON values only; opaque Python objects are forbidden"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True)
class ParameterRef:
    """A reference to one declared trainable scalar parameter."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PARAMETER_NAME.fullmatch(self.name):
            raise AnsatzIRValidationError(
                "parameter name must match [A-Za-z_][A-Za-z0-9_.-]{0,127}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterRef":
        payload = _mapping(payload, "parameter")
        _strict_keys(payload, {"name"}, "parameter")
        if "name" not in payload:
            raise AnsatzIRValidationError("parameter.name is required")
        return cls(name=payload["name"])


@dataclass(frozen=True)
class ParameterTerm:
    """One scaled trainable parameter inside an affine expression."""

    parameter: ParameterRef
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, ParameterRef):
            raise AnsatzIRValidationError("term.parameter must be a ParameterRef")
        coefficient = _finite_number(self.coefficient, "term.coefficient")
        if coefficient == 0.0:
            raise AnsatzIRValidationError("term.coefficient cannot be zero")
        object.__setattr__(self, "coefficient", coefficient)

    def to_dict(self) -> dict[str, Any]:
        return {"parameter": self.parameter.name, "coefficient": self.coefficient}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterTerm":
        payload = _mapping(payload, "expression term")
        _strict_keys(payload, {"parameter", "coefficient"}, "expression term")
        if "parameter" not in payload:
            raise AnsatzIRValidationError("expression term.parameter is required")
        name = payload["parameter"]
        if not isinstance(name, str):
            raise AnsatzIRValidationError("expression term.parameter must be a string")
        return cls(
            parameter=ParameterRef(name),
            coefficient=payload.get("coefficient", 1.0),
        )


@dataclass(frozen=True)
class ParameterExpression:
    """A safe affine expression over declared trainable parameters.

    General symbolic expressions are intentionally excluded.  Affine
    expressions cover parameter sharing and Hamiltonian-coefficient scaling
    while remaining straightforward for an evaluator to audit.
    """

    terms: tuple[ParameterTerm, ...] = ()
    constant: float = 0.0

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not all(isinstance(term, ParameterTerm) for term in terms):
            raise AnsatzIRValidationError("expression.terms must contain ParameterTerm objects")
        names = [term.parameter.name for term in terms]
        if len(names) != len(set(names)):
            raise AnsatzIRValidationError("an expression cannot reference one parameter twice")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "constant", _finite_number(self.constant, "expression.constant"))

    @classmethod
    def parameter(
        cls,
        parameter: str | ParameterRef,
        coefficient: float = 1.0,
    ) -> "ParameterExpression":
        reference = parameter if isinstance(parameter, ParameterRef) else ParameterRef(parameter)
        return cls(terms=(ParameterTerm(reference, coefficient),))

    @classmethod
    def literal(cls, value: float) -> "ParameterExpression":
        return cls(constant=value)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(term.parameter.name for term in self.terms)

    @property
    def is_trainable(self) -> bool:
        return bool(self.terms)

    @property
    def is_zero_at_origin(self) -> bool:
        return self.constant == 0.0

    def scaled(self, coefficient: float) -> "ParameterExpression":
        coefficient = _finite_number(coefficient, "expression scale")
        if coefficient == 0.0:
            return ParameterExpression.literal(0.0)
        return ParameterExpression(
            terms=tuple(
                ParameterTerm(term.parameter, term.coefficient * coefficient)
                for term in self.terms
            ),
            constant=self.constant * coefficient,
        )

    def plus(self, other: "ParameterExpression | float") -> "ParameterExpression":
        right = other if isinstance(other, ParameterExpression) else ParameterExpression.literal(other)
        coefficients: dict[str, float] = {}
        references: dict[str, ParameterRef] = {}
        order: list[str] = []
        for term in self.terms + right.terms:
            name = term.parameter.name
            if name not in coefficients:
                coefficients[name] = 0.0
                references[name] = term.parameter
                order.append(name)
            coefficients[name] += term.coefficient
        terms = tuple(
            ParameterTerm(references[name], coefficients[name])
            for name in order
            if coefficients[name] != 0.0
        )
        return ParameterExpression(terms=terms, constant=self.constant + right.constant)

    def __mul__(self, coefficient: float) -> "ParameterExpression":
        return self.scaled(coefficient)

    def __rmul__(self, coefficient: float) -> "ParameterExpression":
        return self.scaled(coefficient)

    def __add__(self, other: "ParameterExpression | float") -> "ParameterExpression":
        return self.plus(other)

    def __radd__(self, other: "ParameterExpression | float") -> "ParameterExpression":
        return self.plus(other)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [term.to_dict() for term in self.terms],
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ParameterExpression":
        payload = _mapping(payload, "parameter expression")
        keys = set(payload)
        if keys <= {"parameter", "coefficient"} and "parameter" in payload:
            name = payload["parameter"]
            if not isinstance(name, str):
                raise AnsatzIRValidationError("parameter expression.parameter must be a string")
            return cls.parameter(name, payload.get("coefficient", 1.0))
        if keys == {"literal"}:
            return cls.literal(payload["literal"])

        _strict_keys(payload, {"terms", "constant"}, "parameter expression")
        terms_payload = _sequence(payload.get("terms", ()), "parameter expression.terms")
        return cls(
            terms=tuple(ParameterTerm.from_dict(item) for item in terms_payload),
            constant=payload.get("constant", 0.0),
        )


# Short alias for callers that prefer gate-language terminology.
AngleExpression = ParameterExpression


@dataclass(frozen=True)
class ReferenceSpec:
    """Trusted reference preparation applied before all variational layers.

    Entries in ``qubits`` are Qiskit qubit indices, not Pauli-label positions.
    """

    macro: str = "X"
    qubits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.macro, str) or not self.macro:
            raise AnsatzIRValidationError("reference.macro must be a non-empty string")
        qubits = tuple(self.qubits)
        for index, qubit in enumerate(qubits):
            _integer(qubit, f"reference.qubits[{index}]")
        if len(qubits) != len(set(qubits)):
            raise AnsatzIRValidationError("reference qubits must be unique")
        object.__setattr__(self, "qubits", qubits)

    def to_dict(self) -> dict[str, Any]:
        return {"macro": self.macro, "qubits": list(self.qubits)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceSpec":
        payload = _mapping(payload, "reference")
        _strict_keys(payload, {"macro", "qubits"}, "reference")
        qubits = _sequence(payload.get("qubits", ()), "reference.qubits")
        return cls(macro=payload.get("macro", "X"), qubits=tuple(qubits))


@dataclass(frozen=True)
class OperationSpec:
    """One invocation of a trusted logical macro.

    For ``PauliRotation``, the local Pauli word is paired left-to-right with
    the listed ``qubits``.  Thus ``qubits=(0, 1), pauli='XY'`` means X on q0
    and Y on q1; its full-width Qiskit label is ``YX``.
    """

    macro: str
    qubits: tuple[int, ...]
    parameters: Mapping[str, ParameterExpression]
    options: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.macro, str) or not self.macro:
            raise AnsatzIRValidationError("operation.macro must be a non-empty string")
        qubits = tuple(self.qubits)
        for index, qubit in enumerate(qubits):
            _integer(qubit, f"operation.qubits[{index}]")
        if len(qubits) != len(set(qubits)):
            raise AnsatzIRValidationError("operation qubits must be unique")

        if not isinstance(self.parameters, Mapping):
            raise AnsatzIRValidationError("operation.parameters must be an object")
        parameters: dict[str, ParameterExpression] = {}
        for name, expression in self.parameters.items():
            if not isinstance(name, str) or not name:
                raise AnsatzIRValidationError("operation parameter names must be non-empty strings")
            if not isinstance(expression, ParameterExpression):
                raise AnsatzIRValidationError(
                    f"operation parameter {name!r} must be a ParameterExpression"
                )
            parameters[name] = expression

        if not isinstance(self.options, Mapping):
            raise AnsatzIRValidationError("operation.options must be an object")
        options = {
            key: _freeze_json(value, f"operation.options.{key}")
            for key, value in self.options.items()
            if isinstance(key, str)
        }
        if len(options) != len(self.options):
            raise AnsatzIRValidationError("operation option names must be strings")

        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "options", MappingProxyType(options))

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationSpec":
        payload = _mapping(payload, "operation")
        _strict_keys(payload, {"macro", "qubits", "parameters", "options"}, "operation")
        if "macro" not in payload:
            raise AnsatzIRValidationError("operation.macro is required")
        qubits = _sequence(payload.get("qubits", ()), "operation.qubits")
        parameters_payload = _mapping(payload.get("parameters", {}), "operation.parameters")
        options_payload = _mapping(payload.get("options", {}), "operation.options")
        return cls(
            macro=payload["macro"],
            qubits=tuple(qubits),
            parameters={
                name: ParameterExpression.from_dict(expression)
                for name, expression in parameters_payload.items()
            },
            options=dict(options_payload),
        )


@dataclass(frozen=True)
class LayerSpec:
    """An ordered group of logical operations."""

    operations: tuple[OperationSpec, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if not all(isinstance(operation, OperationSpec) for operation in operations):
            raise AnsatzIRValidationError("layer.operations must contain OperationSpec objects")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise AnsatzIRValidationError("layer.name must be a non-empty string when provided")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operations": [operation.to_dict() for operation in self.operations]
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayerSpec":
        payload = _mapping(payload, "layer")
        _strict_keys(payload, {"name", "operations"}, "layer")
        operations = _sequence(payload.get("operations", ()), "layer.operations")
        return cls(
            operations=tuple(OperationSpec.from_dict(item) for item in operations),
            name=payload.get("name"),
        )


@dataclass(frozen=True)
class AnsatzSpec:
    """Complete, serializable logical ansatz specification."""

    num_qubits: int
    parameters: tuple[ParameterRef, ...] = ()
    reference: ReferenceSpec | None = None
    layers: tuple[LayerSpec, ...] = ()
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
            ParameterRef(parameter) if isinstance(parameter, str) else parameter
            for parameter in self.parameters
        )
        if not all(isinstance(parameter, ParameterRef) for parameter in parameters):
            raise AnsatzIRValidationError("ansatz.parameters must contain ParameterRef objects")
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)):
            raise AnsatzIRValidationError("declared parameter names must be unique")

        if self.reference is not None and not isinstance(self.reference, ReferenceSpec):
            raise AnsatzIRValidationError("ansatz.reference must be a ReferenceSpec or None")
        layers = tuple(self.layers)
        if not all(isinstance(layer, LayerSpec) for layer in layers):
            raise AnsatzIRValidationError("ansatz.layers must contain LayerSpec objects")

        object.__setattr__(self, "num_qubits", num_qubits)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "version", version)

    @property
    def operations(self) -> tuple[OperationSpec, ...]:
        return tuple(self.iter_operations())

    def iter_operations(self) -> Iterator[OperationSpec]:
        for layer in self.layers:
            yield from layer.operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "num_qubits": self.num_qubits,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "reference": None if self.reference is None else self.reference.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnsatzSpec":
        payload = _mapping(payload, "ansatz")
        _strict_keys(
            payload,
            {"version", "name", "num_qubits", "parameters", "reference", "layers", "operations"},
            "ansatz",
        )
        if "num_qubits" not in payload:
            raise AnsatzIRValidationError("ansatz.num_qubits is required")
        if "layers" in payload and "operations" in payload:
            raise AnsatzIRValidationError("ansatz must use either layers or operations, not both")

        parameters_payload = _sequence(payload.get("parameters", ()), "ansatz.parameters")
        parameters: list[ParameterRef] = []
        for item in parameters_payload:
            if isinstance(item, str):
                parameters.append(ParameterRef(item))
            else:
                parameters.append(ParameterRef.from_dict(item))

        if "operations" in payload:
            operations_payload = _sequence(payload["operations"], "ansatz.operations")
            operations = tuple(OperationSpec.from_dict(item) for item in operations_payload)
            layers = (LayerSpec(operations=operations),) if operations else ()
        else:
            layers_payload = _sequence(payload.get("layers", ()), "ansatz.layers")
            layers = tuple(LayerSpec.from_dict(item) for item in layers_payload)

        reference_payload = payload.get("reference")
        reference = (
            None if reference_payload is None else ReferenceSpec.from_dict(reference_payload)
        )
        return cls(
            num_qubits=payload["num_qubits"],
            parameters=tuple(parameters),
            reference=reference,
            layers=layers,
            name=payload.get("name", "ansatz"),
            version=payload.get("version", 1),
        )


def parameter_expression(
    parameter: str | ParameterRef,
    coefficient: float = 1.0,
) -> ParameterExpression:
    """Convenience constructor for a one-parameter affine expression."""

    return ParameterExpression.parameter(parameter, coefficient)


def layer(operations: Iterable[OperationSpec], name: str | None = None) -> LayerSpec:
    """Convenience constructor that freezes an operation iterable."""

    return LayerSpec(tuple(operations), name=name)
