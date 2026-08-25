"""Safe compiler and evaluator-derived audit for typed AutoVQE ansatz IR."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from qiskit.circuit import Parameter, QuantumCircuit

from .ansatz_ir import (
    AnsatzIRValidationError,
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
)
from .macros import (
    MacroValidationError,
    get_trusted_macro,
    trusted_macro_zero_is_identity,
)


class AnsatzCompilerError(AnsatzIRValidationError):
    """Raised when a typed ansatz fails compiler-owned safety checks."""


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnsatzCompilerError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnsatzCompilerError(f"{context} must be a finite number")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnsatzCompilerError(f"{context} must be an integer")
    return value


def _string_sequence(value: Any, context: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnsatzCompilerError(f"{context} must be an array of strings")
    result = tuple(value)
    if not all(isinstance(item, str) for item in result):
        raise AnsatzCompilerError(f"{context} must be an array of strings")
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnsatzCompilerError(f"{context} must be an object")
    return value


@dataclass(frozen=True)
class FixedLiteral:
    """One candidate-supplied numeric literal found in an angle expression."""

    path: str
    value: float
    role: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise AnsatzCompilerError("fixed literal path must be a non-empty string")
        if self.role not in {"scale", "offset", "option"}:
            raise AnsatzCompilerError(f"unsupported fixed literal role: {self.role!r}")
        object.__setattr__(self, "value", _finite_number(self.value, "fixed literal value"))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "role": self.role}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixedLiteral":
        payload = _mapping(payload, "fixed literal")
        unknown = set(payload) - {"path", "value", "role"}
        if unknown:
            raise AnsatzCompilerError(
                f"fixed literal contains unsupported fields: {sorted(unknown)}"
            )
        try:
            return cls(
                path=payload["path"],
                value=payload["value"],
                role=payload["role"],
            )
        except KeyError as exc:
            raise AnsatzCompilerError(f"fixed literal missing field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class AnsatzAudit:
    """Parameter and resource audit derived by the compiler."""

    unique_trainable_params: int
    trainable_parameter_names: tuple[str, ...]
    parameter_occurrences: Mapping[str, int]
    unused_parameters: tuple[str, ...]
    spec_nodes: int
    spec_node_counts: Mapping[str, int]
    fixed_literals: tuple[FixedLiteral, ...]
    logical_macros: Mapping[str, int]
    operations: int

    def __post_init__(self) -> None:
        unique = _integer(self.unique_trainable_params, "audit.unique_trainable_params")
        if unique < 0:
            raise AnsatzCompilerError("audit.unique_trainable_params cannot be negative")
        names = tuple(self.trainable_parameter_names)
        if not all(isinstance(name, str) for name in names):
            raise AnsatzCompilerError("audit parameter names must be strings")
        if unique != len(names) or len(names) != len(set(names)):
            raise AnsatzCompilerError(
                "audit unique_trainable_params must match distinct parameter names"
            )

        occurrences = self._validated_counts(
            self.parameter_occurrences, "audit.parameter_occurrences"
        )
        if set(occurrences) != set(names):
            raise AnsatzCompilerError(
                "audit.parameter_occurrences keys must match trainable parameter names"
            )
        node_counts = self._validated_counts(self.spec_node_counts, "audit.spec_node_counts")
        macros = self._validated_counts(self.logical_macros, "audit.logical_macros")
        unused = tuple(self.unused_parameters)
        if not all(isinstance(name, str) for name in unused):
            raise AnsatzCompilerError("audit.unused_parameters must contain strings")
        literals = tuple(self.fixed_literals)
        if not all(isinstance(literal, FixedLiteral) for literal in literals):
            raise AnsatzCompilerError("audit.fixed_literals must contain FixedLiteral objects")
        spec_nodes = _integer(self.spec_nodes, "audit.spec_nodes")
        if spec_nodes < 0 or spec_nodes != sum(node_counts.values()):
            raise AnsatzCompilerError("audit.spec_nodes must equal the node-count sum")
        operations = _integer(self.operations, "audit.operations")
        if operations < 0:
            raise AnsatzCompilerError("audit operation count cannot be negative")

        object.__setattr__(self, "unique_trainable_params", unique)
        object.__setattr__(self, "trainable_parameter_names", names)
        object.__setattr__(self, "parameter_occurrences", MappingProxyType(occurrences))
        object.__setattr__(self, "unused_parameters", unused)
        object.__setattr__(self, "spec_nodes", spec_nodes)
        object.__setattr__(self, "spec_node_counts", MappingProxyType(node_counts))
        object.__setattr__(self, "fixed_literals", literals)
        object.__setattr__(self, "logical_macros", MappingProxyType(macros))
        object.__setattr__(self, "operations", operations)

    @staticmethod
    def _validated_counts(value: Any, context: str) -> dict[str, int]:
        payload = _mapping(value, context)
        result: dict[str, int] = {}
        for name, count in payload.items():
            if not isinstance(name, str):
                raise AnsatzCompilerError(f"{context} keys must be strings")
            checked = _integer(count, f"{context}.{name}")
            if checked < 0:
                raise AnsatzCompilerError(f"{context}.{name} cannot be negative")
            result[name] = checked
        return result

    @property
    def fixed_literal_count(self) -> int:
        return len(self.fixed_literals)

    @property
    def logical_macro_count(self) -> int:
        return sum(self.logical_macros.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_trainable_params": self.unique_trainable_params,
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "parameter_occurrences": dict(self.parameter_occurrences),
            "unused_parameters": list(self.unused_parameters),
            "spec_nodes": self.spec_nodes,
            "spec_node_counts": dict(self.spec_node_counts),
            "fixed_literals": [literal.to_dict() for literal in self.fixed_literals],
            "logical_macros": dict(self.logical_macros),
            "operations": self.operations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnsatzAudit":
        payload = _mapping(payload, "ansatz audit")
        allowed = {
            "unique_trainable_params",
            "trainable_parameter_names",
            "parameter_occurrences",
            "unused_parameters",
            "spec_nodes",
            "spec_node_counts",
            "fixed_literals",
            "logical_macros",
            "operations",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise AnsatzCompilerError(
                f"ansatz audit contains unsupported fields: {sorted(unknown)}"
            )
        missing = allowed - set(payload)
        if missing:
            raise AnsatzCompilerError(f"ansatz audit missing fields: {sorted(missing)}")
        literals_payload = payload["fixed_literals"]
        if isinstance(literals_payload, (str, bytes)) or not isinstance(
            literals_payload, Sequence
        ):
            raise AnsatzCompilerError("audit.fixed_literals must be an array")
        return cls(
            unique_trainable_params=payload["unique_trainable_params"],
            trainable_parameter_names=_string_sequence(
                payload["trainable_parameter_names"], "audit.trainable_parameter_names"
            ),
            parameter_occurrences=payload["parameter_occurrences"],
            unused_parameters=_string_sequence(
                payload["unused_parameters"], "audit.unused_parameters"
            ),
            spec_nodes=payload["spec_nodes"],
            spec_node_counts=payload["spec_node_counts"],
            fixed_literals=tuple(FixedLiteral.from_dict(item) for item in literals_payload),
            logical_macros=payload["logical_macros"],
            operations=payload["operations"],
        )


@dataclass(frozen=True)
class CompiledAnsatz:
    """Compiled Qiskit circuit plus evaluator-owned parameter map and audit."""

    circuit: QuantumCircuit
    parameters: Mapping[str, Parameter]
    audit: AnsatzAudit

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("compiled circuit must be a QuantumCircuit")
        if not isinstance(self.parameters, Mapping) or not all(
            isinstance(name, str) and isinstance(parameter, Parameter)
            for name, parameter in self.parameters.items()
        ):
            raise TypeError("compiled parameters must map names to Qiskit Parameters")
        if not isinstance(self.audit, AnsatzAudit):
            raise TypeError("compiled audit must be an AnsatzAudit")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _coerce_spec(spec: AnsatzSpec | Mapping[str, Any]) -> AnsatzSpec:
    if isinstance(spec, AnsatzSpec):
        return spec
    if isinstance(spec, Mapping):
        return AnsatzSpec.from_dict(spec)
    raise AnsatzCompilerError(
        "compiler input must be an AnsatzSpec or its serialized dictionary; "
        "opaque circuits and custom objects are forbidden"
    )


def _check_qubits(qubits: tuple[int, ...], num_qubits: int, path: str) -> None:
    if len(qubits) != len(set(qubits)):
        raise AnsatzCompilerError(f"{path} contains duplicate qubits")
    for index, qubit in enumerate(qubits):
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise AnsatzCompilerError(f"{path}[{index}] must be an integer")
        if not 0 <= qubit < num_qubits:
            raise AnsatzCompilerError(
                f"{path}[{index}]={qubit} is outside [0, {num_qubits})"
            )


def _numeric_option_literals(value: Any, path: str) -> list[FixedLiteral]:
    literals: list[FixedLiteral] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return literals
    if isinstance(value, (int, float)):
        literals.append(FixedLiteral(path=path, value=float(value), role="option"))
        return literals
    if isinstance(value, Mapping):
        for key, child in value.items():
            literals.extend(_numeric_option_literals(child, f"{path}/{key}"))
        return literals
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            literals.extend(_numeric_option_literals(child, f"{path}/{index}"))
        return literals
    # OperationSpec already rejects non-JSON values.  Keep this defensive error
    # in case an object was constructed through an unsafe bypass.
    raise AnsatzCompilerError(f"{path} contains an opaque value")


def _expression_literals(expression: ParameterExpression, path: str) -> list[FixedLiteral]:
    literals: list[FixedLiteral] = []
    for index, term in enumerate(expression.terms):
        if term.coefficient != 1.0:
            literals.append(
                FixedLiteral(
                    path=f"{path}/terms/{index}/coefficient",
                    value=term.coefficient,
                    role="scale",
                )
            )
    if expression.constant != 0.0 or not expression.terms:
        literals.append(
            FixedLiteral(
                path=f"{path}/constant",
                value=expression.constant,
                role="offset",
            )
        )
    return literals


def _matrix_rank(rows: Sequence[Sequence[float]], *, tolerance: float = 1e-12) -> int:
    matrix = [list(map(float, row)) for row in rows]
    if not matrix:
        return 0
    width = len(matrix[0])
    rank = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(rank, len(matrix))
                if abs(matrix[row][column]) > tolerance
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        scale = matrix[rank][column]
        matrix[rank] = [value / scale for value in matrix[rank]]
        for row in range(rank + 1, len(matrix)):
            factor = matrix[row][column]
            if abs(factor) <= tolerance:
                continue
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    matrix[row], matrix[rank], strict=True
                )
            ]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _validate_operation(
    operation: OperationSpec,
    *,
    declared: set[str],
    num_qubits: int,
    path: str,
    occurrences: Counter[str],
    literals: list[FixedLiteral],
) -> None:
    _check_qubits(operation.qubits, num_qubits, f"{path}/qubits")
    try:
        macro = get_trusted_macro(operation.macro)
        macro.validate_operation(operation)
    except MacroValidationError as exc:
        raise AnsatzCompilerError(f"{path}: {exc}") from exc

    for argument_name, expression in operation.parameters.items():
        expression_path = f"{path}/parameters/{argument_name}"
        unknown_parameters = set(expression.parameter_names) - declared
        if unknown_parameters:
            raise AnsatzCompilerError(
                f"{expression_path} references undeclared parameters: "
                f"{sorted(unknown_parameters)}"
            )
        for parameter_name in expression.parameter_names:
            occurrences[parameter_name] += 1
        literals.extend(_expression_literals(expression, expression_path))

        if macro.variational:
            if not expression.is_trainable:
                raise AnsatzCompilerError(
                    f"{expression_path} must reference a trainable parameter; "
                    "constant-only variational gates are forbidden"
                )
            if not expression.is_zero_at_origin:
                raise AnsatzCompilerError(
                    f"{expression_path} has a fixed offset and is not identity "
                    "when trainable parameters are zero"
                )

    for option_name, option_value in operation.options.items():
        literals.extend(
            _numeric_option_literals(option_value, f"{path}/options/{option_name}")
        )

    if macro.variational:
        if not macro.identity_at_zero or not trusted_macro_zero_is_identity(macro.name):
            raise AnsatzCompilerError(
                f"trusted variational macro {macro.name} fails its zero-identity contract"
            )


def validate_ansatz(spec: AnsatzSpec | Mapping[str, Any]) -> AnsatzAudit:
    """Validate a typed ansatz and derive a representation/resource audit.

    The returned counts are computed from the complete IR.  Candidate-provided
    counts or descriptions are never accepted as inputs.
    """

    checked = _coerce_spec(spec)
    declared_names = tuple(parameter.name for parameter in checked.parameters)
    declared = set(declared_names)
    occurrences: Counter[str] = Counter()
    fixed_literals: list[FixedLiteral] = []
    logical_macros: Counter[str] = Counter()

    operation_count = 0
    expression_count = 0
    term_count = 0
    for operation_index, operation in enumerate(checked.operations):
        path = f"/operations/{operation_index}"
        _validate_operation(
            operation,
            declared=declared,
            num_qubits=checked.num_qubits,
            path=path,
            occurrences=occurrences,
            literals=fixed_literals,
        )
        logical_macros[operation.macro] += 1
        operation_count += 1
        expression_count += len(operation.parameters)
        term_count += sum(
            len(expression.terms) for expression in operation.parameters.values()
        )

    unused = tuple(name for name in declared_names if occurrences[name] == 0)
    if unused:
        raise AnsatzCompilerError(
            "declared trainable parameters must be used; dummy parameters found: "
            f"{list(unused)}"
        )

    used_names = tuple(name for name in declared_names if occurrences[name] > 0)
    parameter_index = {name: index for index, name in enumerate(used_names)}
    incidence_rows: list[list[float]] = []
    for operation in checked.operations:
        for expression in operation.parameters.values():
            row = [0.0] * len(used_names)
            for term in expression.terms:
                row[parameter_index[term.parameter.name]] = float(term.coefficient)
            incidence_rows.append(row)
    if _matrix_rank(incidence_rows) != len(used_names):
        raise AnsatzCompilerError(
            "declared trainable parameters must be linearly independent in the "
            "circuit angles"
        )
    node_counts = {
        "ansatz": 1,
        "parameter_declaration": len(checked.parameters),
        "operation": operation_count,
        "expression": expression_count,
        "parameter_term": term_count,
    }
    return AnsatzAudit(
        unique_trainable_params=len(used_names),
        trainable_parameter_names=used_names,
        parameter_occurrences={name: occurrences[name] for name in used_names},
        unused_parameters=unused,
        spec_nodes=sum(node_counts.values()),
        spec_node_counts=node_counts,
        fixed_literals=tuple(fixed_literals),
        logical_macros=dict(logical_macros),
        operations=operation_count,
    )


def _resolve_expression(
    expression: ParameterExpression,
    parameters: Mapping[str, Parameter],
) -> Any:
    value: Any = expression.constant
    for term in expression.terms:
        value = value + term.coefficient * parameters[term.parameter.name]
    return value


def compile_ansatz(spec: AnsatzSpec | Mapping[str, Any]) -> CompiledAnsatz:
    """Validate and compile an ansatz using only the trusted macro registry."""

    checked = _coerce_spec(spec)
    audit = validate_ansatz(checked)
    parameters = {
        name: Parameter(name) for name in audit.trainable_parameter_names
    }
    circuit = QuantumCircuit(checked.num_qubits, name=checked.name)

    for operation_index, operation in enumerate(checked.operations):
        try:
            get_trusted_macro(operation.macro).emit_operation(
                circuit,
                operation,
                lambda expression: _resolve_expression(expression, parameters),
            )
        except MacroValidationError as exc:
            raise AnsatzCompilerError(
                f"failed to compile /operations/{operation_index}: {exc}"
            ) from exc

    return CompiledAnsatz(circuit=circuit, parameters=parameters, audit=audit)
