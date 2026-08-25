"""Trusted logical macro registry for the AutoVQE ansatz compiler.

Only macros in :data:`TRUSTED_MACROS` can be compiled.  There is intentionally
no public registration hook: candidate specifications may select a macro, but
cannot supply its implementation.  Every variational macro uses the common
convention ``U(angle) = exp(-i * angle * G)`` for its documented Hermitian
generator ``G``.
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

from .ansatz_ir import OperationSpec, ParameterExpression


AngleResolver = Callable[[ParameterExpression], Any]


class MacroValidationError(ValueError):
    """Raised when a trusted macro is invoked with an invalid specification."""


class TrustedMacro:
    """A compiler-owned logical operation.

    Instances are kept in a read-only registry.  Candidate code can reference
    their names, but the compiler always invokes these trusted emitters.
    """

    __slots__ = (
        "name",
        "min_arity",
        "max_arity",
        "parameter_names",
        "option_names",
        "variational",
        "identity_at_zero",
        "description",
        "_operation_validator",
        "_operation_emitter",
        "_zero_probe",
    )

    def __init__(
        self,
        *,
        name: str,
        min_arity: int,
        max_arity: int | None,
        parameter_names: tuple[str, ...] = (),
        option_names: tuple[str, ...] = (),
        variational: bool = False,
        identity_at_zero: bool = False,
        description: str = "",
        operation_validator: Callable[[OperationSpec], None] | None = None,
        operation_emitter: Callable[[QuantumCircuit, OperationSpec, AngleResolver], None]
        | None = None,
        zero_probe: Callable[[], OperationSpec] | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "min_arity", min_arity)
        object.__setattr__(self, "max_arity", max_arity)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "option_names", option_names)
        object.__setattr__(self, "variational", variational)
        object.__setattr__(self, "identity_at_zero", identity_at_zero)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "_operation_validator", operation_validator)
        object.__setattr__(self, "_operation_emitter", operation_emitter)
        object.__setattr__(self, "_zero_probe", zero_probe)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("trusted macro definitions are immutable")

    def _validate_arity(self, arity: int, context: str) -> None:
        if arity < self.min_arity or (
            self.max_arity is not None and arity > self.max_arity
        ):
            if self.max_arity == self.min_arity:
                expected = str(self.min_arity)
            elif self.max_arity is None:
                expected = f"at least {self.min_arity}"
            else:
                expected = f"between {self.min_arity} and {self.max_arity}"
            raise MacroValidationError(
                f"{context} {self.name} requires {expected} qubits, got {arity}"
            )

    def validate_operation(self, operation: OperationSpec) -> None:
        if operation.macro != self.name:
            raise MacroValidationError(
                f"operation names {operation.macro}, but validator is for {self.name}"
            )
        self._validate_arity(len(operation.qubits), "operation")
        actual_parameters = set(operation.parameters)
        expected_parameters = set(self.parameter_names)
        if actual_parameters != expected_parameters:
            raise MacroValidationError(
                f"{self.name} parameters must be {sorted(expected_parameters)}, "
                f"got {sorted(actual_parameters)}"
            )
        actual_options = set(operation.options)
        expected_options = set(self.option_names)
        if actual_options != expected_options:
            raise MacroValidationError(
                f"{self.name} options must be {sorted(expected_options)}, "
                f"got {sorted(actual_options)}"
            )
        if self._operation_validator is not None:
            self._operation_validator(operation)

    def emit_operation(
        self,
        circuit: QuantumCircuit,
        operation: OperationSpec,
        resolve_angle: AngleResolver,
    ) -> None:
        self.validate_operation(operation)
        if self._operation_emitter is None:
            raise MacroValidationError(f"{self.name} has no operation emitter")
        self._operation_emitter(circuit, operation, resolve_angle)

    def zero_probe(self) -> OperationSpec:
        if self._zero_probe is None:
            raise MacroValidationError(f"{self.name} has no zero-identity probe")
        return self._zero_probe()


def _validate_pauli_rotation(operation: OperationSpec) -> None:
    pauli = operation.options.get("pauli")
    if not isinstance(pauli, str):
        raise MacroValidationError("PauliRotation option 'pauli' must be a string")
    if len(pauli) != len(operation.qubits):
        raise MacroValidationError(
            "PauliRotation pauli word length must equal its qubit support length"
        )
    if not pauli or any(letter not in {"X", "Y", "Z"} for letter in pauli):
        raise MacroValidationError(
            "PauliRotation pauli word must contain active X, Y, or Z factors only"
        )


def _emit_pauli_rotation(
    circuit: QuantumCircuit,
    operation: OperationSpec,
    resolve_angle: AngleResolver,
) -> None:
    pauli = operation.options["pauli"]
    qubits = operation.qubits
    angle = resolve_angle(operation.parameters["angle"])

    for qubit, letter in zip(qubits, pauli):
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

    for qubit, letter in reversed(tuple(zip(qubits, pauli))):
        if letter == "X":
            circuit.h(qubit)
        elif letter == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def _emit_xy_exchange(
    circuit: QuantumCircuit,
    operation: OperationSpec,
    resolve_angle: AngleResolver,
) -> None:
    """Emit exp[-i angle (XX + YY)] on a two-qubit support."""

    angle = resolve_angle(operation.parameters["angle"])
    # Qiskit XXPlusYY(theta, 0) = exp[-i theta (XX + YY) / 4].
    circuit.append(XXPlusYYGate(4.0 * angle, 0.0), list(operation.qubits))


def _emit_isotropic_exchange(
    circuit: QuantumCircuit,
    operation: OperationSpec,
    resolve_angle: AngleResolver,
) -> None:
    """Emit exp[-i angle (XX + YY + ZZ)] on a two-qubit support."""

    angle = resolve_angle(operation.parameters["angle"])
    left, right = operation.qubits
    # Qiskit RPP(phi) = exp(-i phi PP / 2), and XX, YY, ZZ commute.
    circuit.rxx(2.0 * angle, left, right)
    circuit.ryy(2.0 * angle, left, right)
    circuit.rzz(2.0 * angle, left, right)


def _zero_operation(
    macro: str,
    qubits: tuple[int, ...],
    *,
    options: Mapping[str, Any] | None = None,
) -> OperationSpec:
    return OperationSpec(
        macro=macro,
        qubits=qubits,
        parameters={"angle": ParameterExpression.literal(0.0)},
        options={} if options is None else options,
    )


TRUSTED_MACROS: Mapping[str, TrustedMacro] = MappingProxyType({
    "PauliRotation": TrustedMacro(
        name="PauliRotation",
        min_arity=1,
        max_arity=None,
        parameter_names=("angle",),
        option_names=("pauli",),
        variational=True,
        identity_at_zero=True,
        description="exp[-i angle P] for an active Pauli word P",
        operation_validator=_validate_pauli_rotation,
        operation_emitter=_emit_pauli_rotation,
        zero_probe=lambda: _zero_operation(
            "PauliRotation", (0, 1), options={"pauli": "XY"}
        ),
    ),
    "XYExchange": TrustedMacro(
        name="XYExchange",
        min_arity=2,
        max_arity=2,
        parameter_names=("angle",),
        variational=True,
        identity_at_zero=True,
        description="exp[-i angle (XX + YY)] via Qiskit XXPlusYY(4*angle, beta=0)",
        operation_emitter=_emit_xy_exchange,
        zero_probe=lambda: _zero_operation("XYExchange", (0, 1)),
    ),
    "IsotropicExchange": TrustedMacro(
        name="IsotropicExchange",
        min_arity=2,
        max_arity=2,
        parameter_names=("angle",),
        variational=True,
        identity_at_zero=True,
        description="exp[-i angle (XX + YY + ZZ)]",
        operation_emitter=_emit_isotropic_exchange,
        zero_probe=lambda: _zero_operation("IsotropicExchange", (0, 1)),
    ),
})


def get_trusted_macro(name: str) -> TrustedMacro:
    """Return a compiler-owned macro definition, rejecting unknown names."""

    try:
        return TRUSTED_MACROS[name]
    except (KeyError, TypeError) as exc:
        raise MacroValidationError(f"unknown or untrusted macro: {name!r}") from exc


def trusted_macro_names() -> tuple[str, ...]:
    """List the trusted variational macro names."""

    return tuple(TRUSTED_MACROS)


@lru_cache(maxsize=None)
def trusted_macro_zero_is_identity(name: str, atol: float = 1e-10) -> bool:
    """Numerically verify the trusted implementation's zero-angle identity.

    This checks the implementation rather than relying only on registry
    metadata.  It is cached because macro definitions are immutable.
    """

    macro = get_trusted_macro(name)
    if not macro.variational:
        return True
    if not macro.identity_at_zero:
        return False
    operation = macro.zero_probe()
    macro.validate_operation(operation)
    num_qubits = max(operation.qubits) + 1
    circuit = QuantumCircuit(num_qubits)
    macro.emit_operation(circuit, operation, lambda expression: expression.constant)
    actual = Operator(circuit).data
    expected = np.eye(2**num_qubits, dtype=complex)
    # Global phase is irrelevant for a logical variational macro.
    return bool(Operator(actual).equiv(Operator(expected), atol=atol))


def validate_trusted_registry() -> None:
    """Raise if any variational trusted macro fails its zero-angle contract."""

    for name, macro in TRUSTED_MACROS.items():
        if macro.variational and not trusted_macro_zero_is_identity(name):
            raise MacroValidationError(
                f"trusted variational macro {name} is not identity at zero"
            )
