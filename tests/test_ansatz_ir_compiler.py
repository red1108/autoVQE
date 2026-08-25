from __future__ import annotations

import json
import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzIRValidationError,
    AnsatzSpec,
    LayerSpec,
    OperationSpec,
    ParameterExpression,
    ParameterRef,
    ReferenceSpec,
)
from autovqe.compiler import (
    AnsatzAudit,
    AnsatzCompilerError,
    compile_ansatz,
    validate_ansatz,
)
from autovqe.macros import (
    TRUSTED_MACROS,
    trusted_macro_names,
    trusted_macro_zero_is_identity,
    validate_trusted_registry,
)


def example_spec() -> AnsatzSpec:
    theta = ParameterRef("theta")
    phi = ParameterRef("phi")
    return AnsatzSpec(
        name="typed_example",
        num_qubits=3,
        parameters=(theta, phi),
        reference=ReferenceSpec(macro="X", qubits=(0,)),
        layers=(
            LayerSpec(
                name="hopping",
                operations=(
                    OperationSpec(
                        macro="PauliRotation",
                        qubits=(0, 1),
                        parameters={
                            "angle": ParameterExpression.parameter(theta, 0.5)
                        },
                        options={"pauli": "XZ"},
                    ),
                    OperationSpec(
                        macro="XYExchange",
                        qubits=(1, 2),
                        parameters={"angle": ParameterExpression.parameter(phi)},
                    ),
                ),
            ),
            LayerSpec(
                name="spin",
                operations=(
                    OperationSpec(
                        macro="IsotropicExchange",
                        qubits=(0, 2),
                        parameters={"angle": ParameterExpression.parameter(theta)},
                    ),
                ),
            ),
        ),
    )


class AnsatzIRTests(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        spec = example_spec()
        payload = spec.to_dict()
        encoded = json.dumps(payload)
        restored = AnsatzSpec.from_dict(json.loads(encoded))
        self.assertEqual(restored, spec)
        self.assertEqual(len(restored.operations), 3)

    def test_top_level_operations_shorthand(self) -> None:
        payload = {
            "version": 1,
            "name": "short",
            "num_qubits": 2,
            "parameters": ["theta"],
            "reference": {"macro": "X", "qubits": [0]},
            "operations": [
                {
                    "macro": "XYExchange",
                    "qubits": [0, 1],
                    "parameters": {"angle": {"parameter": "theta"}},
                    "options": {},
                }
            ],
        }
        spec = AnsatzSpec.from_dict(payload)
        self.assertEqual(len(spec.layers), 1)
        self.assertEqual(spec.operations[0].macro, "XYExchange")
        self.assertEqual(compile_ansatz(payload).audit.unique_trainable_params, 1)

    def test_unknown_serialized_fields_are_rejected(self) -> None:
        payload = example_spec().to_dict()
        payload["unitary"] = [[1, 0], [0, 1]]
        with self.assertRaises(AnsatzIRValidationError):
            AnsatzSpec.from_dict(payload)

    def test_duplicate_support_is_rejected_by_ir(self) -> None:
        with self.assertRaises(AnsatzIRValidationError):
            OperationSpec(
                macro="XYExchange",
                qubits=(0, 0),
                parameters={"angle": ParameterExpression.parameter("theta")},
            )


class TrustedMacroTests(unittest.TestCase):
    def test_registry_is_exact_and_read_only(self) -> None:
        self.assertEqual(
            set(trusted_macro_names()),
            {"PauliRotation", "XYExchange", "IsotropicExchange", "X"},
        )
        with self.assertRaises(TypeError):
            TRUSTED_MACROS["custom"] = TRUSTED_MACROS["X"]  # type: ignore[index]

    def test_all_variational_macros_are_identity_at_zero(self) -> None:
        validate_trusted_registry()
        for name in ("PauliRotation", "XYExchange", "IsotropicExchange"):
            self.assertTrue(trusted_macro_zero_is_identity(name))


class CompilerAuditTests(unittest.TestCase):
    def test_compiler_derives_audit_instead_of_accepting_counts(self) -> None:
        compiled = compile_ansatz(example_spec())
        audit = compiled.audit
        self.assertEqual(audit.unique_trainable_params, 2)
        self.assertEqual(audit.trainable_parameter_names, ("theta", "phi"))
        self.assertEqual(dict(audit.parameter_occurrences), {"theta": 2, "phi": 1})
        self.assertEqual(audit.unused_parameters, ())
        self.assertEqual(audit.layers, 2)
        self.assertEqual(audit.operations, 3)
        self.assertEqual(audit.spec_nodes, 15)
        self.assertEqual(
            dict(audit.logical_macros),
            {
                "X": 1,
                "PauliRotation": 1,
                "XYExchange": 1,
                "IsotropicExchange": 1,
            },
        )
        self.assertEqual(audit.fixed_literal_count, 1)
        self.assertEqual(audit.fixed_literals[0].role, "scale")
        self.assertEqual(audit.fixed_literals[0].value, 0.5)
        self.assertEqual(set(compiled.parameters), {"theta", "phi"})
        self.assertEqual(compiled.circuit.num_qubits, 3)

    def test_audit_round_trip(self) -> None:
        audit = validate_ansatz(example_spec())
        self.assertEqual(AnsatzAudit.from_dict(audit.to_dict()), audit)

    def test_zero_parameters_leave_only_reference_preparation(self) -> None:
        compiled = compile_ansatz(example_spec())
        bound = compiled.circuit.assign_parameters(
            {parameter: 0.0 for parameter in compiled.parameters.values()},
            inplace=False,
        )
        expected = QuantumCircuit(3)
        expected.x(0)
        self.assertTrue(Operator(bound).equiv(Operator(expected)))

    def test_pauli_rotation_matches_exp_minus_i_angle_p(self) -> None:
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            layers=(
                LayerSpec(
                    operations=(
                        OperationSpec(
                            macro="PauliRotation",
                            qubits=(0, 1),
                            parameters={
                                "angle": ParameterExpression.parameter("theta")
                            },
                            options={"pauli": "XY"},
                        ),
                    )
                ),
            ),
        )
        compiled = compile_ansatz(spec)
        angle = 0.271
        actual = compiled.circuit.assign_parameters(
            {compiled.parameters["theta"]: angle}, inplace=False
        )
        # PauliRotation("XY", qubits=(0,1)) means X on q0 and Y on q1.
        expected = QuantumCircuit(2)
        expected.append(
            PauliEvolutionGate(SparsePauliOp("YX"), time=angle),
            [0, 1],
        )
        self.assertTrue(Operator(actual).equiv(Operator(expected)))


class CompilerSafetyTests(unittest.TestCase):
    def one_operation_spec(
        self,
        operation: OperationSpec,
        *,
        parameters: tuple[str, ...] = ("theta",),
        num_qubits: int = 2,
    ) -> AnsatzSpec:
        return AnsatzSpec(
            num_qubits=num_qubits,
            parameters=parameters,
            layers=(LayerSpec(operations=(operation,)),),
        )

    def test_opaque_circuit_input_is_rejected(self) -> None:
        with self.assertRaises(AnsatzCompilerError):
            compile_ansatz(QuantumCircuit(2))  # type: ignore[arg-type]

    def test_unknown_and_reference_only_operation_macros_are_rejected(self) -> None:
        for macro in ("UnitaryGate", "DenseMatrix", "custom", "X"):
            operation = OperationSpec(
                macro=macro,
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
            )
            with self.subTest(macro=macro), self.assertRaises(AnsatzCompilerError):
                compile_ansatz(self.one_operation_spec(operation))

    def test_dense_or_custom_options_are_rejected(self) -> None:
        operation = OperationSpec(
            macro="PauliRotation",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
            options={
                "pauli": "XX",
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
        )
        with self.assertRaises(AnsatzCompilerError):
            compile_ansatz(self.one_operation_spec(operation))

    def test_support_bounds_are_enforced(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 2),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        with self.assertRaises(AnsatzCompilerError):
            compile_ansatz(self.one_operation_spec(operation, num_qubits=2))

        reference_only = AnsatzSpec(
            num_qubits=2,
            reference=ReferenceSpec(qubits=(2,)),
        )
        with self.assertRaises(AnsatzCompilerError):
            compile_ansatz(reference_only)

    def test_pauli_support_must_be_active_and_match_length(self) -> None:
        for pauli in ("X", "XI", "XA"):
            operation = OperationSpec(
                macro="PauliRotation",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
                options={"pauli": pauli},
            )
            with self.subTest(pauli=pauli), self.assertRaises(AnsatzCompilerError):
                compile_ansatz(self.one_operation_spec(operation))

    def test_constant_only_variational_angle_is_rejected(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.literal(0.314)},
        )
        with self.assertRaisesRegex(AnsatzCompilerError, "constant-only"):
            compile_ansatz(self.one_operation_spec(operation, parameters=()))

    def test_nonzero_fixed_offset_is_rejected(self) -> None:
        expression = ParameterExpression.parameter("theta") + 0.25
        operation = OperationSpec(
            macro="IsotropicExchange",
            qubits=(0, 1),
            parameters={"angle": expression},
        )
        with self.assertRaisesRegex(AnsatzCompilerError, "fixed offset"):
            compile_ansatz(self.one_operation_spec(operation))

    def test_undeclared_and_dummy_parameters_are_rejected(self) -> None:
        undeclared_operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("phi")},
        )
        with self.assertRaisesRegex(AnsatzCompilerError, "undeclared"):
            compile_ansatz(self.one_operation_spec(undeclared_operation))

        valid_operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        with self.assertRaisesRegex(AnsatzCompilerError, "dummy"):
            compile_ansatz(
                self.one_operation_spec(
                    valid_operation,
                    parameters=("theta", "unused"),
                )
            )

    def test_empty_layers_and_unknown_references_are_rejected(self) -> None:
        with self.assertRaisesRegex(AnsatzCompilerError, "cannot be empty"):
            compile_ansatz(
                AnsatzSpec(num_qubits=2, layers=(LayerSpec(operations=()),))
            )
        with self.assertRaisesRegex(AnsatzCompilerError, "at least one X target"):
            compile_ansatz(
                AnsatzSpec(num_qubits=2, reference=ReferenceSpec(macro="X", qubits=()))
            )
        with self.assertRaises(AnsatzCompilerError):
            compile_ansatz(
                AnsatzSpec(
                    num_qubits=2,
                    reference=ReferenceSpec(macro="Initialize", qubits=(0,)),
                )
            )


if __name__ == "__main__":
    unittest.main()
