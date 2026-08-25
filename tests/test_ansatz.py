from __future__ import annotations

import json
import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp

from autovqe.ansatz import (
    ALLOWED_GATES,
    AnsatzCompilerError,
    AnsatzIRValidationError,
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
    ParameterRef,
    ParameterTerm,
    compile_ansatz,
)


def _operation(macro: str, coefficient: float = 1.0) -> OperationSpec:
    return OperationSpec(
        macro=macro,
        qubits=(0, 1),
        parameters={"angle": ParameterExpression.parameter("theta", coefficient)},
        options={"pauli": "XY"} if macro == "PauliRotation" else {},
    )


class LeanAnsatzTests(unittest.TestCase):
    def test_v1_json_round_trip_and_exact_allowlist(self) -> None:
        payload = {
            "version": 1,
            "name": "shared",
            "num_qubits": 2,
            "parameters": ["theta"],
            "operations": [
                {
                    "macro": macro,
                    "qubits": [0, 1],
                    "parameters": {
                        "angle": {"parameter": "theta", "coefficient": coefficient}
                    },
                    "options": {"pauli": "XY"} if macro == "PauliRotation" else {},
                }
                for macro, coefficient in zip(ALLOWED_GATES, (-1.0, 0.5, 1.0))
            ],
        }
        restored = AnsatzSpec.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(AnsatzSpec.from_dict(restored.to_dict()), restored)
        compiled = compile_ansatz(restored)
        self.assertEqual(set(ALLOWED_GATES), {"PauliRotation", "XYExchange", "IsotropicExchange"})
        self.assertEqual(compiled.audit["parameter_occurrences"], {"theta": 3})
        self.assertEqual(compiled.audit["operations"], 3)
        self.assertEqual(compiled.audit["logical_macros"], {name: 1 for name in ALLOWED_GATES})
        self.assertEqual(compiled.audit["unique_trainable_params"], 1)
        self.assertEqual(len(compiled.audit["fixed_literals"]), 2)
        self.assertEqual(set(compiled.parameters), {"theta"})

    def test_all_gates_are_identity_at_parameter_origin(self) -> None:
        for macro in ALLOWED_GATES:
            with self.subTest(macro=macro):
                compiled = compile_ansatz(
                    AnsatzSpec(2, ("theta",), (_operation(macro),))
                )
                bound = compiled.circuit.assign_parameters(
                    {compiled.parameters["theta"]: 0.0}, inplace=False
                )
                self.assertTrue(Operator(bound).equiv(Operator(QuantumCircuit(2))))

    def test_qiskit_angle_conventions(self) -> None:
        generators = {
            "PauliRotation": SparsePauliOp.from_list([("YX", 1.0)]),
            "XYExchange": SparsePauliOp.from_list([("XX", 1.0), ("YY", 1.0)]),
            "IsotropicExchange": SparsePauliOp.from_list(
                [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]
            ),
        }
        angle = 0.371
        for macro, generator in generators.items():
            with self.subTest(macro=macro):
                compiled = compile_ansatz(AnsatzSpec(2, ("theta",), (_operation(macro),)))
                actual = Operator(
                    compiled.circuit.assign_parameters(
                        {compiled.parameters["theta"]: angle}, inplace=False
                    )
                ).data
                values, vectors = np.linalg.eigh(generator.to_matrix())
                expected = (vectors * np.exp(-1.0j * angle * values)) @ vectors.conj().T
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_empty_control_is_valid(self) -> None:
        compiled = compile_ansatz(AnsatzSpec(num_qubits=4))
        self.assertEqual(compiled.audit["operations"], 0)
        self.assertEqual(compiled.audit["unique_trainable_params"], 0)
        self.assertEqual(compiled.audit["gates"], 0)
        self.assertEqual(dict(compiled.parameters), {})

    def test_angle_must_be_one_declared_term_with_zero_offset(self) -> None:
        cases = (
            ParameterExpression.literal(0.2),
            ParameterExpression(
                (
                    ParameterTerm(ParameterRef("theta")),
                    ParameterTerm(ParameterRef("phi")),
                )
            ),
            ParameterExpression((ParameterTerm(ParameterRef("theta")),), 0.1),
            ParameterExpression.parameter("undeclared"),
        )
        for angle in cases:
            operation = OperationSpec(
                "XYExchange", (0, 1), {"angle": angle}
            )
            with self.subTest(angle=angle), self.assertRaises(AnsatzCompilerError):
                compile_ansatz(
                    AnsatzSpec(2, ("theta", "phi"), (operation,))
                )
        with self.assertRaisesRegex(AnsatzCompilerError, "must be used"):
            compile_ansatz(AnsatzSpec(2, ("unused",), ()))

    def test_gate_shape_qubits_and_strict_json_keys_are_enforced(self) -> None:
        invalid_operations = (
            OperationSpec("custom", (0, 1), {"angle": ParameterExpression.parameter("theta")}),
            OperationSpec("XYExchange", (0,), {"angle": ParameterExpression.parameter("theta")}),
            OperationSpec("IsotropicExchange", (0, 2), {"angle": ParameterExpression.parameter("theta")}),
            OperationSpec(
                "PauliRotation",
                (0, 1),
                {"angle": ParameterExpression.parameter("theta")},
                {"pauli": "X"},
            ),
        )
        for operation in invalid_operations:
            with self.subTest(macro=operation.macro), self.assertRaises(AnsatzCompilerError):
                compile_ansatz(AnsatzSpec(2, ("theta",), (operation,)))
        payload = AnsatzSpec(2).to_dict()
        with self.assertRaises(AnsatzIRValidationError):
            AnsatzSpec.from_dict({**payload, "reference_energy": -1.0})


if __name__ == "__main__":
    unittest.main()
