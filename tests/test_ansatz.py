from __future__ import annotations

import json
import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp

from autovqe.ansatz import (
    ALLOWED_GATES,
    ALLOWED_SCALES,
    AnsatzCompilerError,
    AnsatzIRValidationError,
    AnsatzSpec,
    OperationSpec,
    compile_ansatz,
)


def _operation(gate: str, scale: float = 1.0) -> OperationSpec:
    return OperationSpec(
        gate=gate,
        qubits=(0, 1),
        parameter="theta",
        scale=scale,
        pauli="XY" if gate == "PauliRotation" else None,
    )


class LeanAnsatzTests(unittest.TestCase):
    def test_direct_v1_json_round_trip_and_exact_allowlists(self) -> None:
        payload = {
            "version": 1,
            "num_qubits": 2,
            "operations": [
                {
                    "gate": gate,
                    "qubits": [0, 1],
                    "parameter": "theta",
                    "scale": scale,
                    **({"pauli": "XY"} if gate == "PauliRotation" else {}),
                }
                for gate, scale in zip(ALLOWED_GATES, (-1.0, 0.5, 1.0))
            ],
        }
        restored = AnsatzSpec.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(AnsatzSpec.from_dict(restored.to_dict()), restored)
        compiled = compile_ansatz(restored)

        self.assertEqual(
            set(ALLOWED_GATES),
            {"PauliRotation", "XYExchange", "IsotropicExchange"},
        )
        self.assertEqual(
            set(ALLOWED_SCALES), {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}
        )
        self.assertEqual(compiled.audit["parameter_occurrences"], {"theta": 3})
        self.assertEqual(compiled.audit["operations"], 3)
        self.assertEqual(compiled.audit["unique_trainable_params"], 1)
        self.assertEqual(set(compiled.parameters), {"theta"})

    def test_all_gates_are_identity_at_parameter_origin(self) -> None:
        for gate in ALLOWED_GATES:
            with self.subTest(gate=gate):
                compiled = compile_ansatz(AnsatzSpec(2, (_operation(gate),)))
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
        for gate, generator in generators.items():
            with self.subTest(gate=gate):
                compiled = compile_ansatz(AnsatzSpec(2, (_operation(gate),)))
                actual = Operator(
                    compiled.circuit.assign_parameters(
                        {compiled.parameters["theta"]: angle}, inplace=False
                    )
                ).data
                values, vectors = np.linalg.eigh(generator.to_matrix())
                expected = (vectors * np.exp(-1.0j * angle * values)) @ vectors.conj().T
                np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_parameters_are_derived_from_raw_operations(self) -> None:
        candidate = AnsatzSpec(
            2,
            (
                OperationSpec("PauliRotation", (0,), "shared", pauli="X"),
                OperationSpec("PauliRotation", (1,), "other", pauli="Y"),
                OperationSpec("PauliRotation", (1,), "shared", pauli="Z"),
            ),
        )
        compiled = compile_ansatz(candidate)
        self.assertEqual(candidate.parameter_names, ("shared", "other"))
        self.assertEqual(
            compiled.audit,
            {
                "unique_trainable_params": 2,
                "trainable_parameter_names": ["shared", "other"],
                "parameter_occurrences": {"shared": 2, "other": 1},
                "operations": 3,
            },
        )

        empty = compile_ansatz(AnsatzSpec(num_qubits=4))
        self.assertEqual(empty.audit["operations"], 0)
        self.assertEqual(empty.audit["unique_trainable_params"], 0)
        self.assertEqual(dict(empty.parameters), {})

    def test_scale_and_parameter_policy_are_closed(self) -> None:
        for scale in ALLOWED_SCALES:
            with self.subTest(scale=scale):
                self.assertEqual(
                    compile_ansatz(AnsatzSpec(2, (_operation("XYExchange", scale),)))
                    .audit["unique_trainable_params"],
                    1,
                )
        with self.assertRaisesRegex(AnsatzCompilerError, "scale must be one of"):
            compile_ansatz(AnsatzSpec(2, (_operation("XYExchange", 0.25),)))
        with self.assertRaises(AnsatzIRValidationError):
            OperationSpec("XYExchange", (0, 1), "theta", 0.0)
        with self.assertRaises(AnsatzIRValidationError):
            OperationSpec("XYExchange", (0, 1), "not a label")

    def test_gate_shape_qubits_and_strict_json_keys_are_enforced(self) -> None:
        invalid_operations = (
            OperationSpec("custom", (0, 1), "theta"),
            OperationSpec("XYExchange", (0,), "theta"),
            OperationSpec("IsotropicExchange", (0, 2), "theta"),
            OperationSpec("PauliRotation", (0, 1), "theta", pauli="X"),
            OperationSpec("XYExchange", (0, 1), "theta", pauli="XX"),
        )
        for operation in invalid_operations:
            with self.subTest(gate=operation.gate), self.assertRaises(AnsatzCompilerError):
                compile_ansatz(AnsatzSpec(2, (operation,)))

        payload = AnsatzSpec(2).to_dict()
        with self.assertRaises(AnsatzIRValidationError):
            AnsatzSpec.from_dict({**payload, "parameters": ["theta"]})
        with self.assertRaises(AnsatzIRValidationError):
            OperationSpec.from_dict(
                {
                    "macro": "XYExchange",
                    "qubits": [0, 1],
                    "parameter": "theta",
                }
            )


if __name__ == "__main__":
    unittest.main()
