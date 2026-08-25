from __future__ import annotations

import unittest

from qiskit.quantum_info import SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
    ParameterRef,
    ParameterTerm,
)
from autovqe.backend import BackendTarget
from autovqe.contracts import PauliTerm, PublicProblem
from autovqe.evaluator import (
    EvaluationError,
    EvaluationProtocol,
    audit_public_candidate,
    evaluate_ansatz,
)


class EvaluatorResourceAuditTests(unittest.TestCase):
    def test_invalid_candidate_returns_a_failed_compile_only_audit(self) -> None:
        result = audit_public_candidate(
            PublicProblem.create(
                num_qubits=1,
                pauli_terms=(PauliTerm("Z", 1.0),),
            ),
            {"num_qubits": 1, "operations": "not-an-array"},
        )

        self.assertFalse(result.valid)
        self.assertTrue(result.violations)
        self.assertEqual(result.metrics, {})

    def test_linear_cancellation_cannot_zero_the_audit_cost(self) -> None:
        # One audit point can cancel the first angle.  A second independent
        # angle keeps the declared parameter basis honest while checking that
        # resource accounting never relies on one convenient binding.
        cancellation_at_old_binding = ParameterExpression(
            terms=(
                ParameterTerm(ParameterRef("a"), -2.0),
                ParameterTerm(ParameterRef("b")),
            )
        )
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("a", "b"),
            operations=(
                OperationSpec(
                    macro="XYExchange",
                    qubits=(0, 1),
                    parameters={"angle": cancellation_at_old_binding},
                ),
                OperationSpec(
                    macro="PauliRotation",
                    qubits=(0,),
                    parameters={"angle": ParameterExpression.parameter("a")},
                    options={"pauli": "Z"},
                ),
            ),
        )
        protocol = EvaluationProtocol(
            max_evals=4,
            restarts=1,
            seed=23,
            audit_binding_count=3,
        )
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("ZI", 1.0)]),
            spec,
            backend_target=BackendTarget(
                basis_gates=["rx", "ry", "rz", "cx"],
                coupling_map=[[0, 1], [1, 0]],
            ),
            protocol=protocol,
        )

        self.assertTrue(result.result.valid, result.result.violations)
        metrics = result.result.metrics
        self.assertGreater(metrics["template_twoq_count"], 0)
        self.assertGreater(metrics["audit_worst_twoq_count"], 0)
        self.assertFalse(any(key.startswith("generic_") for key in metrics))
        self.assertEqual(metrics["audit_binding_count"], 3)

    def test_resource_audit_requires_at_least_three_bindings(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "at least 3"):
            EvaluationProtocol(audit_binding_count=2).validate()

    def test_constant_hamiltonian_is_the_explicit_flat_activity_case(self) -> None:
        spec = AnsatzSpec(
            num_qubits=1,
            parameters=("theta",),
            operations=(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=(0,),
                    parameters={
                        "angle": ParameterExpression.parameter("theta")
                    },
                    options={"pauli": "X"},
                ),
            ),
        )
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("I", 3.0)]),
            spec,
            protocol=EvaluationProtocol(max_evals=4, restarts=1),
        ).result

        self.assertTrue(result.valid, result.violations)
        self.assertTrue(result.constant_hamiltonian)
        self.assertEqual(result.hamiltonian_active_norm, 0.0)
        self.assertLessEqual(result.objective_energy_span, 1e-12)
        self.assertIsNone(result.objective_activity_fraction)

    def test_candidate_specific_audit_metrics_are_reproducible(self) -> None:
        theta = ParameterExpression.parameter("theta")
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            operations=(
                OperationSpec(
                    macro="XYExchange",
                    qubits=(0, 1),
                    parameters={"angle": theta},
                ),
            ),
        )
        protocol = EvaluationProtocol(max_evals=3, restarts=1, seed=41)
        backend = BackendTarget(
            basis_gates=["rx", "ry", "rz", "cx"],
            coupling_map=[[0, 1], [1, 0]],
        )
        hamiltonian = SparsePauliOp.from_list([("ZI", 1.0)])

        first = evaluate_ansatz(
            hamiltonian, spec, backend_target=backend, protocol=protocol
        )
        second = evaluate_ansatz(
            hamiltonian, spec, backend_target=backend, protocol=protocol
        )

        self.assertTrue(first.result.valid, first.result.violations)
        self.assertEqual(
            set(first.result.to_dict()),
            {
                "valid",
                "best_energy",
                "trace_summary",
                "objective_calls",
                "optimizer",
                "seed",
                "optimized_parameter_binding",
                "audit",
                "metrics",
                "violations",
                "objective_energy_span",
                "hamiltonian_active_norm",
                "objective_activity_fraction",
                "constant_hamiltonian",
            },
        )
        self.assertEqual(first.result.metrics, second.result.metrics)
        self.assertEqual(first.result.objective_energy_span, 0.0)
        self.assertEqual(first.result.hamiltonian_active_norm, 1.0)
        self.assertEqual(first.result.objective_activity_fraction, 0.0)
        self.assertFalse(first.result.constant_hamiltonian)
        self.assertLess(len(first.result.trace_summary), first.result.objective_calls + 1)


if __name__ == "__main__":
    unittest.main()
