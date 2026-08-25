from __future__ import annotations

import unittest

from qiskit.quantum_info import SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzSpec,
    LayerSpec,
    OperationSpec,
    ParameterExpression,
    ParameterRef,
    ParameterTerm,
)
from autovqe.evaluator import EvaluationError, EvaluationProtocol, evaluate_ansatz
from autovqe.prepare import BackendTarget


class EvaluatorResourceAuditTests(unittest.TestCase):
    def test_linear_cancellation_cannot_zero_the_generic_cost(self) -> None:
        # The former public binding was a=s, b=2s.  This expression therefore
        # made the gate exactly identity and reported a zero generic cost.
        cancellation_at_old_binding = ParameterExpression(
            terms=(
                ParameterTerm(ParameterRef("a"), -2.0),
                ParameterTerm(ParameterRef("b")),
            )
        )
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("a", "b"),
            layers=(
                LayerSpec(
                    operations=(
                        OperationSpec(
                            macro="XYExchange",
                            qubits=(0, 1),
                            parameters={"angle": cancellation_at_old_binding},
                        ),
                    )
                ),
            ),
        )
        protocol = EvaluationProtocol(
            max_evals=4,
            restarts=1,
            seed=23,
            generic_binding_count=3,
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

        self.assertTrue(result.receipt.valid, result.receipt.violations)
        metrics = result.receipt.metrics
        self.assertGreater(metrics["template_twoq_count"], 0)
        self.assertGreater(metrics["generic_worst_twoq_count"], 0)
        for metric_name in ("singleq_count", "twoq_count", "total_gate_count", "depth"):
            self.assertEqual(
                metrics[f"generic_{metric_name}"],
                metrics[f"generic_worst_{metric_name}"],
            )
        self.assertEqual(metrics["generic_binding_count"], 3)

    def test_generic_resource_audit_requires_at_least_three_bindings(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "at least 3"):
            EvaluationProtocol(generic_binding_count=2).validate()

    def test_candidate_specific_generic_metrics_are_reproducible(self) -> None:
        theta = ParameterExpression.parameter("theta")
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            layers=(
                LayerSpec(
                    operations=(
                        OperationSpec(
                            macro="XYExchange",
                            qubits=(0, 1),
                            parameters={"angle": theta},
                        ),
                    )
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

        self.assertTrue(first.receipt.valid, first.receipt.violations)
        self.assertEqual(first.receipt.candidate_hash, second.receipt.candidate_hash)
        self.assertEqual(first.receipt.metrics, second.receipt.metrics)


if __name__ == "__main__":
    unittest.main()
