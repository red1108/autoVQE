from __future__ import annotations

import unittest

from autovqe.ansatz_ir import (
    AnsatzSpec,
    LayerSpec,
    OperationSpec,
    ParameterExpression,
)
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
    SectorSpec,
)
from autovqe.evaluator import (
    CANONICAL_BASIS_GATES,
    EvaluationProtocol,
    backend_target_from_public,
    canonical_backend_target,
    evaluate_public_problem,
)


def problem_with_backend(backend: BackendSpec) -> PublicProblem:
    return PublicProblem.create(
        num_qubits=2,
        pauli_terms=(PauliTerm("ZI", 1.0),),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        reference=ReferenceSpec(),
        backend=backend,
    )


def exchange_spec() -> AnsatzSpec:
    return AnsatzSpec(
        name="backend_boundary_exchange",
        num_qubits=2,
        parameters=("theta",),
        layers=(
            LayerSpec(
                operations=(
                    OperationSpec(
                        macro="XYExchange",
                        qubits=(0, 1),
                        parameters={
                            "angle": ParameterExpression.parameter("theta")
                        },
                    ),
                ),
            ),
        ),
    )


class EvaluatorBackendBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = EvaluationProtocol(max_evals=4, restarts=1, seed=31)

    def test_empty_backend_uses_logical_target_and_still_emits_canonical_metrics(self) -> None:
        problem = problem_with_backend(BackendSpec())
        self.assertIsNone(backend_target_from_public(problem))

        result = evaluate_public_problem(problem, exchange_spec(), protocol=self.protocol)

        self.assertTrue(result.receipt.valid, result.receipt.violations)
        metrics = result.receipt.metrics
        self.assertEqual(metrics["template_twoq_count"], 1)
        self.assertEqual(metrics["generic_worst_twoq_count"], 1)
        self.assertEqual(metrics["final_twoq_count"], 1)
        for prefix in (
            "canonical_template",
            "canonical_generic_worst",
            "canonical_final",
        ):
            for name in ("singleq_count", "twoq_count", "total_gate_count", "depth"):
                self.assertIn(f"{prefix}_{name}", metrics)
            self.assertGreater(metrics[f"{prefix}_total_gate_count"], 0)

    def test_canonical_metrics_do_not_depend_on_declared_backend(self) -> None:
        empty = problem_with_backend(BackendSpec())
        declared = problem_with_backend(
            BackendSpec(
                basis_gates=("rx", "ry", "rz", "cx"),
                coupling_map=((0, 1), (1, 0)),
            )
        )

        empty_result = evaluate_public_problem(
            empty, exchange_spec(), protocol=self.protocol
        )
        declared_result = evaluate_public_problem(
            declared, exchange_spec(), protocol=self.protocol
        )

        self.assertTrue(empty_result.receipt.valid, empty_result.receipt.violations)
        self.assertTrue(declared_result.receipt.valid, declared_result.receipt.violations)
        empty_canonical = {
            key: value
            for key, value in empty_result.receipt.metrics.items()
            if key.startswith("canonical_")
        }
        declared_canonical = {
            key: value
            for key, value in declared_result.receipt.metrics.items()
            if key.startswith("canonical_")
        }
        self.assertEqual(empty_canonical, declared_canonical)

        canonical = canonical_backend_target()
        self.assertEqual(tuple(canonical.basis_gates), CANONICAL_BASIS_GATES)
        self.assertIsNone(canonical.coupling_map)

    def test_backend_basis_does_not_authorize_an_untrusted_macro(self) -> None:
        problem = problem_with_backend(
            BackendSpec(basis_gates=("rz", "sx", "x", "cx", "unitary"))
        )
        forged = exchange_spec().to_dict()
        forged["layers"][0]["operations"][0]["macro"] = "UnitaryGate"

        result = evaluate_public_problem(problem, forged, protocol=self.protocol)

        self.assertFalse(result.receipt.valid)
        self.assertTrue(result.receipt.violations)
        self.assertIn("unknown or untrusted macro", result.receipt.violations[0])


if __name__ == "__main__":
    unittest.main()
