from __future__ import annotations

import unittest

from autovqe.ansatz_ir import (
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
)
from autovqe.backend import (
    CANONICAL_BASIS_GATES,
    backend_target_from_problem,
    canonical_backend_target,
)
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    InitialStateSpec,
    SectorSpec,
)
from autovqe.evaluator import (
    EvaluationProtocol,
    evaluate_public_problem,
)


def problem_with_backend(backend: BackendSpec) -> PublicProblem:
    return PublicProblem.create(
        num_qubits=2,
        pauli_terms=(PauliTerm("ZI", 1.0),),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        initial_state=InitialStateSpec(),
        backend=backend,
    )


def exchange_spec() -> AnsatzSpec:
    return AnsatzSpec(
        name="backend_boundary_exchange",
        num_qubits=2,
        parameters=("theta",),
        operations=(
            OperationSpec(
                macro="XYExchange",
                qubits=(0, 1),
                parameters={
                    "angle": ParameterExpression.parameter("theta")
                },
            ),
        ),
    )


class EvaluatorBackendBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = EvaluationProtocol(max_evals=4, restarts=1, seed=31)

    def test_empty_backend_uses_logical_target_and_still_emits_canonical_metrics(self) -> None:
        problem = problem_with_backend(BackendSpec())
        self.assertIsNone(backend_target_from_problem(problem))

        result = evaluate_public_problem(problem, exchange_spec(), protocol=self.protocol)

        self.assertTrue(result.result.valid, result.result.violations)
        metrics = result.result.metrics
        self.assertEqual(metrics["template_twoq_count"], 1)
        self.assertEqual(metrics["audit_worst_twoq_count"], 1)
        self.assertEqual(metrics["final_twoq_count"], 1)
        for prefix in (
            "canonical_template",
            "canonical_audit_worst",
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

        self.assertTrue(empty_result.result.valid, empty_result.result.violations)
        self.assertTrue(declared_result.result.valid, declared_result.result.violations)
        empty_canonical = {
            key: value
            for key, value in empty_result.result.metrics.items()
            if key.startswith("canonical_")
        }
        declared_canonical = {
            key: value
            for key, value in declared_result.result.metrics.items()
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
        forged["operations"][0]["macro"] = "UnitaryGate"

        result = evaluate_public_problem(problem, forged, protocol=self.protocol)

        self.assertFalse(result.result.valid)
        self.assertTrue(result.result.violations)
        self.assertIn("unknown or untrusted macro", result.result.violations[0])


if __name__ == "__main__":
    unittest.main()
