from __future__ import annotations

import copy
import unittest

from autovqe.evaluator import (
    EvaluationError,
    EvaluationProtocol,
    audit_public_candidate,
    candidate_identity,
    evaluate_public_problem,
)
from autovqe.problem import BackendSpec, PauliTerm, PublicProblem


def operation(
    pauli: str, qubits: list[int], parameter: str, scale: float = 1.0
) -> dict:
    result = {
        "gate": "PauliRotation",
        "qubits": qubits,
        "parameter": parameter,
        "pauli": pauli,
    }
    if scale != 1.0:
        result["scale"] = scale
    return result


def spec(operations: list[dict]) -> dict:
    return {"version": 1, "num_qubits": 2, "operations": operations}


class LeanEvaluatorIdentityTests(unittest.TestCase):
    def test_semantic_representations_share_one_identity(self) -> None:
        base = spec(
            [operation("X", [0], "a"), operation("Y", [1], "b")]
        )
        renamed = spec(
            [operation("X", [0], "left"), operation("Y", [1], "right")]
        )
        reordered = copy.deepcopy(base)
        reordered["operations"].reverse()
        scaled = copy.deepcopy(base)
        for item in scaled["operations"]:
            item["scale"] = -2.0
        self.assertEqual(candidate_identity(base), candidate_identity(renamed))
        self.assertEqual(candidate_identity(base), candidate_identity(reordered))
        self.assertEqual(candidate_identity(base), candidate_identity(scaled))

        supported = spec([operation("XY", [0, 1], "theta")])
        reversed_support = spec([operation("YX", [1, 0], "theta")])
        self.assertEqual(
            candidate_identity(supported), candidate_identity(reversed_support)
        )

    def test_gate_expansion_and_rotation_splits_are_not_fresh_families(self) -> None:
        exchange = spec(
            [
                {
                    "gate": "XYExchange",
                    "qubits": [1, 0],
                    "parameter": "theta",
                }
            ]
        )
        expanded = spec(
            [
                operation("XX", [0, 1], "theta"),
                operation("YY", [0, 1], "theta"),
            ]
        )
        unsplit = spec([operation("X", [0], "theta")])
        split = spec(
            [
                operation("X", [0], "theta", 0.5),
                operation("X", [0], "theta", 0.5),
            ]
        )
        cancellation_retry = spec(
            [
                operation("X", [0], "theta"),
                operation("X", [0], "theta"),
                operation("X", [0], "theta", -1.0),
            ]
        )
        self.assertEqual(candidate_identity(exchange), candidate_identity(expanded))
        self.assertEqual(candidate_identity(unsplit), candidate_identity(split))
        self.assertEqual(
            candidate_identity(unsplit), candidate_identity(cancellation_retry)
        )

    def test_noncommuting_order_and_parameter_sharing_remain_distinct(self) -> None:
        ordered = spec(
            [operation("X", [0], "a"), operation("Z", [0], "b")]
        )
        reversed_order = copy.deepcopy(ordered)
        reversed_order["operations"].reverse()
        shared = spec(
            [operation("X", [0], "shared"), operation("Y", [1], "shared")]
        )
        independent = spec(
            [operation("X", [0], "a"), operation("Y", [1], "b")]
        )
        self.assertNotEqual(
            candidate_identity(ordered), candidate_identity(reversed_order)
        )
        self.assertNotEqual(candidate_identity(shared), candidate_identity(independent))

    def test_parameter_renaming_does_not_change_optimizer_trace(self) -> None:
        candidate = spec(
            [operation("X", [0], "a"), operation("Y", [1], "b")]
        )
        renamed = spec(
            [operation("X", [0], "left"), operation("Y", [1], "right")]
        )
        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 0.3)),
        )
        protocol = EvaluationProtocol(max_evals=12, restarts=1, seed=19)
        first = evaluate_public_problem(problem, candidate, protocol=protocol)
        second = evaluate_public_problem(problem, renamed, protocol=protocol)
        self.assertEqual(first.trace_summary, second.trace_summary)
        assert first.optimized_parameter_binding is not None
        assert second.optimized_parameter_binding is not None
        self.assertEqual(
            tuple(first.optimized_parameter_binding.values()),
            tuple(second.optimized_parameter_binding.values()),
        )

    def test_global_parameter_rescaling_does_not_change_evaluation(self) -> None:
        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("IZ", 1.0), PauliTerm("IX", 0.2)),
        )
        protocol = EvaluationProtocol(max_evals=12, restarts=1, seed=19)
        scales = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
        candidates = [
            spec([operation("Y", [0], "theta", scale)]) for scale in scales
        ]
        self.assertEqual(len({candidate_identity(item) for item in candidates}), 1)

        results = [
            evaluate_public_problem(problem, item, protocol=protocol)
            for item in candidates
        ]
        reference = results[0]
        self.assertTrue(reference.valid, reference.violations)
        reference_binding = reference.optimized_parameter_binding
        self.assertIsNotNone(reference_binding)
        assert reference_binding is not None
        reference_angle = scales[0] * reference_binding["theta"]
        for scale, result in zip(scales[1:], results[1:], strict=True):
            self.assertTrue(result.valid, result.violations)
            self.assertEqual(result.trace_summary, reference.trace_summary)
            self.assertEqual(result.best_energy, reference.best_energy)
            self.assertEqual(result.resources, reference.resources)
            binding = result.optimized_parameter_binding
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertAlmostEqual(scale * binding["theta"], reference_angle)


class LeanEvaluatorBoundaryTests(unittest.TestCase):
    def test_three_identity_seeded_resource_bindings_are_required(self) -> None:
        candidate = spec([operation("XY", [0, 1], "theta")])
        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("ZI", 1.0),),
            backend=BackendSpec(
                basis_gates=("rx", "ry", "rz", "cx"),
                coupling_map=((0, 1), (1, 0)),
            ),
        )
        protocol = EvaluationProtocol(audit_binding_count=3, seed=23)
        first = audit_public_candidate(problem, candidate, protocol=protocol)
        second = audit_public_candidate(problem, candidate, protocol=protocol)
        self.assertTrue(first.valid, first.violations)
        self.assertEqual(first.resources, second.resources)
        self.assertEqual(
            set(first.resources), {"parameters", "twoq_count", "total_gate_count", "depth"}
        )
        with self.assertRaisesRegex(EvaluationError, "at least 3"):
            EvaluationProtocol(audit_binding_count=2).validate()

    def test_redundant_raw_parameters_are_rejected_after_canonicalization(self) -> None:
        redundant = spec(
            [operation("X", [0], "a"), operation("X", [0], "b")]
        )
        problem = PublicProblem.create(
            num_qubits=2, pauli_terms=(PauliTerm("ZI", 1.0),)
        )
        result = evaluate_public_problem(problem, redundant)
        self.assertFalse(result.valid)
        self.assertIn("redundant", result.violations[0])

    def test_protocol_validation(self) -> None:
        with self.assertRaises(EvaluationError):
            EvaluationProtocol(transpile_optimization_level=True).validate()


if __name__ == "__main__":
    unittest.main()
