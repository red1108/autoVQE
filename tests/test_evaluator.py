from __future__ import annotations

import copy
import unittest

from qiskit.quantum_info import SparsePauliOp

from autovqe.problem import BackendSpec, InitialStateSpec, PauliTerm, PublicProblem
from autovqe.evaluator import (
    EvaluationError,
    EvaluationProtocol,
    audit_public_candidate,
    candidate_identity,
    evaluate_ansatz,
    evaluate_public_problem,
)


def operation(
    pauli: str, qubits: list[int], parameter: str, coefficient: float = 1.0
) -> dict:
    return {
        "macro": "PauliRotation",
        "qubits": qubits,
        "parameters": {
            "angle": {"parameter": parameter, "coefficient": coefficient}
        },
        "options": {"pauli": pauli},
    }


def spec(parameters: list[str], operations: list[dict], name: str = "candidate") -> dict:
    return {
        "version": 1,
        "name": name,
        "num_qubits": 2,
        "parameters": parameters,
        "operations": operations,
    }


class LeanEvaluatorIdentityTests(unittest.TestCase):
    def test_semantic_representations_share_one_identity(self) -> None:
        base = spec(
            ["a", "b"],
            [operation("X", [0], "a"), operation("Y", [1], "b")],
        )
        declarations = copy.deepcopy(base)
        declarations["parameters"].reverse()
        reordered = copy.deepcopy(base)
        reordered["operations"].reverse()
        scaled = copy.deepcopy(base)
        for item in scaled["operations"]:
            item["parameters"]["angle"]["coefficient"] = -2.0
        self.assertEqual(candidate_identity(base), candidate_identity(declarations))
        self.assertEqual(candidate_identity(base), candidate_identity(reordered))
        self.assertEqual(candidate_identity(base), candidate_identity(scaled))

        supported = spec(["theta"], [operation("XY", [0, 1], "theta")])
        reversed_support = spec(
            ["theta"], [operation("YX", [1, 0], "theta")]
        )
        self.assertEqual(
            candidate_identity(supported), candidate_identity(reversed_support)
        )

    def test_macro_expansion_and_rotation_splits_are_not_fresh_families(self) -> None:
        exchange = spec(
            ["theta"],
            [
                {
                    "macro": "XYExchange",
                    "qubits": [1, 0],
                    "parameters": {"angle": {"parameter": "theta"}},
                    "options": {},
                }
            ],
        )
        expanded = spec(
            ["theta"],
            [
                operation("XX", [0, 1], "theta"),
                operation("YY", [0, 1], "theta"),
            ],
        )
        unsplit = spec(["theta"], [operation("X", [0], "theta")])
        split = spec(
            ["theta"],
            [
                operation("X", [0], "theta", 0.5),
                operation("X", [0], "theta", 0.5),
            ],
        )
        cancellation_retry = spec(
            ["theta"],
            [
                operation("X", [0], "theta"),
                operation("X", [0], "theta"),
                operation("X", [0], "theta", -1.0),
            ],
        )
        self.assertEqual(candidate_identity(exchange), candidate_identity(expanded))
        self.assertEqual(candidate_identity(unsplit), candidate_identity(split))
        self.assertEqual(
            candidate_identity(unsplit), candidate_identity(cancellation_retry)
        )

    def test_noncommuting_order_and_parameter_sharing_remain_distinct(self) -> None:
        ordered = spec(
            ["a", "b"],
            [operation("X", [0], "a"), operation("Z", [0], "b")],
        )
        reversed_order = copy.deepcopy(ordered)
        reversed_order["operations"].reverse()
        shared = spec(
            ["shared"],
            [operation("X", [0], "shared"), operation("Y", [1], "shared")],
        )
        independent = spec(
            ["a", "b"],
            [operation("X", [0], "a"), operation("Y", [1], "b")],
        )
        self.assertNotEqual(
            candidate_identity(ordered), candidate_identity(reversed_order)
        )
        self.assertNotEqual(candidate_identity(shared), candidate_identity(independent))

    def test_declaration_order_does_not_change_the_optimizer_trace(self) -> None:
        candidate = spec(
            ["a", "b"],
            [operation("X", [0], "a"), operation("Y", [1], "b")],
        )
        reordered = copy.deepcopy(candidate)
        reordered["parameters"].reverse()
        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 0.3)),
        )
        protocol = EvaluationProtocol(max_evals=12, restarts=1, seed=19)
        first = evaluate_public_problem(problem, candidate, protocol=protocol).result
        second = evaluate_public_problem(problem, reordered, protocol=protocol).result
        self.assertEqual(first.trace_summary, second.trace_summary)
        self.assertEqual(
            first.optimized_parameter_binding, second.optimized_parameter_binding
        )

    def test_global_parameter_rescaling_does_not_change_evaluation(self) -> None:
        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("IZ", 1.0), PauliTerm("IX", 0.2)),
        )
        protocol = EvaluationProtocol(max_evals=12, restarts=1, seed=19)
        coefficients = (1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
        candidates = [
            spec(["theta"], [operation("Y", [0], "theta", coefficient)])
            for coefficient in coefficients
        ]
        self.assertEqual(len({candidate_identity(item) for item in candidates}), 1)

        results = [
            evaluate_public_problem(problem, item, protocol=protocol).result
            for item in candidates
        ]
        reference = results[0]
        self.assertTrue(reference.valid, reference.violations)
        reference_binding = reference.optimized_parameter_binding
        self.assertIsNotNone(reference_binding)
        assert reference_binding is not None
        reference_angle = coefficients[0] * reference_binding["theta"]
        for coefficient, result in zip(coefficients[1:], results[1:], strict=True):
            self.assertTrue(result.valid, result.violations)
            self.assertEqual(result.trace_summary, reference.trace_summary)
            self.assertEqual(result.best_energy, reference.best_energy)
            self.assertEqual(result.metrics, reference.metrics)
            binding = result.optimized_parameter_binding
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertAlmostEqual(coefficient * binding["theta"], reference_angle)


class LeanEvaluatorBoundaryTests(unittest.TestCase):
    def test_three_identity_seeded_resource_bindings_are_required(self) -> None:
        candidate = spec(["theta"], [operation("XY", [0, 1], "theta")])
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
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.metrics["audit_binding_count"], 3)
        with self.assertRaisesRegex(EvaluationError, "at least 3"):
            EvaluationProtocol(audit_binding_count=2).validate()

    def test_redundant_physical_parameters_are_rejected(self) -> None:
        redundant = spec(
            ["a", "b"],
            [operation("X", [0], "a"), operation("X", [0], "b")],
        )
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("ZI", 1.0)]), redundant
        ).result
        self.assertFalse(result.valid)
        self.assertIn("redundant", result.violations[0])

    def test_protocol_hamiltonian_and_occupation_validation(self) -> None:
        with self.assertRaises(EvaluationError):
            EvaluationProtocol(transpile_optimization_level=True).validate()
        empty = {"version": 1, "num_qubits": 1, "parameters": [], "operations": []}
        wrong_occupation = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", 1.0)]),
            empty,
            initial_occupation=(0, 0),
        ).result
        bool_occupation = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", 1.0)]),
            empty,
            initial_occupation=(False,),
        ).result
        non_hermitian = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", 1j)]), empty
        ).result
        self.assertFalse(wrong_occupation.valid)
        self.assertFalse(bool_occupation.valid)
        self.assertIn("Hermitian", non_hermitian.violations[0])


if __name__ == "__main__":
    unittest.main()
