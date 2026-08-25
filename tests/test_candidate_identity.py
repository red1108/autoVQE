from __future__ import annotations

import copy
import unittest

from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    InitialStateSpec,
    SectorSpec,
)
from autovqe.evaluator import EvaluationProtocol, candidate_identity, evaluate_public_problem


def _spec() -> dict:
    return {
        "version": 1,
        "name": "presentation_a",
        "num_qubits": 2,
        "parameters": ["left", "right"],
        "operations": [
            {
                "macro": "PauliRotation",
                "qubits": [0],
                "parameters": {
                    "angle": {"parameter": "left", "coefficient": 1}
                },
                "options": {"pauli": "X"},
            },
            {
                "macro": "PauliRotation",
                "qubits": [1],
                "parameters": {
                    "angle": {"parameter": "right", "coefficient": 1.0}
                },
                "options": {"pauli": "Y"},
            },
        ],
    }


class CandidateIdentityTests(unittest.TestCase):
    def test_cosmetic_and_alpha_renaming_changes_share_one_identity(self) -> None:
        original = _spec()
        renamed = copy.deepcopy(original)
        renamed["name"] = "presentation_b"
        renamed["parameters"] = [{"name": "beta"}, {"name": "alpha"}]
        renamed["operations"][0]["parameters"]["angle"][
            "parameter"
        ] = "beta"
        renamed["operations"][1]["parameters"]["angle"][
            "parameter"
        ] = "alpha"
        self.assertEqual(candidate_identity(original), candidate_identity(renamed))

        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 1.0)),
            encoding=EncodingSpec(),
            sector=SectorSpec(),
            initial_state=InitialStateSpec(),
            backend=BackendSpec(),
        )
        protocol = EvaluationProtocol(max_evals=8, restarts=1, seed=19)
        first = evaluate_public_problem(problem, original, protocol=protocol).result
        second = evaluate_public_problem(problem, renamed, protocol=protocol).result
        self.assertEqual(first.trace_summary, second.trace_summary)
        self.assertEqual(first.metrics, second.metrics)

    def test_physical_operation_order_changes_the_identity(self) -> None:
        original = _spec()
        original["operations"][1]["qubits"] = [0]
        reordered = copy.deepcopy(original)
        reordered["operations"].reverse()
        self.assertNotEqual(candidate_identity(original), candidate_identity(reordered))

    def test_disjoint_operation_reordering_has_one_identity(self) -> None:
        original = _spec()
        reordered = copy.deepcopy(original)
        reordered["operations"].reverse()
        self.assertEqual(candidate_identity(original), candidate_identity(reordered))

    def test_overlapping_commuting_pauli_rotations_have_one_identity(self) -> None:
        original = _spec()
        original["operations"][0].update(
            {"qubits": [0], "options": {"pauli": "Z"}}
        )
        original["operations"][1].update(
            {"qubits": [0, 1], "options": {"pauli": "ZZ"}}
        )
        reordered = copy.deepcopy(original)
        reordered["operations"].reverse()
        self.assertEqual(candidate_identity(original), candidate_identity(reordered))

    def test_pauli_support_order_is_cosmetic(self) -> None:
        original = _spec()
        original["parameters"] = ["theta"]
        original["operations"] = [
            {
                "macro": "PauliRotation",
                "qubits": [0, 1],
                "parameters": {"angle": {"parameter": "theta"}},
                "options": {"pauli": "XY"},
            }
        ]
        reordered = copy.deepcopy(original)
        reordered["operations"][0]["qubits"] = [1, 0]
        reordered["operations"][0]["options"]["pauli"] = "YX"
        self.assertEqual(candidate_identity(original), candidate_identity(reordered))

    def test_exchange_macros_share_identity_with_their_pauli_generators(self) -> None:
        for macro, paulis in (
            ("XYExchange", ("XX", "YY")),
            ("IsotropicExchange", ("XX", "YY", "ZZ")),
        ):
            with self.subTest(macro=macro):
                shorthand = {
                    "version": 1,
                    "name": "shorthand",
                    "num_qubits": 2,
                    "parameters": ["theta"],
                    "operations": [
                        {
                            "macro": macro,
                            "qubits": [1, 0],
                            "parameters": {"angle": {"parameter": "theta"}},
                            "options": {},
                        }
                    ],
                }
                expanded = copy.deepcopy(shorthand)
                expanded["name"] = "expanded"
                expanded["operations"] = [
                    {
                        "macro": "PauliRotation",
                        "qubits": [0, 1],
                        "parameters": {"angle": {"parameter": "theta"}},
                        "options": {"pauli": pauli},
                    }
                    for pauli in paulis
                ]
                self.assertEqual(
                    candidate_identity(shorthand), candidate_identity(expanded)
                )

    def test_global_parameter_sign_and_scale_do_not_create_a_new_family(self) -> None:
        original = _spec()
        reparameterized = copy.deepcopy(original)
        for operation in reparameterized["operations"]:
            operation["parameters"]["angle"]["coefficient"] = -2.0
        self.assertEqual(
            candidate_identity(original),
            candidate_identity(reparameterized),
        )

    def test_invertible_parameter_mixing_does_not_create_a_new_family(self) -> None:
        original = _spec()
        mixed = copy.deepcopy(original)
        mixed["parameters"] = ["u", "v"]
        mixed["operations"][0]["parameters"]["angle"] = {
            "terms": [
                {"parameter": "u", "coefficient": 1.0},
                {"parameter": "v", "coefficient": 1.0},
            ],
            "constant": 0.0,
        }
        mixed["operations"][1]["parameters"]["angle"] = {
            "terms": [
                {"parameter": "u", "coefficient": 1.0},
                {"parameter": "v", "coefficient": -1.0},
            ],
            "constant": 0.0,
        }
        self.assertEqual(candidate_identity(original), candidate_identity(mixed))

    def test_parameter_sharing_changes_the_identity(self) -> None:
        original = _spec()
        shared = copy.deepcopy(original)
        shared["parameters"] = ["shared"]
        for operation in shared["operations"]:
            operation["parameters"]["angle"]["parameter"] = "shared"
        self.assertNotEqual(candidate_identity(original), candidate_identity(shared))


if __name__ == "__main__":
    unittest.main()
