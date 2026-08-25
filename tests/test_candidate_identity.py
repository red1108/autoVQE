from __future__ import annotations

import copy
import unittest

from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
    SectorSpec,
)
from autovqe.evaluator import EvaluationProtocol, candidate_identity, evaluate_public_problem


def _spec() -> dict:
    return {
        "version": 1,
        "name": "presentation_a",
        "num_qubits": 2,
        "parameters": ["left", "right"],
        "reference": None,
        "layers": [
            {
                "name": "display_layer",
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
        ],
    }


class CandidateIdentityTests(unittest.TestCase):
    def test_cosmetic_and_alpha_renaming_changes_share_one_identity(self) -> None:
        original = _spec()
        renamed = copy.deepcopy(original)
        renamed["name"] = "presentation_b"
        renamed["parameters"] = [{"name": "beta"}, {"name": "alpha"}]
        renamed["layers"][0]["name"] = "another_label"
        renamed["layers"][0]["operations"][0]["parameters"]["angle"][
            "parameter"
        ] = "beta"
        renamed["layers"][0]["operations"][1]["parameters"]["angle"][
            "parameter"
        ] = "alpha"
        self.assertEqual(candidate_identity(original), candidate_identity(renamed))

        problem = PublicProblem.create(
            num_qubits=2,
            pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 1.0)),
            encoding=EncodingSpec(),
            sector=SectorSpec(),
            reference=ReferenceSpec(),
            backend=BackendSpec(),
        )
        protocol = EvaluationProtocol(max_evals=8, restarts=1, seed=19)
        first = evaluate_public_problem(problem, original, protocol=protocol).receipt
        second = evaluate_public_problem(problem, renamed, protocol=protocol).receipt
        self.assertEqual(first.energy_trace, second.energy_trace)
        self.assertEqual(first.metrics, second.metrics)

    def test_physical_operation_order_changes_the_identity(self) -> None:
        original = _spec()
        original["layers"][0]["operations"][1]["qubits"] = [0]
        reordered = copy.deepcopy(original)
        reordered["layers"][0]["operations"].reverse()
        self.assertNotEqual(candidate_identity(original), candidate_identity(reordered))

    def test_parameter_sharing_changes_the_identity(self) -> None:
        original = _spec()
        shared = copy.deepcopy(original)
        shared["parameters"] = ["shared"]
        for operation in shared["layers"][0]["operations"]:
            operation["parameters"]["angle"]["parameter"] = "shared"
        self.assertNotEqual(candidate_identity(original), candidate_identity(shared))


if __name__ == "__main__":
    unittest.main()
