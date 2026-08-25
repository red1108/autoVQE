from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autovqe.problem import (
    PauliTerm,
    canonical_data,
    hamiltonian_from_problem,
    load_problem,
    observe_problem,
)


def payload() -> dict:
    return {
        "name": "lean_test",
        "pauli_terms": [
            {"pauli": "II", "coeff": -1.0},
            {"pauli": "XX", "coeff": 0.25},
            {"pauli": "IZ", "coeff": 0.5},
        ],
        "basis_gates": ["rx", "ry", "rz", "cx"],
        "coupling_map": [[0, 1], [1, 0]],
        "initial_state_hint": [1, 0],
    }


def load(value: dict):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hamiltonian.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return load_problem(path)


class LeanProblemTests(unittest.TestCase):
    def test_loads_backend_state_and_compact_observation(self) -> None:
        problem = load(payload())
        self.assertEqual(problem.num_qubits, 2)
        self.assertEqual(problem.initial_occupation, (1, 0))
        self.assertEqual(problem.backend.coupling_map, ((0, 1), (1, 0)))
        self.assertEqual(hamiltonian_from_problem(problem).num_qubits, 2)

        observation = canonical_data(observe_problem(problem))
        self.assertNotIn("pauli_terms", observation)
        self.assertEqual(observation["structure"]["term_count"], 3)
        self.assertEqual(observation["structure"]["support_graph_edges"], [[0, 1]])

    def test_combines_and_sorts_duplicate_pauli_words(self) -> None:
        value = payload()
        value["pauli_terms"] = [
            {"pauli": "Z", "coeff": 1.0},
            {"pauli": "I", "coeff": -0.5},
            {"pauli": "Z", "coeff": 2.0},
        ]
        value["initial_state_hint"] = [0]
        value["coupling_map"] = []
        self.assertEqual(load(value).pauli_terms, (PauliTerm("I", -0.5), PauliTerm("Z", 3.0)))

    def test_rejects_unknown_answer_fields_and_null_hint(self) -> None:
        for field, bad in (("reference_energy", -2.0), ("expected_ansatz", {})):
            with self.subTest(field=field):
                value = payload()
                value[field] = bad
                with self.assertRaisesRegex(ValueError, "invalid problem document"):
                    load(value)
        value = payload()
        value["initial_state_hint"] = None
        with self.assertRaisesRegex(ValueError, "initial_state_hint"):
            load(value)

    def test_rejects_constant_only_hamiltonian(self) -> None:
        value = payload()
        value["pauli_terms"] = [{"pauli": "II", "coeff": 2.0}]
        with self.assertRaisesRegex(ValueError, "constant-only"):
            load(value)

    def test_rejects_duplicate_keys_nonfinite_and_unknown_symmetry(self) -> None:
        documents = (
            '{"name":"a","name":"b","pauli_terms":[]}',
            '{"name":"a","pauli_terms":[{"pauli":"Z","coeff":NaN}]}',
            '{"name":"a","pauli_terms":[{"pauli":"Z","coeff":1}],"symmetry":{"answer":1}}',
        )
        for document in documents:
            with self.subTest(document=document):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "hamiltonian.json"
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_problem(path)


if __name__ == "__main__":
    unittest.main()
