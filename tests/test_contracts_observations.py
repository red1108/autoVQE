from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    InitialStateSpec,
    PauliTerm,
    PublicProblem,
    SectorSpec,
    canonical_data,
    canonical_json,
)
from autovqe.observations import observe_problem
from autovqe.problem import (
    hamiltonian_from_problem,
    load_problem,
    load_problem_document,
)


def _payload() -> dict:
    return {
        "name": "small_test_problem",
        "pauli_terms": [
            {"pauli": "II", "coeff": -1.052373245772859},
            {"pauli": "IZ", "coeff": 0.39793742484318045},
            {"pauli": "ZI", "coeff": -0.39793742484318045},
            {"pauli": "ZZ", "coeff": -0.01128010425623538},
            {"pauli": "XX", "coeff": 0.18093119978423156},
        ],
        "basis_gates": ["rx", "ry", "rz", "cx"],
        "coupling_map": [[0, 1], [1, 0]],
        "initial_state_hint": [1, 0],
    }


def _load(payload: dict | None = None) -> PublicProblem:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "problem.json"
        path.write_text(json.dumps(payload or _payload()), encoding="utf-8")
        return load_problem(path)


class CanonicalSerializationTests(unittest.TestCase):
    def test_canonical_json_is_sorted_compact_and_normalizes_negative_zero(self) -> None:
        value = {"z": -0.0, "a": [2, 1], "middle": {"b": True, "a": None}}
        self.assertEqual(
            canonical_json(value),
            '{"a":[2,1],"middle":{"a":null,"b":true},"z":0.0}',
        )

    def test_canonical_json_rejects_non_finite_numbers(self) -> None:
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    canonical_json({"value": value})


class ProblemContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = _load()

    def test_loader_returns_the_single_validated_problem_model(self) -> None:
        self.assertIsInstance(self.problem, PublicProblem)
        self.assertEqual(self.problem.problem_id, "small_test_problem")
        self.assertEqual(self.problem.num_qubits, 2)
        self.assertEqual(self.problem.initial_state.occupation, (1, 0))
        self.assertEqual(
            self.problem.backend.basis_gates,
            ("rx", "ry", "rz", "cx"),
        )
        self.assertEqual(self.problem.backend.coupling_map, ((0, 1), (1, 0)))
        self.assertFalse(hasattr(self.problem, "reference_energy"))
        self.assertFalse(hasattr(self.problem, "reference_state"))

    def test_problem_loader_accepts_a_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8-sig")
            loaded = load_problem(path)
        self.assertEqual(loaded.problem_id, "small_test_problem")
        self.assertEqual(hamiltonian_from_problem(self.problem).num_qubits, 2)

    def test_observation_is_compact_and_does_not_embed_pauli_terms(self) -> None:
        observation = observe_problem(self.problem)
        payload = canonical_data(observation)

        self.assertNotIn("pauli_terms", payload)
        self.assertEqual(payload["problem_id"], self.problem.problem_id)
        self.assertEqual(payload["structure"]["term_count"], 5)

    def test_structural_observation_is_mechanical(self) -> None:
        structure = observe_problem(self.problem).structure
        self.assertEqual(structure.term_count, 5)
        self.assertEqual(structure.identity_term_count, 1)
        self.assertEqual(structure.max_locality, 2)
        self.assertIn((2, 2), structure.locality_counts)
        self.assertEqual(structure.support_graph_edge_count, 1)
        self.assertEqual(structure.support_graph_degrees, ((0, 1), (1, 1)))
        self.assertEqual(structure.support_graph_components, ((0, 1),))
        self.assertTrue(structure.support_graph_edges_complete)
        self.assertEqual(structure.support_graph_edges, ((0, 1),))

    def test_large_support_graph_is_summarized_without_partial_edges(self) -> None:
        num_qubits = 12
        terms = []
        for left in range(num_qubits):
            for right in range(left + 1, num_qubits):
                label = ["I"] * num_qubits
                label[num_qubits - left - 1] = "Z"
                label[num_qubits - right - 1] = "Z"
                terms.append(PauliTerm("".join(label), 1.0))
        problem = PublicProblem.create(
            problem_id="dense_support",
            num_qubits=num_qubits,
            pauli_terms=terms,
        )

        structure = observe_problem(problem).structure

        self.assertEqual(structure.support_graph_edge_count, 66)
        self.assertEqual(
            structure.support_graph_degrees,
            tuple((qubit, 11) for qubit in range(num_qubits)),
        )
        self.assertEqual(
            structure.support_graph_components,
            (tuple(range(num_qubits)),),
        )
        self.assertFalse(structure.support_graph_edges_complete)
        self.assertEqual(structure.support_graph_edges, ())

    def test_loader_accepts_only_typed_v1_symmetry_metadata(self) -> None:
        payload = _payload()
        payload["symmetry"] = {
            "active_electrons": 1,
            "active_orbitals": 1,
            "spin_orbitals": 2,
            "mapping": "jordan_wigner",
        }
        problem = _load(payload)

        self.assertEqual(problem.encoding.mapping, "jordan_wigner")
        self.assertEqual(problem.encoding.active_orbitals, 1)
        self.assertEqual(problem.encoding.spin_orbitals, 2)
        self.assertEqual(problem.sector.values, (("particle_number", 1),))

    def test_unknown_answer_and_dominant_occupation_fields_are_rejected(self) -> None:
        cases = [
            ("reference_energy", -999.0),
            ("reference_state", [1.0, 0.0]),
            ("expected_ansatz", {"name": "answer"}),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                payload = _payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, "invalid problem document"):
                    _load(payload)

        payload = _payload()
        payload["symmetry"] = {"dominant_occupation": "10"}
        with self.assertRaisesRegex(ValueError, "invalid symmetry fields"):
            _load(payload)

    def test_invalid_top_level_and_symmetry_field_types_are_rejected(self) -> None:
        top_level_cases = {
            "name": 7,
            "source_note": False,
            "basis_gates": "rx,ry",
            "coupling_map": None,
            "initial_state_hint": None,
        }
        for field, value in top_level_cases.items():
            with self.subTest(field=field):
                payload = _payload()
                payload[field] = value
                with self.assertRaises(ValueError):
                    _load(payload)

        symmetry_cases = {
            "symmetry": [],
            "mapping": 3,
            "spin_orbitals": 2.0,
            "active_electrons": True,
            "spin_projection": "zero",
            "parity": [],
        }
        for field, value in symmetry_cases.items():
            with self.subTest(field=field):
                payload = _payload()
                if field == "symmetry":
                    payload[field] = value
                else:
                    payload["symmetry"] = {field: value}
                with self.assertRaises(ValueError):
                    _load(payload)

    def test_document_loader_returns_validated_canonical_raw_input(self) -> None:
        payload = _payload()
        payload["source_note"] = "public provenance only"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            problem, document = load_problem_document(path)

        self.assertIsInstance(problem, PublicProblem)
        self.assertEqual(document, canonical_data(payload))
        self.assertIn("coeff", document["pauli_terms"][0])
        self.assertNotIn("real", document["pauli_terms"][0])

    def test_duplicate_pauli_terms_are_combined(self) -> None:
        payload = _payload()
        payload["pauli_terms"] = [
            {"pauli": "Z", "coeff": 1.0},
            {"pauli": "Z", "coeff": 2.0},
            {"pauli": "I", "coeff": -0.5},
        ]
        payload["initial_state_hint"] = [0]
        payload["coupling_map"] = []
        problem = _load(payload)

        self.assertEqual(
            problem.pauli_terms,
            (PauliTerm("I", -0.5), PauliTerm("Z", 3.0)),
        )

    def test_public_problem_validates_initial_state_and_backend_bounds(self) -> None:
        common = {
            "num_qubits": 2,
            "pauli_terms": (PauliTerm("ZI", 1.0),),
            "encoding": EncodingSpec(),
            "sector": SectorSpec(),
        }
        with self.assertRaisesRegex(ValueError, "occupation"):
            PublicProblem.create(
                **common,
                initial_state=InitialStateSpec(
                    kind="computational_basis",
                    occupation=(1,),
                ),
                backend=BackendSpec(),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            PublicProblem.create(
                **common,
                initial_state=InitialStateSpec(),
                backend=BackendSpec(coupling_map=((0, 2),)),
            )
        with self.assertRaisesRegex(ValueError, "qiskit_little_endian"):
            EncodingSpec(qubit_order="big_endian")


if __name__ == "__main__":
    unittest.main()
