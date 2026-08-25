from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path

from autovqe import prepare
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    FORBIDDEN_AGENT_KEYS,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
    SectorSpec,
    assert_agent_safe,
    canonical_json,
)
from autovqe.observations import adapt_prepare_problem, public_problem_from_prepare


ROOT = Path(__file__).resolve().parents[1]


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

    def test_agent_safety_walks_nested_mappings(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_energy"):
            assert_agent_safe({"nested": [{"reference_energy": -1.0}]})
        with self.assertRaisesRegex(ValueError, "model_class"):
            assert_agent_safe({"model_class": "memorized"})


class ProblemContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = prepare.load_problem(ROOT / "examples" / "h2_2q.json")

    def test_adapter_splits_public_private_and_agent_safe_views(self) -> None:
        views = adapt_prepare_problem(self.problem)

        self.assertEqual(views.public, views.public_problem)
        self.assertEqual(views.private, views.private_context)
        self.assertEqual(views.safe, views.observation_bundle)
        self.assertEqual(views.public_problem.problem_id, self.problem.name)
        self.assertEqual(views.public_problem.num_qubits, 2)
        self.assertEqual(views.public_problem.reference.occupation, (1, 0))
        self.assertEqual(views.public_problem.backend.basis_gates, ("rx", "ry", "rz", "cx"))
        self.assertEqual(views.public_problem.backend.coupling_map, ((0, 1), (1, 0)))

        self.assertEqual(views.private_context.source_name, self.problem.name)
        self.assertEqual(views.private_context.reference_energy, self.problem.reference_energy)
        self.assertEqual(
            views.private_context.reference_state,
            tuple(complex(value) for value in self.problem.reference_state),
        )

        payload = views.observation_bundle.to_canonical_json()
        lowered = payload.lower()
        for forbidden in FORBIDDEN_AGENT_KEYS:
            self.assertNotIn(f'"{forbidden}"', lowered)
        self.assertIn(self.problem.name, payload)
        self.assertFalse(hasattr(views.observation_bundle, "model_class"))
        self.assertFalse(hasattr(views.observation_bundle, "recommendation"))

    def test_public_problem_name_and_observation_json_are_stable(self) -> None:
        first = adapt_prepare_problem(self.problem)
        second_problem = prepare.load_problem(ROOT / "examples" / "h2_2q.json")
        second = adapt_prepare_problem(second_problem)

        self.assertEqual(first.public_problem.problem_id, second.public_problem.problem_id)
        self.assertEqual(
            first.observation_bundle.to_canonical_json(),
            second.observation_bundle.to_canonical_json(),
        )

    def test_structural_observation_is_mechanical_not_recommendational(self) -> None:
        structure = adapt_prepare_problem(self.problem).observation_bundle.structure
        self.assertEqual(structure.term_count, len(self.problem.hamiltonian.paulis))
        self.assertEqual(structure.identity_term_count, 1)
        self.assertEqual(structure.max_locality, 2)
        self.assertIn((2, 2), structure.locality_counts)
        self.assertEqual(structure.support_graph_edges, ((0, 1),))
        self.assertFalse(structure.has_complex_coefficients)

    def test_only_allowlisted_symmetry_metadata_crosses_public_boundary(self) -> None:
        poisoned = replace(
            self.problem,
            symmetry={
                "active_electrons": 1,
                "active_orbitals": 1,
                "spin_orbitals": 2,
                "mapping": "jordan_wigner",
                "model_class": "fixture_special_case",
                "recommendation": "initialize_exact_state",
                "reference_energy": -999.0,
                "reference_state": [1.0, 0.0, 0.0, 0.0],
            },
        )
        public = public_problem_from_prepare(poisoned)
        payload = canonical_json(public)

        self.assertEqual(public.encoding.mapping, "jordan_wigner")
        self.assertEqual(public.encoding.active_orbitals, 1)
        self.assertEqual(public.encoding.spin_orbitals, 2)
        self.assertEqual(public.sector.values, (("particle_number", 1),))
        for forbidden in FORBIDDEN_AGENT_KEYS:
            self.assertNotIn(f'"{forbidden}"', payload.lower())
        self.assertNotIn("fixture_special_case", payload)
        self.assertNotIn("initialize_exact_state", payload)

    def test_private_context_is_rejected_by_agent_safety_guard(self) -> None:
        private_context = adapt_prepare_problem(self.problem).private_context
        with self.assertRaisesRegex(ValueError, "reference_energy"):
            assert_agent_safe(private_context)

    def test_public_problem_validates_reference_and_backend_bounds(self) -> None:
        common = {
            "num_qubits": 2,
            "pauli_terms": (PauliTerm("ZI", 1.0),),
            "encoding": EncodingSpec(),
            "sector": SectorSpec(),
        }
        with self.assertRaisesRegex(ValueError, "occupation"):
            PublicProblem.create(
                **common,
                reference=ReferenceSpec(kind="computational_basis", occupation=(1,)),
                backend=BackendSpec(),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            PublicProblem.create(
                **common,
                reference=ReferenceSpec(),
                backend=BackendSpec(coupling_map=((0, 2),)),
            )
        with self.assertRaisesRegex(ValueError, "qiskit_little_endian"):
            EncodingSpec(qubit_order="big_endian")


if __name__ == "__main__":
    unittest.main()
