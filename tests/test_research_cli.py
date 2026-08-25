from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autovqe import research_cli
from autovqe.controller import MAX_EXTERNAL_ACTION_BYTES


def _problem() -> dict:
    return {
        "name": "one_qubit",
        "pauli_terms": [{"pauli": "Z", "coeff": 1.0}],
        "basis_gates": ["rz", "sx", "x", "cx"],
        "initial_state_hint": [0],
    }


def _candidate(*, pauli: str = "Y", name: str = "single_rotation") -> dict:
    return {
        "version": 1,
        "name": name,
        "num_qubits": 1,
        "parameters": ["theta"],
        "operations": [
            {
                "macro": "PauliRotation",
                "qubits": [0],
                "parameters": {"angle": {"parameter": "theta"}},
                "options": {"pauli": pauli},
            }
        ],
    }


def _null_candidate() -> dict:
    return {
        "version": 1,
        "name": "diagnostic_control",
        "num_qubits": 1,
        "parameters": [],
        "operations": [],
    }


class ResearchCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.problem_path = self.root / "problem.json"
        self.problem_path.write_text(
            json.dumps(_problem(), indent=2) + "\n",
            encoding="utf-8",
        )
        self.run_dir = self.root / "run"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _init(self) -> dict:
        return research_cli.initialize_run(
            self.problem_path,
            self.run_dir,
            total_budget=20.0,
        )

    def test_init_is_compact_and_keeps_a_problem_snapshot_for_drift_checks(self) -> None:
        original = _problem()
        original["source_note"] = "public provenance"
        self.problem_path.write_text(
            json.dumps(original, indent=2) + "\n",
            encoding="utf-8",
        )
        initialized = self._init()
        rendered = research_cli.render_json(initialized)
        self.assertNotIn("pauli_terms", rendered)
        self.assertNotIn("optimized_parameter_binding", rendered)

        observation = json.loads(
            (self.run_dir / research_cli.OBSERVATION_FILE).read_text()
        )
        snapshot = json.loads(
            (self.run_dir / research_cli.PROBLEM_FILE).read_text()
        )
        self.assertNotIn("pauli_terms", observation)
        self.assertEqual(snapshot["pauli_terms"][0]["pauli"], "Z")
        self.assertEqual(snapshot, original)
        self.assertIn("coeff", snapshot["pauli_terms"][0])
        self.assertNotIn("real", snapshot["pauli_terms"][0])

        context = json.loads((self.run_dir / research_cli.RUN_FILE).read_text())
        self.assertEqual(
            set(context),
            {"schema_version", "problem_path", "total_budget"},
        )
        self.assertEqual(research_cli.run_status(self.run_dir)["events"], 0)
        self.assertEqual(initialized["state"]["hypotheses"], {})
        self.assertEqual(initialized["state"]["candidates"], {})
        with self.assertRaisesRegex(research_cli.ResearchCliError, "already exists"):
            self._init()

    def test_changed_problem_is_rejected(self) -> None:
        self._init()
        changed = _problem()
        changed["pauli_terms"][0]["coeff"] = 2.0
        self.problem_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "changed"):
            research_cli.run_status(self.run_dir)

    def test_model_equivalent_raw_problem_change_is_rejected(self) -> None:
        original = _problem()
        original["source_note"] = "first provenance"
        self.problem_path.write_text(json.dumps(original), encoding="utf-8")
        self._init()

        changed = _problem()
        changed["source_note"] = "different provenance"
        self.problem_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "changed"):
            research_cli.run_status(self.run_dir)

    def test_action_file_rejects_invalid_json_shapes_and_size(self) -> None:
        self._init()
        action = self.root / "action.json"

        action.write_text(
            json.dumps(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "bom",
                    "claim": {"kind": "ansatz_structure", "family": "bom"},
                }
            ),
            encoding="utf-8-sig",
        )
        accepted = research_cli.execute_action_file(self.run_dir, action)
        self.assertTrue(accepted["result"]["accepted"])

        action.write_text('{"type":"x","type":"y"}', encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "duplicate"):
            research_cli.execute_action_file(self.run_dir, action)

        action.write_text('{"cost":NaN}', encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "non-finite"):
            research_cli.execute_action_file(self.run_dir, action)

        action.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "JSON object"):
            research_cli.execute_action_file(self.run_dir, action)

        action.write_bytes(b" " * (MAX_EXTERNAL_ACTION_BYTES + 1))
        with self.assertRaisesRegex(research_cli.ResearchCliError, "exceeds"):
            research_cli.execute_action_file(self.run_dir, action)

        with self.assertRaisesRegex(research_cli.ResearchCliError, "regular JSON file"):
            research_cli.execute_action_file(self.run_dir, self.root)

    def test_closed_loop_exposes_parameters_only_in_terminal_result(self) -> None:
        self._init()
        actions = [
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "rotation",
                "claim": {
                    "kind": "ansatz_structure",
                    "family": "single Y rotation",
                },
            },
            {
                "type": "submit_candidate",
                "candidate_id": "candidate",
                "hypothesis_id": "rotation",
                "spec": _candidate(),
                "metadata": {
                    "prediction": "rotation lowers the energy from its zero-angle baseline",
                },
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
            },
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "control_branch",
                "claim": {"kind": "null_control"},
            },
            {
                "type": "submit_candidate",
                "candidate_id": "control",
                "hypothesis_id": "control_branch",
                "spec": _null_candidate(),
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "control",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "control",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "control",
            },
        ]
        for action in actions:
            output = research_cli.execute_action(self.run_dir, action)
            self.assertNotIn(
                "optimized_parameter_binding",
                research_cli.render_json(output),
            )

        status = research_cli.run_status(self.run_dir)
        rendered_status = research_cli.render_json(status)
        self.assertNotIn("optimized_parameter_binding", rendered_status)
        self.assertNotIn('"spec"', rendered_status)
        self.assertNotIn('"probes"', rendered_status)
        self.assertNotIn('"evaluations"', rendered_status)
        candidate_summary = status["state"]["candidates"]["candidate"]
        self.assertEqual(
            candidate_summary["next_action"],
            "commit_or_dispose_after_comparison",
        )
        self.assertEqual(
            candidate_summary["latest_evaluation"]["evaluation_id"],
            "evaluation:candidate:promotion",
        )
        self.assertEqual(
            candidate_summary["audit_summary"]["evaluation_id"],
            "evaluation:candidate:audit",
        )
        self.assertNotIn("evaluation:candidate:smoke", rendered_status)
        self.assertEqual(
            status["state"]["candidates"]["control"]["latest_evaluation"]["stage"],
            "promotion",
        )
        full_status = research_cli.run_status(self.run_dir, full=True)
        self.assertIn('"spec"', research_cli.render_json(full_status))
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(full_status),
        )
        history_text = (self.run_dir / research_cli.HISTORY_FILE).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("optimized_parameter_binding", history_text)
        with self.assertRaisesRegex(research_cli.ResearchCliError, "not terminal"):
            research_cli.run_result(self.run_dir)

        research_cli.execute_action(
            self.run_dir,
            {"type": "commit", "candidate_id": "candidate"},
        )
        result = research_cli.run_result(self.run_dir)
        self.assertEqual(result["decision"], "positive_commit")
        self.assertEqual(result["candidate_id"], "candidate")
        self.assertLess(result["energy"], -0.9)
        self.assertEqual(set(result["optimized_parameters"]), {"theta"})
        self.assertIn("unique_trainable_params", result["audit"])
        self.assertTrue(result["resources"])
        self.assertEqual(
            result["comparison"]["mode"],
            "evaluated_competitor",
        )
        self.assertEqual(
            result["evidence"]["evaluation:candidate:promotion"]["kind"],
            "evaluation",
        )
        self.assertIsNone(result["reference_score"])
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(result["evidence"]),
        )

    def test_compact_status_keeps_only_the_latest_probe_on_its_branch(self) -> None:
        problem = {
            "name": "two_qubit_probe",
            "pauli_terms": [{"pauli": "ZZ", "coeff": 1.0}],
            "basis_gates": ["rz", "sx", "x", "cx"],
            "initial_state_hint": [0, 0],
        }
        self.problem_path.write_text(json.dumps(problem) + "\n", encoding="utf-8")
        self._init()
        research_cli.execute_action(
            self.run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "total_z",
                "claim": {
                    "kind": "exact_pauli_symmetry",
                    "generator": {
                        "type": "global_pauli_sum",
                        "pauli": "Z",
                        "selector": "all_sites",
                    },
                },
            },
        )
        research_cli.execute_action(
            self.run_dir,
            {"type": "request_probe", "hypothesis_id": "total_z"},
        )

        status = research_cli.run_status(self.run_dir)
        rendered = research_cli.render_json(status)
        branch = status["state"]["hypotheses"]["total_z"]

        self.assertNotIn('"probes"', rendered)
        self.assertEqual(branch["latest_probe"]["probe_id"], "probe:total_z")
        self.assertEqual(branch["latest_probe"]["verdict"], "supported")
        self.assertIn("residual", branch["latest_probe"]["metrics"])

    def test_negative_result_includes_the_cited_evaluator_record(self) -> None:
        problem = {
            "name": "two_qubit_ground_state",
            "pauli_terms": [
                {"pauli": "ZI", "coeff": -1.0},
                {"pauli": "IZ", "coeff": -1.0},
            ],
            "basis_gates": ["rz", "sx", "x", "cx"],
            "initial_state_hint": [0, 0],
        }
        self.problem_path.write_text(json.dumps(problem) + "\n", encoding="utf-8")
        self._init()

        def active_candidate(pauli: str, qubits: list[int], name: str) -> dict:
            return {
                "version": 1,
                "name": name,
                "num_qubits": 2,
                "parameters": ["theta"],
                "operations": [
                    {
                        "macro": "PauliRotation",
                        "qubits": qubits,
                        "parameters": {"angle": {"parameter": "theta"}},
                        "options": {"pauli": pauli},
                    }
                ],
            }

        actions = [
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "rotation_a",
                "claim": {
                    "kind": "ansatz_structure",
                    "family": "active family A",
                },
            },
            {
                "type": "submit_candidate",
                "candidate_id": "candidate_a",
                "hypothesis_id": "rotation_a",
                "spec": active_candidate("X", [0], "active_rotation_a"),
                "metadata": {
                    "prediction": "rotation lowers the energy",
                },
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate_a",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate_a",
            },
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "rotation_b",
                "claim": {
                    "kind": "ansatz_structure",
                    "family": "active family B",
                },
            },
            {
                "type": "submit_candidate",
                "candidate_id": "candidate_b",
                "hypothesis_id": "rotation_b",
                "spec": active_candidate("Y", [1], "active_rotation_b"),
                "metadata": {
                    "prediction": "the independent structure lowers the energy",
                },
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate_b",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate_b",
            },
            {
                "type": "retire",
                "entity": "hypothesis",
                "entity_id": "rotation_a",
                "reason": "no surviving candidate",
            },
            {
                "type": "retire",
                "entity": "hypothesis",
                "entity_id": "rotation_b",
                "reason": "no surviving candidate",
            },
            {
                "type": "close_negative",
                "reason": "no branch survived the local rule",
            },
        ]
        for action in actions:
            research_cli.execute_action(self.run_dir, action)

        result = research_cli.run_result(self.run_dir)
        self.assertEqual(result["decision"], "negative_close")
        self.assertIn("closes only the recorded investigated branches", result["scope"])
        self.assertNotIn("promotion rule", result["scope"])
        self.assertEqual(
            result["evidence"]["evaluation:candidate_a:smoke"]["kind"],
            "evaluation",
        )
        self.assertEqual(
            result["evidence"]["evaluation:candidate_a:smoke"]["candidate_id"],
            "candidate_a",
        )
        self.assertEqual(len(result["coverage"]["numerical_candidate_ids"]), 2)
        self.assertEqual(len(result["coverage"]["structure_lineage_ids"]), 2)
        self.assertEqual(result["coverage"]["search_mode"], "structural_breadth")
        self.assertGreater(
            result["evidence"]["evaluation:candidate_a:smoke"][
                "metrics"
            ]["objective_activity_fraction"],
            1e-6,
        )
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(result),
        )


if __name__ == "__main__":
    unittest.main()
