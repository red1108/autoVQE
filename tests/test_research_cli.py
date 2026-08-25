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


def _candidate() -> dict:
    return {
        "version": 1,
        "name": "single_rotation",
        "num_qubits": 1,
        "parameters": ["theta"],
        "reference": None,
        "operations": [
            {
                "macro": "PauliRotation",
                "qubits": [0],
                "parameters": {"angle": {"parameter": "theta"}},
                "options": {"pauli": "Y"},
            }
        ],
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

    def test_init_and_status_are_small_local_state_without_reference_answers(self) -> None:
        initialized = self._init()
        rendered = research_cli.render_json(initialized)
        self.assertNotIn("reference_energy", rendered)
        self.assertNotIn("reference_state", rendered)
        self.assertNotIn("optimized_parameter_binding", rendered)

        context = json.loads((self.run_dir / research_cli.RUN_FILE).read_text())
        self.assertEqual(
            set(context),
            {"schema_version", "problem_path", "total_budget"},
        )
        self.assertEqual(research_cli.run_status(self.run_dir)["events"], 0)
        with self.assertRaisesRegex(research_cli.ResearchCliError, "already exists"):
            self._init()

    def test_changed_problem_is_rejected(self) -> None:
        self._init()
        changed = _problem()
        changed["pauli_terms"][0]["coeff"] = 2.0
        self.problem_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(research_cli.ResearchCliError, "changed"):
            research_cli.run_status(self.run_dir)

    def test_action_file_rejects_invalid_json_shapes_and_size(self) -> None:
        self._init()
        action = self.root / "action.json"

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
                    "enforcement": "unconstrained",
                    "prediction": "rotation lowers the energy from its zero-angle baseline",
                },
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
                "evaluation_id": "candidate.audit",
                "stage": "audit",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
                "evaluation_id": "candidate.smoke",
                "stage": "smoke",
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
                "evaluation_id": "candidate.promotion",
                "stage": "promotion",
            },
        ]
        for action in actions:
            output = research_cli.execute_action(self.run_dir, action)
            self.assertNotIn(
                "optimized_parameter_binding",
                research_cli.render_json(output),
            )

        status = research_cli.run_status(self.run_dir)
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(status),
        )
        with self.assertRaisesRegex(research_cli.ResearchCliError, "not terminal"):
            research_cli.run_result(self.run_dir)

        research_cli.execute_action(
            self.run_dir,
            {
                "type": "commit",
                "candidate_id": "candidate",
                "evidence_ids": ["candidate.promotion"],
                "comparison": {
                    "mode": "documented_non_dominance",
                    "reason": "only eligible promoted candidate in this local branch",
                    "evidence_ids": ["candidate.promotion"],
                },
            },
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
            "documented_non_dominance",
        )
        self.assertEqual(
            result["evidence"]["candidate.promotion"]["kind"],
            "evaluation",
        )
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(result["evidence"]),
        )

    def test_negative_result_includes_the_cited_evaluator_record(self) -> None:
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
                    "enforcement": "unconstrained",
                    "prediction": "rotation lowers the energy",
                },
            },
            {
                "type": "evaluate_candidate",
                "candidate_id": "candidate",
                "evaluation_id": "candidate.audit",
                "stage": "audit",
            },
            {
                "type": "retire",
                "entity": "candidate",
                "entity_id": "candidate",
                "reason": "branch is not competitive",
            },
            {
                "type": "retire",
                "entity": "hypothesis",
                "entity_id": "rotation",
                "reason": "no surviving candidate",
            },
            {
                "type": "close_negative",
                "reason": "no branch survived the local rule",
                "evidence_ids": ["candidate.audit"],
            },
        ]
        for action in actions:
            research_cli.execute_action(self.run_dir, action)

        result = research_cli.run_result(self.run_dir)
        self.assertEqual(result["decision"], "negative_close")
        self.assertEqual(
            result["evidence"]["candidate.audit"]["kind"],
            "evaluation",
        )
        self.assertEqual(
            result["evidence"]["candidate.audit"]["candidate_id"],
            "candidate",
        )
        self.assertNotIn(
            "optimized_parameter_binding",
            research_cli.render_json(result),
        )


if __name__ == "__main__":
    unittest.main()
