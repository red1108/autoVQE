from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autovqe.research as research
from autovqe.evaluator import EvaluationResult
from autovqe.research import (
    ResearchError,
    execute_action,
    initialize_run,
    run_result,
    run_status,
)


H2_PROBLEM = {
    "name": "h2_test",
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


def _write_problem(path: Path, problem: dict | None = None) -> None:
    path.write_text(json.dumps(problem or H2_PROBLEM), encoding="utf-8")


def _rotation(pauli: str, qubits: list[int], parameter: str) -> dict:
    return {
        "gate": "PauliRotation",
        "qubits": qubits,
        "parameter": parameter,
        "pauli": pauli,
    }


def _spec(operations: list[dict]) -> dict:
    return {
        "version": 1,
        "num_qubits": 2,
        "operations": operations,
    }


def _submit_structure(
    run_dir: Path,
    hypothesis_id: str,
    candidate_id: str,
    spec: dict,
) -> dict:
    execute_action(
        run_dir,
        {
            "type": "propose_hypothesis",
            "hypothesis_id": hypothesis_id,
            "family": hypothesis_id,
            "prediction": "the proposed structure improves the initial state",
        },
    )
    return execute_action(
        run_dir,
        {
            "type": "submit_candidate",
            "candidate_id": candidate_id,
            "hypothesis_id": hypothesis_id,
            "spec": spec,
        },
    )


def _fixed_evaluation(energy: float, activity: float = 1.0) -> EvaluationResult:
    return EvaluationResult(
        valid=True,
        best_energy=energy,
        baseline_energy=0.0,
        trace_summary=((1, energy),),
        objective_calls=32,
        optimized_parameter_binding={"theta": 0.0},
        audit={"unique_trainable_params": 1},
        resources={"parameters": 1, "twoq_count": 1, "total_gate_count": 1, "depth": 1},
        objective_activity_fraction=activity,
    )


class ResearchLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "hamiltonian.json"
        _write_problem(self.source)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def initialize(self, name: str = "run") -> Path:
        run_dir = self.root / name
        initialize_run(self.source, run_dir, total_budget=100)
        return run_dir

    def test_binding_is_hidden_until_terminal_replay(self) -> None:
        run_dir = self.initialize()
        self.source.write_text("{}\n", encoding="utf-8")
        target = _spec([_rotation("YX", [0, 1], "theta")])
        comparator = _spec(
            [
                _rotation("XX", [0, 1], "alpha"),
                _rotation("Z", [0], "beta"),
            ],
        )
        for hypothesis_id, candidate_id, spec in (
            ("yx-branch", "yx", target),
            ("ordered-branch", "xx-z", comparator),
        ):
            submitted = _submit_structure(run_dir, hypothesis_id, candidate_id, spec)
            self.assertTrue(submitted["result"]["audit_passed"])
            smoke = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertTrue(smoke["result"]["passed"])

        execute_action(run_dir, {"type": "evaluate_candidate", "candidate_id": "yx"})
        execute_action(run_dir, {"type": "evaluate_candidate", "candidate_id": "xx-z"})
        execute_action(run_dir, {"type": "commit", "candidate_id": "yx"})

        self.assertNotIn(
            "optimized_parameter_binding", json.dumps(run_status(run_dir))
        )
        self.assertNotIn(
            "optimized_parameter_binding",
            (run_dir / "events.jsonl").read_text(encoding="utf-8"),
        )
        result = run_result(run_dir)
        self.assertEqual(result["decision"], "positive_commit")
        self.assertEqual(result["candidate_id"], "yx")
        self.assertEqual(set(result["optimized_parameters"]), {"theta"})
        self.assertLess(result["energy"], -1.85)
        with self.assertRaisesRegex(ResearchError, "terminal"):
            execute_action(run_dir, {"type": "commit", "candidate_id": "yx"})

    def test_agent_cannot_submit_evaluator_events(self) -> None:
        run_dir = self.initialize()
        for action_type in ("record_evaluation", "record_symmetry_probe"):
            with self.subTest(action_type=action_type):
                with self.assertRaisesRegex(ResearchError, "evaluator-owned"):
                    execute_action(run_dir, {"type": action_type})

    def test_only_falsifiable_structure_hypotheses_are_accepted(self) -> None:
        run_dir = self.initialize()
        with self.assertRaisesRegex(ResearchError, "prediction or falsifier"):
            execute_action(
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "unfalsifiable",
                    "family": "empty claim",
                },
            )
        with self.assertRaisesRegex(ResearchError, "invalid external action fields"):
            execute_action(
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "legacy-control",
                    "claim": {"kind": "null_control"},
                },
            )

    def test_failed_automatic_audit_allows_a_fresh_candidate(self) -> None:
        run_dir = self.initialize()
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "repairable",
                "family": "ordered excitation",
                "falsifier": "no allowed rotation improves the initial state",
            },
        )
        rejected = execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "empty",
                "hypothesis_id": "repairable",
                "spec": _spec([]),
            },
        )
        self.assertFalse(rejected["result"]["audit_passed"])
        state = run_status(run_dir)["state"]
        self.assertEqual(state["candidates"]["empty"]["status"], "RETIRED")
        self.assertEqual(
            state["candidates"]["empty"]["latest_evaluation"]["stage"], "audit"
        )

        accepted = execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "replacement",
                "hypothesis_id": "repairable",
                "spec": _spec(
                    [_rotation("YX", [0, 1], "theta")],
                ),
            },
        )
        self.assertTrue(accepted["result"]["audit_passed"])
        with self.assertRaisesRegex(ResearchError, "unsupported external action"):
            execute_action(
                run_dir,
                {
                    "type": "revise_candidate",
                    "source_id": "empty",
                    "new_id": "replacement-2",
                },
            )

    def test_structure_family_names_cannot_create_fake_independence(self) -> None:
        run_dir = self.initialize()
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "first",
                "family": "Boundary   HVA",
                "prediction": "the boundary order lowers energy",
            },
        )
        with self.assertRaisesRegex(ResearchError, "duplicates existing hypothesis"):
            execute_action(
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "cosmetic-copy",
                    "family": " boundary hva ",
                    "falsifier": "the copied family does not improve",
                },
            )

    def test_supported_symmetry_can_specialize_equivalent_exchange(self) -> None:
        source = self.root / "exchange.json"
        _write_problem(
            source,
            {
                "name": "exchange",
                "pauli_terms": [
                    {"pauli": "XX", "coeff": 1.0},
                    {"pauli": "YY", "coeff": 1.0},
                    {"pauli": "ZZ", "coeff": 0.5},
                ],
                "initial_state_hint": [1, 0],
            },
        )
        run_dir = self.root / "exchange-run"
        initialize_run(source, run_dir, total_budget=100)
        probe_id = execute_action(
            run_dir,
            {
                "type": "request_symmetry_probe",
                "probe_id": "global-z",
                "generator": {"type": "global_pauli_sum", "pauli": "Z"},
            },
        )["result"]["probe_id"]
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "exchange-structure",
                "family": "native number-conserving exchange",
                "prediction": "the exchange direction lowers the baseline",
            },
        )
        specialized = _spec(
            [
                {
                    "gate": "XYExchange",
                    "qubits": [0, 1],
                    "parameter": "theta",
                }
            ],
        )
        submitted = execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "specialized",
                "hypothesis_id": "exchange-structure",
                "spec": specialized,
                "symmetry_evidence_ids": [probe_id],
            },
        )
        self.assertTrue(submitted["result"]["audit_passed"])
        self.assertEqual(submitted["result"]["resources"]["twoq_count"], 2)
        self.assertEqual(
            run_status(run_dir)["state"]["symmetry_probes"][probe_id]["verdict"],
            "supported",
        )

    def test_failed_comparator_promotion_does_not_block_recovery(self) -> None:
        run_dir = self.initialize()
        _submit_structure(
            run_dir,
            "target-root",
            "target",
            _spec([_rotation("YX", [0, 1], "theta")]),
        )
        _submit_structure(
            run_dir,
            "comparator-root",
            "comparator",
            _spec([_rotation("XY", [0, 1], "theta")]),
        )
        energies = iter((-1.0, -1.0, -1.0, -0.5))
        with patch.object(
            research,
            "evaluate_public_problem",
            side_effect=lambda problem, spec, protocol: _fixed_evaluation(next(energies)),
        ):
            for candidate_id in ("target", "comparator", "target"):
                result = execute_action(
                    run_dir,
                    {"type": "evaluate_candidate", "candidate_id": candidate_id},
                )
                self.assertTrue(result["result"]["passed"])
            failed = execute_action(
                run_dir,
                {"type": "evaluate_candidate", "candidate_id": "comparator"},
            )
            self.assertFalse(failed["result"]["passed"])

        recovered = execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "recovery-root",
                "family": "recovery ordering",
                "prediction": "a third ordering improves the baseline",
            },
        )
        self.assertTrue(recovered["result"]["accepted"])

    def test_commit_checks_better_promotions_from_the_same_root(self) -> None:
        run_dir = self.initialize()
        _submit_structure(
            run_dir,
            "shared-root",
            "target",
            _spec([_rotation("YX", [0, 1], "theta")]),
        )
        execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "sibling",
                "hypothesis_id": "shared-root",
                "spec": _spec(
                    [
                        _rotation("YX", [0, 1], "theta"),
                        _rotation("Z", [0], "phase"),
                    ],
                ),
            },
        )
        _submit_structure(
            run_dir,
            "fair-root",
            "fair",
            _spec([_rotation("XY", [0, 1], "theta")]),
        )
        energies = iter((-1.0, -2.0, -0.9, -1.0, -0.9, -2.0))
        with patch.object(
            research,
            "evaluate_public_problem",
            side_effect=lambda problem, spec, protocol: _fixed_evaluation(next(energies)),
        ):
            for candidate_id in (
                "target",
                "sibling",
                "fair",
                "target",
                "fair",
                "sibling",
            ):
                evaluated = execute_action(
                    run_dir,
                    {"type": "evaluate_candidate", "candidate_id": candidate_id},
                )
                self.assertTrue(evaluated["result"]["passed"])

        with self.assertRaisesRegex(ResearchError, "dominated by comparator sibling"):
            execute_action(run_dir, {"type": "commit", "candidate_id": "target"})

    def test_negative_close_needs_two_active_failed_structures(self) -> None:
        run_dir = self.initialize()
        with patch.object(
            research,
            "evaluate_public_problem",
            side_effect=lambda problem, spec, protocol: _fixed_evaluation(0.1, 0.1),
        ):
            for index, pauli in enumerate(("YX", "XY"), 1):
                hypothesis_id = f"branch-{index}"
                candidate_id = f"candidate-{index}"
                _submit_structure(
                    run_dir,
                    hypothesis_id,
                    candidate_id,
                    _spec(
                        [_rotation(pauli, [0, 1], "theta")],
                    ),
                )
                failed = execute_action(
                    run_dir,
                    {"type": "evaluate_candidate", "candidate_id": candidate_id},
                )
                self.assertFalse(failed["result"]["passed"])
                execute_action(
                    run_dir,
                    {
                        "type": "retire_hypothesis",
                        "hypothesis_id": hypothesis_id,
                        "reason": "candidate was numerically falsified",
                    },
                )
            closed = execute_action(
                run_dir,
                {
                    "type": "close_negative",
                    "reason": "two distinct structures failed smoke",
                },
            )
        self.assertEqual(
            closed["result"]["coverage"]["search_mode"], "structural_breadth"
        )
        self.assertEqual(run_result(run_dir)["decision"], "negative_close")

    def test_hypothesis_revisions_cannot_manufacture_negative_close_breadth(self) -> None:
        run_dir = self.initialize()
        with patch.object(
            research,
            "evaluate_public_problem",
            side_effect=lambda problem, spec, protocol: _fixed_evaluation(0.1),
        ):
            _submit_structure(
                run_dir,
                "root-a",
                "candidate-a",
                _spec(
                    [_rotation("YX", [0, 1], "theta")],
                ),
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-a"}
            )
            execute_action(
                run_dir,
                {
                    "type": "revise_hypothesis",
                    "source_id": "root-a",
                    "new_id": "structure-hop",
                    "family": "reordered descendant",
                    "prediction": "the revised ordering improves the baseline",
                    "reason": "revise the structure without creating independence",
                },
            )
            execute_action(
                run_dir,
                {
                    "type": "submit_candidate",
                    "candidate_id": "candidate-b",
                    "hypothesis_id": "structure-hop",
                    "spec": _spec(
                        [_rotation("XY", [0, 1], "theta")],
                    ),
                },
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-b"}
            )
        execute_action(
            run_dir,
            {
                "type": "retire_hypothesis",
                "hypothesis_id": "structure-hop",
                "reason": "the descendant candidate failed smoke",
            },
        )
        with self.assertRaisesRegex(ResearchError, "two independent"):
            execute_action(
                run_dir,
                {"type": "close_negative", "reason": "one lineage changed labels"},
            )


if __name__ == "__main__":
    unittest.main()
