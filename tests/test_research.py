from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import autovqe.research as research
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
        "macro": "PauliRotation",
        "qubits": qubits,
        "parameters": {
            "angle": {
                "constant": 0.0,
                "terms": [{"parameter": parameter, "coefficient": 1.0}],
            }
        },
        "options": {"pauli": pauli},
    }


def _spec(name: str, parameters: list[str], operations: list[dict]) -> dict:
    return {
        "version": 1,
        "name": name,
        "num_qubits": 2,
        "parameters": [{"name": parameter} for parameter in parameters],
        "operations": operations,
    }


def _submit_structure(
    run_dir: Path,
    hypothesis_id: str,
    candidate_id: str,
    spec: dict,
) -> None:
    execute_action(
        run_dir,
        {
            "type": "propose_hypothesis",
            "hypothesis_id": hypothesis_id,
            "claim": {"kind": "ansatz_structure", "family": hypothesis_id},
        },
    )
    execute_action(
        run_dir,
        {
            "type": "submit_candidate",
            "candidate_id": candidate_id,
            "hypothesis_id": hypothesis_id,
            "spec": spec,
            "metadata": {"prediction": "fixed evaluation improves the baseline"},
        },
    )


class _FixedEvaluation:
    def __init__(self, energy: float) -> None:
        self.valid = True
        self.best_energy = energy
        self.violations: tuple[str, ...] = ()
        self.optimized_parameter_binding = {"theta": 0.0}
        self.metrics = {
            f"{prefix}_{suffix}": 1
            for prefix in (
                "template",
                "audit_worst",
                "canonical_template",
                "canonical_audit_worst",
            )
            for suffix in ("twoq_count", "total_gate_count", "depth")
        }

    def to_dict(self) -> dict:
        return {
            "valid": True,
            "best_energy": self.best_energy,
            "trace_summary": [[1, self.best_energy]],
            "objective_calls": 32,
            "optimizer": "fixed-test",
            "seed": 7,
            "optimized_parameter_binding": dict(self.optimized_parameter_binding),
            "audit": {"unique_trainable_params": 1},
            "metrics": dict(self.metrics),
            "violations": [],
            "objective_energy_span": 1.0,
            "hamiltonian_active_norm": 1.0,
            "objective_activity_fraction": 1.0,
            "constant_hamiltonian": False,
        }


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
        target = _spec("yx", ["theta"], [_rotation("YX", [0, 1], "theta")])
        comparator = _spec(
            "xx-z",
            ["alpha", "beta"],
            [
                _rotation("XX", [0, 1], "alpha"),
                _rotation("Z", [0], "beta"),
            ],
        )
        for hypothesis_id, candidate_id, spec in (
            ("yx-branch", "yx", target),
            ("ordered-branch", "xx-z", comparator),
        ):
            _submit_structure(run_dir, hypothesis_id, candidate_id, spec)
            audit = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertEqual(audit["result"]["stage"], "audit")
            smoke = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertTrue(smoke["result"]["passed"])

        execute_action(run_dir, {"type": "evaluate_candidate", "candidate_id": "yx"})
        execute_action(run_dir, {"type": "evaluate_candidate", "candidate_id": "xx-z"})
        execute_action(run_dir, {"type": "commit", "candidate_id": "yx"})

        self.assertNotIn(
            "optimized_parameter_binding", json.dumps(run_status(run_dir, full=True))
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
        with self.assertRaisesRegex(ResearchError, "evaluator-owned"):
            execute_action(run_dir, {"type": "record_evaluation"})

    def test_null_control_is_not_a_structural_comparator(self) -> None:
        run_dir = self.initialize()
        _submit_structure(
            run_dir,
            "target-structure",
            "target",
            _spec("target", ["theta"], [_rotation("YX", [0, 1], "theta")]),
        )
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "empty-control",
                "claim": {"kind": "null_control"},
            },
        )
        execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "empty",
                "hypothesis_id": "empty-control",
                "spec": _spec("empty", [], []),
            },
        )
        for candidate_id in ("target", "empty"):
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            smoke = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertTrue(smoke["result"]["passed"])
        with self.assertRaisesRegex(ResearchError, "different structure root"):
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "target"}
            )

    def test_structure_family_names_cannot_create_fake_independence(self) -> None:
        run_dir = self.initialize()
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "first",
                "claim": {"kind": "ansatz_structure", "family": "Boundary   HVA"},
            },
        )
        with self.assertRaisesRegex(ResearchError, "duplicates existing hypothesis"):
            execute_action(
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "cosmetic-copy",
                    "claim": {"kind": "ansatz_structure", "family": " boundary hva "},
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
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "global-z",
                "claim": {
                    "kind": "exact_pauli_symmetry",
                    "generator": {"type": "global_pauli_sum", "pauli": "Z"},
                },
            },
        )
        probe_id = execute_action(
            run_dir, {"type": "request_probe", "hypothesis_id": "global-z"}
        )["result"]["probe_id"]
        with self.assertRaisesRegex(ResearchError, "primary hypothesis"):
            execute_action(
                run_dir,
                {
                    "type": "submit_candidate",
                    "candidate_id": "misowned",
                    "hypothesis_id": "global-z",
                    "spec": _spec(
                        "misowned", ["theta"], [_rotation("XX", [0, 1], "theta")]
                    ),
                    "metadata": {"prediction": "symmetry is not a structure branch"},
                },
            )
        execute_action(
            run_dir,
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "exchange-structure",
                "claim": {
                    "kind": "ansatz_structure",
                    "family": "shared XX+YY exchange",
                },
            },
        )
        generic = _spec(
            "generic",
            ["theta"],
            [
                _rotation("XX", [0, 1], "theta"),
                _rotation("YY", [0, 1], "theta"),
            ],
        )
        execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "generic",
                "hypothesis_id": "exchange-structure",
                "spec": generic,
                "metadata": {"prediction": "shared exchange lowers the baseline"},
            },
        )
        specialized = _spec(
            "specialized",
            ["theta"],
            [
                {
                    "macro": "XYExchange",
                    "qubits": [0, 1],
                    "parameters": {
                        "angle": {
                            "constant": 0.0,
                            "terms": [
                                {"parameter": "theta", "coefficient": 1.0}
                            ],
                        }
                    },
                    "options": {},
                }
            ],
        )
        execute_action(
            run_dir,
            {
                "type": "revise",
                "entity": "candidate",
                "source_id": "generic",
                "new_id": "specialized",
                "replacement": specialized,
                "reason": "supported conservation permits native exchange",
                "metadata": {"prediction": "same family with fewer routed gates"},
                "symmetry_evidence_ids": [probe_id],
            },
        )
        audit = execute_action(
            run_dir, {"type": "evaluate_candidate", "candidate_id": "specialized"}
        )
        self.assertTrue(audit["result"]["passed"])
        observed = audit["result"]["resource_policy"]["observed"]
        self.assertEqual(observed["conservative_twoq_count"], 2)

    def test_failed_comparator_promotion_does_not_block_recovery(self) -> None:
        run_dir = self.initialize()
        _submit_structure(
            run_dir,
            "target-root",
            "target",
            _spec("target", ["theta"], [_rotation("YX", [0, 1], "theta")]),
        )
        _submit_structure(
            run_dir,
            "comparator-root",
            "comparator",
            _spec("comparator", ["theta"], [_rotation("XY", [0, 1], "theta")]),
        )
        for candidate_id in ("target", "comparator"):
            audit = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertTrue(audit["result"]["passed"])

        energies = iter((-1.0, -1.0, -1.0, -0.5))
        with (
            patch.object(research.ResearchController, "_baseline", return_value=0.0),
            patch.object(
                research,
                "evaluate_public_problem",
                side_effect=lambda problem, spec, protocol: SimpleNamespace(
                    result=_FixedEvaluation(next(energies))
                ),
            ),
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
                "claim": {"kind": "ansatz_structure", "family": "recovery ordering"},
            },
        )
        self.assertTrue(recovered["result"]["accepted"])

    def test_commit_checks_better_promotions_from_the_same_root(self) -> None:
        run_dir = self.initialize()
        _submit_structure(
            run_dir,
            "shared-root",
            "target",
            _spec("target", ["theta"], [_rotation("YX", [0, 1], "theta")]),
        )
        execute_action(
            run_dir,
            {
                "type": "submit_candidate",
                "candidate_id": "sibling",
                "hypothesis_id": "shared-root",
                "spec": _spec(
                    "sibling",
                    ["theta", "phase"],
                    [
                        _rotation("YX", [0, 1], "theta"),
                        _rotation("Z", [0], "phase"),
                    ],
                ),
                "metadata": {"prediction": "phase conditioning improves this root"},
            },
        )
        _submit_structure(
            run_dir,
            "fair-root",
            "fair",
            _spec("fair", ["theta"], [_rotation("XY", [0, 1], "theta")]),
        )
        for candidate_id in ("target", "sibling", "fair"):
            audit = execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertTrue(audit["result"]["passed"])

        energies = iter((-1.0, -2.0, -0.9, -1.0, -0.9, -2.0))
        with (
            patch.object(research.ResearchController, "_baseline", return_value=0.0),
            patch.object(
                research,
                "evaluate_public_problem",
                side_effect=lambda problem, spec, protocol: SimpleNamespace(
                    result=_FixedEvaluation(next(energies))
                ),
            ),
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
        class FailedEvaluation:
            valid = True
            best_energy = 0.1
            violations: tuple[str, ...] = ()
            metrics = {
                f"{prefix}_{suffix}": 1
                for prefix in (
                    "template",
                    "audit_worst",
                    "canonical_template",
                    "canonical_audit_worst",
                )
                for suffix in ("twoq_count", "total_gate_count", "depth")
            }

            def to_dict(self) -> dict:
                return {
                    "valid": True,
                    "best_energy": self.best_energy,
                    "trace_summary": [[1, self.best_energy]],
                    "objective_calls": 32,
                    "optimizer": "fixed-test",
                    "seed": 7,
                    "optimized_parameter_binding": {"theta": 0.0},
                    "audit": {"unique_trainable_params": 1},
                    "metrics": self.metrics,
                    "violations": [],
                    "objective_energy_span": 0.1,
                    "hamiltonian_active_norm": 1.0,
                    "objective_activity_fraction": 0.1,
                    "constant_hamiltonian": False,
                }

        run_dir = self.initialize()
        with (
            patch.object(research.ResearchController, "_baseline", return_value=0.0),
            patch.object(
                research,
                "evaluate_public_problem",
                side_effect=lambda problem, spec, protocol: SimpleNamespace(
                    result=FailedEvaluation()
                ),
            ),
        ):
            for index, pauli in enumerate(("YX", "XY"), 1):
                hypothesis_id = f"branch-{index}"
                candidate_id = f"candidate-{index}"
                _submit_structure(
                    run_dir,
                    hypothesis_id,
                    candidate_id,
                    _spec(
                        candidate_id,
                        ["theta"],
                        [_rotation(pauli, [0, 1], "theta")],
                    ),
                )
                execute_action(
                    run_dir,
                    {"type": "evaluate_candidate", "candidate_id": candidate_id},
                )
                failed = execute_action(
                    run_dir,
                    {"type": "evaluate_candidate", "candidate_id": candidate_id},
                )
                self.assertFalse(failed["result"]["passed"])
                execute_action(
                    run_dir,
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
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

    def test_kind_hops_cannot_manufacture_negative_close_breadth(self) -> None:
        run_dir = self.initialize()
        with (
            patch.object(research.ResearchController, "_baseline", return_value=0.0),
            patch.object(
                research,
                "evaluate_public_problem",
                side_effect=lambda problem, spec, protocol: SimpleNamespace(
                    result=_FixedEvaluation(0.1)
                ),
            ),
        ):
            _submit_structure(
                run_dir,
                "root-a",
                "candidate-a",
                _spec(
                    "candidate-a",
                    ["theta"],
                    [_rotation("YX", [0, 1], "theta")],
                ),
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-a"}
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-a"}
            )
            execute_action(
                run_dir,
                {
                    "type": "revise",
                    "entity": "hypothesis",
                    "source_id": "root-a",
                    "new_id": "symmetry-hop",
                    "replacement": {
                        "kind": "exact_pauli_symmetry",
                        "generator": {"type": "global_pauli_sum", "pauli": "Z"},
                    },
                    "reason": "test whether changing claim kind creates a new root",
                },
            )
            execute_action(
                run_dir,
                {
                    "type": "revise",
                    "entity": "hypothesis",
                    "source_id": "symmetry-hop",
                    "new_id": "structure-hop",
                    "replacement": {
                        "kind": "ansatz_structure",
                        "family": "post-symmetry ordering",
                    },
                    "reason": "return to a structural claim in the same lineage",
                },
            )
            execute_action(
                run_dir,
                {
                    "type": "submit_candidate",
                    "candidate_id": "candidate-b",
                    "hypothesis_id": "structure-hop",
                    "spec": _spec(
                        "candidate-b",
                        ["theta"],
                        [_rotation("XY", [0, 1], "theta")],
                    ),
                    "metadata": {"falsifier": "fixed smoke does not improve baseline"},
                },
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-b"}
            )
            execute_action(
                run_dir, {"type": "evaluate_candidate", "candidate_id": "candidate-b"}
            )
        execute_action(
            run_dir,
            {
                "type": "retire",
                "entity": "hypothesis",
                "entity_id": "structure-hop",
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
