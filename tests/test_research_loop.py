from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autovqe.history import HistoryIntegrityError, JsonlRunHistory
from autovqe.research import (
    BudgetExceeded,
    EventFormatError,
    Lifecycle,
    ResearchLoop,
    TransitionError,
    normalize_event,
    replay_history,
    validate_negative_close_coverage,
)


class ResearchLoopTests(unittest.TestCase):
    def test_scripted_closed_loop_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.jsonl"
            loop = ResearchLoop(path, total_budget=30)
            actions = [
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h.raw",
                    "claim": {"generator": "sum_z", "exact": True},
                    "cost": 1,
                },
                {
                    "type": "record_probe",
                    "hypothesis_id": "h.raw",
                    "probe_id": "p.commutator.0",
                    "verdict": "refuted",
                    "result": {"normalized_commutator": 0.2},
                    "cost": 1,
                },
                {
                    "type": "revise",
                    "entity": "hypothesis",
                    "source_id": "h.raw",
                    "new_id": "h.refined",
                    "replacement": {"generator": "sum_xyz", "exact": True},
                    "reason": "the first generator missed matched exchange terms",
                    "cost": 0.5,
                },
                {
                    "type": "record_probe",
                    "hypothesis_id": "h.refined",
                    "probe_id": "p.commutator.1",
                    "verdict": "supported",
                    "result": {"normalized_commutator": 1.0e-14},
                    "cost": 1,
                },
                {
                    "type": "submit_candidate",
                    "candidate_id": "c.first",
                    "hypothesis_id": "h.refined",
                    "spec": {"macro": "exchange", "layers": 1},
                    "cost": 1,
                },
                {
                    "type": "record_evaluation",
                    "candidate_id": "c.first",
                    "evaluation_id": "e.audit.0",
                    "stage": "audit",
                    "passed": True,
                    "metrics": {"leakage": 0.0},
                    "cost": 1,
                },
                {
                    "type": "record_evaluation",
                    "candidate_id": "c.first",
                    "evaluation_id": "e.smoke.0",
                    "stage": "smoke",
                    "passed": True,
                    "metrics": {"energy": -1.0},
                    "cost": 2,
                },
                {
                    "type": "revise",
                    "entity": "candidate",
                    "source_id": "c.first",
                    "new_id": "c.revised",
                    "replacement": {"macro": "exchange", "layers": 2},
                    "reason": "smoke trace was still improving at the budget boundary",
                    "metadata": {
                        "prediction": "the added layer continues the observed improvement"
                    },
                    "cost": 0.5,
                },
                {
                    "type": "submit_candidate",
                    "candidate_id": "c.control",
                    "hypothesis_id": "h.refined",
                    "spec": {"macro": "hea_control", "layers": 1},
                    "cost": 0.5,
                },
                {
                    "type": "retire",
                    "entity": "candidate",
                    "entity_id": "c.control",
                    "reason": "dominated by the matched structured candidate",
                    "cost": 0,
                },
                {
                    "type": "record_evaluation",
                    "candidate_id": "c.revised",
                    "evaluation_id": "e.audit.1",
                    "stage": "audit",
                    "passed": True,
                    "metrics": {"leakage": 0.0},
                    "cost": 1,
                },
                {
                    "type": "record_evaluation",
                    "candidate_id": "c.revised",
                    "evaluation_id": "e.smoke.1",
                    "stage": "smoke",
                    "passed": True,
                    "metrics": {"energy": -1.2},
                    "cost": 2,
                },
                {
                    "type": "record_evaluation",
                    "candidate_id": "c.revised",
                    "evaluation_id": "e.promotion.1",
                    "stage": "promotion",
                    "passed": True,
                    "metrics": {"median_energy": -1.25, "seeds": 3},
                    "cost": 3,
                },
                {
                    "type": "commit",
                    "candidate_id": "c.revised",
                    "metadata": {"policy": "locked_hidden_eval"},
                },
            ]

            final_state = loop.run_script(actions)

            self.assertTrue(final_state.committed)
            self.assertEqual(final_state.committed_candidate_id, "c.revised")
            self.assertEqual(final_state.hypotheses["h.raw"].status, Lifecycle.REVISED)
            self.assertEqual(final_state.hypotheses["h.refined"].status, Lifecycle.SUPPORTED)
            self.assertEqual(final_state.candidates["c.first"].status, Lifecycle.REVISED)
            self.assertEqual(final_state.candidates["c.control"].status, Lifecycle.RETIRED)
            self.assertEqual(final_state.candidates["c.revised"].status, Lifecycle.PROMOTED)
            self.assertIn("prediction", final_state.candidates["c.revised"].metadata)
            self.assertAlmostEqual(final_state.spent_budget, 14.5)

            history = JsonlRunHistory(path)
            events = history.read_events()
            self.assertEqual(len(events), len(actions))
            self.assertEqual([event.seq for event in events], list(range(len(actions))))
            self.assertEqual(
                set(events[0].to_record()),
                {"version", "seq", "type", "payload", "cost"},
            )
            with self.assertRaises(TypeError):
                events[0].payload["hypothesis_id"] = "tampered"  # type: ignore[index]

            replayed = replay_history(history, total_budget=30)
            reopened = ResearchLoop(path, total_budget=30).state
            self.assertEqual(replayed.to_dict(), final_state.to_dict())
            self.assertEqual(reopened.to_dict(), final_state.to_dict())

            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "h.after_commit",
                        "claim": {"kind": "forbidden"},
                    }
                )
            self.assertEqual(len(history.read_events()), len(actions))

    def test_budget_failure_is_not_appended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = JsonlRunHistory(Path(directory) / "budget.jsonl")
            loop = ResearchLoop(history, total_budget=2)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "u1"},
                    "cost": 1.5,
                }
            )

            with self.assertRaises(BudgetExceeded):
                loop.dispatch(
                    {
                        "type": "record_probe",
                        "hypothesis_id": "h1",
                        "probe_id": "p1",
                        "verdict": "supported",
                        "result": {"commutator": 0.0},
                        "cost": 0.75,
                    }
                )

            self.assertEqual(len(history.read_events()), 1)
            self.assertAlmostEqual(loop.state.spent_budget, 1.5)
            self.assertEqual(loop.state.hypotheses["h1"].status, Lifecycle.PROPOSED)

    def test_invalid_lifecycle_is_rejected_without_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = JsonlRunHistory(Path(directory) / "transitions.jsonl")
            loop = ResearchLoop(history, total_budget=20)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "permutation"},
                }
            )
            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "c1",
                        "hypothesis_id": "h1",
                        "spec": {"layers": 1},
                    }
                )

            loop.dispatch(
                {
                    "type": "record_probe",
                    "hypothesis_id": "h1",
                    "probe_id": "p1",
                    "verdict": "supported",
                    "result": {},
                }
            )
            loop.dispatch(
                {
                    "type": "submit_candidate",
                    "candidate_id": "c1",
                    "hypothesis_id": "h1",
                    "spec": {"layers": 1},
                }
            )
            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": "c1",
                        "evaluation_id": "e1",
                        "stage": "smoke",
                        "passed": True,
                        "metrics": {},
                    }
                )
            with self.assertRaises(TransitionError):
                loop.dispatch({"type": "commit", "candidate_id": "c1"})

            self.assertEqual(len(history.read_events()), 3)
            self.assertEqual(loop.state.candidates["c1"].status, Lifecycle.CANDIDATE)

    def test_failed_candidate_can_be_revised_without_erasing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "revision.jsonl", total_budget=10)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "structure",
                    "claim": {"kind": "ansatz_structure"},
                }
            )
            loop.dispatch(
                {
                    "type": "submit_candidate",
                    "candidate_id": "first",
                    "hypothesis_id": "structure",
                    "spec": {"operations": []},
                }
            )
            loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": "first",
                    "evaluation_id": "evaluation:first:audit",
                    "stage": "audit",
                    "passed": False,
                    "metrics": {"violations": ["too deep"]},
                }
            )
            loop.dispatch(
                {
                    "type": "revise",
                    "entity": "candidate",
                    "source_id": "first",
                    "new_id": "second",
                    "replacement": {"operations": [{"macro": "smaller"}]},
                    "reason": "reduce the audited resource cost",
                }
            )

            self.assertEqual(loop.state.candidates["first"].status, Lifecycle.REVISED)
            self.assertEqual(loop.state.candidates["second"].parent_id, "first")
            self.assertIn("evaluation:first:audit", loop.state.evaluations)

    def test_grounded_negative_close_is_terminal_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.jsonl"
            loop = ResearchLoop(path, total_budget=5)
            for hypothesis_id in ("h1", "h2"):
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": hypothesis_id,
                        "claim": {
                            "kind": "ansatz_structure",
                            "family": hypothesis_id,
                        },
                    }
                )

            def fail_candidate(
                candidate_id: str,
                hypothesis_id: str,
            ) -> None:
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": hypothesis_id,
                        "spec": {"family": candidate_id},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e.{candidate_id}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {"valid": True},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e.{candidate_id}.smoke",
                        "stage": "smoke",
                        "passed": False,
                        "metrics": {
                            "valid": True,
                            "objective_calls": 8,
                            "objective_energy_span": 0.2,
                            "hamiltonian_active_norm": 1.0,
                            "objective_activity_fraction": 0.2,
                            "constant_hamiltonian": False,
                        },
                    }
                )

            fail_candidate("c1", "h1")
            fail_candidate("c2", "h2")
            for hypothesis_id in ("h1", "h2"):
                loop.dispatch(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
                        "reason": "the numerical ansatz tests refuted the branch",
                    }
                )
            loop.dispatch(
                {
                    "type": "close_negative",
                    "reason": "all investigated branches were refuted",
                    "evidence_ids": [
                        "e.c1.smoke",
                        "e.c2.smoke",
                    ],
                }
            )

            self.assertTrue(loop.state.terminal)
            self.assertFalse(loop.state.committed)
            self.assertTrue(loop.state.negative_closed)
            self.assertEqual(loop.state.terminal_decision, "negative_close")
            self.assertEqual(
                replay_history(JsonlRunHistory(path), total_budget=5).to_dict(),
                loop.state.to_dict(),
            )
            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "too_late",
                        "claim": {"kind": "testable"},
                    }
                )

    def test_flat_failures_cannot_forge_negative_coverage_during_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "flat.jsonl", total_budget=5)
            for index in (1, 2):
                hypothesis_id = f"h{index}"
                candidate_id = f"c{index}"
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": hypothesis_id,
                        "claim": {"kind": "ansatz_structure", "family": index},
                    }
                )
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": hypothesis_id,
                        "spec": {"family": candidate_id},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e{index}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {"valid": True},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e{index}.smoke",
                        "stage": "smoke",
                        "passed": False,
                        "metrics": {
                            "valid": True,
                            "objective_calls": 8,
                            "objective_energy_span": 0.0,
                            "hamiltonian_active_norm": 1.0,
                            "objective_activity_fraction": 0.0,
                            "constant_hamiltonian": False,
                        },
                    }
                )
                loop.dispatch(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
                        "reason": "flat numerical failure",
                    }
                )
            with self.assertRaisesRegex(TransitionError, "objective activity"):
                loop.dispatch(
                    {
                        "type": "close_negative",
                        "reason": "flat objectives are not coverage",
                        "evidence_ids": ["e1.smoke", "e2.smoke"],
                    }
                )

    def test_hypothesis_revision_does_not_fake_a_second_structure_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "lineage.jsonl", total_budget=5)

            def add_active_failure(
                hypothesis_id: str,
                candidate_id: str,
                evaluation_id: str,
            ) -> None:
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": hypothesis_id,
                        "spec": {"family": candidate_id},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"{evaluation_id}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {"valid": True},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"{evaluation_id}.smoke",
                        "stage": "smoke",
                        "passed": False,
                        "metrics": {
                            "valid": True,
                            "objective_calls": 8,
                            "objective_energy_span": 0.2,
                            "hamiltonian_active_norm": 1.0,
                            "objective_activity_fraction": 0.2,
                            "constant_hamiltonian": False,
                        },
                    }
                )

            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "ansatz_structure", "family": "first"},
                }
            )
            add_active_failure("h1", "c1", "e1")
            loop.dispatch(
                {
                    "type": "revise",
                    "entity": "hypothesis",
                    "source_id": "h1",
                    "new_id": "h1.revised",
                    "replacement": {
                        "kind": "ansatz_structure",
                        "family": "revised",
                    },
                    "reason": "refine the same structural branch",
                }
            )
            add_active_failure("h1.revised", "c2", "e2")
            loop.dispatch(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "h1.revised",
                    "reason": "the revised branch also failed",
                }
            )
            with self.assertRaisesRegex(TransitionError, "found 1"):
                loop.dispatch(
                    {
                        "type": "close_negative",
                        "reason": "one lineage is not breadth",
                        "evidence_ids": ["e1.smoke", "e2.smoke"],
                    }
                )

    def test_revised_hypothesis_is_covered_by_its_root_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "root-coverage.jsonl", total_budget=5)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "ansatz_structure", "family": "initial"},
                }
            )
            loop.dispatch(
                {
                    "type": "revise",
                    "entity": "hypothesis",
                    "source_id": "h1",
                    "new_id": "h1.revised",
                    "replacement": {
                        "kind": "ansatz_structure",
                        "family": "refined",
                    },
                    "reason": "refine before numerical evaluation",
                }
            )
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h2",
                    "claim": {"kind": "ansatz_structure", "family": "independent"},
                }
            )

            evidence_ids: list[str] = []
            for hypothesis_id, candidate_id in (
                ("h1.revised", "c1"),
                ("h2", "c2"),
            ):
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": hypothesis_id,
                        "spec": {"family": candidate_id},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e.{candidate_id}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {"valid": True},
                    }
                )
                evaluation_id = f"e.{candidate_id}.smoke"
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": evaluation_id,
                        "stage": "smoke",
                        "passed": False,
                        "metrics": {
                            "valid": True,
                            "objective_calls": 8,
                            "objective_energy_span": 0.2,
                            "hamiltonian_active_norm": 1.0,
                            "objective_activity_fraction": 0.2,
                            "constant_hamiltonian": False,
                        },
                    }
                )
                loop.dispatch(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
                        "reason": "objective-active numerical failure",
                    }
                )
                evidence_ids.append(evaluation_id)

            coverage = validate_negative_close_coverage(
                loop.state, evidence_ids
            )
            loop.dispatch(
                {
                    "type": "close_negative",
                    "reason": "both root lineages were numerically refuted",
                    "evidence_ids": evidence_ids,
                }
            )

            self.assertEqual(
                coverage["covered_lineages"],
                ["h1", "h2"],
            )

    def test_constant_hamiltonian_allows_two_flat_structure_lineages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "constant.jsonl", total_budget=5)
            evidence_ids: list[str] = []
            for index in (1, 2):
                hypothesis_id = f"h{index}"
                candidate_id = f"c{index}"
                evaluation_id = f"e{index}.smoke"
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": hypothesis_id,
                        "claim": {"kind": "ansatz_structure", "family": index},
                    }
                )
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": hypothesis_id,
                        "spec": {"family": candidate_id},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"e{index}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {"valid": True},
                    }
                )
                loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": evaluation_id,
                        "stage": "smoke",
                        "passed": False,
                        "metrics": {
                            "valid": True,
                            "objective_calls": 8,
                            "objective_energy_span": 0.0,
                            "hamiltonian_active_norm": 0.0,
                            "objective_activity_fraction": None,
                            "constant_hamiltonian": True,
                        },
                    }
                )
                loop.dispatch(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
                        "reason": "constant Hamiltonian branch exhausted",
                    }
                )
                evidence_ids.append(evaluation_id)
            loop.dispatch(
                {
                    "type": "close_negative",
                    "reason": "a constant Hamiltonian has no active objective",
                    "evidence_ids": evidence_ids,
                }
            )
            self.assertTrue(loop.state.negative_closed)

    def test_promoted_disposition_requires_actual_dominance_during_replay(self) -> None:
        def metrics(energy: float) -> dict:
            return {
                "valid": True,
                "best_energy": energy,
                "resource_policy": {
                    "eligible": True,
                    "observed": {
                        "conservative_twoq_count": 2,
                        "conservative_total_gate_count": 4,
                        "conservative_depth": 3,
                    },
                },
                "audit": {"unique_trainable_params": 1},
            }

        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "dominance.jsonl", total_budget=5)
            for suffix in ("target", "comparator"):
                loop.dispatch(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": f"h.{suffix}",
                        "claim": {"kind": "ansatz_structure", "family": suffix},
                    }
                )
                loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": suffix,
                        "hypothesis_id": f"h.{suffix}",
                        "spec": {"family": suffix},
                    }
                )
                for stage in ("audit", "smoke", "promotion"):
                    loop.dispatch(
                        {
                            "type": "record_evaluation",
                            "candidate_id": suffix,
                            "evaluation_id": f"e.{suffix}.{stage}",
                            "stage": stage,
                            "passed": True,
                            "metrics": (
                                {"valid": True}
                                if stage == "audit"
                                else metrics(-10.0 if suffix == "target" else 100.0)
                            ),
                        }
                    )

            with self.assertRaisesRegex(TransitionError, "actually dominates"):
                loop.dispatch(
                    {
                        "type": "retire",
                        "entity": "candidate",
                        "entity_id": "target",
                        "reason": "forged comparison",
                        "evidence_ids": [
                            "e.target.promotion",
                            "e.comparator.promotion",
                        ],
                    }
                )
            self.assertEqual(loop.state.candidates["target"].status, Lifecycle.PROMOTED)

    def test_negative_close_rejects_live_branches_and_unknown_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loop = ResearchLoop(Path(directory) / "negative.jsonl", total_budget=5)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "testable"},
                }
            )
            loop.dispatch(
                {
                    "type": "record_probe",
                    "hypothesis_id": "h1",
                    "probe_id": "p1",
                    "verdict": "refuted",
                    "result": {},
                }
            )
            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "close_negative",
                        "reason": "premature",
                        "evidence_ids": ["p1"],
                    }
                )
            loop.dispatch(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "h1",
                    "reason": "refuted",
                }
            )
            with self.assertRaises(TransitionError):
                loop.dispatch(
                    {
                        "type": "close_negative",
                        "reason": "ungrounded",
                        "evidence_ids": ["missing"],
                    }
                )
            self.assertFalse(loop.state.terminal)

    def test_nonsequential_history_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tamper.jsonl"
            loop = ResearchLoop(path, total_budget=5)
            loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "charge"},
                    "cost": 1,
                }
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            record["seq"] = 1
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaises(HistoryIntegrityError):
                JsonlRunHistory(path).read_events()

    def test_event_normalizer_is_strict_and_json_only(self) -> None:
        with self.assertRaises(EventFormatError):
            normalize_event({"type": "invent_result", "cost": 0})
        with self.assertRaises(EventFormatError):
            normalize_event(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "u1"},
                    "unexpected": True,
                }
            )
        with self.assertRaises(EventFormatError):
            normalize_event(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"score": float("nan")},
                }
            )
        with self.assertRaises(EventFormatError):
            normalize_event(
                {
                    "type": "record_evaluation",
                    "candidate_id": "c1",
                    "evaluation_id": "e1",
                    "stage": "audit",
                    "passed": 1,
                    "metrics": {},
                }
            )
        with self.assertRaises(EventFormatError):
            normalize_event(
                {
                    "type": "close_negative",
                    "reason": "ungrounded",
                    "evidence_ids": [],
                }
            )
        with self.assertRaises(EventFormatError):
            normalize_event(
                {
                    "type": "close_negative",
                    "reason": "duplicate grounding",
                    "evidence_ids": ["p1", "p1"],
                }
            )


if __name__ == "__main__":
    unittest.main()
