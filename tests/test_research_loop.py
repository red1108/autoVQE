from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from autovqe.history import HistoryIntegrityError, JsonlRunHistory
from autovqe.research import (
    ActionParseError,
    BudgetExceeded,
    Lifecycle,
    ResearchLoop,
    TransitionError,
    parse_action,
    replay_history,
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

    def test_grounded_negative_close_is_terminal_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.jsonl"
            loop = ResearchLoop(path, total_budget=5)
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
                    "probe_id": "p.refuted",
                    "verdict": "refuted",
                    "result": {"residual": 0.5},
                }
            )
            loop.dispatch(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "h1",
                    "reason": "the falsification probe refuted the claim",
                }
            )
            loop.dispatch(
                {
                    "type": "close_negative",
                    "reason": "all investigated branches were refuted",
                    "evidence_ids": ["p.refuted"],
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

    def test_action_parser_is_strict_and_json_only(self) -> None:
        with self.assertRaises(ActionParseError):
            parse_action({"type": "invent_result", "cost": 0})
        with self.assertRaises(ActionParseError):
            parse_action(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "u1"},
                    "unexpected": True,
                }
            )
        with self.assertRaises(ActionParseError):
            parse_action(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"score": float("nan")},
                }
            )
        with self.assertRaises(ActionParseError):
            parse_action(
                {
                    "type": "record_evaluation",
                    "candidate_id": "c1",
                    "evaluation_id": "e1",
                    "stage": "audit",
                    "passed": 1,
                    "metrics": {},
                }
            )
        with self.assertRaises(ActionParseError):
            parse_action(
                {
                    "type": "close_negative",
                    "reason": "ungrounded",
                    "evidence_ids": [],
                }
            )
        with self.assertRaises(ActionParseError):
            parse_action(
                {
                    "type": "close_negative",
                    "reason": "duplicate grounding",
                    "evidence_ids": ["p1", "p1"],
                }
            )


if __name__ == "__main__":
    unittest.main()
