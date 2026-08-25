from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autovqe.ansatz_ir import (
    AnsatzSpec,
    LayerSpec,
    OperationSpec,
    ParameterExpression,
    ReferenceSpec as AnsatzReferenceSpec,
)
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
    SectorSpec,
)
from autovqe.controller import ControllerError, ResearchController
from autovqe.evaluator import EvaluationProtocol, evaluate_public_problem


def two_qubit_problem() -> PublicProblem:
    return PublicProblem.create(
        num_qubits=2,
        pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 1.0)),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        reference=ReferenceSpec(),
        backend=BackendSpec(
            basis_gates=("rx", "ry", "rz", "cx"),
            coupling_map=((0, 1), (1, 0)),
        ),
    )


def parity_ansatz() -> AnsatzSpec:
    return AnsatzSpec(
        name="parity_pair_rotation",
        num_qubits=2,
        parameters=("theta",),
        layers=(
            LayerSpec(
                operations=(
                    OperationSpec(
                        macro="PauliRotation",
                        qubits=(0, 1),
                        parameters={
                            "angle": ParameterExpression.parameter("theta")
                        },
                        options={"pauli": "XX"},
                    ),
                ),
            ),
        ),
    )


def xy_exchange_ansatz() -> AnsatzSpec:
    return AnsatzSpec(
        name="xy_exchange",
        num_qubits=2,
        parameters=("theta",),
        layers=(
            LayerSpec(
                operations=(
                    OperationSpec(
                        macro="XYExchange",
                        qubits=(0, 1),
                        parameters={
                            "angle": ParameterExpression.parameter("theta")
                        },
                    ),
                ),
            ),
        ),
    )


def parity_generator_recipe() -> dict:
    return {
        "type": "pauli_sum",
        "terms": [{"pauli": "ZZ", "coeff": 1.0}],
    }


class TrustedEvaluatorTests(unittest.TestCase):
    def test_evaluator_optimizes_and_derives_all_reported_counts(self) -> None:
        result = evaluate_public_problem(
            two_qubit_problem(),
            parity_ansatz(),
            protocol=EvaluationProtocol(max_evals=64, restarts=2, seed=13),
        )
        self.assertTrue(result.receipt.valid, result.receipt.violations)
        self.assertLess(result.receipt.best_energy, -1.99)
        self.assertEqual(result.receipt.audit["unique_trainable_params"], 1)
        self.assertEqual(result.receipt.audit["parameter_occurrences"], {"theta": 1})
        self.assertGreater(result.receipt.metrics["generic_twoq_count"], 0)
        self.assertEqual(result.receipt.objective_calls, len(result.receipt.energy_trace))
        self.assertNotIn("best_values", result.receipt.to_dict())

    def test_candidate_cannot_smuggle_reported_energy_or_counts(self) -> None:
        forged = parity_ansatz().to_dict()
        forged["energy"] = -999.0
        forged["num_params"] = 0
        result = evaluate_public_problem(
            two_qubit_problem(),
            forged,
            protocol=EvaluationProtocol(max_evals=8, restarts=1),
        )
        self.assertFalse(result.receipt.valid)
        self.assertIsNone(result.receipt.best_energy)
        self.assertTrue(result.receipt.violations)


class ClosedControllerTests(unittest.TestCase):
    def support_parity(self, controller: ResearchController) -> None:
        controller.dispatch_external(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "parity",
                "claim": {
                    "kind": "exact_pauli_symmetry",
                    "generator": parity_generator_recipe(),
                },
            }
        )
        controller.dispatch_external(
            {
                "type": "request_probe",
                "hypothesis_id": "parity",
                "probe_id": "comm_zz",
                "probe": {
                    "type": "normalized_commutator",
                    "generator": parity_generator_recipe(),
                },
            }
        )

    def test_semantically_equivalent_candidate_is_not_a_fresh_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(),
                Path(directory) / "events.jsonl",
                total_budget=20.0,
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "structure",
                    "claim": {"kind": "ansatz_structure", "family": "test family"},
                }
            )
            original = parity_ansatz().to_dict()
            accepted = controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "original",
                    "hypothesis_id": "structure",
                    "spec": original,
                    "metadata": {
                        "enforcement": "unconstrained",
                        "prediction": "the original structure improves its baseline",
                    },
                }
            )
            self.assertEqual(accepted.result, {"accepted": True})
            renamed = parity_ansatz().to_dict()
            renamed["name"] = "cosmetic_new_name"
            renamed["parameters"] = [{"name": "alpha"}]
            renamed["layers"][0]["name"] = "cosmetic_layer"
            renamed["layers"][0]["operations"][0]["parameters"]["angle"][
                "terms"
            ][0]["parameter"] = "alpha"
            with self.assertRaisesRegex(ControllerError, "semantically equivalent"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "renamed_duplicate",
                        "hypothesis_id": "structure",
                        "spec": renamed,
                        "metadata": {
                            "enforcement": "unconstrained",
                            "prediction": "a cosmetic rename should not be new evidence",
                        },
                    }
                )

    def test_probe_to_candidate_to_commit_feedback_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(),
                Path(directory) / "events.jsonl",
                total_budget=20.0,
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "parity",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": parity_generator_recipe(),
                    },
                    # External cost is ignored; accounting is evaluator-owned.
                    "cost": 999,
                }
            )
            probe = controller.dispatch_external(
                {
                    "type": "request_probe",
                    "hypothesis_id": "parity",
                    "probe_id": "comm_zz",
                    "probe": {
                        "type": "normalized_commutator",
                        "generator": parity_generator_recipe(),
                    },
                }
            )
            self.assertTrue(probe.result["metrics"]["exact"])
            self.assertEqual(
                probe.state["hypotheses"]["parity"]["status"], "SUPPORTED"
            )
            with self.assertRaisesRegex(ControllerError, "one fixed deterministic"):
                controller.dispatch_external(
                    {
                        "type": "request_probe",
                        "hypothesis_id": "parity",
                        "probe_id": "comm_zz_repeat",
                        "probe": {
                            "type": "normalized_commutator",
                            "generator": parity_generator_recipe(),
                        },
                    }
                )

            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "pair_xx",
                    "hypothesis_id": "parity",
                    "spec": parity_ansatz().to_dict(),
                    "metadata": {
                        "enforcement": "preserve",
                        "prediction": "the structured move improves on its zero-angle baseline",
                    },
                }
            )
            for stage in ("audit", "smoke", "promotion"):
                receipt = controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "pair_xx",
                        "evaluation_id": f"pair_xx_{stage}",
                        "stage": stage,
                    }
                )
                self.assertTrue(receipt.state["evaluations"][f"pair_xx_{stage}"]["passed"])

            with self.assertRaises(ControllerError):
                controller.dispatch_external(
                    {
                        "type": "retire",
                        "entity": "candidate",
                        "entity_id": "pair_xx",
                        "reason": "attempt to erase a passed promotion",
                    }
                )
            with self.assertRaises(ControllerError):
                controller.dispatch_external(
                    {
                        "type": "revise",
                        "entity": "candidate",
                        "source_id": "pair_xx",
                        "new_id": "pair_xx_revised",
                        "replacement": parity_ansatz().to_dict(),
                        "reason": "attempt to replace a passed promotion",
                    }
                )
            with self.assertRaises(ControllerError):
                controller.dispatch_external(
                    {"type": "commit", "candidate_id": "pair_xx"}
                )
            committed = controller.dispatch_external(
                {
                    "type": "commit",
                    "candidate_id": "pair_xx",
                    "evidence_ids": ["pair_xx_promotion"],
                    "comparison": {
                        "mode": "documented_non_dominance",
                        "reason": "the current protocol produced no evaluated candidate that dominates this promotion",
                        "evidence_ids": ["pair_xx_promotion"],
                    },
                }
            )
            self.assertEqual(committed.state["committed_candidate_id"], "pair_xx")
            self.assertEqual(committed.state["terminal_decision"], "positive_commit")
            self.assertLess(committed.state["spent_budget"], 20.0)

    def test_negative_close_requires_terminal_branches_and_substantive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("autovqe.controller.MAX_HISTORY_EVENTS", 4):
                controller = ResearchController(
                    two_qubit_problem(), Path(directory) / "events.jsonl"
                )
                noncommuting = {
                    "type": "pauli_sum",
                    "terms": [{"pauli": "XX", "coeff": 1.0}],
                }
                controller.dispatch_external(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "not_a_symmetry",
                        "claim": {
                            "kind": "exact_pauli_symmetry",
                            "generator": noncommuting,
                        },
                    }
                )
                controller.dispatch_external(
                    {
                        "type": "request_probe",
                        "hypothesis_id": "not_a_symmetry",
                        "probe_id": "comm_xx",
                        "probe": {
                            "type": "normalized_commutator",
                            "generator": noncommuting,
                        },
                    }
                )
                with self.assertRaises(ControllerError):
                    controller.dispatch_external(
                        {
                            "type": "close_negative",
                            "reason": "the only hypothesis was refuted",
                            "evidence_ids": ["comm_xx"],
                        }
                    )
                controller.dispatch_external(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": "not_a_symmetry",
                        "reason": "the trusted commutator probe refuted it",
                    }
                )
                before_close = controller.state.last_seq
                with self.assertRaises(ControllerError):
                    controller.dispatch_external(
                        {
                            "type": "propose_hypothesis",
                            "hypothesis_id": "would_consume_terminal_slot",
                            "claim": {
                                "kind": "exact_pauli_symmetry",
                                "generator": noncommuting,
                            },
                        }
                    )
                self.assertEqual(controller.state.last_seq, before_close)
                closed = controller.dispatch_external(
                    {
                        "type": "close_negative",
                        "reason": "all investigated branches were refuted",
                        "evidence_ids": ["comm_xx"],
                    }
                )
                self.assertEqual(closed.state["terminal_decision"], "negative_close")
                self.assertEqual(closed.state["last_seq"], 3)
                with self.assertRaises(ControllerError):
                    controller.dispatch_external(
                        {
                            "type": "propose_hypothesis",
                            "hypothesis_id": "too_late",
                            "claim": {"kind": "null_control"},
                        }
                    )

    def test_automatic_admission_alone_cannot_ground_negative_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            proposed = controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "control",
                    "claim": {"kind": "null_control"},
                }
            )
            admission_id = proposed.state["hypotheses"]["control"]["probe_ids"][0]
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "control",
                    "reason": "no experiment was run",
                }
            )
            with self.assertRaises(ControllerError):
                controller.dispatch_external(
                    {
                        "type": "close_negative",
                        "reason": "unsupported negative conclusion",
                        "evidence_ids": [admission_id],
                    }
                )
            self.assertFalse(controller.state.terminal)

    def test_negative_close_requires_evidence_for_every_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            noncommuting = {
                "type": "pauli_sum",
                "terms": [{"pauli": "XX", "coeff": 1.0}],
            }
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "refuted",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": noncommuting,
                    },
                }
            )
            controller.dispatch_external(
                {
                    "type": "request_probe",
                    "hypothesis_id": "refuted",
                    "probe_id": "p.refuted",
                    "probe": {
                        "type": "normalized_commutator",
                        "generator": noncommuting,
                    },
                }
            )
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "refuted",
                    "reason": "commutator refuted it",
                }
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "untested_structure",
                    "claim": {"kind": "ansatz_structure", "family": "untested"},
                }
            )
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "untested_structure",
                    "reason": "retired without an experiment",
                }
            )
            with self.assertRaisesRegex(ControllerError, "untested_structure"):
                controller.dispatch_external(
                    {
                        "type": "close_negative",
                        "reason": "one branch was tested and one was not",
                        "evidence_ids": ["p.refuted"],
                    }
                )
            self.assertFalse(controller.state.terminal)

    def test_commit_rejects_candidate_without_preregistered_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            # Build a valid lifecycle cheaply through the internal event reducer;
            # the assertion under test is the external trusted commit boundary.
            controller.loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "test"},
                }
            )
            controller.loop.dispatch(
                {
                    "type": "record_probe",
                    "hypothesis_id": "h1",
                    "probe_id": "p1",
                    "verdict": "supported",
                    "result": {},
                }
            )
            controller.loop.dispatch(
                {
                    "type": "submit_candidate",
                    "candidate_id": "c1",
                    "hypothesis_id": "h1",
                    "spec": {"test": True},
                    "metadata": {"enforcement": "unconstrained"},
                }
            )
            for stage in ("audit", "smoke", "promotion"):
                controller.loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": "c1",
                        "evaluation_id": f"e.{stage}",
                        "stage": stage,
                        "passed": True,
                        "metrics": {},
                    }
                )
            before = controller.state.last_seq
            with self.assertRaises(ControllerError):
                controller.dispatch_external(
                    {
                        "type": "commit",
                        "candidate_id": "c1",
                        "evidence_ids": ["e.promotion"],
                        "comparison": {
                            "mode": "documented_non_dominance",
                            "reason": "no observed dominator",
                            "evidence_ids": ["e.promotion"],
                        },
                    }
                )
            self.assertEqual(controller.state.last_seq, before)

    def test_promotable_submission_and_revision_preregister_new_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "structure",
                    "claim": {"kind": "ansatz_structure", "family": "test"},
                }
            )
            with self.assertRaisesRegex(ControllerError, "preregister"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "missing_prediction",
                        "hypothesis_id": "structure",
                        "spec": parity_ansatz().to_dict(),
                        "metadata": {"enforcement": "unconstrained"},
                    }
                )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "first",
                    "hypothesis_id": "structure",
                    "spec": parity_ansatz().to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "prediction": "the first candidate improves its baseline",
                    },
                }
            )
            replacement = parity_ansatz().to_dict()
            replacement["layers"][0]["operations"][0]["parameters"]["angle"][
                "terms"
            ][0]["coefficient"] = 2.0
            base_revision = {
                "type": "revise",
                "entity": "candidate",
                "source_id": "first",
                "new_id": "second",
                "replacement": replacement,
                "reason": "test a different generator scale",
            }
            with self.assertRaisesRegex(ControllerError, "preregister"):
                controller.dispatch_external(base_revision)
            revised = controller.dispatch_external(
                {
                    **base_revision,
                    "metadata": {
                        "falsifier": "no improvement at the revised scale refutes it"
                    },
                }
            )
            new_metadata = revised.state["candidates"]["second"]["metadata"]
            self.assertEqual(new_metadata["enforcement"], "unconstrained")
            self.assertIn("falsifier", new_metadata)

    def test_commit_accepts_grounded_evaluated_competitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.loop.dispatch(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "test"},
                }
            )
            controller.loop.dispatch(
                {
                    "type": "record_probe",
                    "hypothesis_id": "h1",
                    "probe_id": "p1",
                    "verdict": "supported",
                    "result": {},
                }
            )
            for candidate_id in ("winner", "control"):
                controller.loop.dispatch(
                    {
                        "type": "submit_candidate",
                        "candidate_id": candidate_id,
                        "hypothesis_id": "h1",
                        "spec": {"test": candidate_id},
                        "metadata": {
                            "prediction": f"preregistered {candidate_id} outcome"
                        },
                    }
                )
                controller.loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"{candidate_id}.audit",
                        "stage": "audit",
                        "passed": True,
                        "metrics": {},
                    }
                )
                controller.loop.dispatch(
                    {
                        "type": "record_evaluation",
                        "candidate_id": candidate_id,
                        "evaluation_id": f"{candidate_id}.smoke",
                        "stage": "smoke",
                        "passed": True,
                        "metrics": {},
                    }
                )
            controller.loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": "winner",
                    "evaluation_id": "winner.promotion",
                    "stage": "promotion",
                    "passed": True,
                    "metrics": {},
                }
            )
            receipt = controller.dispatch_external(
                {
                    "type": "commit",
                    "candidate_id": "winner",
                    "evidence_ids": ["winner.promotion", "control.smoke"],
                    "comparison": {
                        "mode": "evaluated_competitor",
                        "candidate_id": "control",
                        "evidence_ids": ["control.smoke"],
                    },
                }
            )
            self.assertEqual(receipt.state["terminal_decision"], "positive_commit")

    def test_preserving_claim_rejects_a_symmetry_breaking_operation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            self.support_parity(controller)
            breaking = AnsatzSpec(
                num_qubits=2,
                parameters=("theta",),
                layers=(
                    LayerSpec(
                        operations=(
                            OperationSpec(
                                macro="PauliRotation",
                                qubits=(0,),
                                parameters={
                                    "angle": ParameterExpression.parameter("theta")
                                },
                                options={"pauli": "X"},
                            ),
                        ),
                    ),
                ),
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "breaks_parity",
                    "hypothesis_id": "parity",
                    "spec": breaking.to_dict(),
                    "metadata": {
                        "enforcement": "preserve",
                        "falsifier": "any nonzero operation-charge commutator refutes it",
                    },
                }
            )
            receipt = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "breaks_parity",
                    "evaluation_id": "breaks_parity_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(receipt.state["evaluations"]["breaks_parity_audit"]["passed"])
            self.assertIn("breaks its claimed", receipt.result["violations"][0])

    def test_conservation_macros_require_probed_exact_symmetry_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "named_family",
                    "claim": {
                        "kind": "ansatz_structure",
                        "family": "number_conserving",
                    },
                }
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "unsupported_exchange",
                    "hypothesis_id": "named_family",
                    "spec": xy_exchange_ansatz().to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "prediction": "exchange structure should improve the baseline",
                    },
                }
            )
            rejected = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "unsupported_exchange",
                    "evaluation_id": "unsupported_exchange_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(rejected.result["valid"])
            self.assertIn(
                "controller-SUPPORTED exact_pauli_symmetry parent",
                rejected.result["violations"][0],
            )

            self.support_parity(controller)
            probed_spec = xy_exchange_ansatz().to_dict()
            probed_spec["layers"][0]["operations"][0]["parameters"]["angle"][
                "terms"
            ][0]["coefficient"] = 2.0
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "probed_exchange",
                    "hypothesis_id": "parity",
                    "spec": probed_spec,
                    "metadata": {
                        "enforcement": "preserve",
                        "prediction": "the probed charge is preserved operation by operation",
                    },
                }
            )
            accepted = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "probed_exchange",
                    "evaluation_id": "probed_exchange_audit",
                    "stage": "audit",
                }
            )
            self.assertTrue(accepted.result["valid"], accepted.result)

    def test_unapproved_fixed_multiplier_is_rejected_at_trusted_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            self.support_parity(controller)
            spec = parity_ansatz().to_dict()
            spec["layers"][0]["operations"][0]["parameters"]["angle"]["terms"][0][
                "coefficient"
            ] = 0.123456789
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "encoded_angle",
                    "hypothesis_id": "parity",
                    "spec": spec,
                    "metadata": {
                        "enforcement": "preserve",
                        "falsifier": "an unapproved fixed multiplier invalidates the design",
                    },
                }
            )
            receipt = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "encoded_angle",
                    "evaluation_id": "encoded_angle_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(receipt.state["evaluations"]["encoded_angle_audit"]["passed"])
            self.assertIn("unapproved fixed", receipt.result["violations"][0])

    def test_agent_cannot_record_its_own_probe_or_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            for action_type in ("record_probe", "record_evaluation"):
                with self.subTest(action_type=action_type), self.assertRaises(
                    ControllerError
                ):
                    controller.dispatch_external({"type": action_type})

    def test_agent_metadata_cannot_claim_private_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            with self.assertRaisesRegex(ControllerError, "reference_energy"):
                controller.dispatch_external(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "leak",
                        "claim": {"reference_energy": -999.0},
                    }
                )

    def test_reference_moments_cannot_certify_an_exact_symmetry(self) -> None:
        problem = PublicProblem.create(
            num_qubits=1,
            pauli_terms=(PauliTerm("X", 1.0),),
            encoding=EncodingSpec(),
            sector=SectorSpec(),
            reference=ReferenceSpec(kind="computational_basis", occupation=(0,)),
            backend=BackendSpec(),
        )
        generator = {
            "type": "pauli_sum",
            "terms": [{"pauli": "Z", "coeff": 1.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(problem, Path(directory) / "events.jsonl")
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "false_z",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": generator,
                    },
                }
            )
            with self.assertRaisesRegex(ControllerError, "normalized_commutator"):
                controller.dispatch_external(
                    {
                        "type": "request_probe",
                        "hypothesis_id": "false_z",
                        "probe_id": "moments_only",
                        "probe": {
                            "type": "reference_moments",
                            "generator": generator,
                        },
                    }
                )
            receipt = controller.dispatch_external(
                {
                    "type": "request_probe",
                    "hypothesis_id": "false_z",
                    "probe_id": "comm_z",
                    "probe": {
                        "type": "normalized_commutator",
                        "generator": generator,
                    },
                }
            )
            self.assertFalse(receipt.result["controller_passed"])
            self.assertEqual(
                receipt.state["hypotheses"]["false_z"]["status"], "PROBED"
            )

    def test_exact_symmetry_candidate_cannot_opt_out_of_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            self.support_parity(controller)
            with self.assertRaisesRegex(ControllerError, "enforcement='preserve'"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "opt_out",
                        "hypothesis_id": "parity",
                        "spec": parity_ansatz().to_dict(),
                    }
                )

    def test_candidate_controlled_reference_is_rejected_by_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "generic",
                    "claim": {"kind": "ansatz_structure", "family": "control"},
                }
            )
            spec = AnsatzSpec(
                num_qubits=2,
                parameters=("theta",),
                reference=AnsatzReferenceSpec(macro="X", qubits=(0, 1)),
                layers=(
                    LayerSpec(
                        operations=(
                            OperationSpec(
                                macro="PauliRotation",
                                qubits=(0, 1),
                                parameters={
                                    "angle": ParameterExpression.parameter("theta")
                                },
                                options={"pauli": "XX"},
                            ),
                        )
                    ),
                ),
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "hardcoded_bits",
                    "hypothesis_id": "generic",
                    "spec": spec.to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "falsifier": "a candidate-controlled reference must fail audit",
                    },
                }
            )
            audit = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "hardcoded_bits",
                    "evaluation_id": "hardcoded_bits_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(audit.state["evaluations"]["hardcoded_bits_audit"]["passed"])
            self.assertIn("reference preparation", audit.result["violations"][0])

    def test_zero_parameter_zero_operation_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "generic",
                    "claim": {"kind": "ansatz_structure", "family": "empty"},
                }
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "empty",
                    "hypothesis_id": "generic",
                    "spec": AnsatzSpec(num_qubits=2).to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "falsifier": "an empty nonvariational candidate must fail audit",
                    },
                }
            )
            audit = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "empty",
                    "evaluation_id": "empty_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(audit.state["evaluations"]["empty_audit"]["passed"])
            self.assertIn("at least one operation", audit.result["violations"][0])

    def test_state_inert_parameterized_candidate_fails_smoke(self) -> None:
        problem = PublicProblem.create(
            num_qubits=1,
            pauli_terms=(PauliTerm("Z", 1.0),),
            encoding=EncodingSpec(),
            sector=SectorSpec(),
            reference=ReferenceSpec(),
            backend=BackendSpec(basis_gates=("rz", "sx", "x")),
        )
        inert = AnsatzSpec(
            num_qubits=1,
            parameters=("theta",),
            layers=(
                LayerSpec(
                    operations=(
                        OperationSpec(
                            macro="PauliRotation",
                            qubits=(0,),
                            parameters={
                                "angle": ParameterExpression.parameter("theta")
                            },
                            options={"pauli": "Z"},
                        ),
                    )
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(problem, Path(directory) / "events.jsonl")
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "phase_only",
                    "claim": {"kind": "ansatz_structure", "family": "phase_only"},
                }
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "inert",
                    "hypothesis_id": "phase_only",
                    "spec": inert.to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "falsifier": "no baseline energy improvement refutes useful motion",
                    },
                }
            )
            audit = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "inert",
                    "evaluation_id": "inert_audit",
                    "stage": "audit",
                }
            )
            self.assertTrue(audit.state["evaluations"]["inert_audit"]["passed"])
            smoke = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "inert",
                    "evaluation_id": "inert_smoke",
                    "stage": "smoke",
                }
            )
            self.assertFalse(smoke.state["evaluations"]["inert_smoke"]["passed"])
            self.assertAlmostEqual(smoke.result["energy_improvement"], 0.0)

    def test_budget_and_duplicate_ids_are_checked_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(),
                Path(directory) / "events.jsonl",
                total_budget=2.4,
            )
            self.support_parity(controller)
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "pair_xx",
                    "hypothesis_id": "parity",
                    "spec": parity_ansatz().to_dict(),
                    "metadata": {
                        "enforcement": "preserve",
                        "prediction": "the pair rotation improves its baseline",
                    },
                }
            )
            controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "pair_xx",
                    "evaluation_id": "audit_once",
                    "stage": "audit",
                }
            )
            with self.assertRaisesRegex(ControllerError, "already exists"):
                controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "pair_xx",
                        "evaluation_id": "audit_once",
                        "stage": "audit",
                    }
                )
            with self.assertRaisesRegex(ControllerError, "already has fixed audit"):
                controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "pair_xx",
                        "evaluation_id": "audit_repeat_new_id",
                        "stage": "audit",
                    }
                )
            before = controller.state.last_seq
            with self.assertRaisesRegex(ControllerError, "remaining budget"):
                controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "pair_xx",
                        "evaluation_id": "too_expensive",
                        "stage": "smoke",
                    }
                )
            self.assertEqual(controller.state.last_seq, before)

    def test_single_parameter_cannot_hide_an_unbounded_gate_fanout(self) -> None:
        repeated = tuple(
            OperationSpec(
                macro="PauliRotation",
                qubits=(0,),
                parameters={"angle": ParameterExpression.parameter("theta")},
                options={"pauli": "Z"},
            )
            for _ in range(65)
        )
        spec = AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            layers=(LayerSpec(operations=repeated),),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                two_qubit_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "fanout",
                    "claim": {"kind": "ansatz_structure", "family": "shared"},
                }
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "too_shared",
                    "hypothesis_id": "fanout",
                    "spec": spec.to_dict(),
                    "metadata": {
                        "enforcement": "unconstrained",
                        "falsifier": "excessive fanout must fail trusted audit",
                    },
                }
            )
            audit = controller.dispatch_external(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "too_shared",
                    "evaluation_id": "too_shared_audit",
                    "stage": "audit",
                }
            )
            self.assertFalse(audit.state["evaluations"]["too_shared_audit"]["passed"])
            self.assertIn("fan-out exceeds 64", audit.result["violations"][0])


if __name__ == "__main__":
    unittest.main()
