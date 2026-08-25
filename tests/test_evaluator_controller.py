from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autovqe.ansatz_ir import AnsatzSpec, OperationSpec, ParameterExpression
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    InitialStateSpec,
    PauliTerm,
    PublicProblem,
    SectorSpec,
)
from autovqe.controller import ControllerError, ResearchController, StepResult
from autovqe.evaluator import EvaluationResult, EvaluationRun, ResourceAudit


def two_qubit_problem() -> PublicProblem:
    return PublicProblem.create(
        num_qubits=2,
        pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 1.0)),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        initial_state=InitialStateSpec(),
        backend=BackendSpec(
            basis_gates=("rx", "ry", "rz", "cx"),
            coupling_map=((0, 1), (1, 0)),
        ),
    )


def rotation_ansatz(pauli: str = "XX", *, name: str = "pair") -> AnsatzSpec:
    return AnsatzSpec(
        name=name,
        num_qubits=2,
        parameters=("theta",),
        operations=(
            OperationSpec(
                macro="PauliRotation",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
                options={"pauli": pauli},
            ),
        ),
    )


def xy_exchange_ansatz() -> AnsatzSpec:
    return AnsatzSpec(
        name="exchange",
        num_qubits=2,
        parameters=("theta",),
        operations=(
            OperationSpec(
                macro="XYExchange",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
            ),
        ),
    )


def parity_generator() -> dict:
    return {
        "type": "pauli_sum",
        "terms": [{"pauli": "ZZ", "coeff": 1.0}],
    }


def resource_policy(twoq: int, total: int, depth: int) -> dict:
    return {
        "eligible": True,
        "observed": {
            "conservative_twoq_count": twoq,
            "conservative_total_gate_count": total,
            "conservative_depth": depth,
        },
    }


class ControllerTests(unittest.TestCase):
    def make_controller(
        self,
        directory: str,
        *,
        problem: PublicProblem | None = None,
        budget: float = 100.0,
    ) -> ResearchController:
        return ResearchController(
            problem or two_qubit_problem(),
            Path(directory) / "events.jsonl",
            total_budget=budget,
        )

    def propose_structure(
        self, controller: ResearchController, hypothesis_id: str = "structure"
    ) -> StepResult:
        return controller.dispatch_external(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": hypothesis_id,
                "claim": {"kind": "ansatz_structure", "family": hypothesis_id},
            }
        )

    def support_parity(self, controller: ResearchController) -> None:
        controller.dispatch_external(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "parity",
                "claim": {
                    "kind": "exact_pauli_symmetry",
                    "generator": parity_generator(),
                },
            }
        )
        controller.dispatch_external(
            {"type": "request_probe", "hypothesis_id": "parity"}
        )

    def submit(
        self,
        controller: ResearchController,
        candidate_id: str,
        spec: AnsatzSpec,
        *,
        hypothesis_id: str = "structure",
    ) -> StepResult:
        return controller.dispatch_external(
            {
                "type": "submit_candidate",
                "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id,
                "spec": spec.to_dict(),
                "metadata": {
                    "prediction": "the proposed structure improves its zero-angle baseline"
                },
            }
        )

    def test_structure_is_ready_without_fake_probe_and_result_is_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            step = self.propose_structure(controller)
            self.assertIsInstance(step, StepResult)
            self.assertEqual(
                step.state_summary["hypotheses"]["structure"]["status"], "READY"
            )
            self.assertEqual(controller.state.probes, {})
            payload = step.to_dict()
            self.assertNotIn("state", payload)
            self.assertNotIn("spec", str(payload))
            self.assertNotIn("trace", str(payload))

    def test_probe_is_derived_and_has_a_deterministic_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "parity",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": parity_generator(),
                    },
                }
            )
            with self.assertRaisesRegex(ControllerError, "extra"):
                controller.dispatch_external(
                    {
                        "type": "request_probe",
                        "hypothesis_id": "parity",
                        "probe_id": "agent-chosen",
                    }
                )
            result = controller.dispatch_external(
                {"type": "request_probe", "hypothesis_id": "parity"}
            )
            self.assertEqual(result.result["probe_id"], "probe:parity")
            self.assertTrue(result.result["passed"])
            self.assertEqual(
                result.state_summary["hypotheses"]["parity"]["status"],
                "SUPPORTED",
            )
            self.assertEqual(set(controller.state.probes), {"probe:parity"})

    def test_agent_ids_leave_room_for_controller_evidence_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            structure_id = "h" * 96
            candidate_id = "c" * 96
            self.propose_structure(controller, structure_id)
            self.submit(
                controller,
                candidate_id,
                rotation_ansatz(),
                hypothesis_id=structure_id,
            )
            audited = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": candidate_id}
            )
            self.assertEqual(
                audited.result["evaluation_id"],
                f"evaluation:{candidate_id}:audit",
            )

            symmetry_id = "p" * 96
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": symmetry_id,
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": parity_generator(),
                    },
                }
            )
            probed = controller.dispatch_external(
                {"type": "request_probe", "hypothesis_id": symmetry_id}
            )
            self.assertEqual(probed.result["probe_id"], f"probe:{symmetry_id}")

            with self.assertRaisesRegex(ControllerError, "0,95"):
                self.propose_structure(controller, "x" * 97)

    def test_candidate_must_be_typed_and_semantic_duplicates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "original", rotation_ansatz())
            renamed = AnsatzSpec(
                name="renamed",
                num_qubits=2,
                parameters=("alpha",),
                operations=(
                    OperationSpec(
                        macro="PauliRotation",
                        qubits=(0, 1),
                        parameters={
                            "angle": ParameterExpression.parameter("alpha")
                        },
                        options={"pauli": "XX"},
                    ),
                ),
            )
            with self.assertRaisesRegex(ControllerError, "semantically equivalent"):
                self.submit(controller, "renamed", renamed)
            forged = rotation_ansatz("YY").to_dict()
            forged["reported_num_parameters"] = 0
            with self.assertRaisesRegex(ControllerError, "invalid candidate"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "forged",
                        "hypothesis_id": "structure",
                        "spec": forged,
                        "metadata": {"prediction": "forged counts should be ignored"},
                    }
                )
            with self.assertRaisesRegex(ControllerError, "unsupported fields"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "metadata_forgery",
                        "hypothesis_id": "structure",
                        "spec": rotation_ansatz("YY").to_dict(),
                        "metadata": {
                            "prediction": "a legitimate prediction",
                            "reported_energy": -999.0,
                        },
                    }
                )

    def test_evaluation_stage_and_id_are_controller_owned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            with self.assertRaisesRegex(ControllerError, "extra"):
                controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "candidate",
                        "stage": "promotion",
                    }
                )
            audit = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "candidate"}
            )
            self.assertEqual(audit.result["stage"], "audit")
            self.assertEqual(
                audit.result["evaluation_id"], "evaluation:candidate:audit"
            )
            self.assertTrue(audit.result["passed"], audit.result)
            smoke = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "candidate"}
            )
            self.assertEqual(smoke.result["stage"], "smoke")
            self.assertTrue(smoke.result["passed"], smoke.result)
            self.assertNotIn("optimized_parameter_binding", smoke.result)
            self.assertIn("trace_summary", smoke.result)
            self.assertLessEqual(len(smoke.result["trace_summary"]), 8)
            self.assertNotIn("energy_trace", smoke.result)

    def test_baseline_failure_is_a_harness_error_not_scientific_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "candidate"}
            )
            before = controller.state

            with patch.object(
                controller,
                "_baseline_energy",
                side_effect=RuntimeError("backend unavailable"),
            ), self.assertRaisesRegex(ControllerError, "baseline evaluation failed"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "candidate"}
                )

            after = controller.state
            self.assertEqual(after.last_seq, before.last_seq)
            self.assertEqual(after.spent_budget, before.spent_budget)
            self.assertEqual(
                after.candidates["candidate"].status,
                before.candidates["candidate"].status,
            )
            self.assertNotIn("evaluation:candidate:smoke", after.evaluations)

            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "candidate",
                    "entity_id": "candidate",
                    "reason": "harness error is not falsification",
                }
            )
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "structure",
                    "reason": "harness error is not falsification",
                }
            )
            with self.assertRaisesRegex(ControllerError, "structure"):
                controller.dispatch_external(
                    {"type": "close_negative", "reason": "must fail closed"}
                )

    def test_optimizer_failure_does_not_consume_scientific_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "candidate"}
            )
            before = controller.state
            failed_run = EvaluationRun(
                result=EvaluationResult(
                    valid=False,
                    best_energy=None,
                    trace_summary=(),
                    objective_calls=0,
                    optimizer="COBYLA",
                    seed=7,
                    optimized_parameter_binding=None,
                    audit={},
                    metrics={},
                    violations=("optimizer backend failed",),
                ),
                best_values=(),
                final_circuit=None,
            )
            with patch(
                "autovqe.controller.evaluate_public_problem",
                return_value=failed_run,
            ), self.assertRaisesRegex(ControllerError, "without producing scientific evidence"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "candidate"}
                )
            after = controller.state
            self.assertEqual(after.last_seq, before.last_seq)
            self.assertEqual(after.spent_budget, before.spent_budget)
            self.assertEqual(after.candidates["candidate"].status.value, "AUDITED")

    def test_resource_failure_is_recorded_before_any_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "oversized", rotation_ansatz())
            huge = ResourceAudit(
                valid=True,
                audit={},
                metrics={
                    "template_twoq_count": 999,
                    "audit_worst_twoq_count": 999,
                    "canonical_template_twoq_count": 1,
                    "canonical_audit_worst_twoq_count": 1,
                    "template_total_gate_count": 999,
                    "audit_worst_total_gate_count": 999,
                    "canonical_template_total_gate_count": 1,
                    "canonical_audit_worst_total_gate_count": 1,
                    "template_depth": 999,
                    "audit_worst_depth": 999,
                    "canonical_template_depth": 1,
                    "canonical_audit_worst_depth": 1,
                },
            )
            with patch("autovqe.controller.audit_public_candidate", return_value=huge), patch(
                "autovqe.controller.evaluate_public_problem"
            ) as optimize:
                audit = controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "oversized"}
                )
            optimize.assert_not_called()
            self.assertFalse(audit.result["passed"])
            self.assertFalse(audit.result["resource_policy"]["eligible"])
            self.assertEqual(
                audit.state_summary["candidates"]["oversized"]["status"],
                "RETIRED",
            )

    def test_unexpected_audit_failure_does_not_retire_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            before = controller.state
            with patch(
                "autovqe.controller.compile_ansatz",
                side_effect=RuntimeError("compiler service failed"),
            ), self.assertRaisesRegex(ControllerError, "audit infrastructure failed"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "candidate"}
                )
            after = controller.state
            self.assertEqual(after.last_seq, before.last_seq)
            self.assertEqual(after.spent_budget, before.spent_budget)
            self.assertEqual(after.candidates["candidate"].status.value, "CANDIDATE")

    def test_evaluator_audit_failure_does_not_retire_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            before = controller.state
            with patch(
                "autovqe.evaluator._physical_metrics",
                side_effect=RuntimeError("transpiler service failed"),
            ), self.assertRaisesRegex(ControllerError, "audit infrastructure failed"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "candidate"}
                )
            after = controller.state
            self.assertEqual(after.last_seq, before.last_seq)
            self.assertEqual(after.spent_budget, before.spent_budget)
            self.assertEqual(after.candidates["candidate"].status.value, "CANDIDATE")

    def test_special_conservation_gate_requires_supported_exact_symmetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "unsupported", xy_exchange_ansatz())
            rejected = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "unsupported"}
            )
            self.assertFalse(rejected.result["passed"])
            self.assertIn("SUPPORTED exact_pauli_symmetry", rejected.result["violations"][0])

    def test_failed_atomic_representation_can_be_repaired_without_retrying_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            problem = PublicProblem.create(
                num_qubits=2,
                pauli_terms=(PauliTerm("ZI", 1.0), PauliTerm("IZ", 2.0)),
                backend=BackendSpec(
                    basis_gates=("rx", "ry", "rz", "cx"),
                    coupling_map=((0, 1), (1, 0)),
                ),
            )
            controller = self.make_controller(directory, problem=problem)
            self.propose_structure(controller)
            controller.dispatch_external(
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
                }
            )
            controller.dispatch_external(
                {"type": "request_probe", "hypothesis_id": "total_z"}
            )
            theta = ParameterExpression.parameter("theta")
            expanded = AnsatzSpec(
                name="expanded_exchange",
                num_qubits=2,
                parameters=("theta",),
                operations=tuple(
                    OperationSpec(
                        macro="PauliRotation",
                        qubits=(0, 1),
                        parameters={"angle": theta},
                        options={"pauli": pauli},
                    )
                    for pauli in ("XX", "YY")
                ),
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "expanded",
                    "hypothesis_id": "structure",
                    "spec": expanded.to_dict(),
                    "symmetry_evidence_ids": ["probe:total_z"],
                    "metadata": {"prediction": "tied generators preserve total Z"},
                }
            )
            failed = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "expanded"}
            )
            self.assertFalse(failed.result["passed"])

            repaired = controller.dispatch_external(
                {
                    "type": "revise",
                    "entity": "candidate",
                    "source_id": "expanded",
                    "new_id": "atomic",
                    "replacement": xy_exchange_ansatz().to_dict(),
                    "reason": "use the trusted atomic representation for symmetry audit",
                    "symmetry_evidence_ids": ["probe:total_z"],
                    "metadata": {"prediction": "atomic exchange preserves total Z"},
                }
            )
            self.assertTrue(repaired.result["accepted"])
            audited = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "atomic"}
            )
            self.assertTrue(audited.result["passed"], audited.result)

        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.support_parity(controller)
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "supported",
                    "hypothesis_id": "structure",
                    "spec": xy_exchange_ansatz().to_dict(),
                    "symmetry_evidence_ids": ["probe:parity"],
                    "metadata": {
                        "prediction": "exchange preserves the supported parity"
                    },
                }
            )
            accepted = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "supported"}
            )
            self.assertTrue(accepted.result["passed"], accepted.result)
            self.assertIn("symmetry_audit", accepted.result)
            self.assertEqual(
                controller.state.candidates["supported"].symmetry_evidence_ids,
                ("probe:parity",),
            )

    def test_spectator_symmetry_cannot_justify_a_special_gate(self) -> None:
        problem = PublicProblem.create(
            num_qubits=3,
            pauli_terms=(PauliTerm("IIX", 1.0), PauliTerm("ZII", 0.2)),
            initial_state=InitialStateSpec(
                kind="computational_basis", occupation=(0, 0, 0)
            ),
            backend=BackendSpec(basis_gates=("rx", "ry", "rz", "cx")),
        )
        exchange = AnsatzSpec(
            num_qubits=3,
            parameters=("theta",),
            operations=(
                OperationSpec(
                    macro="XYExchange",
                    qubits=(0, 1),
                    parameters={
                        "angle": ParameterExpression.parameter("theta")
                    },
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory, problem=problem)
            self.propose_structure(controller)
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "spectator",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": {
                            "type": "pauli_sum",
                            "terms": [{"pauli": "ZII"}],
                        },
                    },
                }
            )
            controller.dispatch_external(
                {"type": "request_probe", "hypothesis_id": "spectator"}
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "vacuous_exchange",
                    "hypothesis_id": "structure",
                    "spec": exchange.to_dict(),
                    "symmetry_evidence_ids": ["probe:spectator"],
                    "metadata": {"prediction": "test spectator relevance"},
                }
            )
            audit = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "vacuous_exchange"}
            )
            self.assertFalse(audit.result["passed"])
            self.assertIn("no relevant cited symmetry", audit.result["violations"][0])

    def test_operation_level_symmetry_breaking_is_rejected(self) -> None:
        breaking = AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            operations=(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=(0,),
                    parameters={"angle": ParameterExpression.parameter("theta")},
                    options={"pauli": "X"},
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.support_parity(controller)
            self.submit(
                controller, "breaking", breaking, hypothesis_id="parity"
            )
            audit = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "breaking"}
            )
            self.assertFalse(audit.result["passed"])
            self.assertIn("breaks a cited exact symmetry", audit.result["violations"][0])

    def test_failed_candidate_can_be_revised_without_erasing_evidence(self) -> None:
        problem = PublicProblem.create(
            num_qubits=1,
            pauli_terms=(PauliTerm("Z", 1.0),),
            initial_state=InitialStateSpec(),
            backend=BackendSpec(basis_gates=("rz", "sx", "x")),
        )
        inert = AnsatzSpec(
            num_qubits=1,
            parameters=("theta",),
            operations=(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=(0,),
                    parameters={"angle": ParameterExpression.parameter("theta")},
                    options={"pauli": "Z"},
                ),
            ),
        )
        revised = AnsatzSpec(
            num_qubits=1,
            parameters=("phi",),
            operations=(
                OperationSpec(
                    macro="PauliRotation",
                    qubits=(0,),
                    parameters={"angle": ParameterExpression.parameter("phi")},
                    options={"pauli": "X"},
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory, problem=problem)
            self.propose_structure(controller)
            self.submit(controller, "inert", inert)
            self.assertTrue(
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "inert"}
                ).result["passed"]
            )
            smoke = controller.dispatch_external(
                {"type": "evaluate_candidate", "candidate_id": "inert"}
            )
            self.assertFalse(smoke.result["passed"])
            revision = controller.dispatch_external(
                {
                    "type": "revise",
                    "entity": "candidate",
                    "source_id": "inert",
                    "new_id": "active_rotation",
                    "replacement": revised.to_dict(),
                    "reason": "phase-only motion did not change energy",
                    "metadata": {
                        "prediction": "an X rotation explores lower-energy states"
                    },
                }
            )
            self.assertEqual(
                revision.state_summary["candidates"]["active_rotation"]["status"],
                "CANDIDATE",
            )
            self.assertIn("evaluation:inert:smoke", controller.state.evaluations)

    def _seed_commit_state(
        self,
        controller: ResearchController,
        *,
        target_energy: float,
        target_resources: tuple[int, int, int],
        comparator_energy: float,
        comparator_resources: tuple[int, int, int],
        include_comparator: bool = True,
        target_parameters: int = 1,
        comparator_parameters: int = 1,
    ) -> None:
        loop = controller.loop
        loop.dispatch(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "target_h",
                "claim": {"kind": "ansatz_structure", "family": "target"},
            }
        )
        loop.dispatch(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "control_h",
                "claim": {"kind": "ansatz_structure", "family": "control"},
            }
        )
        for candidate_id, pauli, hypothesis_id in (
            ("target", "XX", "target_h"),
            ("control", "YY", "control_h"),
        ):
            loop.dispatch(
                {
                    "type": "submit_candidate",
                    "candidate_id": candidate_id,
                    "hypothesis_id": hypothesis_id,
                    "spec": rotation_ansatz(pauli).to_dict(),
                    "metadata": {
                        "prediction": "registered before evaluation",
                        "enforcement": "unconstrained",
                    },
                }
            )
            loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": candidate_id,
                    "evaluation_id": f"evaluation:{candidate_id}:audit",
                    "stage": "audit",
                    "passed": True,
                    "metrics": {"valid": True},
                }
            )
        target_point = resource_policy(*target_resources)
        control_point = resource_policy(*comparator_resources)
        loop.dispatch(
            {
                "type": "record_evaluation",
                "candidate_id": "target",
                "evaluation_id": "evaluation:target:smoke",
                "stage": "smoke",
                "passed": True,
                "metrics": {
                    "valid": True,
                    "best_energy": target_energy,
                    "resource_policy": target_point,
                    "audit": {"unique_trainable_params": target_parameters},
                },
            }
        )
        loop.dispatch(
            {
                "type": "record_evaluation",
                "candidate_id": "target",
                "evaluation_id": "evaluation:target:promotion",
                "stage": "promotion",
                "passed": True,
                "metrics": {
                    "valid": True,
                    "best_energy": target_energy,
                    "resource_policy": target_point,
                    "audit": {"unique_trainable_params": target_parameters},
                },
            }
        )
        if include_comparator:
            loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": "control",
                    "evaluation_id": "evaluation:control:smoke",
                    "stage": "smoke",
                    "passed": True,
                    "metrics": {
                        "valid": True,
                        "best_energy": comparator_energy,
                        "resource_policy": control_point,
                        "audit": {
                            "unique_trainable_params": comparator_parameters
                        },
                    },
                }
            )
            loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": "control",
                    "evaluation_id": "evaluation:control:promotion",
                    "stage": "promotion",
                    "passed": True,
                    "metrics": {
                        "valid": True,
                        "best_energy": comparator_energy,
                        "resource_policy": control_point,
                        "audit": {
                            "unique_trainable_params": comparator_parameters
                        },
                    },
                }
            )

    def test_commit_derives_real_comparison_and_rejects_dominated_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self._seed_commit_state(
                controller,
                target_energy=-1.5,
                target_resources=(10, 30, 20),
                comparator_energy=-1.0,
                comparator_resources=(5, 20, 10),
            )
            committed = controller.dispatch_external(
                {"type": "commit", "candidate_id": "target"}
            )
            comparison = committed.result["comparison"]
            self.assertEqual(comparison["target"]["best_energy"], -1.5)
            self.assertEqual(
                comparison["evaluations"][0]["resources"]
                ["conservative_twoq_count"],
                5,
            )
            self.assertEqual(
                committed.state_summary["terminal_decision"], "positive_commit"
            )
            self.assertNotIn("documented_non_dominance", str(committed.to_dict()))

        cases = (
            (-0.5, (1, 2, 2), -1.0, (20, 50, 40), "energetically worse"),
            (-1.0, (10, 30, 20), -1.0, (5, 20, 10), "Pareto-dominated"),
        )
        for target_energy, target_r, control_energy, control_r, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                controller = self.make_controller(directory)
                self._seed_commit_state(
                    controller,
                    target_energy=target_energy,
                    target_resources=target_r,
                    comparator_energy=control_energy,
                    comparator_resources=control_r,
                )
                with self.assertRaisesRegex(ControllerError, message):
                    controller.dispatch_external(
                        {"type": "commit", "candidate_id": "target"}
                    )

        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self._seed_commit_state(
                controller,
                target_energy=-1.0,
                target_resources=(5, 10, 8),
                comparator_energy=-1.0,
                comparator_resources=(5, 10, 8),
                target_parameters=8,
                comparator_parameters=1,
            )
            with self.assertRaisesRegex(ControllerError, "Pareto-dominated"):
                controller.dispatch_external(
                    {"type": "commit", "candidate_id": "target"}
                )

    def test_dominated_promoted_candidate_can_be_retired_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self._seed_commit_state(
                controller,
                target_energy=-0.5,
                target_resources=(10, 30, 20),
                comparator_energy=-1.0,
                comparator_resources=(5, 20, 10),
            )
            retired = controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "candidate",
                    "entity_id": "target",
                    "reason": "fair comparison dominated the promotion",
                }
            )
            self.assertTrue(retired.result["accepted"])
            self.assertEqual(controller.state.candidates["target"].status.value, "RETIRED")
            self.assertEqual(
                set(controller.state.candidates["target"].disposition_evidence_ids),
                {
                    "evaluation:target:promotion",
                    "evaluation:control:promotion",
                },
            )

    def test_promotion_requires_a_smoked_different_hypothesis_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller)
            self.submit(controller, "target", rotation_ansatz())
            self.assertTrue(
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "target"}
                ).result["passed"]
            )
            self.assertTrue(
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "target"}
                ).result["passed"]
            )
            before = controller.state.spent_budget
            with self.assertRaisesRegex(ControllerError, "different-hypothesis"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "target"}
                )
            self.assertEqual(controller.state.spent_budget, before)

    def test_first_promotion_reserves_the_next_comparison_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory, budget=12.0)
            self._seed_commit_state(
                controller,
                target_energy=-1.0,
                target_resources=(2, 4, 3),
                comparator_energy=-0.5,
                comparator_resources=(2, 4, 3),
                include_comparator=False,
            )
            controller.loop.dispatch(
                {
                    "type": "record_evaluation",
                    "candidate_id": "control",
                    "evaluation_id": "evaluation:control:smoke",
                    "stage": "smoke",
                    "passed": True,
                    "metrics": {
                        "valid": True,
                        "best_energy": -0.5,
                        "resource_policy": resource_policy(2, 4, 3),
                        "audit": {"unique_trainable_params": 1},
                    },
                }
            )
            before = controller.state
            with self.assertRaisesRegex(ControllerError, "reserved fair comparison"):
                controller.dispatch_external(
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "budget_leak",
                        "claim": {"kind": "null_control"},
                    }
                )
            after = controller.state
            self.assertEqual(after.last_seq, before.last_seq)
            self.assertEqual(after.spent_budget, before.spent_budget)

    def test_commit_requires_an_evaluated_comparator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self._seed_commit_state(
                controller,
                target_energy=-1.0,
                target_resources=(1, 2, 2),
                comparator_energy=-0.5,
                comparator_resources=(2, 4, 4),
                include_comparator=False,
            )
            with self.assertRaisesRegex(ControllerError, "different-hypothesis"):
                controller.dispatch_external(
                    {"type": "commit", "candidate_id": "target"}
                )

    def test_negative_close_derives_evidence_and_requires_branch_coverage(self) -> None:
        noncommuting = {
            "type": "pauli_sum",
            "terms": [{"pauli": "XX", "coeff": 1.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "false_symmetry",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": noncommuting,
                    },
                }
            )
            probe = controller.dispatch_external(
                {"type": "request_probe", "hypothesis_id": "false_symmetry"}
            )
            self.assertFalse(probe.result["passed"])
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "false_symmetry",
                    "reason": "commutator probe refuted it",
                }
            )
            with self.assertRaisesRegex(ControllerError, "extra"):
                controller.dispatch_external(
                    {
                        "type": "close_negative",
                        "reason": "closed",
                        "evidence_ids": ["agent-picked"],
                    }
                )
            with self.assertRaisesRegex(ControllerError, "objective-active"):
                controller.dispatch_external(
                    {
                        "type": "close_negative",
                        "reason": "a cheap symmetry refutation is not an ansatz search",
                    }
                )
            self.assertFalse(controller.state.terminal)

        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller, "untested")
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "untested",
                    "reason": "no experiment was run",
                }
            )
            with self.assertRaisesRegex(ControllerError, "untested"):
                controller.dispatch_external(
                    {"type": "close_negative", "reason": "ungrounded"}
                )

        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            self.propose_structure(controller, "audit_only")
            self.submit(
                controller,
                "compiled_only",
                rotation_ansatz(),
                hypothesis_id="audit_only",
            )
            self.assertTrue(
                controller.dispatch_external(
                    {
                        "type": "evaluate_candidate",
                        "candidate_id": "compiled_only",
                    }
                ).result["passed"]
            )
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "candidate",
                    "entity_id": "compiled_only",
                    "reason": "agent prose is not falsification",
                }
            )
            controller.dispatch_external(
                {
                    "type": "retire",
                    "entity": "hypothesis",
                    "entity_id": "audit_only",
                    "reason": "agent prose is not falsification",
                }
            )
            with self.assertRaisesRegex(ControllerError, "audit_only"):
                controller.dispatch_external(
                    {"type": "close_negative", "reason": "audit is not evidence"}
                )

    def test_phase_only_candidates_cannot_claim_negative_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory)
            for qubit in (0, 1):
                hypothesis_id = f"shortcut_{qubit}"
                candidate_id = f"phase_only_{qubit}"
                self.propose_structure(controller, hypothesis_id)
                self.submit(
                    controller,
                    candidate_id,
                    AnsatzSpec(
                        name=candidate_id,
                        num_qubits=2,
                        parameters=("theta",),
                        operations=(
                            OperationSpec(
                                macro="PauliRotation",
                                qubits=(qubit,),
                                parameters={
                                    "angle": ParameterExpression.parameter("theta")
                                },
                                options={"pauli": "Z"},
                            ),
                        ),
                    ),
                    hypothesis_id=hypothesis_id,
                )
                self.assertTrue(
                    controller.dispatch_external(
                        {"type": "evaluate_candidate", "candidate_id": candidate_id}
                    ).result["passed"]
                )
                smoke = controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": candidate_id}
                )
                self.assertFalse(smoke.result["passed"])
                self.assertEqual(smoke.result["objective_activity_fraction"], 0.0)
                controller.dispatch_external(
                    {
                        "type": "retire",
                        "entity": "hypothesis",
                        "entity_id": hypothesis_id,
                        "reason": "the deliberately inert candidate failed smoke",
                    }
                )
            with self.assertRaisesRegex(ControllerError, "objective activity"):
                controller.dispatch_external(
                    {"type": "close_negative", "reason": "cheap shortcut"}
                )
            self.assertFalse(controller.state.terminal)
            self.assertAlmostEqual(controller.state.spent_budget, 4.9)

    def test_agent_cannot_write_evidence_or_skip_budget_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller(directory, budget=0.3)
            for action_type in ("record_probe", "record_evaluation"):
                with self.subTest(action_type=action_type), self.assertRaisesRegex(
                    ControllerError, "evaluator-owned"
                ):
                    controller.dispatch_external({"type": action_type})
            self.propose_structure(controller)
            self.submit(controller, "candidate", rotation_ansatz())
            before = controller.state.last_seq
            with self.assertRaisesRegex(ControllerError, "remaining budget"):
                controller.dispatch_external(
                    {"type": "evaluate_candidate", "candidate_id": "candidate"}
                )
            self.assertEqual(controller.state.last_seq, before)


if __name__ == "__main__":
    unittest.main()
