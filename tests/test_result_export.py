from __future__ import annotations

import argparse
import json
import math
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autovqe import prepare, research_cli
from autovqe.compiler import compile_ansatz
from autovqe.evaluator import hamiltonian_from_public
from autovqe.observations import adapt_prepare_problem
from autovqe.probes import energy_from_circuit
from meta_agent import operator


class TrustedResultExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "agent"
        self.evaluator = self.root / "evaluator"
        self.problem = self.root / "problem.json"
        self.problem.write_text(
            json.dumps(
                {
                    "name": "result_export_test",
                    "pauli_terms": [
                        {"pauli": "ZI", "coeff": 1.0},
                        {"pauli": "IZ", "coeff": 1.0},
                    ],
                    "basis_gates": ["rx", "ry", "rz", "cx"],
                    "coupling_map": [[0, 1], [1, 0]],
                }
            ),
            encoding="utf-8",
        )
        self.campaign = self.root / "campaign.json"
        self.campaign.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "result_export_test",
                    "research_mode": "discovery",
                    "problem": str(self.problem),
                    "total_budget": 20.0,
                    "model_label": "test-model",
                    "local_agent_bundle": str(self.bundle),
                    "local_evaluator_run": str(self.evaluator),
                }
            ),
            encoding="utf-8",
        )
        operator.prepare_campaign(
            argparse.Namespace(
                campaign=self.campaign,
                security="local_unsealed",
                agent_bundle=None,
                evaluator_run=None,
                allow_dirty_evaluator=False,
                model_label=None,
            )
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _spec() -> dict:
        return {
            "version": 1,
            "name": "pair_xx",
            "num_qubits": 2,
            "parameters": [{"name": "theta"}],
            "reference": None,
            "layers": [
                {
                    "operations": [
                        {
                            "macro": "PauliRotation",
                            "qubits": [0, 1],
                            "parameters": {
                                "angle": {
                                    "terms": [
                                        {"parameter": "theta", "coefficient": 1.0}
                                    ],
                                    "constant": 0.0,
                                }
                            },
                            "options": {"pauli": "XX"},
                        }
                    ]
                }
            ],
        }

    def _action(self, action: dict) -> dict:
        return research_cli.execute_action(
            self.problem,
            self.evaluator,
            action,
            require_sealed=False,
        )

    def _export(self, output: Path | None = None) -> dict:
        return operator.export_result(
            argparse.Namespace(
                evaluator_run=self.evaluator,
                output=output,
                allow_unsealed=True,
            )
        )

    def _positive_terminal(self) -> None:
        self._action(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "structure",
                "claim": {"kind": "ansatz_structure", "family": "pair rotation"},
            }
        )
        self._action(
            {
                "type": "submit_candidate",
                "candidate_id": "candidate",
                "hypothesis_id": "structure",
                "spec": self._spec(),
                "metadata": {
                    "enforcement": "unconstrained",
                    "prediction": "the pair rotation improves the zero-angle baseline",
                    # This is intentionally untrusted and must not reach the export.
                    "optimized_parameter_binding": {"theta": -999.0},
                },
            }
        )
        for stage in ("audit", "smoke", "promotion"):
            result = self._action(
                {
                    "type": "evaluate_candidate",
                    "candidate_id": "candidate",
                    "evaluation_id": f"candidate.{stage}",
                    "stage": stage,
                }
            )
            self.assertTrue(result["state"]["evaluations"][f"candidate.{stage}"]["passed"])
        self._action(
            {
                "type": "commit",
                "candidate_id": "candidate",
                "evidence_ids": ["candidate.promotion"],
                "comparison": {
                    "mode": "documented_non_dominance",
                    "reason": "no recorded candidate dominates this promotion",
                    "evidence_ids": ["candidate.promotion"],
                },
                # Another agent-authored value which the export must ignore.
                "metadata": {"claimed_best_energy": -999.0},
            }
        )

    def test_positive_export_is_evaluator_derived_reproducible_and_untrusted(self) -> None:
        with self.assertRaisesRegex(operator.OperatorError, "requires a sealed campaign"):
            operator.export_result(
                argparse.Namespace(
                    evaluator_run=self.evaluator,
                    output=None,
                    allow_unsealed=False,
                )
            )
        with self.assertRaisesRegex(operator.OperatorError, "requires a controller-accepted"):
            self._export()

        self._positive_terminal()
        output = self.evaluator / "published-result.json"
        summary = self._export(output)
        first_bytes = output.read_bytes()
        repeated = self._export(output)
        self.assertEqual(output.read_bytes(), first_bytes)
        self.assertEqual(summary, repeated)

        artifact = json.loads(first_bytes)
        self.assertEqual(artifact["decision"], "positive_commit")
        self.assertEqual(
            artifact["trust"]["classification"], "UNTRUSTED_LOCAL_INTEGRATION"
        )
        self.assertFalse(artifact["trust"]["benchmark_grade"])
        self.assertEqual(artifact["result"]["candidate_id"], "candidate")
        promotion = artifact["result"]["promotion"]
        self.assertEqual(promotion["evaluation_id"], "candidate.promotion")
        self.assertEqual(
            promotion["candidate_semantic_sha256"],
            artifact["result"]["candidate_semantic_sha256"],
        )
        binding = promotion["optimized_parameter_binding"]
        self.assertEqual(set(binding), {"theta"})
        self.assertTrue(math.isfinite(binding["theta"]))
        self.assertEqual(first_bytes.count(b'"optimized_parameter_binding"'), 1)
        self.assertNotIn(b'"best_values"', first_bytes)
        self.assertNotIn("-999", first_bytes.decode("utf-8"))

        compiled = compile_ansatz(artifact["result"]["ansatz_spec"])
        bound = compiled.circuit.assign_parameters(
            {compiled.parameters[name]: value for name, value in binding.items()},
            inplace=False,
        )
        public_problem = adapt_prepare_problem(
            prepare.load_problem(self.problem)
        ).public_problem
        reproduced_energy = energy_from_circuit(
            bound, hamiltonian_from_public(public_problem)
        )
        self.assertAlmostEqual(
            reproduced_energy, promotion["energy"]["best_energy"], places=10
        )

        core = dict(artifact)
        supplied_hash = core.pop("artifact_sha256")
        self.assertEqual(
            supplied_hash,
            operator._sha256_bytes(
                operator._RESULT_ARTIFACT_DOMAIN + operator._canonical_bytes(core)
            ),
        )

    def test_negative_export_has_no_fabricated_positive_result(self) -> None:
        self._action(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "structure",
                "claim": {"kind": "ansatz_structure", "family": "invalid control"},
            }
        )
        self._action(
            {
                "type": "submit_candidate",
                "candidate_id": "invalid",
                "hypothesis_id": "structure",
                "spec": {"num_qubits": 2},
                "metadata": {
                    "enforcement": "unconstrained",
                    "falsifier": "trusted compilation rejects the malformed IR",
                },
            }
        )
        audit = self._action(
            {
                "type": "evaluate_candidate",
                "candidate_id": "invalid",
                "evaluation_id": "invalid.audit",
                "stage": "audit",
            }
        )
        self.assertFalse(audit["state"]["evaluations"]["invalid.audit"]["passed"])
        self._action(
            {
                "type": "retire",
                "entity": "hypothesis",
                "entity_id": "structure",
                "reason": "its only candidate failed trusted compilation",
            }
        )
        self._action(
            {
                "type": "close_negative",
                "reason": "the investigated branch failed its trusted audit",
                "evidence_ids": ["invalid.audit"],
            }
        )
        summary = self._export()
        artifact = json.loads(Path(summary["artifact"]).read_text(encoding="utf-8"))
        self.assertEqual(artifact["decision"], "negative_close")
        self.assertEqual(
            artifact["result"]["evidence"]["invalid.audit"]["kind"],
            "evaluation",
        )
        self.assertNotIn("ansatz_spec", artifact["result"])
        self.assertNotIn("promotion", artifact["result"])
        self.assertNotIn(
            "optimized_parameter_binding",
            Path(summary["artifact"]).read_text(encoding="utf-8"),
        )

    def test_export_refuses_agent_bundle_and_different_existing_output(self) -> None:
        self._positive_terminal()
        with self.assertRaisesRegex(operator.OperatorError, "agent bundle"):
            self._export(self.bundle / "result.json")
        occupied = self.evaluator / "occupied.json"
        occupied.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(operator.OperatorError, "different content"):
            self._export(occupied)

        raw_problem = json.loads(self.problem.read_text(encoding="utf-8"))
        raw_problem["name"] = "changed_after_terminal"
        self.problem.write_text(json.dumps(raw_problem), encoding="utf-8")
        with self.assertRaisesRegex(operator.OperatorError, "raw problem changed"):
            self._export(self.evaluator / "must-not-exist.json")

    def test_export_rejects_paths_outside_the_evaluator_run(self) -> None:
        self._positive_terminal()
        outside = self.root / "public-result.json"
        with self.assertRaisesRegex(operator.OperatorError, "evaluator-owned run"):
            self._export(outside)
        self.assertFalse(outside.exists())

    def test_export_rejects_target_and_parent_links(self) -> None:
        self._positive_terminal()
        outside = self.root / "outside"
        outside.mkdir()
        outside_file = outside / "outside.json"
        outside_file.write_text("do-not-touch", encoding="utf-8")

        target_link = self.evaluator / "target-link.json"
        dangling_link = self.evaluator / "dangling-link.json"
        parent_link = self.evaluator / "parent-link"
        try:
            target_link.symlink_to(outside_file)
            dangling_link.symlink_to(outside / "missing.json")
            parent_link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("symbolic-link creation is not available")

        for unsafe in (
            target_link,
            dangling_link,
            parent_link / "through-parent.json",
        ):
            with self.subTest(path=unsafe), self.assertRaisesRegex(
                operator.OperatorError, "symbolic link|reparse point"
            ):
                self._export(unsafe)
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "do-not-touch")
        self.assertFalse((outside / "missing.json").exists())
        self.assertFalse((outside / "through-parent.json").exists())

    def test_export_detects_existing_target_swap_during_idempotence_check(self) -> None:
        self._positive_terminal()
        output = self.evaluator / "swap-result.json"
        self._export(output)
        real_read = operator._read_regular_bytes
        swapped = False

        def swap_after_read(path: Path, *, max_bytes: int) -> bytes:
            nonlocal swapped
            value = real_read(path, max_bytes=max_bytes)
            if path == output and not swapped:
                replacement = self.evaluator / "replacement.json"
                replacement.write_text("attacker replacement", encoding="utf-8")
                replacement.replace(output)
                swapped = True
            return value

        with mock.patch.object(
            operator, "_read_regular_bytes", side_effect=swap_after_read
        ), self.assertRaisesRegex(operator.OperatorError, "changed while being verified"):
            self._export(output)
        self.assertTrue(swapped)

    def test_export_uses_exclusive_creation_when_target_appears(self) -> None:
        self._positive_terminal()
        output = self.evaluator / "create-race.json"
        real_open = operator.os.open
        raced = False

        def appear_before_create(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal raced
            is_target_create = bool(
                flags & operator.os.O_CREAT
                and flags & operator.os.O_EXCL
                and (
                    (dir_fd is not None and path == output.name)
                    or (dir_fd is None and Path(path) == output)
                )
            )
            if is_target_create and not raced:
                output.write_text("racing file", encoding="utf-8")
                raced = True
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch.object(
            operator.os, "open", side_effect=appear_before_create
        ), self.assertRaisesRegex(operator.OperatorError, "exclusive creation"):
            self._export(output)
        self.assertTrue(raced)
        self.assertEqual(output.read_text(encoding="utf-8"), "racing file")

    def test_secure_writer_detects_parent_identity_replacement(self) -> None:
        parent = self.evaluator / "identity-parent"
        parent.mkdir()
        output, chain = operator._validated_result_output(
            parent / "result.json",
            evaluator_run=self.evaluator,
            agent_bundle=self.bundle,
        )
        moved = self.evaluator / "identity-parent-original"
        parent.rename(moved)
        parent.mkdir()
        with self.assertRaisesRegex(operator.OperatorError, "parent changed"):
            operator._secure_write_terminal_result(
                output,
                b"trusted payload\n",
                parent_chain=chain,
            )
        self.assertFalse(output.exists())
        self.assertFalse((moved / "result.json").exists())

    def test_windows_reparse_parent_and_target_flags_are_rejected(self) -> None:
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not reparse_flag:
            self.skipTest("Windows reparse attributes are unavailable")
        parent = self.evaluator / "mock-reparse-parent"
        parent.mkdir()
        target = parent / "result.json"
        target.write_text("existing", encoding="utf-8")
        real_lstat = Path.lstat
        for marked_path in (parent, target):
            with self.subTest(path=marked_path):
                def marked_lstat(path: Path):
                    metadata = real_lstat(path)
                    if path != marked_path:
                        return metadata
                    marked = mock.Mock()
                    marked.st_mode = metadata.st_mode
                    marked.st_dev = metadata.st_dev
                    marked.st_ino = metadata.st_ino
                    marked.st_file_attributes = reparse_flag
                    return marked

                with mock.patch.object(
                    Path, "lstat", autospec=True, side_effect=marked_lstat
                ):
                    with self.assertRaisesRegex(
                        operator.OperatorError, "reparse point"
                    ):
                        operator._validated_result_output(
                            target,
                            evaluator_run=self.evaluator,
                            agent_bundle=self.bundle,
                        )


if __name__ == "__main__":
    unittest.main()
