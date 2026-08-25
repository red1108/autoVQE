from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from autovqe import research_cli
from autovqe.contracts import assert_agent_safe
from meta_agent import client, operator


REPO_ROOT = Path(__file__).resolve().parents[1]


class MetaAgentBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "agent"
        self.evaluator = self.root / "evaluator"
        self.problem = self.root / "private_problem.json"
        self.problem.write_bytes((REPO_ROOT / "examples" / "h2_2q.json").read_bytes())
        self.campaign = self.root / "campaign.json"
        self.campaign.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": "bridge_test",
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self) -> dict[str, object]:
        return operator.prepare_campaign(
            argparse.Namespace(
                campaign=self.campaign,
                security="local_unsealed",
                agent_bundle=None,
                evaluator_run=None,
                allow_dirty_evaluator=False,
            )
        )

    def _serve_once(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            code = operator.serve(
                argparse.Namespace(
                    evaluator_run=self.evaluator,
                    poll_interval=0.01,
                    idle_timeout=5.0,
                    once=True,
                    exit_on_commit=False,
                    allow_unsealed=True,
                )
            )
        self.assertEqual(code, 0)

    def _client_submit(self, action: dict[str, object], name: str = "action.json") -> dict:
        action_path = self.bundle / "actions" / name
        action_path.write_text(json.dumps(action), encoding="utf-8")
        server_error: list[BaseException] = []

        def target() -> None:
            try:
                self._serve_once()
            except BaseException as exc:  # Propagate thread failures to the test.
                server_error.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        completed = subprocess.run(
            [
                sys.executable,
                str(self.bundle / "client.py"),
                "submit",
                "--action",
                str(action_path),
                "--timeout",
                "10",
                "--poll-interval",
                "0.01",
            ],
            cwd=self.bundle,
            check=False,
            capture_output=True,
            text=True,
        )
        thread.join(timeout=10)
        if server_error:
            raise server_error[0]
        self.assertFalse(thread.is_alive(), "operator bridge did not stop")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_prepare_emits_an_isolated_example_free_bundle(self) -> None:
        prepared = self._prepare()
        self.assertEqual(prepared["security_mode"], "local_unsealed")
        self.assertEqual(
            {path.name for path in self.bundle.iterdir()},
            {
                "actions",
                "inbox",
                "outbox",
                "observation.json",
                "AGENT_CONTRACT.md",
                "GOAL.md",
                "client.py",
                "AGENTS.md",
                "session.json",
                "journal.md",
            },
        )
        goal = (self.bundle / "GOAL.md").read_text(encoding="utf-8")
        self.assertTrue(goal.startswith("/goal "))
        self.assertNotRegex(goal, r"\{\{[A-Z0-9_]+\}\}")
        self.assertNotIn(str(self.problem), goal)
        observation = json.loads((self.bundle / "observation.json").read_text())
        assert_agent_safe(observation)
        session = json.loads((self.bundle / "session.json").read_text())
        self.assertNotIn("problem_path", session)
        self.assertNotIn("evaluator_run", session)
        self.assertNotIn("signature", session)
        self.assertEqual(
            session["publication_auth_scheme"], "merkle_lamport_sha256_v1"
        )
        self.assertRegex(session["publication_auth_root"], r"^[0-9a-f]{64}$")
        self.assertNotIn("seed", json.dumps(session).lower())
        signed_status = json.loads(
            (self.bundle / "inbox" / "latest_status.json").read_text()
        )
        self.assertEqual(
            signed_status["publication_capacity"]["remaining_slots_after_publish"],
            session["publication_auth_slots"]
            - signed_status["authentication"]["key_index"]
            - 1,
        )
        signer = json.loads(
            (self.evaluator / operator.PUBLICATION_SIGNER).read_text(encoding="utf-8")
        )
        exposed_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in self.bundle.rglob("*")
            if path.is_file()
        )
        self.assertNotIn(signer["seed_hex"], exposed_text)

    def test_client_protected_reads_do_not_use_path_read_bytes(self) -> None:
        self._prepare()
        output = io.StringIO()
        with mock.patch.object(
            Path, "read_bytes", side_effect=AssertionError("unsafe unbounded read")
        ):
            with contextlib.redirect_stdout(output):
                self.assertEqual(client.show_status(self.bundle), 0)
        self.assertEqual(json.loads(output.getvalue())["message_type"], "status")

    def test_client_detects_a_file_identity_change_during_secure_open(self) -> None:
        protected = self.root / "identity.json"
        protected.write_text("{}", encoding="utf-8")
        real_fstat = os.fstat

        def changed_identity(descriptor: int) -> os.stat_result:
            values = list(real_fstat(descriptor))
            values[1] += 1
            return os.stat_result(values)

        with mock.patch.object(client.os, "fstat", side_effect=changed_identity):
            with self.assertRaisesRegex(client.ClientError, "changed while being opened"):
                client._read_json_object(protected)

    def test_client_operator_round_trip_and_idempotent_republish(self) -> None:
        self._prepare()
        action = {
            "type": "propose_hypothesis",
            "hypothesis_id": "bridge_h1",
            "claim": {"kind": "ansatz_structure", "family": "bridge test"},
        }
        receipt = self._client_submit(action)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["status"]["checkpoint"]["ledger_events"], 2)
        self.assertNotIn("signature", json.dumps(receipt))
        self.assertNotIn("context_hash", json.dumps(receipt))

        archived = next((self.evaluator / "bridge_requests").glob("req_*.json"))
        request_id = archived.stem
        published_receipt = self.bundle / "inbox" / f"{request_id}.receipt.json"
        trusted_receipt = self.evaluator / "bridge_receipts" / f"{request_id}.json"
        original_signed_receipt = published_receipt.read_bytes()
        self.assertEqual(original_signed_receipt, trusted_receipt.read_bytes())
        receipt_key_index = json.loads(original_signed_receipt)["authentication"][
            "key_index"
        ]
        status_key_index = json.loads(
            (self.bundle / "inbox" / "latest_status.json").read_text()
        )["authentication"]["key_index"]
        self.assertNotEqual(receipt_key_index, status_key_index)
        replay = self.bundle / "outbox" / f"{request_id}.ready.json"
        replay.write_bytes(archived.read_bytes())
        self._serve_once()
        self.assertEqual(published_receipt.read_bytes(), original_signed_receipt)
        status = operator.operator_status(
            argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=True)
        )
        self.assertEqual(status["checkpoint"]["ledger_events"], 2)

    def test_optimizer_binding_is_private_until_terminal_operator_export(self) -> None:
        self._prepare()
        self._client_submit(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "binding_structure",
                "claim": {
                    "kind": "ansatz_structure",
                    "family": "two-qubit pair rotation",
                },
            },
            "binding-proposal.json",
        )
        self._client_submit(
            {
                "type": "submit_candidate",
                "candidate_id": "binding_candidate",
                "hypothesis_id": "binding_structure",
                "spec": {
                    "version": 1,
                    "name": "binding_candidate",
                    "num_qubits": 2,
                    "parameters": [{"name": "theta"}],
                    "reference": {"macro": "X", "qubits": [0]},
                    "layers": [
                        {
                            "operations": [
                                {
                                    "macro": "PauliRotation",
                                    "qubits": [0, 1],
                                    "parameters": {
                                        "angle": {
                                            "terms": [
                                                {
                                                    "parameter": "theta",
                                                    "coefficient": 1.0,
                                                }
                                            ],
                                            "constant": 0.0,
                                        }
                                    },
                                    "options": {"pauli": "XY"},
                                }
                            ]
                        }
                    ],
                },
                "metadata": {
                    "enforcement": "unconstrained",
                    "prediction": "the pair rotation improves the reference energy",
                },
            },
            "binding-candidate.json",
        )
        self._client_submit(
            {
                "type": "evaluate_candidate",
                "candidate_id": "binding_candidate",
                "evaluation_id": "binding.audit",
                "stage": "audit",
            },
            "binding-audit.json",
        )
        smoke_receipt = self._client_submit(
            {
                "type": "evaluate_candidate",
                "candidate_id": "binding_candidate",
                "evaluation_id": "binding.smoke",
                "stage": "smoke",
            },
            "binding-smoke.json",
        )
        self.assertTrue(
            smoke_receipt["result"]["state"]["terminal_decision"] is None
        )

        # The authoritative replay keeps the evaluator-owned point for a later
        # terminal export.
        trusted_status = research_cli.run_status(
            self.evaluator,
            require_sealed=False,
        )
        trusted_metrics = trusted_status["state"]["evaluations"]["binding.smoke"][
            "metrics"
        ]
        self.assertEqual(
            set(trusted_metrics["optimized_parameter_binding"]), {"theta"}
        )

        # Neither the signed action receipt nor a fresh signed status gives the
        # still-running agent the optimizer's selected values.
        operator.operator_status(
            argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=True)
        )
        public_documents = [
            json.dumps(smoke_receipt),
            (self.bundle / "inbox" / "latest_status.json").read_text(
                encoding="utf-8"
            ),
            *[
                path.read_text(encoding="utf-8")
                for path in (self.bundle / "inbox").glob("*.receipt.json")
            ],
        ]
        for document in public_documents:
            self.assertNotIn("optimized_parameter_binding", document)
            self.assertNotIn("best_values", document)

    def test_client_rejects_tampered_authenticated_status(self) -> None:
        self._prepare()
        status_path = self.bundle / "inbox" / "latest_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["status"]["checkpoint"]["ledger_events"] = 999
        status_path.write_text(json.dumps(status), encoding="utf-8")
        with self.assertRaisesRegex(client.ClientError, "authentication"):
            client.show_status(self.bundle)

    def test_client_rejects_tampered_authenticated_receipt(self) -> None:
        self._prepare()
        self._client_submit(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": "signed_h1",
                "claim": {"kind": "ansatz_structure", "family": "signed test"},
            }
        )
        receipt_path = next((self.bundle / "inbox").glob("req_*.receipt.json"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["action_sha256"] = "0" * 64
        session = client._load_session(self.bundle)
        with self.assertRaisesRegex(client.ClientError, "authentication"):
            client._verify_publication(receipt, session, message_type="receipt")

    def test_bundle_client_validate_is_local_preflight_only(self) -> None:
        self._prepare()
        action_path = self.bundle / "actions" / "preflight.json"
        action_path.write_text(
            json.dumps({"type": "unknown_but_strict_json"}), encoding="utf-8"
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(client.validate_action(self.bundle, action_path), 0)
        validated = json.loads(output.getvalue())
        self.assertTrue(validated["ok"])
        self.assertIn("semantics were not run", validated["scope"])

    def test_stale_cursor_is_rejected_without_a_ledger_mutation(self) -> None:
        self._prepare()
        session = json.loads((self.bundle / "session.json").read_text())
        latest = json.loads((self.bundle / "inbox" / "latest_status.json").read_text())
        checkpoint = latest["status"]["checkpoint"]
        cursor = {
            "ledger_events": checkpoint["ledger_events"],
            "ledger_tip": checkpoint["ledger_tip"],
        }

        def envelope(index: int) -> tuple[str, dict]:
            request_id = f"req_{index:024x}"
            action = {
                "type": "propose_hypothesis",
                "hypothesis_id": f"stale_h{index}",
                "claim": {"kind": "ansatz_structure", "family": f"stale {index}"},
            }
            action_hash = operator._sha256_bytes(operator._canonical_bytes(action))
            return request_id, {
                "protocol_version": 1,
                "session_id": session["session_id"],
                "campaign_id": session["campaign_id"],
                "request_id": request_id,
                "manifest_sha256": session["manifest_sha256"],
                "expected_cursor": cursor,
                "action_sha256": action_hash,
                "action": action,
            }

        for index in (1, 2):
            request_id, request = envelope(index)
            (self.bundle / "outbox" / f"{request_id}.ready.json").write_text(
                json.dumps(request), encoding="utf-8"
            )
        self._serve_once()
        self._serve_once()
        second_receipt = json.loads(
            (self.bundle / "inbox" / f"req_{2:024x}.receipt.json").read_text()
        )
        self.assertFalse(second_receipt["ok"])
        self.assertEqual(second_receipt["error"]["type"], "CursorConflict")
        status = operator.operator_status(
            argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=True)
        )
        self.assertEqual(status["checkpoint"]["ledger_events"], 2)

    def test_near_limit_request_archive_is_exact_and_crash_replayable(self) -> None:
        self._prepare()
        manifest, key, anchor_dir = operator._load_manifest(
            self.evaluator, require_sealed=False
        )
        session = json.loads((self.bundle / "session.json").read_text())
        latest = json.loads((self.bundle / "inbox" / "latest_status.json").read_text())
        checkpoint = latest["status"]["checkpoint"]
        action = {"type": "large_invalid_action", "padding": [0] * 400_000}
        action_bytes = operator._canonical_bytes(action)
        self.assertLessEqual(len(action_bytes), operator.MAX_EXTERNAL_ACTION_BYTES)
        request_id = f"req_{91:024x}"
        envelope = {
            "protocol_version": 1,
            "session_id": session["session_id"],
            "campaign_id": session["campaign_id"],
            "request_id": request_id,
            "manifest_sha256": session["manifest_sha256"],
            "expected_cursor": {
                "ledger_events": checkpoint["ledger_events"],
                "ledger_tip": checkpoint["ledger_tip"],
            },
            "action_sha256": operator._sha256_bytes(action_bytes),
            "action": action,
        }
        envelope_bytes = operator._canonical_bytes(envelope)
        self.assertLessEqual(len(envelope_bytes), operator.MAX_BRIDGE_BYTES)
        ready = self.bundle / "outbox" / f"{request_id}.ready.json"
        ready.write_bytes(envelope_bytes)

        with mock.patch.object(
            operator,
            "execute_action",
            side_effect=KeyboardInterrupt("simulated receipt-prewrite crash"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                operator._process_ready(
                    ready, manifest, key=key, anchor_dir=anchor_dir
                )
        archive = self.evaluator / "bridge_requests" / f"{request_id}.json"
        self.assertEqual(archive.read_bytes(), envelope_bytes)
        self.assertLessEqual(archive.stat().st_size, operator.MAX_BRIDGE_BYTES)

        operator._recover_pending(manifest, key=key, anchor_dir=anchor_dir)
        with mock.patch.object(
            operator, "execute_action", side_effect=RuntimeError("replayed safely")
        ):
            receipt = operator._process_ready(
                ready, manifest, key=key, anchor_dir=anchor_dir
            )
        self.assertFalse(receipt["ok"])
        self.assertEqual(receipt["error"]["message"], "replayed safely")
        self.assertFalse(ready.exists())

    def test_protected_bundle_tampering_blocks_the_operator(self) -> None:
        self._prepare()
        with (self.bundle / "GOAL.md").open("a", encoding="utf-8") as handle:
            handle.write("\nchanged\n")
        manifest, _, _ = operator._load_manifest(self.evaluator, require_sealed=False)
        with self.assertRaisesRegex(operator.OperatorError, "protected file"):
            operator._verify_bundle(manifest)

    def test_private_problem_change_blocks_the_operator(self) -> None:
        self._prepare()
        raw = json.loads(self.problem.read_text(encoding="utf-8"))
        raw["reference_note"] = "changed after campaign preparation"
        self.problem.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(operator.OperatorError, "raw problem changed"):
            operator._load_manifest(self.evaluator, require_sealed=False)

    def test_malformed_request_is_rejected_without_stopping_or_mutating(self) -> None:
        self._prepare()
        request_id = f"req_{3:024x}"
        ready = self.bundle / "outbox" / f"{request_id}.ready.json"
        ready.write_text('{"protocol_version":1,"protocol_version":1}', encoding="utf-8")
        self._serve_once()
        receipt = json.loads(
            (self.bundle / "inbox" / f"{request_id}.receipt.json").read_text()
        )
        self.assertFalse(receipt["ok"])
        self.assertEqual(receipt["error"]["type"], "RequestRejected")
        self.assertFalse(ready.exists())
        status = operator.operator_status(
            argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=True)
        )
        self.assertEqual(status["checkpoint"]["ledger_events"], 0)

    def test_meta_operator_defaults_to_sealed(self) -> None:
        args = operator.build_parser().parse_args(
            ["prepare", "--campaign", str(self.campaign)]
        )
        self.assertEqual(args.security, "sealed")

    def test_operator_refuses_unsealed_run_without_explicit_override(self) -> None:
        self._prepare()
        with self.assertRaisesRegex(operator.OperatorError, "requires a sealed campaign"):
            operator.operator_status(
                argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=False)
            )

    def test_ready_directory_is_quarantined_without_mutating_the_ledger(self) -> None:
        self._prepare()
        request_id = f"req_{4:024x}"
        ready = self.bundle / "outbox" / f"{request_id}.ready.json"
        ready.mkdir()
        self._serve_once()
        self.assertFalse(ready.exists())
        self.assertTrue(
            any(path.name.endswith(ready.name) for path in ready.parent.glob(".rejected-*"))
        )
        receipt = json.loads(
            (self.bundle / "inbox" / f"{request_id}.receipt.json").read_text()
        )
        self.assertFalse(receipt["ok"])
        status = operator.operator_status(
            argparse.Namespace(evaluator_run=self.evaluator, allow_unsealed=True)
        )
        self.assertEqual(status["checkpoint"]["ledger_events"], 0)

    def test_sealed_anchor_must_not_overlap_the_agent_bundle(self) -> None:
        sealed_bundle = (self.root / "sealed-agent-overlap").resolve()
        environment = {
            "AUTOVQE_EVALUATOR_KEY": "test-operator-key-with-32-bytes",
            "AUTOVQE_EVALUATOR_ANCHOR_DIR": str(sealed_bundle / "anchors"),
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(operator.OperatorError, "anchor.*disjoint"):
                operator.prepare_campaign(
                    argparse.Namespace(
                        campaign=self.campaign,
                        security="sealed",
                        agent_bundle=sealed_bundle,
                        evaluator_run=(self.root / "sealed-evaluator-overlap").resolve(),
                        allow_dirty_evaluator=True,
                        model_label=None,
                    )
                )

    def test_sealed_anchor_environment_path_must_be_absolute(self) -> None:
        environment = {
            "AUTOVQE_EVALUATOR_KEY": "test-operator-key-with-32-bytes",
            "AUTOVQE_EVALUATOR_ANCHOR_DIR": "relative-anchors",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(operator.OperatorError, "anchor.*absolute"):
                operator.prepare_campaign(
                    argparse.Namespace(
                        campaign=self.campaign,
                        security="sealed",
                        agent_bundle=(self.root / "absolute-agent").resolve(),
                        evaluator_run=(self.root / "absolute-evaluator").resolve(),
                        allow_dirty_evaluator=True,
                        model_label=None,
                    )
                )

    def test_default_manifest_load_rejects_a_sealed_to_local_downgrade(self) -> None:
        anchors = self.root / "downgrade-anchors"
        sealed_bundle = self.root / "downgrade-agent"
        sealed_evaluator = self.root / "downgrade-evaluator"
        environment = {
            "AUTOVQE_EVALUATOR_KEY": "test-operator-key-with-32-bytes",
            "AUTOVQE_EVALUATOR_ANCHOR_DIR": str(anchors.resolve()),
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            operator.prepare_campaign(
                argparse.Namespace(
                    campaign=self.campaign,
                    security="sealed",
                    agent_bundle=sealed_bundle.resolve(),
                    evaluator_run=sealed_evaluator.resolve(),
                    allow_dirty_evaluator=True,
                    model_label=None,
                )
            )
            manifest_path = sealed_evaluator / operator.OPERATOR_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["security_mode"] = "local_unsealed"
            manifest["signature"] = None
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(operator.OperatorError, "requires a sealed campaign"):
                operator._load_manifest(sealed_evaluator)

    def test_separated_sealed_bridge_round_trip(self) -> None:
        anchors = self.root / "anchors"
        sealed_bundle = self.root / "sealed-agent"
        sealed_evaluator = self.root / "sealed-evaluator"
        environment = {
            "AUTOVQE_EVALUATOR_KEY": "test-operator-key-with-32-bytes",
            "AUTOVQE_EVALUATOR_ANCHOR_DIR": str(anchors.resolve()),
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            prepared = operator.prepare_campaign(
                argparse.Namespace(
                    campaign=self.campaign,
                    security="sealed",
                    agent_bundle=sealed_bundle.resolve(),
                    evaluator_run=sealed_evaluator.resolve(),
                    allow_dirty_evaluator=True,
                )
            )
            self.bundle = sealed_bundle
            self.evaluator = sealed_evaluator
            receipt = self._client_submit(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "sealed_h1",
                    "claim": {
                        "kind": "ansatz_structure",
                        "family": "sealed bridge test",
                    },
                }
            )
        self.assertEqual(prepared["security_mode"], "sealed")
        self.assertTrue(receipt["ok"])
        self.assertTrue(receipt["status"]["security"]["rollback_protected"])
        self.assertNotIn("signature", json.dumps(receipt))
        self.assertEqual(len(tuple(anchors.glob("autovqe-*.events.jsonl"))), 1)


if __name__ == "__main__":
    unittest.main()
