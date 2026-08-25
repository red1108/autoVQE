from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autovqe import research_cli
from autovqe.ledger import JsonlEventLedger


class ResearchCliTests(unittest.TestCase):
    def test_sealed_anchor_rejects_related_relative_and_git_worktree_paths(self) -> None:
        context = {"run_id": "a" * 32}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            run_dir = base / "runs" / "one"
            run_dir.mkdir(parents=True)
            invalid = {
                "equal": run_dir,
                "ancestor": run_dir.parent,
                "descendant": run_dir / "anchors",
                "relative": Path("relative-evaluator-anchors"),
            }
            for label, anchor_dir in invalid.items():
                with self.subTest(label=label), self.assertRaisesRegex(
                    research_cli.ResearchCliError,
                    "absolute|equal|contain",
                ):
                    research_cli._anchor_path(run_dir, context, anchor_dir)

            worktree = research_cli._git_worktree_root(
                Path(research_cli.__file__).resolve().parent
            )
            self.assertIsNotNone(worktree)
            assert worktree is not None
            with self.assertRaisesRegex(research_cli.ResearchCliError, "Git worktree"):
                research_cli._anchor_path(
                    run_dir,
                    context,
                    worktree / "agent-controlled-anchors",
                )

    def test_action_file_enforces_raw_cap_before_json_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            action_path = Path(directory) / "oversized.json"
            action_path.write_bytes(b"{" + b" " * research_cli.MAX_EXTERNAL_ACTION_BYTES)
            with mock.patch("autovqe.research_cli.json.loads") as parse:
                with self.assertRaisesRegex(research_cli.ResearchCliError, "byte cap"):
                    research_cli.execute_action_file(
                        "unused-problem.json",
                        Path(directory) / "unused-run",
                        action_path,
                        require_sealed=False,
                    )
                parse.assert_not_called()

    def test_action_file_rejects_non_regular_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaisesRegex(research_cli.ResearchCliError, "regular file"):
                research_cli.execute_action_file(
                    "unused-problem.json",
                    base / "unused-run",
                    base,
                    require_sealed=False,
                )

            target = base / "target.json"
            target.write_text('{"type":"retire"}', encoding="utf-8")
            link = base / "action-link.json"
            try:
                link.symlink_to(target)
            except OSError:
                return
            with self.assertRaisesRegex(research_cli.ResearchCliError, "symbolic link"):
                research_cli.execute_action_file(
                    "unused-problem.json",
                    base / "unused-run",
                    link,
                    require_sealed=False,
                )

    def test_action_file_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.json"
            invalid_documents = {
                "duplicate": '{"type":"retire","type":"commit"}',
                "nan": '{"type":"retire","value":NaN}',
                "infinity": '{"type":"retire","value":Infinity}',
                "negative_infinity": '{"type":"retire","value":-Infinity}',
                "overflow_to_infinity": '{"type":"retire","value":1e999}',
            }
            for label, document in invalid_documents.items():
                with self.subTest(label=label):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(
                        research_cli.ResearchCliError,
                        "duplicate JSON key|non-finite JSON number",
                    ):
                        research_cli.execute_action_file(
                            "unused-problem.json",
                            Path(directory) / "unused-run",
                            path,
                            require_sealed=False,
                        )

    def test_atomic_writes_do_not_use_predictable_temporary_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "state.json"
            predictable = destination.with_suffix(destination.suffix + ".tmp")
            predictable.write_text("do-not-touch", encoding="utf-8")
            colliding = base / f".{destination.name}.{'0' * 32}.tmp"
            colliding.write_text("collision", encoding="utf-8")
            with mock.patch(
                "autovqe.research_cli.secrets.token_hex",
                side_effect=["0" * 32, "1" * 32],
            ):
                research_cli._write_json(destination, {"ok": True})
            self.assertEqual(predictable.read_text(encoding="utf-8"), "do-not-touch")
            self.assertEqual(colliding.read_text(encoding="utf-8"), "collision")
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"ok": True})

            run_dir = base / "run"
            run_dir.mkdir()
            mirror = run_dir / research_cli.LEDGER_FILE
            mirror.write_text("old", encoding="utf-8")
            mirror_temporary = mirror.with_suffix(mirror.suffix + ".tmp")
            mirror_temporary.write_text("do-not-touch", encoding="utf-8")
            research_cli._sync_ledger_mirror(run_dir, ())
            self.assertEqual(mirror.read_text(encoding="utf-8"), "")
            self.assertEqual(
                mirror_temporary.read_text(encoding="utf-8"), "do-not-touch"
            )

    def test_init_writes_only_safe_observation_and_replays_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            initialized = research_cli.initialize_run(
                "examples/h2_2q.json",
                run_dir,
                total_budget=12.0,
            )
            observation = (run_dir / research_cli.OBSERVATION_FILE).read_text(
                encoding="utf-8"
            )
            self.assertNotIn("reference_energy", observation)
            self.assertNotIn("reference_state", observation)
            self.assertNotIn("model_class", observation)
            self.assertNotIn("recommended", observation)
            self.assertEqual(initialized["state"]["remaining_budget"], 12.0)

            research_cli.execute_action(
                "examples/h2_2q.json",
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "h1",
                    "claim": {"kind": "null_control"},
                    "cost": 1000,
                },
                require_sealed=False,
            )
            status = research_cli.run_status(run_dir, require_sealed=False)
            # The controller records both the proposal and its evaluator-owned
            # non-algebraic admission marker.
            self.assertEqual(status["ledger_events"], 2)
            self.assertEqual(status["state"]["spent_budget"], 0.1)
            self.assertEqual(status["security"]["mode"], "local_unsealed")
            self.assertFalse(status["security"]["tamper_evident"])
            with self.assertRaisesRegex(research_cli.ResearchCliError, "sealed"):
                research_cli.run_status(run_dir)

    def test_context_prevents_problem_swap_and_reinitialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            research_cli.initialize_run(
                "examples/h2_2q.json", run_dir, total_budget=10.0
            )
            with self.assertRaises(research_cli.ResearchCliError):
                research_cli.initialize_run(
                    "examples/h2_2q.json", run_dir, total_budget=10.0
                )
            with self.assertRaises(research_cli.ResearchCliError):
                research_cli.load_controller(
                    "examples/ising_1d_5q.json",
                    run_dir,
                    require_sealed=False,
                )

    def test_action_file_must_contain_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            research_cli.initialize_run(
                "examples/h2_2q.json", run_dir, total_budget=10.0
            )
            action_path = Path(directory) / "action.json"
            action_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertRaises(research_cli.ResearchCliError):
                research_cli.execute_action_file(
                    "examples/h2_2q.json",
                    run_dir,
                    action_path,
                    require_sealed=False,
                )

    def test_expected_cursor_is_compared_immediately_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            initialized = research_cli.initialize_run(
                "examples/h2_2q.json", run_dir, total_budget=10.0
            )
            checkpoint = initialized["checkpoint"]
            stale_cursor = {
                "ledger_events": checkpoint["ledger_events"],
                "ledger_tip": checkpoint["ledger_tip"],
            }
            research_cli.execute_action(
                "examples/h2_2q.json",
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "first_cursor_action",
                    "claim": {"kind": "ansatz_structure", "family": "first"},
                },
                require_sealed=False,
                expected_cursor=stale_cursor,
            )
            with self.assertRaisesRegex(
                research_cli.ResearchCliError, "no longer matches"
            ):
                research_cli.execute_action(
                    "examples/h2_2q.json",
                    run_dir,
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": "stale_cursor_action",
                        "claim": {"kind": "ansatz_structure", "family": "stale"},
                    },
                    require_sealed=False,
                    expected_cursor=stale_cursor,
                )
            status = research_cli.run_status(run_dir, require_sealed=False)
            self.assertEqual(status["checkpoint"]["ledger_events"], 2)

    def test_sealed_context_and_checkpoint_detect_tampering(self) -> None:
        key = b"test-evaluator-key-material-32-bytes"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sealed"
            anchor_dir = Path(directory) / "anchors"
            initialized = research_cli.initialize_run(
                "examples/h2_2q.json",
                run_dir,
                total_budget=10.0,
                evaluator_key=key,
                anchor_dir=anchor_dir,
            )
            self.assertEqual(initialized["security"]["mode"], "sealed")
            with self.assertRaisesRegex(research_cli.ResearchCliError, "evaluator key"):
                research_cli.run_status(run_dir, require_sealed=True)

            context_path = run_dir / research_cli.CONTEXT_FILE
            context = json.loads(context_path.read_text(encoding="utf-8"))
            context["total_budget"] = 1000000.0
            context_path.write_text(json.dumps(context), encoding="utf-8")
            with self.assertRaisesRegex(research_cli.ResearchCliError, "HMAC"):
                research_cli.run_status(
                    run_dir,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    require_sealed=True,
                )

    def test_sealed_checkpoint_detects_ledger_rollback(self) -> None:
        key = b"another-test-evaluator-key-material"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sealed"
            anchor_dir = Path(directory) / "anchors"
            research_cli.initialize_run(
                "examples/h2_2q.json",
                run_dir,
                total_budget=10.0,
                evaluator_key=key,
                anchor_dir=anchor_dir,
            )
            research_cli.execute_action(
                "examples/h2_2q.json",
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "generic",
                    "claim": {"kind": "ansatz_structure", "family": "baseline"},
                },
                evaluator_key=key,
                anchor_dir=anchor_dir,
                require_sealed=True,
            )
            context = json.loads(
                (run_dir / research_cli.CONTEXT_FILE).read_text(encoding="utf-8")
            )
            self.assertTrue(
                (anchor_dir / f"autovqe-{context['run_id']}.events.jsonl").is_file()
            )
            ledger_path = run_dir / research_cli.LEDGER_FILE
            ledger_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                research_cli.ResearchCliError, "mirror|checkpoint"
            ):
                research_cli.run_status(
                    run_dir,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    require_sealed=True,
                )

    def test_external_anchor_rejects_a_valid_old_checkpoint_pair(self) -> None:
        key = b"rollback-protected-evaluator-key"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sealed"
            anchor_dir = Path(directory) / "anchors"
            research_cli.initialize_run(
                "examples/h2_2q.json",
                run_dir,
                total_budget=10.0,
                evaluator_key=key,
                anchor_dir=anchor_dir,
            )

            def propose(identifier: str) -> None:
                research_cli.execute_action(
                    "examples/h2_2q.json",
                    run_dir,
                    {
                        "type": "propose_hypothesis",
                        "hypothesis_id": identifier,
                        "claim": {
                            "kind": "ansatz_structure",
                            "family": identifier,
                        },
                    },
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    require_sealed=True,
                )

            propose("first")
            ledger_path = run_dir / research_cli.LEDGER_FILE
            checkpoint_path = run_dir / research_cli.CHECKPOINT_FILE
            old_ledger = ledger_path.read_text(encoding="utf-8")
            old_checkpoint = checkpoint_path.read_text(encoding="utf-8")
            propose("second")

            # Both restored files are internally valid and carry old HMACs.
            # Only the external monotonic head distinguishes them from current.
            ledger_path.write_text(old_ledger, encoding="utf-8")
            checkpoint_path.write_text(old_checkpoint, encoding="utf-8")
            with self.assertRaisesRegex(
                research_cli.ResearchCliError, "mirror|monotonic head"
            ):
                research_cli.run_status(
                    run_dir,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    require_sealed=True,
                )

    def test_run_path_registry_rejects_swapping_in_another_sealed_run(self) -> None:
        key = b"path-bound-evaluator-key-material"
        with tempfile.TemporaryDirectory() as directory:
            anchor_dir = Path(directory) / "anchors"
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            for run_dir in (first, second):
                research_cli.initialize_run(
                    "examples/h2_2q.json",
                    run_dir,
                    total_budget=10.0,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                )
            for filename in (
                research_cli.CONTEXT_FILE,
                research_cli.CHECKPOINT_FILE,
                research_cli.OBSERVATION_FILE,
            ):
                shutil.copyfile(second / filename, first / filename)
            with self.assertRaisesRegex(research_cli.ResearchCliError, "run_id|context"):
                research_cli.run_status(
                    first,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    require_sealed=True,
                )

    def test_anchor_advance_rejects_a_longer_genesis_fork(self) -> None:
        key = b"ancestry-check-evaluator-key"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "sealed"
            anchor_dir = Path(directory) / "anchors"
            research_cli.initialize_run(
                "examples/h2_2q.json",
                run_dir,
                total_budget=10.0,
                evaluator_key=key,
                anchor_dir=anchor_dir,
            )
            research_cli.execute_action(
                "examples/h2_2q.json",
                run_dir,
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "anchored",
                    "claim": {"kind": "ansatz_structure", "family": "anchored"},
                },
                evaluator_key=key,
                anchor_dir=anchor_dir,
                require_sealed=True,
            )
            context = json.loads(
                (run_dir / research_cli.CONTEXT_FILE).read_text(encoding="utf-8")
            )
            previous = json.loads(
                (run_dir / research_cli.CHECKPOINT_FILE).read_text(encoding="utf-8")
            )

            fork_path = Path(directory) / "fork.jsonl"
            fork = JsonlEventLedger(fork_path)
            for index in range(3):
                fork.append("fork_event", {"index": index}, cost=0.0)
            authoritative = research_cli._external_ledger_path(
                run_dir, context, anchor_dir
            )
            shutil.copyfile(fork_path, authoritative)

            with self.assertRaisesRegex(research_cli.ResearchCliError, "descendant"):
                research_cli._write_checkpoint(
                    run_dir,
                    context,
                    evaluator_key=key,
                    anchor_dir=anchor_dir,
                    expected_previous=previous,
                )


if __name__ == "__main__":
    unittest.main()
