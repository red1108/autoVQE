from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import stat
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Mapping

from . import prepare
from .contracts import canonical_data
from .controller import MAX_EXTERNAL_ACTION_BYTES, ResearchController
from .ledger import GENESIS_HASH, JsonlEventLedger
from .observations import adapt_prepare_problem
from .research import ResearchLoop


CONTEXT_FILE = "context.json"
OBSERVATION_FILE = "observation.json"
LEDGER_FILE = "events.jsonl"
CHECKPOINT_FILE = "checkpoint.json"
RUN_SCHEMA_VERSION = 3
EVALUATOR_KEY_ENV = "AUTOVQE_EVALUATOR_KEY"
EVALUATOR_ANCHOR_DIR_ENV = "AUTOVQE_EVALUATOR_ANCHOR_DIR"
LOCAL_UNSEALED = "local_unsealed"
SEALED = "sealed"
_CONTEXT_DOMAIN = b"autovqe-context-v3\0"
_CHECKPOINT_DOMAIN = b"autovqe-checkpoint-v3\0"
_ANCHOR_DOMAIN = b"autovqe-anchor-v3\0"
_REGISTRY_DOMAIN = b"autovqe-registry-v3\0"
_ATOMIC_WRITE_ATTEMPTS = 16


class ResearchCliError(RuntimeError):
    pass


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            canonical_data(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ResearchCliError(f"value is not canonical JSON: {exc}") from exc


def _normalized_key(value: str | bytes | None) -> bytes | None:
    if value is None:
        return None
    key = value.encode("utf-8") if isinstance(value, str) else bytes(value)
    if len(key) < 16:
        raise ResearchCliError("evaluator sealing key must contain at least 16 bytes")
    return key


def evaluator_key_from_environment(*, required: bool = False) -> bytes | None:
    raw = os.environ.get(EVALUATOR_KEY_ENV)
    if raw is None:
        if required:
            raise ResearchCliError(
                f"sealed mode requires the {EVALUATOR_KEY_ENV} environment variable"
            )
        return None
    return _normalized_key(raw)


def evaluator_anchor_dir_from_environment(*, required: bool = False) -> Path | None:
    raw = os.environ.get(EVALUATOR_ANCHOR_DIR_ENV)
    if raw is None:
        if required:
            raise ResearchCliError(
                f"sealed mode requires the {EVALUATOR_ANCHOR_DIR_ENV} environment variable"
            )
        return None
    if not raw.strip():
        raise ResearchCliError(f"{EVALUATOR_ANCHOR_DIR_ENV} must not be empty")
    return Path(raw)


def _signature(
    payload: Mapping[str, Any],
    key: bytes,
    *,
    domain: bytes,
) -> str:
    return hmac.new(key, domain + _canonical_bytes(payload), hashlib.sha256).hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes through an exclusive, unpredictable same-directory file."""

    descriptor: int | None = None
    temporary: Path | None = None
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    for _ in range(_ATOMIC_WRITE_ATTEMPTS):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
        try:
            descriptor = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor is None or temporary is None:
        raise ResearchCliError(f"could not allocate an atomic temporary file for {path}")

    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while publishing research state")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        temporary = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    _atomic_replace_bytes(path, (encoded + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchCliError(f"missing research file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchCliError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResearchCliError(f"{path} must contain a JSON object")
    return raw


def _read_action_file(path: Path) -> dict[str, Any]:
    """Read one untrusted action without following links or parsing oversized data."""

    try:
        before_open = path.lstat()
    except FileNotFoundError as exc:
        raise ResearchCliError(f"missing action file: {path}") from exc
    except OSError as exc:
        raise ResearchCliError(f"cannot inspect action file {path}: {exc}") from exc
    if stat.S_ISLNK(before_open.st_mode):
        raise ResearchCliError(f"action file must not be a symbolic link: {path}")
    if not stat.S_ISREG(before_open.st_mode):
        raise ResearchCliError(f"action file must be a regular file: {path}")
    if before_open.st_size > MAX_EXTERNAL_ACTION_BYTES:
        raise ResearchCliError(
            f"action file exceeds {MAX_EXTERNAL_ACTION_BYTES} byte cap"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResearchCliError(f"cannot open action file {path}: {exc}") from exc
    try:
        after_open = os.fstat(descriptor)
        if not stat.S_ISREG(after_open.st_mode):
            raise ResearchCliError(f"action file must be a regular file: {path}")
        if (before_open.st_dev, before_open.st_ino) != (
            after_open.st_dev,
            after_open.st_ino,
        ):
            raise ResearchCliError(f"action file changed while it was being opened: {path}")

        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = os.read(
                descriptor,
                min(65536, MAX_EXTERNAL_ACTION_BYTES + 1 - received),
            )
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if received > MAX_EXTERNAL_ACTION_BYTES:
                raise ResearchCliError(
                    f"action file exceeds {MAX_EXTERNAL_ACTION_BYTES} byte cap"
                )
    finally:
        os.close(descriptor)

    try:
        rendered = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResearchCliError(f"action file is not valid UTF-8: {path}") from exc

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResearchCliError(f"duplicate JSON key in action file: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ResearchCliError(f"non-finite JSON number in action file: {value}")

    try:
        raw = json.loads(
            rendered,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ResearchCliError(f"invalid JSON in {path}: {exc}") from exc
    except RecursionError as exc:
        raise ResearchCliError(f"JSON nesting is too deep in action file: {path}") from exc
    if not isinstance(raw, dict):
        raise ResearchCliError(f"{path} must contain a JSON object")

    pending: list[Any] = [raw]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ResearchCliError("non-finite JSON number in action file")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return raw


def _context_path(run_dir: Path) -> Path:
    return run_dir / CONTEXT_FILE


def _ledger_path(run_dir: Path) -> Path:
    return run_dir / LEDGER_FILE


def _checkpoint_path(run_dir: Path) -> Path:
    return run_dir / CHECKPOINT_FILE


def _git_worktree_root(path: Path) -> Path | None:
    """Find a containing Git worktree without invoking agent-controlled Git."""

    candidate = path.resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return None


def _is_at_or_below(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _validated_anchor_directory(
    run_dir: Path,
    anchor_dir: str | Path | None,
) -> Path:
    if anchor_dir is None:
        raise ResearchCliError("sealed mode requires an external evaluator anchor directory")
    supplied = Path(anchor_dir)
    if not supplied.is_absolute():
        raise ResearchCliError("evaluator anchor directory must be an absolute path")

    directory = supplied.resolve()
    run_resolved = run_dir.resolve()
    if _is_at_or_below(directory, run_resolved) or _is_at_or_below(
        run_resolved, directory
    ):
        raise ResearchCliError(
            "evaluator anchor directory must not equal, contain, or be contained by "
            "the run directory"
        )

    possible_worktrees = {
        root
        for root in (
            _git_worktree_root(Path.cwd()),
            _git_worktree_root(Path(__file__).resolve().parent),
            _git_worktree_root(run_resolved),
        )
        if root is not None
    }
    if any(_is_at_or_below(directory, root) for root in possible_worktrees):
        raise ResearchCliError(
            "evaluator anchor directory must be outside the agent Git worktree"
        )
    return directory


def _anchor_path(
    run_dir: Path,
    context: Mapping[str, Any],
    anchor_dir: str | Path | None,
) -> Path:
    directory = _validated_anchor_directory(run_dir, anchor_dir)
    run_id = context.get("run_id")
    if not isinstance(run_id, str) or len(run_id) != 32:
        raise ResearchCliError("sealed context has an invalid run_id")
    return directory / f"autovqe-{run_id}.anchor.json"


def _external_ledger_path(
    run_dir: Path,
    context: Mapping[str, Any],
    anchor_dir: str | Path | None,
) -> Path:
    anchor_path = _anchor_path(run_dir, context, anchor_dir)
    return anchor_path.parent / f"autovqe-{context['run_id']}.events.jsonl"


def _run_path_hash(run_dir: Path) -> str:
    rendered = run_dir.resolve().as_posix()
    if os.name == "nt":
        rendered = rendered.casefold()
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _registry_path(run_dir: Path, anchor_dir: str | Path | None) -> Path:
    directory = _validated_anchor_directory(run_dir, anchor_dir)
    return directory / f"autovqe-path-{_run_path_hash(run_dir)}.registry.json"


@contextmanager
def _exclusive_anchor_lock(anchor_path: Path):
    anchor_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = anchor_path.with_suffix(anchor_path.suffix + ".lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise ResearchCliError(
            "evaluator anchor is locked by another action or needs operator recovery"
        ) from exc
    os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _security_summary(mode: str) -> dict[str, Any]:
    if mode == SEALED:
        return {
            "mode": SEALED,
            "tamper_evident": True,
            "rollback_protected": True,
            "trust_boundary": "external_evaluator_secret_and_monotonic_anchor",
        }
    return {
        "mode": LOCAL_UNSEALED,
        "tamper_evident": False,
        "rollback_protected": False,
        "warning": (
            "local_unsealed is for development only; an agent with filesystem "
            "write access can rewrite context, checkpoint, and ledger together"
        ),
    }


def _anchor_core(
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": context["run_id"],
        "context_hash": checkpoint["context_hash"],
        "ledger_events": checkpoint["ledger_events"],
        "ledger_tip": checkpoint["ledger_tip"],
    }


def _registry_core(
    run_dir: Path,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_path_hash": _run_path_hash(run_dir),
        "run_id": context["run_id"],
        "security_mode": SEALED,
        "context_hash": _content_hash(context),
    }


def _write_or_verify_registry(
    run_dir: Path,
    context: Mapping[str, Any],
    *,
    evaluator_key: str | bytes | None,
    anchor_dir: str | Path | None,
    initialize: bool,
) -> dict[str, Any]:
    key = _normalized_key(evaluator_key)
    if key is None:
        raise ResearchCliError("sealed registry requires the evaluator key")
    path = _registry_path(run_dir, anchor_dir)
    core = _registry_core(run_dir, context)
    if initialize:
        if path.exists():
            raise ResearchCliError(f"evaluator run-path registry already exists: {path}")
        registry = {
            **core,
            "signature": _signature(core, key, domain=_REGISTRY_DOMAIN),
        }
        _write_json(path, registry)
        return registry

    registry = _read_json(path)
    required = set(core) | {"signature"}
    if set(registry) != required:
        raise ResearchCliError("invalid evaluator run-path registry fields")
    signature = registry.get("signature")
    if not isinstance(signature, str):
        raise ResearchCliError("evaluator run-path registry is missing its HMAC")
    supplied_core = {field: registry[field] for field in core}
    expected = _signature(supplied_core, key, domain=_REGISTRY_DOMAIN)
    if not hmac.compare_digest(signature, expected):
        raise ResearchCliError("evaluator run-path registry HMAC verification failed")
    if supplied_core != core:
        raise ResearchCliError(
            "run directory is bound to a different sealed run_id or context"
        )
    return registry


def _authoritative_ledger_path(
    run_dir: Path,
    context: Mapping[str, Any],
    anchor_dir: str | Path | None,
) -> Path:
    if context["security_mode"] == SEALED:
        return _external_ledger_path(run_dir, context, anchor_dir)
    return _ledger_path(run_dir)


def _sync_ledger_mirror(run_dir: Path, events: tuple[Any, ...]) -> None:
    """Atomically publish the evaluator-owned sealed ledger as a read-only mirror."""

    path = _ledger_path(run_dir)
    if not events and not path.exists():
        return
    encoded = "".join(f"{event.to_json()}\n" for event in events)
    _atomic_replace_bytes(path, encoded.encode("utf-8"))


def _read_and_verify_anchor(
    run_dir: Path,
    context: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    evaluator_key: str | bytes | None,
    anchor_dir: str | Path | None,
    require_match: bool,
) -> dict[str, Any] | None:
    if context["security_mode"] != SEALED:
        return None
    key = _normalized_key(evaluator_key)
    if key is None:
        raise ResearchCliError("sealed anchor requires the evaluator key")
    path = _anchor_path(run_dir, context, anchor_dir)
    anchor = _read_json(path)
    required = {
        "schema_version",
        "run_id",
        "context_hash",
        "ledger_events",
        "ledger_tip",
        "signature",
    }
    if set(anchor) != required:
        raise ResearchCliError("invalid evaluator anchor fields")
    signature = anchor.get("signature")
    if not isinstance(signature, str):
        raise ResearchCliError("evaluator anchor is missing its HMAC signature")
    core = {field: anchor[field] for field in required if field != "signature"}
    expected = _signature(core, key, domain=_ANCHOR_DOMAIN)
    if not hmac.compare_digest(signature, expected):
        raise ResearchCliError("evaluator anchor HMAC verification failed")
    if anchor["schema_version"] != RUN_SCHEMA_VERSION:
        raise ResearchCliError("evaluator anchor schema does not match the run")
    if anchor["run_id"] != context["run_id"]:
        raise ResearchCliError("evaluator anchor run_id does not match context")
    if anchor["context_hash"] != checkpoint["context_hash"]:
        raise ResearchCliError("evaluator anchor does not bind the current context")
    if require_match and (
        anchor["ledger_events"] != checkpoint["ledger_events"]
        or anchor["ledger_tip"] != checkpoint["ledger_tip"]
    ):
        raise ResearchCliError(
            "ledger/checkpoint is not the externally anchored monotonic head"
        )
    return anchor


def _write_checkpoint(
    run_dir: Path,
    context: Mapping[str, Any],
    *,
    evaluator_key: str | bytes | None,
    anchor_dir: str | Path | None = None,
    initialize_anchor: bool = False,
    expected_previous: Mapping[str, Any] | None = None,
    lock_held: bool = False,
) -> dict[str, Any]:
    key = _normalized_key(evaluator_key)
    if context["security_mode"] == SEALED and key is None:
        raise ResearchCliError("sealed checkpoint requires the evaluator key")

    def make_checkpoint(events: tuple[Any, ...]) -> dict[str, Any]:
        core = {
            "schema_version": RUN_SCHEMA_VERSION,
            "security_mode": context["security_mode"],
            "context_hash": _content_hash(context),
            "ledger_events": len(events),
            "ledger_tip": events[-1].event_hash if events else GENESIS_HASH,
        }
        signature = (
            _signature(core, key, domain=_CHECKPOINT_DOMAIN)
            if context["security_mode"] == SEALED and key is not None
            else None
        )
        return {**core, "signature": signature}

    if context["security_mode"] != SEALED:
        events = JsonlEventLedger(_ledger_path(run_dir)).verify()
        checkpoint = make_checkpoint(events)
        _write_json(_checkpoint_path(run_dir), checkpoint)
        return checkpoint

    anchor_path = _anchor_path(run_dir, context, anchor_dir)
    lock = nullcontext() if lock_held else _exclusive_anchor_lock(anchor_path)
    with lock:
        authoritative_path = _external_ledger_path(run_dir, context, anchor_dir)
        if initialize_anchor and authoritative_path.exists():
            raise ResearchCliError(
                f"evaluator-owned ledger already exists: {authoritative_path}"
            )
        events = JsonlEventLedger(authoritative_path).verify()
        checkpoint = make_checkpoint(events)
        if initialize_anchor:
            if anchor_path.exists():
                raise ResearchCliError(f"evaluator anchor already exists: {anchor_path}")
            if expected_previous is not None:
                raise ResearchCliError("initial anchor cannot have a previous checkpoint")
            _write_or_verify_registry(
                run_dir,
                context,
                evaluator_key=key,
                anchor_dir=anchor_dir,
                initialize=True,
            )
        else:
            if expected_previous is None:
                raise ResearchCliError("sealed head advance requires the previous checkpoint")
            _write_or_verify_registry(
                run_dir,
                context,
                evaluator_key=key,
                anchor_dir=anchor_dir,
                initialize=False,
            )
            previous_anchor = _read_and_verify_anchor(
                run_dir,
                context,
                expected_previous,
                evaluator_key=key,
                anchor_dir=anchor_dir,
                require_match=True,
            )
            assert previous_anchor is not None
            old_count = int(previous_anchor["ledger_events"])
            old_tip = str(previous_anchor["ledger_tip"])
            if len(events) <= old_count:
                raise ResearchCliError("evaluator anchor counter must advance monotonically")
            if old_count == 0:
                if old_tip != GENESIS_HASH:
                    raise ResearchCliError("empty evaluator anchor has a non-genesis tip")
            elif events[old_count - 1].event_hash != old_tip:
                raise ResearchCliError(
                    "new evaluator ledger is not a descendant of the anchored head"
                )

        # The external lock serializes compare-and-swap updates.  Cross-file
        # crashes fail closed (anchor/checkpoint mismatch) and require trusted
        # operator recovery; they never silently accept a fork.
        _sync_ledger_mirror(run_dir, events)
        _write_json(_checkpoint_path(run_dir), checkpoint)
        anchor_core = _anchor_core(context, checkpoint)
        anchor = {
            **anchor_core,
            "signature": _signature(anchor_core, key, domain=_ANCHOR_DOMAIN),
        }
        _write_json(anchor_path, anchor)
    return checkpoint


def initialize_run(
    problem_path: str | Path,
    run_dir: str | Path,
    *,
    total_budget: float,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(run_dir)
    if (
        isinstance(total_budget, bool)
        or not isinstance(total_budget, (int, float))
        or not math.isfinite(float(total_budget))
        or float(total_budget) <= 0
    ):
        raise ResearchCliError("total_budget must be a finite positive number")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ResearchCliError(f"research run already exists: {destination}")
    key = _normalized_key(evaluator_key)
    mode = SEALED if key is not None else LOCAL_UNSEALED
    if mode == SEALED and anchor_dir is None:
        raise ResearchCliError("sealed mode requires an external evaluator anchor directory")
    if mode == LOCAL_UNSEALED and anchor_dir is not None:
        raise ResearchCliError("an evaluator anchor requires sealed mode")
    if mode == SEALED:
        _validated_anchor_directory(destination, anchor_dir)

    destination.mkdir(parents=True, exist_ok=True)
    prepared = prepare.load_problem(problem_path)
    views = adapt_prepare_problem(prepared)
    context_core = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": secrets.token_hex(16),
        "problem_id": views.public_problem.problem_id,
        "observation_hash": views.observation_bundle.content_hash(),
        "total_budget": float(total_budget),
        "security_mode": mode,
    }
    context = {
        **context_core,
        "signature": (
            _signature(context_core, key, domain=_CONTEXT_DOMAIN)
            if key is not None
            else None
        ),
    }
    _write_json(_context_path(destination), context)
    _write_json(destination / OBSERVATION_FILE, views.observation_bundle)
    # Touch through the ledger API only conceptually; an absent file is the
    # canonical empty hash chain and avoids an ambiguous blank JSONL line.
    ledger = JsonlEventLedger(_ledger_path(destination))
    ledger.verify()
    checkpoint = _write_checkpoint(
        destination,
        context,
        evaluator_key=key,
        anchor_dir=anchor_dir,
        initialize_anchor=mode == SEALED,
    )
    state = ResearchLoop(ledger, total_budget=total_budget).state
    return {
        "run_dir": str(destination),
        "context": context,
        "checkpoint": checkpoint,
        "security": _security_summary(mode),
        "observation_file": str(destination / OBSERVATION_FILE),
        "ledger_file": str(_ledger_path(destination)),
        "state": state.to_dict(),
    }


def _load_run_context(
    run_dir: Path,
    *,
    evaluator_key: str | bytes | None = None,
    require_sealed: bool = False,
) -> dict[str, Any]:
    context = _read_json(_context_path(run_dir))
    required = {
        "schema_version",
        "run_id",
        "problem_id",
        "observation_hash",
        "total_budget",
        "security_mode",
        "signature",
    }
    if set(context) != required:
        raise ResearchCliError(
            f"invalid context fields: missing={sorted(required - set(context))} "
            f"extra={sorted(set(context) - required)}"
        )
    if context["schema_version"] != RUN_SCHEMA_VERSION:
        raise ResearchCliError(
            f"unsupported research run schema: {context['schema_version']!r}"
        )
    run_id = context["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise ResearchCliError("context run_id must be a 128-bit lowercase hex value")
    key = _normalized_key(evaluator_key)
    mode = context["security_mode"]
    if mode not in {LOCAL_UNSEALED, SEALED}:
        raise ResearchCliError(f"invalid security mode: {mode!r}")
    if (require_sealed or key is not None) and mode != SEALED:
        raise ResearchCliError("sealed research context is required")
    if mode == SEALED:
        if key is None:
            raise ResearchCliError("sealed research context requires the evaluator key")
        signature = context["signature"]
        if not isinstance(signature, str):
            raise ResearchCliError("sealed context is missing its HMAC signature")
        core = {field: context[field] for field in required if field != "signature"}
        expected = _signature(core, key, domain=_CONTEXT_DOMAIN)
        if not hmac.compare_digest(signature, expected):
            raise ResearchCliError("research context HMAC verification failed")
    elif context["signature"] is not None:
        raise ResearchCliError("local_unsealed context must not contain a signature")
    try:
        budget = float(context["total_budget"])
    except (TypeError, ValueError) as exc:
        raise ResearchCliError("context total_budget must be a positive number") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise ResearchCliError("context total_budget must be a finite positive number")
    return context


def _verify_checkpoint(
    run_dir: Path,
    context: Mapping[str, Any],
    *,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = _read_json(_checkpoint_path(run_dir))
    required = {
        "schema_version",
        "security_mode",
        "context_hash",
        "ledger_events",
        "ledger_tip",
        "signature",
    }
    if set(checkpoint) != required:
        raise ResearchCliError("invalid checkpoint fields")
    if checkpoint["schema_version"] != RUN_SCHEMA_VERSION:
        raise ResearchCliError("checkpoint schema does not match the research run")
    if checkpoint["security_mode"] != context["security_mode"]:
        raise ResearchCliError("checkpoint security mode does not match context")
    if checkpoint["context_hash"] != _content_hash(context):
        raise ResearchCliError("checkpoint does not bind the current context")

    if context["security_mode"] == SEALED:
        _write_or_verify_registry(
            run_dir,
            context,
            evaluator_key=evaluator_key,
            anchor_dir=anchor_dir,
            initialize=False,
        )
        events = JsonlEventLedger(
            _external_ledger_path(run_dir, context, anchor_dir)
        ).verify()
        mirror_events = JsonlEventLedger(_ledger_path(run_dir)).verify()
        if tuple(event.event_hash for event in mirror_events) != tuple(
            event.event_hash for event in events
        ):
            raise ResearchCliError(
                "run-directory ledger mirror does not match the evaluator-owned ledger"
            )
    else:
        events = JsonlEventLedger(_ledger_path(run_dir)).verify()
    actual_tip = events[-1].event_hash if events else GENESIS_HASH
    if checkpoint["ledger_events"] != len(events) or checkpoint["ledger_tip"] != actual_tip:
        raise ResearchCliError("ledger does not match the evaluator checkpoint")

    key = _normalized_key(evaluator_key)
    if context["security_mode"] == SEALED:
        if key is None:
            raise ResearchCliError("sealed checkpoint requires the evaluator key")
        signature = checkpoint["signature"]
        if not isinstance(signature, str):
            raise ResearchCliError("sealed checkpoint is missing its HMAC signature")
        core = {field: checkpoint[field] for field in required if field != "signature"}
        expected = _signature(core, key, domain=_CHECKPOINT_DOMAIN)
        if not hmac.compare_digest(signature, expected):
            raise ResearchCliError("research checkpoint HMAC verification failed")
    elif checkpoint["signature"] is not None:
        raise ResearchCliError("local_unsealed checkpoint must not contain a signature")
    if context["security_mode"] == SEALED:
        _read_and_verify_anchor(
            run_dir,
            context,
            checkpoint,
            evaluator_key=key,
            anchor_dir=anchor_dir,
            require_match=True,
        )
    return checkpoint


def load_controller(
    problem_path: str | Path,
    run_dir: str | Path,
    *,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
    require_sealed: bool = True,
) -> ResearchController:
    directory = Path(run_dir)
    context = _load_run_context(
        directory,
        evaluator_key=evaluator_key,
        require_sealed=require_sealed,
    )
    _verify_checkpoint(
        directory,
        context,
        evaluator_key=evaluator_key,
        anchor_dir=anchor_dir,
    )
    prepared = prepare.load_problem(problem_path)
    views = adapt_prepare_problem(prepared)
    if views.public_problem.problem_id != context["problem_id"]:
        raise ResearchCliError("problem content does not match the initialized research run")
    if views.observation_bundle.content_hash() != context["observation_hash"]:
        raise ResearchCliError("agent observation does not match the initialized research run")
    return ResearchController(
        views.public_problem,
        _authoritative_ledger_path(directory, context, anchor_dir),
        total_budget=float(context["total_budget"]),
    )


def execute_action(
    problem_path: str | Path,
    run_dir: str | Path,
    action: Mapping[str, Any],
    *,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
    require_sealed: bool = True,
    expected_cursor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    initial_context = _load_run_context(
        directory,
        evaluator_key=evaluator_key,
        require_sealed=require_sealed,
    )

    def perform(*, lock_held: bool) -> dict[str, Any]:
        controller = load_controller(
            problem_path,
            directory,
            evaluator_key=evaluator_key,
            anchor_dir=anchor_dir,
            require_sealed=require_sealed,
        )
        context = _load_run_context(
            directory,
            evaluator_key=evaluator_key,
            require_sealed=require_sealed,
        )
        previous_checkpoint = _verify_checkpoint(
            directory,
            context,
            evaluator_key=evaluator_key,
            anchor_dir=anchor_dir,
        )
        if expected_cursor is not None:
            if set(expected_cursor) != {"ledger_events", "ledger_tip"}:
                raise ResearchCliError(
                    "expected cursor fields must be exactly ledger_events and ledger_tip"
                )
            expected_events = expected_cursor.get("ledger_events")
            expected_tip = expected_cursor.get("ledger_tip")
            if (
                isinstance(expected_events, bool)
                or not isinstance(expected_events, int)
                or expected_events < 0
                or not isinstance(expected_tip, str)
                or len(expected_tip) != 64
                or any(character not in "0123456789abcdef" for character in expected_tip)
            ):
                raise ResearchCliError("expected cursor is invalid")
            actual_cursor = {
                "ledger_events": previous_checkpoint["ledger_events"],
                "ledger_tip": previous_checkpoint["ledger_tip"],
            }
            if dict(expected_cursor) != actual_cursor:
                raise ResearchCliError(
                    "expected cursor no longer matches the authoritative ledger head"
                )
        receipt = controller.dispatch_external(action).to_dict()
        receipt["checkpoint"] = _write_checkpoint(
            directory,
            context,
            evaluator_key=evaluator_key,
            anchor_dir=anchor_dir,
            expected_previous=(
                previous_checkpoint if context["security_mode"] == SEALED else None
            ),
            lock_held=lock_held,
        )
        receipt["security"] = _security_summary(str(context["security_mode"]))
        return receipt

    if initial_context["security_mode"] == SEALED:
        with _exclusive_anchor_lock(_anchor_path(directory, initial_context, anchor_dir)):
            return perform(lock_held=True)
    return perform(lock_held=False)


def execute_action_file(
    problem_path: str | Path,
    run_dir: str | Path,
    action_path: str | Path,
    *,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
    require_sealed: bool = True,
    expected_cursor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return execute_action(
        problem_path,
        run_dir,
        _read_action_file(Path(action_path)),
        evaluator_key=evaluator_key,
        anchor_dir=anchor_dir,
        require_sealed=require_sealed,
        expected_cursor=expected_cursor,
    )


def run_status(
    run_dir: str | Path,
    *,
    evaluator_key: str | bytes | None = None,
    anchor_dir: str | Path | None = None,
    require_sealed: bool = True,
) -> dict[str, Any]:
    directory = Path(run_dir)
    context = _load_run_context(
        directory,
        evaluator_key=evaluator_key,
        require_sealed=require_sealed,
    )
    checkpoint = _verify_checkpoint(
        directory,
        context,
        evaluator_key=evaluator_key,
        anchor_dir=anchor_dir,
    )
    ledger = JsonlEventLedger(
        _authoritative_ledger_path(directory, context, anchor_dir)
    )
    state = ResearchLoop(ledger, total_budget=float(context["total_budget"])).state
    return {
        "context": context,
        "checkpoint": checkpoint,
        "security": _security_summary(str(context["security_mode"])),
        "ledger_events": len(ledger.verify()),
        "state": state.to_dict(),
    }


def render_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
