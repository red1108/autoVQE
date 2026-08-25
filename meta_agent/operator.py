"""Trusted operator for isolated Codex-driven AutoVQE campaigns.

The operator must run from a pinned checkout outside the action-producing
agent's write boundary. It owns the problem path, evaluator run, optional HMAC
key/anchor, and the only call into ``execute_action``. The generated bundle is
safe to expose to Codex and contains a standard-library-only file client.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autovqe.ansatz_ir import AnsatzSpec
from autovqe.controller import MAX_EXTERNAL_ACTION_BYTES
from autovqe.evaluator import candidate_hash
from autovqe.research_cli import (
    EVALUATOR_ANCHOR_DIR_ENV,
    EVALUATOR_KEY_ENV,
    LOCAL_UNSEALED,
    SEALED,
    evaluator_anchor_dir_from_environment,
    evaluator_key_from_environment,
    execute_action,
    initialize_run,
    run_status,
)


PROTOCOL_VERSION = 1
OPERATOR_MANIFEST = "operator_manifest.json"
BRIDGE_STATE = "bridge_pending.json"
MAX_BRIDGE_BYTES = MAX_EXTERNAL_ACTION_BYTES + 100_000
MAX_PROTECTED_FILE_BYTES = 10_000_000
MAX_TRUSTED_PROBLEM_BYTES = 50_000_000
MAX_PUBLIC_STATUS_BYTES = 2_000_000
MAX_PUBLISHED_STATUS_BYTES = 5_000_000
MAX_PUBLISHED_RECEIPT_BYTES = 10_000_000
MAX_RESULT_ARTIFACT_BYTES = 20_000_000
MAX_AUTH_TREE_BYTES = 500_000
PUBLICATION_AUTH_SCHEME = "merkle_lamport_sha256_v1"
PUBLICATION_AUTH_SLOTS = 512
PUBLICATION_SIGNER = "publication_signer.json"
PUBLICATION_TREE = "publication_auth_tree.json"
PUBLICATION_SIGNER_LOCK = "publication_signer.lock"
_MANIFEST_DOMAIN = b"autovqe-meta-operator-v1\0"
_PUBLICATION_MESSAGE_DOMAIN = b"autovqe-publication-message-v1\0"
_PUBLICATION_SECRET_DOMAIN = b"autovqe-publication-lamport-secret-v1\0"
_PUBLICATION_LEAF_DOMAIN = b"autovqe-publication-lamport-leaf-v1\0"
_PUBLICATION_NODE_DOMAIN = b"autovqe-publication-merkle-node-v1\0"
_PUBLICATION_STATE_DOMAIN = b"autovqe-publication-signer-state-v1\0"
_RESULT_ARTIFACT_DOMAIN = b"autovqe-terminal-result-v1\0"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_REQUEST_RE = re.compile(r"^(req_[0-9a-f]{24})\.ready\.json$")
_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z0-9_]+\}\}")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_EVALUATOR_PUBLICATION_FIELDS = frozenset(
    {"optimized_parameter_binding", "best_values"}
)
_REPO_ROOT = Path(__file__).resolve().parents[1]


class OperatorError(RuntimeError):
    """Raised when campaign preparation or bridge verification fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(value: str) -> None:
    raise OperatorError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OperatorError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded regular file without accepting a link swap."""

    try:
        before_open = path.lstat()
    except FileNotFoundError as exc:
        raise OperatorError(f"missing JSON file: {path}") from exc
    if stat.S_ISLNK(before_open.st_mode):
        raise OperatorError(f"symbolic links are not accepted: {path}")
    if not stat.S_ISREG(before_open.st_mode):
        raise OperatorError(f"JSON input must be a regular file: {path}")
    if before_open.st_size > max_bytes:
        raise OperatorError(
            f"JSON input exceeds {max_bytes} raw bytes: {before_open.st_size}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OperatorError(f"cannot securely open JSON input {path}: {exc}") from exc
    try:
        after_open = os.fstat(descriptor)
        if not stat.S_ISREG(after_open.st_mode):
            raise OperatorError(f"JSON input must be a regular file: {path}")
        if (before_open.st_dev, before_open.st_ino) != (
            after_open.st_dev,
            after_open.st_ino,
        ):
            raise OperatorError(f"JSON input changed while being opened: {path}")
        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if received > max_bytes:
                raise OperatorError(f"JSON input exceeds {max_bytes} raw bytes")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    return raw


def _read_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    raw = _read_regular_bytes(path, max_bytes=max_bytes)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise OperatorError(f"invalid UTF-8 JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OperatorError(f"{path} must contain a JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OperatorError(f"value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, max_bytes: int) -> str:
    return _sha256_bytes(_read_regular_bytes(path, max_bytes=max_bytes))


def _publication_secret(
    seed: bytes, key_index: int, bit_index: int, branch: int
) -> bytes:
    coordinates = (
        key_index.to_bytes(4, "big")
        + bit_index.to_bytes(2, "big")
        + branch.to_bytes(1, "big")
    )
    return hmac.new(
        seed, _PUBLICATION_SECRET_DOMAIN + coordinates, hashlib.sha256
    ).digest()


def _publication_leaf(seed: bytes, key_index: int) -> bytes:
    public_values: list[bytes] = []
    for bit_index in range(256):
        for branch in (0, 1):
            public_values.append(
                hashlib.sha256(
                    _publication_secret(seed, key_index, bit_index, branch)
                ).digest()
            )
    return hashlib.sha256(
        _PUBLICATION_LEAF_DOMAIN + b"".join(public_values)
    ).digest()


def _build_publication_tree(seed: bytes) -> list[list[bytes]]:
    leaves = [
        _publication_leaf(seed, key_index)
        for key_index in range(PUBLICATION_AUTH_SLOTS)
    ]
    levels = [leaves]
    current = leaves
    while len(current) > 1:
        current = [
            hashlib.sha256(
                _PUBLICATION_NODE_DOMAIN + current[index] + current[index + 1]
            ).digest()
            for index in range(0, len(current), 2)
        ]
        levels.append(current)
    return levels


def _signer_state_mac(core: Mapping[str, Any], seed: bytes) -> str:
    return hmac.new(
        seed,
        _PUBLICATION_STATE_DOMAIN + _canonical_bytes(core),
        hashlib.sha256,
    ).hexdigest()


def _write_signer_state(path: Path, core: Mapping[str, Any], seed: bytes) -> None:
    _atomic_write_json(path, {**core, "state_mac": _signer_state_mac(core, seed)})


def _create_publication_auth(
    evaluator_run: Path, *, session_id: str, campaign_id: str
) -> tuple[str, str]:
    seed = os.urandom(32)
    levels = _build_publication_tree(seed)
    root = levels[-1][0].hex()
    tree = {
        "schema_version": 1,
        "scheme": PUBLICATION_AUTH_SCHEME,
        "slots": PUBLICATION_AUTH_SLOTS,
        "root_sha256": root,
        "levels": [[node.hex() for node in level] for level in levels],
    }
    tree_path = evaluator_run / PUBLICATION_TREE
    _atomic_write_json(tree_path, tree)
    signer_core = {
        "schema_version": 1,
        "scheme": PUBLICATION_AUTH_SCHEME,
        "session_id": session_id,
        "campaign_id": campaign_id,
        "slots": PUBLICATION_AUTH_SLOTS,
        "root_sha256": root,
        "next_key_index": 0,
        "seed_hex": seed.hex(),
    }
    _write_signer_state(evaluator_run / PUBLICATION_SIGNER, signer_core, seed)
    return root, _sha256_file(tree_path, max_bytes=MAX_AUTH_TREE_BYTES)


def _load_publication_tree(manifest: Mapping[str, Any]) -> list[list[bytes]]:
    tree_path = Path(manifest["evaluator_run"]) / PUBLICATION_TREE
    if _sha256_file(tree_path, max_bytes=MAX_AUTH_TREE_BYTES) != manifest[
        "publication_tree_sha256"
    ]:
        raise OperatorError("publication authentication tree was modified")
    tree = _read_json_object(tree_path, max_bytes=MAX_AUTH_TREE_BYTES)
    required = {
        "schema_version",
        "scheme",
        "slots",
        "root_sha256",
        "levels",
    }
    if set(tree) != required or tree["schema_version"] != 1:
        raise OperatorError("publication authentication tree is invalid")
    if (
        tree["scheme"] != PUBLICATION_AUTH_SCHEME
        or tree["slots"] != PUBLICATION_AUTH_SLOTS
        or tree["root_sha256"] != manifest["publication_auth_root"]
    ):
        raise OperatorError("publication authentication tree binding mismatch")
    raw_levels = tree["levels"]
    expected_level_count = PUBLICATION_AUTH_SLOTS.bit_length()
    if not isinstance(raw_levels, list) or len(raw_levels) != expected_level_count:
        raise OperatorError("publication authentication tree levels are invalid")
    levels: list[list[bytes]] = []
    for level_index, raw_level in enumerate(raw_levels):
        expected_nodes = PUBLICATION_AUTH_SLOTS >> level_index
        if not isinstance(raw_level, list) or len(raw_level) != expected_nodes:
            raise OperatorError("publication authentication tree width is invalid")
        if not all(isinstance(node, str) and _SHA256_RE.fullmatch(node) for node in raw_level):
            raise OperatorError("publication authentication tree hash is invalid")
        levels.append([bytes.fromhex(node) for node in raw_level])
    if levels[-1][0].hex() != manifest["publication_auth_root"]:
        raise OperatorError("publication authentication root mismatch")
    return levels


def _load_signer_state(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    path = Path(manifest["evaluator_run"]) / PUBLICATION_SIGNER
    state = _read_json_object(path, max_bytes=100_000)
    core_fields = {
        "schema_version",
        "scheme",
        "session_id",
        "campaign_id",
        "slots",
        "root_sha256",
        "next_key_index",
        "seed_hex",
    }
    if set(state) != core_fields | {"state_mac"}:
        raise OperatorError("publication signer state fields are invalid")
    seed_hex = state["seed_hex"]
    if not isinstance(seed_hex, str) or not _SHA256_RE.fullmatch(seed_hex):
        raise OperatorError("publication signer seed is invalid")
    seed = bytes.fromhex(seed_hex)
    core = {field: state[field] for field in core_fields}
    supplied_mac = state["state_mac"]
    if not isinstance(supplied_mac, str) or not hmac.compare_digest(
        supplied_mac, _signer_state_mac(core, seed)
    ):
        raise OperatorError("publication signer state authentication failed")
    if (
        state["schema_version"] != 1
        or state["scheme"] != manifest["publication_auth_scheme"]
        or state["session_id"] != manifest["session_id"]
        or state["campaign_id"] != manifest["campaign_id"]
        or state["slots"] != manifest["publication_auth_slots"]
        or state["root_sha256"] != manifest["publication_auth_root"]
    ):
        raise OperatorError("publication signer state binding mismatch")
    next_index = state["next_key_index"]
    if isinstance(next_index, bool) or not isinstance(next_index, int):
        raise OperatorError("publication signer cursor is invalid")
    if not 0 <= next_index <= PUBLICATION_AUTH_SLOTS:
        raise OperatorError("publication signer cursor is out of range")
    return state, seed


def _validate_publication_binding(
    manifest: Mapping[str, Any], publication: Mapping[str, Any], message_type: str
) -> None:
    if publication.get("message_type") != message_type:
        raise OperatorError("publication message type mismatch")
    if publication.get("protocol_version") != PROTOCOL_VERSION:
        raise OperatorError("publication protocol version mismatch")
    if publication.get("session_id") != manifest["session_id"]:
        raise OperatorError("publication session binding mismatch")
    if publication.get("campaign_id") != manifest["campaign_id"]:
        raise OperatorError("publication campaign binding mismatch")
    if publication.get("session_manifest_sha256") != manifest[
        "session_manifest_sha256"
    ]:
        raise OperatorError("publication manifest binding mismatch")


def _authenticate_publication(
    manifest: Mapping[str, Any], publication: Mapping[str, Any], *, message_type: str
) -> dict[str, Any]:
    if "authentication" in publication:
        raise OperatorError("publication is already authenticated")
    _validate_publication_binding(manifest, publication, message_type)
    lock_path = Path(manifest["evaluator_run"]) / PUBLICATION_SIGNER_LOCK
    try:
        lock_descriptor = os.open(
            lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError as exc:
        raise OperatorError(
            "publication signer is busy or needs trusted stale-lock recovery"
        ) from exc
    try:
        os.write(lock_descriptor, f"pid={os.getpid()} at={_utc_now()}\n".encode("ascii"))
        os.close(lock_descriptor)
        lock_descriptor = -1
        state, seed = _load_signer_state(manifest)
        levels = _load_publication_tree(manifest)
        key_index = state["next_key_index"]
        if key_index >= PUBLICATION_AUTH_SLOTS:
            raise OperatorError("publication authentication keys are exhausted")
        next_core = {
            key: value
            for key, value in state.items()
            if key not in {"state_mac", "next_key_index"}
        }
        next_core["next_key_index"] = key_index + 1
        _write_signer_state(
            Path(manifest["evaluator_run"]) / PUBLICATION_SIGNER,
            next_core,
            seed,
        )

        unsigned = dict(publication)
        if message_type == "status":
            if "publication_capacity" in unsigned:
                raise OperatorError("status publication capacity is operator-owned")
            unsigned["publication_capacity"] = {
                "total_slots": PUBLICATION_AUTH_SLOTS,
                "remaining_slots_after_publish": PUBLICATION_AUTH_SLOTS
                - key_index
                - 1,
            }
        message_digest = hashlib.sha256(
            _PUBLICATION_MESSAGE_DOMAIN + _canonical_bytes(unsigned)
        ).digest()
        revealed_secrets: list[str] = []
        opposite_hashes: list[str] = []
        public_values: list[bytes] = []
        for bit_index in range(256):
            selected = (message_digest[bit_index // 8] >> (7 - bit_index % 8)) & 1
            secrets = (
                _publication_secret(seed, key_index, bit_index, 0),
                _publication_secret(seed, key_index, bit_index, 1),
            )
            hashes = tuple(hashlib.sha256(value).digest() for value in secrets)
            revealed_secrets.append(secrets[selected].hex())
            opposite_hashes.append(hashes[1 - selected].hex())
            public_values.extend(hashes)
        leaf = hashlib.sha256(
            _PUBLICATION_LEAF_DOMAIN + b"".join(public_values)
        ).digest()
        if not hmac.compare_digest(leaf, levels[0][key_index]):
            raise OperatorError("publication signer seed does not match public tree")
        proof: list[str] = []
        cursor = key_index
        for level in levels[:-1]:
            proof.append(level[cursor ^ 1].hex())
            cursor //= 2
        authentication = {
            "scheme": PUBLICATION_AUTH_SCHEME,
            "key_index": key_index,
            "message_sha256": message_digest.hex(),
            "revealed_secrets": revealed_secrets,
            "opposite_hashes": opposite_hashes,
            "merkle_proof": proof,
        }
        authenticated = {**unsigned, "authentication": authentication}
        byte_limit = (
            MAX_PUBLISHED_STATUS_BYTES
            if message_type == "status"
            else MAX_PUBLISHED_RECEIPT_BYTES
        )
        if len(_render_json_bytes(authenticated)) > byte_limit:
            raise OperatorError(
                f"authenticated {message_type} exceeds the client publication limit"
            )
        return authenticated
    finally:
        if "lock_descriptor" in locals() and lock_descriptor >= 0:
            os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def _verify_authenticated_publication(
    manifest: Mapping[str, Any], publication: Mapping[str, Any], *, message_type: str
) -> dict[str, Any]:
    _validate_publication_binding(manifest, publication, message_type)
    authentication = publication.get("authentication")
    required = {
        "scheme",
        "key_index",
        "message_sha256",
        "revealed_secrets",
        "opposite_hashes",
        "merkle_proof",
    }
    if not isinstance(authentication, Mapping) or set(authentication) != required:
        raise OperatorError("publication authentication fields are invalid")
    unsigned = {key: value for key, value in publication.items() if key != "authentication"}
    digest = hashlib.sha256(
        _PUBLICATION_MESSAGE_DOMAIN + _canonical_bytes(unsigned)
    ).digest()
    if (
        authentication["scheme"] != manifest["publication_auth_scheme"]
        or not isinstance(authentication["message_sha256"], str)
        or not hmac.compare_digest(authentication["message_sha256"], digest.hex())
    ):
        raise OperatorError("publication message authentication mismatch")
    key_index = authentication["key_index"]
    if (
        isinstance(key_index, bool)
        or not isinstance(key_index, int)
        or not 0 <= key_index < manifest["publication_auth_slots"]
    ):
        raise OperatorError("publication authentication key index is invalid")
    revealed = authentication["revealed_secrets"]
    opposite = authentication["opposite_hashes"]
    proof = authentication["merkle_proof"]
    proof_length = manifest["publication_auth_slots"].bit_length() - 1
    if (
        not isinstance(revealed, list)
        or len(revealed) != 256
        or not isinstance(opposite, list)
        or len(opposite) != 256
        or not isinstance(proof, list)
        or len(proof) != proof_length
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in revealed)
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in opposite)
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in proof)
    ):
        raise OperatorError("publication hash-based signature is invalid")
    public_values: list[bytes] = []
    for bit_index in range(256):
        selected = (digest[bit_index // 8] >> (7 - bit_index % 8)) & 1
        selected_hash = hashlib.sha256(bytes.fromhex(revealed[bit_index])).digest()
        other_hash = bytes.fromhex(opposite[bit_index])
        pair = [b"", b""]
        pair[selected] = selected_hash
        pair[1 - selected] = other_hash
        public_values.extend(pair)
    node = hashlib.sha256(
        _PUBLICATION_LEAF_DOMAIN + b"".join(public_values)
    ).digest()
    cursor = key_index
    for sibling_hex in proof:
        sibling = bytes.fromhex(sibling_hex)
        if cursor & 1:
            node = hashlib.sha256(
                _PUBLICATION_NODE_DOMAIN + sibling + node
            ).digest()
        else:
            node = hashlib.sha256(
                _PUBLICATION_NODE_DOMAIN + node + sibling
            ).digest()
        cursor //= 2
    if not hmac.compare_digest(node.hex(), manifest["publication_auth_root"]):
        raise OperatorError("publication signature does not match campaign public root")
    return unsigned


def _path_identity_hash(path: Path) -> str:
    rendered = path.resolve().as_posix()
    if os.name == "nt":
        rendered = rendered.casefold()
    return _sha256_bytes(rendered.encode("utf-8"))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_descriptor = os.open(path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _render_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _render_json_bytes(value))


def _resolve(value: str | Path, *, base: Path = _REPO_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _campaign_config(path: Path) -> dict[str, Any]:
    config = _read_json_object(path, max_bytes=100_000)
    required = {
        "schema_version",
        "campaign_id",
        "research_mode",
        "problem",
        "total_budget",
        "model_label",
        "local_agent_bundle",
        "local_evaluator_run",
    }
    if set(config) != required:
        raise OperatorError(
            "campaign config fields must be exactly " + ", ".join(sorted(required))
        )
    if config["schema_version"] != 1:
        raise OperatorError("unsupported campaign config schema")
    if not isinstance(config["campaign_id"], str) or not _ID_RE.fullmatch(
        config["campaign_id"]
    ):
        raise OperatorError("campaign_id is invalid")
    if config["research_mode"] not in {"discovery", "literature_assisted"}:
        raise OperatorError("research_mode must be discovery or literature_assisted")
    if not isinstance(config["model_label"], str) or not config["model_label"].strip():
        raise OperatorError("model_label must be a non-empty string")
    budget = config["total_budget"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) <= 0
    ):
        raise OperatorError("total_budget must be a finite positive number")
    for field in ("problem", "local_agent_bundle", "local_evaluator_run"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise OperatorError(f"{field} must be a non-empty path string")
    return config


def _source_files() -> tuple[Path, ...]:
    files: set[Path] = {
        _REPO_ROOT / "pyproject.toml",
        _REPO_ROOT / "uv.lock",
        Path(__file__).resolve(),
        (_REPO_ROOT / "meta_agent" / "client.py").resolve(),
        (_REPO_ROOT / "meta_agent" / "CODEX_GOAL.md").resolve(),
        (_REPO_ROOT / "meta_agent" / "AGENT_CONTRACT.md").resolve(),
    }
    files.update(path.resolve() for path in (_REPO_ROOT / "autovqe").rglob("*.py"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise OperatorError(f"trusted source file is missing: {missing[0]}")
    return tuple(sorted(files, key=lambda path: path.as_posix()))


def _source_tree_hash() -> str:
    digest = hashlib.sha256()
    for path in _source_files():
        relative = path.relative_to(_REPO_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = _read_regular_bytes(path, max_bytes=MAX_PROTECTED_FILE_BYTES)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git_identity() -> tuple[str | None, bool]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    head = run("rev-parse", "HEAD")
    commit = head.stdout.strip() if head.returncode == 0 else None
    status_result = run("status", "--porcelain", "--untracked-files=all")
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def _manifest_signature(core: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, _MANIFEST_DOMAIN + _canonical_bytes(core), hashlib.sha256).hexdigest()


def _render_goal(template: str, replacements: Mapping[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    leftovers = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if leftovers:
        raise OperatorError(f"unresolved goal placeholders: {leftovers}")
    return rendered


def _bundle_agents_text(research_mode: str) -> str:
    internet_rule = (
        "Do not use web search or outside literature during this discovery campaign."
        if research_mode == "discovery"
        else "Literature may be used only when each influence is recorded in journal.md."
    )
    return f"""# AutoVQE isolated campaign instructions

This directory is the complete Codex workspace for one research campaign.
Follow `GOAL.md` and `AGENT_CONTRACT.md` exactly.

- Read only files inside this workspace. Never traverse parent directories.
- Write only direct children of `actions/` and `journal.md`.
- Submit state changes only with
  `uv run --no-project python client.py submit --action actions/<file>.json`.
- Before submission, preflight the file with
  `uv run --no-project python client.py validate --action actions/<file>.json`;
  validation checks only local JSON/path/size rules and is not evaluator evidence.
- Read evaluator state only with
  `uv run --no-project python client.py status` and bridge receipts.
- Do not edit `GOAL.md`, `AGENT_CONTRACT.md`, `AGENTS.md`, `client.py`,
  `session.json`, `observation.json`, `inbox/`, or `outbox/`.
- Do not seek the target source file, exact answer, evaluator implementation,
  key, anchor, run directory, repository, git history, or prior outputs.
- Never treat a local file edit or self-computed score as evaluator evidence.
- {internet_rule}
"""


def _status_payload(manifest: Mapping[str, Any], status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "status",
        "session_id": manifest["session_id"],
        "campaign_id": manifest["campaign_id"],
        "session_manifest_sha256": manifest["session_manifest_sha256"],
        "published_at_utc": _utc_now(),
        "status": _public_status(status),
    }


def _bounded_public_value(value: Any, *, max_bytes: int) -> Any:
    encoded = _canonical_bytes(value)
    if len(encoded) <= max_bytes:
        return json.loads(encoded.decode("utf-8"))
    return {
        "omitted": True,
        "canonical_bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }


def _redact_private_evaluator_values(value: Any) -> Any:
    """Remove terminal-export-only evaluator values from agent publications."""

    if isinstance(value, Mapping):
        return {
            key: _redact_private_evaluator_values(child)
            for key, child in value.items()
            if key not in _PRIVATE_EVALUATOR_PUBLICATION_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_redact_private_evaluator_values(child) for child in value]
    return value


def _candidate_spec_summary(spec: Any) -> dict[str, Any]:
    encoded = _canonical_bytes(spec)
    summary: dict[str, Any] = {
        "omitted": True,
        "canonical_bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }
    if isinstance(spec, Mapping):
        for field in ("version", "num_qubits"):
            if isinstance(spec.get(field), (str, int, float)) and not isinstance(
                spec.get(field), bool
            ):
                summary[field] = spec[field]
        parameters = spec.get("parameters")
        operations = spec.get("operations")
        layers = spec.get("layers")
        if isinstance(parameters, list):
            summary["declared_parameters"] = len(parameters)
        if isinstance(operations, list):
            summary["top_level_operations"] = len(operations)
        if isinstance(layers, list):
            summary["layers"] = len(layers)
    return summary


def _public_state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    state = _redact_private_evaluator_values(state)
    if not isinstance(state, Mapping):
        raise OperatorError("evaluator state must be an object")
    summary = {
        key: _bounded_public_value(value, max_bytes=8_000)
        for key, value in state.items()
        if key not in {"hypotheses", "candidates", "probes", "evaluations"}
    }
    for field in ("hypotheses", "candidates", "probes", "evaluations"):
        records = state.get(field, {})
        summary[f"{field}_count"] = len(records) if isinstance(records, Mapping) else 0
    return summary


def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    state = _redact_private_evaluator_values(state)
    if not isinstance(state, Mapping):
        raise OperatorError("evaluator state must be an object")
    result = _public_state_summary(state)
    hypotheses = state.get("hypotheses", {})
    candidates = state.get("candidates", {})
    probes = state.get("probes", {})
    evaluations = state.get("evaluations", {})
    if not all(
        isinstance(value, Mapping)
        for value in (hypotheses, candidates, probes, evaluations)
    ):
        raise OperatorError("evaluator state collections are invalid")

    result["hypotheses"] = {
        str(key): {
            **{
                field: _bounded_public_value(value, max_bytes=8_000)
                for field, value in record.items()
                if field not in {"claim", "metadata"}
            },
            "claim": _bounded_public_value(record.get("claim", {}), max_bytes=8_000),
            "metadata": _bounded_public_value(
                record.get("metadata", {}), max_bytes=4_000
            ),
        }
        for key, record in hypotheses.items()
        if isinstance(record, Mapping)
    }
    result["candidates"] = {
        str(key): {
            **{
                field: _bounded_public_value(value, max_bytes=8_000)
                for field, value in record.items()
                if field not in {"spec", "metadata"}
            },
            "spec": _candidate_spec_summary(record.get("spec", {})),
            "metadata": _bounded_public_value(
                record.get("metadata", {}), max_bytes=4_000
            ),
        }
        for key, record in candidates.items()
        if isinstance(record, Mapping)
    }
    result["probes"] = {
        str(key): {
            **{
                field: _bounded_public_value(value, max_bytes=8_000)
                for field, value in record.items()
                if field != "result"
            },
            "result": _bounded_public_value(record.get("result", {}), max_bytes=8_000),
        }
        for key, record in probes.items()
        if isinstance(record, Mapping)
    }
    result["evaluations"] = {
        str(key): {
            **{
                field: _bounded_public_value(value, max_bytes=8_000)
                for field, value in record.items()
                if field != "metrics"
            },
            "metrics": _bounded_public_value(
                record.get("metrics", {}), max_bytes=16_000
            ),
        }
        for key, record in evaluations.items()
        if isinstance(record, Mapping)
    }
    return result


def _public_status(status: Mapping[str, Any]) -> dict[str, Any]:
    """Remove HMACs and evaluator-only filesystem/anchor details from status."""

    context = status["context"]
    checkpoint = status["checkpoint"]
    security = status["security"]
    public = {
        "context": {
            "schema_version": context["schema_version"],
            "run_id": context["run_id"],
            "problem_id": context["problem_id"],
            "observation_hash": context["observation_hash"],
            "total_budget": context["total_budget"],
            "security_mode": context["security_mode"],
        },
        "checkpoint": {
            "schema_version": checkpoint["schema_version"],
            "security_mode": checkpoint["security_mode"],
            "ledger_events": checkpoint["ledger_events"],
            "ledger_tip": checkpoint["ledger_tip"],
        },
        "security": {
            key: security[key]
            for key in (
                "mode",
                "tamper_evident",
                "rollback_protected",
                "trust_boundary",
                "warning",
            )
            if key in security
        },
        "ledger_events": status["ledger_events"],
        "state": _public_state(status["state"]),
    }
    if len(_canonical_bytes(public)) > MAX_PUBLIC_STATUS_BYTES:
        public["state"] = {
            **_public_state_summary(status["state"]),
            "history_truncated": True,
        }
    return public


def _public_action_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Allowlist the controller receipt while dropping checkpoint HMAC material."""

    if result is None:
        return None
    required = {
        "action_type",
        "generated_event_hashes",
        "result",
        "state",
        "checkpoint",
        "security",
    }
    if set(result) != required:
        raise OperatorError("evaluator returned an unexpected action receipt shape")
    checkpoint = result["checkpoint"]
    security = result["security"]
    return {
        "action_type": result["action_type"],
        "generated_event_hashes": result["generated_event_hashes"],
        "result": _bounded_public_value(
            _redact_private_evaluator_values(result["result"]),
            max_bytes=64_000,
        ),
        "state": _public_state_summary(result["state"]),
        "checkpoint": {
            key: checkpoint[key]
            for key in (
                "schema_version",
                "security_mode",
                "ledger_events",
                "ledger_tip",
            )
        },
        "security": {
            key: security[key]
            for key in (
                "mode",
                "tamper_evident",
                "rollback_protected",
                "trust_boundary",
                "warning",
            )
            if key in security
        },
    }


def _verified_status(
    manifest: Mapping[str, Any],
    *,
    key: bytes | None,
    anchor_dir: Path | None,
) -> dict[str, Any]:
    return run_status(
        manifest["evaluator_run"],
        evaluator_key=key,
        anchor_dir=anchor_dir,
        require_sealed=manifest["security_mode"] == SEALED,
    )


def _publish_status(
    manifest: Mapping[str, Any],
    status: Mapping[str, Any],
) -> None:
    bundle = Path(manifest["agent_bundle"])
    publication = _authenticate_publication(
        manifest, _status_payload(manifest, status), message_type="status"
    )
    _atomic_write_json(bundle / "inbox" / "latest_status.json", publication)


def prepare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    campaign_path = _resolve(args.campaign)
    config = _campaign_config(campaign_path)
    model_label = getattr(args, "model_label", None) or config["model_label"]
    if not isinstance(model_label, str) or not model_label.strip():
        raise OperatorError("model label must be a non-empty string")
    model_label = model_label.strip()
    security_mode = args.security
    if security_mode not in {LOCAL_UNSEALED, SEALED}:
        raise OperatorError("invalid security mode")

    if security_mode == SEALED:
        if args.agent_bundle is None or args.evaluator_run is None:
            raise OperatorError(
                "sealed preparation requires explicit --agent-bundle and --evaluator-run"
            )
        if not Path(args.agent_bundle).is_absolute() or not Path(args.evaluator_run).is_absolute():
            raise OperatorError("sealed bundle and evaluator run paths must be absolute")
        key = evaluator_key_from_environment(required=True)
        anchor_dir = evaluator_anchor_dir_from_environment(required=True)
        assert anchor_dir is not None
        if not anchor_dir.is_absolute():
            raise OperatorError("sealed evaluator anchor directory must be absolute")
    else:
        key = None
        anchor_dir = None

    problem = _resolve(config["problem"])
    if not problem.is_file():
        raise OperatorError(f"campaign problem does not exist: {problem}")
    agent_bundle = _resolve(args.agent_bundle or config["local_agent_bundle"])
    evaluator_run = _resolve(args.evaluator_run or config["local_evaluator_run"])
    if _overlap(agent_bundle, evaluator_run):
        raise OperatorError("agent bundle and evaluator run must be disjoint directories")
    if security_mode == SEALED:
        assert anchor_dir is not None
        anchor_dir = anchor_dir.resolve()
        if _overlap(agent_bundle, _REPO_ROOT):
            raise OperatorError("sealed agent bundle must be outside the trusted checkout")
        if _overlap(evaluator_run, _REPO_ROOT):
            raise OperatorError("sealed evaluator run must be outside the trusted checkout")
        if _overlap(agent_bundle, problem):
            raise OperatorError("sealed agent bundle must not expose the raw problem path")
        if _overlap(agent_bundle, anchor_dir):
            raise OperatorError(
                "sealed evaluator anchor directory must be disjoint from the agent bundle"
            )
        if _overlap(evaluator_run, anchor_dir):
            raise OperatorError(
                "sealed evaluator anchor directory must be disjoint from the evaluator run"
            )
    for path, label in ((agent_bundle, "agent bundle"), (evaluator_run, "evaluator run")):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise OperatorError(f"{label} already exists and is not empty: {path}")

    git_commit, git_dirty = _git_identity()
    if security_mode == SEALED and git_dirty and not args.allow_dirty_evaluator:
        raise OperatorError(
            "sealed evaluator checkout is dirty; commit/pin it or explicitly use "
            "--allow-dirty-evaluator for a non-benchmark run"
        )
    source_hash = _source_tree_hash()
    initialized = initialize_run(
        problem,
        evaluator_run,
        total_budget=float(config["total_budget"]),
        evaluator_key=key,
        anchor_dir=anchor_dir,
    )
    context = initialized["context"]
    status = run_status(
        evaluator_run,
        evaluator_key=key,
        anchor_dir=anchor_dir,
        require_sealed=security_mode == SEALED,
    )
    session_id = os.urandom(16).hex()
    publication_auth_root, publication_tree_hash = _create_publication_auth(
        evaluator_run,
        session_id=session_id,
        campaign_id=str(config["campaign_id"]),
    )
    observation_bytes = _read_regular_bytes(
        evaluator_run / "observation.json", max_bytes=MAX_PROTECTED_FILE_BYTES
    )
    observation = json.loads(observation_bytes.decode("utf-8"))
    if not isinstance(observation, dict):
        raise OperatorError("generated observation is not an object")

    contract_bytes = _read_regular_bytes(
        _REPO_ROOT / "meta_agent" / "AGENT_CONTRACT.md",
        max_bytes=MAX_PROTECTED_FILE_BYTES,
    )
    client_bytes = _read_regular_bytes(
        _REPO_ROOT / "meta_agent" / "client.py",
        max_bytes=MAX_PROTECTED_FILE_BYTES,
    )
    goal_template = (_REPO_ROOT / "meta_agent" / "CODEX_GOAL.md").read_text(
        encoding="utf-8"
    )
    state = status["state"]
    replacements = {
        "CAMPAIGN_ID": str(config["campaign_id"]),
        "RUN_ID": str(context["run_id"]),
        "PROBLEM_ID": str(context["problem_id"]),
        "OBSERVATION_HASH": str(context["observation_hash"]),
        "TOTAL_BUDGET": str(float(config["total_budget"])),
        "REMAINING_BUDGET": str(state["remaining_budget"]),
        "GATEWAY_COMMAND": (
            "uv run --no-project python client.py submit --action actions/<action>.json"
        ),
        "STATUS_COMMAND": "uv run --no-project python client.py status",
        "WORKSPACE_ROOT": ".",
        "SECURITY_MODE": security_mode,
        "RESEARCH_MODE": str(config["research_mode"]),
        "MODEL_LABEL": model_label,
    }
    goal_bytes = _render_goal(goal_template, replacements).encode("utf-8")
    goal_hash = _sha256_bytes(goal_bytes)
    contract_hash = _sha256_bytes(contract_bytes)
    client_hash = _sha256_bytes(client_bytes)
    observation_file_hash = _sha256_bytes(observation_bytes)
    agents_bytes = _bundle_agents_text(str(config["research_mode"])).encode("utf-8")
    agents_hash = _sha256_bytes(agents_bytes)

    agent_bundle.mkdir(parents=True, exist_ok=True)
    for child in ("actions", "inbox", "outbox"):
        (agent_bundle / child).mkdir(parents=True, exist_ok=True)
    session_core = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session_id,
        "campaign_id": config["campaign_id"],
        "run_id": context["run_id"],
        "problem_id": context["problem_id"],
        "observation_hash": context["observation_hash"],
        "total_budget": float(config["total_budget"]),
        "security_mode": security_mode,
        "research_mode": config["research_mode"],
        "model_label": model_label,
        "goal_sha256": goal_hash,
        "contract_sha256": contract_hash,
        "client_sha256": client_hash,
        "evaluator_source_sha256": source_hash,
        "observation_file_sha256": observation_file_hash,
        "publication_auth_scheme": PUBLICATION_AUTH_SCHEME,
        "publication_auth_root": publication_auth_root,
        "publication_auth_slots": PUBLICATION_AUTH_SLOTS,
    }
    session = {
        **session_core,
        "manifest_sha256": _sha256_bytes(_canonical_bytes(session_core)),
    }
    _atomic_write_bytes(agent_bundle / "observation.json", observation_bytes)
    _atomic_write_bytes(agent_bundle / "AGENT_CONTRACT.md", contract_bytes)
    _atomic_write_bytes(agent_bundle / "GOAL.md", goal_bytes)
    _atomic_write_bytes(agent_bundle / "client.py", client_bytes)
    _atomic_write_bytes(
        agent_bundle / "AGENTS.md",
        agents_bytes,
    )
    _atomic_write_json(agent_bundle / "session.json", session)
    _atomic_write_bytes(
        agent_bundle / "journal.md",
        b"# Campaign journal\n\nRecord predictions before each evaluator action.\n",
    )

    manifest_core = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "campaign_id": config["campaign_id"],
        "research_mode": config["research_mode"],
        "security_mode": security_mode,
        "problem_path": str(problem),
        "problem_file_sha256": _sha256_file(
            problem, max_bytes=MAX_TRUSTED_PROBLEM_BYTES
        ),
        "evaluator_run": str(evaluator_run),
        "agent_bundle": str(agent_bundle),
        "total_budget": float(config["total_budget"]),
        "model_label": model_label,
        "run_id": context["run_id"],
        "problem_id": context["problem_id"],
        "observation_hash": context["observation_hash"],
        "goal_sha256": goal_hash,
        "contract_sha256": contract_hash,
        "client_sha256": client_hash,
        "observation_file_sha256": observation_file_hash,
        "agents_sha256": agents_hash,
        "session_manifest_sha256": session["manifest_sha256"],
        "session_id": session["session_id"],
        "evaluator_source_sha256": source_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "anchor_directory_sha256": (
            _path_identity_hash(anchor_dir) if anchor_dir is not None else None
        ),
        "publication_auth_scheme": PUBLICATION_AUTH_SCHEME,
        "publication_auth_root": publication_auth_root,
        "publication_auth_slots": PUBLICATION_AUTH_SLOTS,
        "publication_tree_sha256": publication_tree_hash,
    }
    manifest = {
        **manifest_core,
        "signature": (
            _manifest_signature(manifest_core, key)
            if security_mode == SEALED and key is not None
            else None
        ),
    }
    _atomic_write_json(evaluator_run / OPERATOR_MANIFEST, manifest)
    _publish_status(manifest, status)
    return {
        "campaign_id": config["campaign_id"],
        "security_mode": security_mode,
        "research_mode": config["research_mode"],
        "agent_bundle": str(agent_bundle),
        "evaluator_run": str(evaluator_run),
        "goal_file": str(agent_bundle / "GOAL.md"),
        "source_sha256": source_hash,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "next": (
            f"open Codex with workspace {agent_bundle}, start the trusted bridge, "
            "then paste GOAL.md as /goal"
        ),
    }


def _load_manifest(
    evaluator_run: Path,
    *,
    require_sealed: bool = True,
) -> tuple[dict[str, Any], bytes | None, Path | None]:
    manifest = _read_json_object(evaluator_run / OPERATOR_MANIFEST, max_bytes=200_000)
    core_fields = {
        "schema_version",
        "protocol_version",
        "campaign_id",
        "research_mode",
        "security_mode",
        "problem_path",
        "problem_file_sha256",
        "evaluator_run",
        "agent_bundle",
        "total_budget",
        "model_label",
        "run_id",
        "problem_id",
        "observation_hash",
        "goal_sha256",
        "contract_sha256",
        "client_sha256",
        "observation_file_sha256",
        "agents_sha256",
        "session_manifest_sha256",
        "session_id",
        "evaluator_source_sha256",
        "git_commit",
        "git_dirty",
        "anchor_directory_sha256",
        "publication_auth_scheme",
        "publication_auth_root",
        "publication_auth_slots",
        "publication_tree_sha256",
    }
    if set(manifest) != core_fields | {"signature"}:
        raise OperatorError("operator manifest fields are invalid")
    if manifest["schema_version"] != 1 or manifest["protocol_version"] != PROTOCOL_VERSION:
        raise OperatorError("unsupported operator manifest schema")
    if (
        manifest["publication_auth_scheme"] != PUBLICATION_AUTH_SCHEME
        or manifest["publication_auth_slots"] != PUBLICATION_AUTH_SLOTS
        or not isinstance(manifest["publication_auth_root"], str)
        or not _SHA256_RE.fullmatch(manifest["publication_auth_root"])
        or not isinstance(manifest["publication_tree_sha256"], str)
        or not _SHA256_RE.fullmatch(manifest["publication_tree_sha256"])
    ):
        raise OperatorError("operator publication authentication fields are invalid")
    if Path(manifest["evaluator_run"]).resolve() != evaluator_run.resolve():
        raise OperatorError("operator manifest evaluator_run path mismatch")
    mode = manifest["security_mode"]
    if require_sealed and mode != SEALED:
        raise OperatorError(
            "operator requires a sealed campaign; pass --allow-unsealed only for a local integration run"
        )
    if mode == SEALED:
        key = evaluator_key_from_environment(required=True)
        anchor_dir = evaluator_anchor_dir_from_environment(required=True)
        assert key is not None
        signature = manifest.get("signature")
        if not isinstance(signature, str):
            raise OperatorError("sealed operator manifest is unsigned")
        core = {field: manifest[field] for field in core_fields}
        expected = _manifest_signature(core, key)
        if not hmac.compare_digest(signature, expected):
            raise OperatorError("operator manifest HMAC verification failed")
        assert anchor_dir is not None
        if _path_identity_hash(anchor_dir) != manifest["anchor_directory_sha256"]:
            raise OperatorError("operator anchor directory does not match the campaign")
    elif mode == LOCAL_UNSEALED:
        key = None
        anchor_dir = None
        if manifest.get("signature") is not None:
            raise OperatorError("local operator manifest must not be signed")
    else:
        raise OperatorError("operator manifest security mode is invalid")
    _verify_trusted_inputs(manifest)
    _load_publication_tree(manifest)
    _load_signer_state(manifest)
    return manifest, key, anchor_dir


def _verify_trusted_inputs(manifest: Mapping[str, Any]) -> None:
    if _source_tree_hash() != manifest["evaluator_source_sha256"]:
        raise OperatorError("trusted evaluator source changed after campaign preparation")
    problem = Path(manifest["problem_path"])
    if not problem.is_file() or _sha256_file(
        problem, max_bytes=MAX_TRUSTED_PROBLEM_BYTES
    ) != manifest["problem_file_sha256"]:
        raise OperatorError("raw problem changed after campaign preparation")


def _verify_bundle(manifest: Mapping[str, Any]) -> None:
    bundle = Path(manifest["agent_bundle"])
    try:
        bundle_metadata = bundle.lstat()
    except FileNotFoundError as exc:
        raise OperatorError("agent bundle is missing") from exc
    if stat.S_ISLNK(bundle_metadata.st_mode) or not stat.S_ISDIR(bundle_metadata.st_mode):
        raise OperatorError("agent bundle must be a non-link directory")
    resolved_bundle = bundle.resolve()
    literal_bundle = Path(os.path.abspath(bundle))
    if os.path.normcase(str(resolved_bundle)) != os.path.normcase(str(literal_bundle)):
        raise OperatorError("agent bundle root resolves through a link or junction")
    for child_name in ("actions", "inbox", "outbox"):
        child = bundle / child_name
        try:
            child_metadata = child.lstat()
        except FileNotFoundError as exc:
            raise OperatorError(f"agent bundle directory is missing: {child_name}") from exc
        if stat.S_ISLNK(child_metadata.st_mode) or not stat.S_ISDIR(child_metadata.st_mode):
            raise OperatorError(
                f"agent bundle child must be a non-link directory: {child_name}"
            )
        if child.resolve().parent != resolved_bundle:
            raise OperatorError(
                f"agent bundle child resolves outside the bundle: {child_name}"
            )
    session = _read_json_object(bundle / "session.json", max_bytes=100_000)
    for field in (
        "campaign_id",
        "run_id",
        "problem_id",
        "observation_hash",
        "total_budget",
        "security_mode",
        "research_mode",
        "model_label",
        "goal_sha256",
        "contract_sha256",
        "client_sha256",
        "observation_file_sha256",
        "evaluator_source_sha256",
        "publication_auth_scheme",
        "publication_auth_root",
        "publication_auth_slots",
    ):
        if session.get(field) != manifest[field]:
            raise OperatorError(f"agent session does not match operator manifest: {field}")
    if session.get("manifest_sha256") != manifest["session_manifest_sha256"]:
        raise OperatorError("agent session manifest hash mismatch")
    session_core = {key: value for key, value in session.items() if key != "manifest_sha256"}
    if _sha256_bytes(_canonical_bytes(session_core)) != session["manifest_sha256"]:
        raise OperatorError("agent session was modified")
    protected = {
        "GOAL.md": "goal_sha256",
        "AGENT_CONTRACT.md": "contract_sha256",
        "client.py": "client_sha256",
        "observation.json": "observation_file_sha256",
        "AGENTS.md": "agents_sha256",
    }
    for filename, field in protected.items():
        if _sha256_file(
            bundle / filename, max_bytes=MAX_PROTECTED_FILE_BYTES
        ) != manifest[field]:
            raise OperatorError(f"agent bundle protected file was modified: {filename}")


def _public_error(exc: BaseException, manifest: Mapping[str, Any]) -> dict[str, str]:
    message = str(exc)
    sensitive = [
        str(manifest["problem_path"]),
        str(manifest["evaluator_run"]),
        str(manifest["agent_bundle"]),
        os.environ.get(EVALUATOR_ANCHOR_DIR_ENV, ""),
        os.environ.get(EVALUATOR_KEY_ENV, ""),
    ]
    for value in sensitive:
        if value:
            message = message.replace(value, "<trusted>")
    return {"type": type(exc).__name__, "message": message}


def _trusted_receipt_path(manifest: Mapping[str, Any], request_id: str) -> Path:
    return Path(manifest["evaluator_run"]) / "bridge_receipts" / f"{request_id}.json"


def _published_receipt_path(manifest: Mapping[str, Any], request_id: str) -> Path:
    return Path(manifest["agent_bundle"]) / "inbox" / f"{request_id}.receipt.json"


def _publish_receipt(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _verify_authenticated_publication(manifest, receipt, message_type="receipt")
    _atomic_write_json(
        _published_receipt_path(manifest, str(receipt["request_id"])), receipt
    )


def _recover_pending(
    manifest: Mapping[str, Any],
    *,
    key: bytes | None,
    anchor_dir: Path | None,
) -> None:
    pending_path = Path(manifest["evaluator_run"]) / BRIDGE_STATE
    if not pending_path.exists():
        return
    pending = _read_json_object(pending_path, max_bytes=100_000)
    required = {
        "schema_version",
        "request_id",
        "action_sha256",
        "before_ledger_events",
        "before_ledger_tip",
    }
    if set(pending) != required or pending["schema_version"] != 1:
        raise OperatorError("bridge pending state is invalid")
    request_id = str(pending["request_id"])
    trusted_receipt = _trusted_receipt_path(manifest, request_id)
    if trusted_receipt.exists():
        receipt = _read_json_object(trusted_receipt, max_bytes=10_000_000)
        _publish_receipt(manifest, receipt)
        pending_path.unlink()
        return
    status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    checkpoint = status["checkpoint"]
    unchanged = (
        checkpoint["ledger_events"] == pending["before_ledger_events"]
        and checkpoint["ledger_tip"] == pending["before_ledger_tip"]
    )
    if unchanged:
        pending_path.unlink()
        return
    receipt = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "receipt",
        "session_id": manifest["session_id"],
        "campaign_id": manifest["campaign_id"],
        "session_manifest_sha256": manifest["session_manifest_sha256"],
        "request_id": request_id,
        "action_sha256": pending["action_sha256"],
        "ok": False,
        "processed_at_utc": _utc_now(),
        "result": None,
        "error": {
            "type": "IndeterminatePriorSubmission",
            "message": (
                "the authoritative ledger advanced before a durable bridge receipt; "
                "the action was not retried—use the attached status"
            ),
        },
        "status": _public_status(status),
    }
    receipt = _authenticate_publication(
        manifest, receipt, message_type="receipt"
    )
    _atomic_write_json(trusted_receipt, receipt)
    _publish_receipt(manifest, receipt)
    _publish_status(manifest, status)
    pending_path.unlink()


def _process_ready(
    ready_path: Path,
    manifest: Mapping[str, Any],
    *,
    key: bytes | None,
    anchor_dir: Path | None,
) -> dict[str, Any]:
    match = _REQUEST_RE.fullmatch(ready_path.name)
    if match is None:
        raise OperatorError(f"invalid ready request filename: {ready_path.name}")
    request_id = match.group(1)
    envelope = _read_json_object(ready_path, max_bytes=MAX_BRIDGE_BYTES)
    required = {
        "protocol_version",
        "session_id",
        "campaign_id",
        "request_id",
        "manifest_sha256",
        "expected_cursor",
        "action_sha256",
        "action",
    }
    if set(envelope) != required:
        raise OperatorError("request envelope fields are invalid")
    if envelope["protocol_version"] != PROTOCOL_VERSION:
        raise OperatorError("request protocol version is invalid")
    if envelope["session_id"] != manifest["session_id"]:
        raise OperatorError("request session_id mismatch")
    if envelope["campaign_id"] != manifest["campaign_id"]:
        raise OperatorError("request campaign_id mismatch")
    if envelope["manifest_sha256"] != manifest["session_manifest_sha256"]:
        raise OperatorError("request session manifest hash mismatch")
    if envelope["request_id"] != request_id:
        raise OperatorError("request_id does not match ready filename")
    action = envelope["action"]
    if not isinstance(action, dict):
        raise OperatorError("request action must be an object")
    action_bytes = _canonical_bytes(action)
    if len(action_bytes) > MAX_EXTERNAL_ACTION_BYTES:
        raise OperatorError("canonical action exceeds the evaluator limit")
    action_sha256 = _sha256_bytes(action_bytes)
    if envelope["action_sha256"] != action_sha256:
        raise OperatorError("request action hash mismatch")
    envelope_bytes = _canonical_bytes(envelope)
    if len(envelope_bytes) > MAX_BRIDGE_BYTES:
        raise OperatorError("canonical request envelope exceeds the bridge limit")

    evaluator_run = Path(manifest["evaluator_run"])
    archive = evaluator_run / "bridge_requests" / f"{request_id}.json"
    trusted_receipt = _trusted_receipt_path(manifest, request_id)
    if archive.exists():
        prior_bytes = _read_regular_bytes(archive, max_bytes=MAX_BRIDGE_BYTES)
        if not hmac.compare_digest(prior_bytes, envelope_bytes):
            raise OperatorError("request_id was reused with different content")
    else:
        _atomic_write_bytes(archive, envelope_bytes)
    if trusted_receipt.exists():
        receipt = _read_json_object(trusted_receipt, max_bytes=10_000_000)
        _publish_receipt(manifest, receipt)
        _discard_untrusted_path(ready_path)
        return receipt

    before = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    expected_cursor = envelope["expected_cursor"]
    if not isinstance(expected_cursor, dict) or set(expected_cursor) != {
        "ledger_events",
        "ledger_tip",
    }:
        raise OperatorError("request expected_cursor is invalid")
    actual_cursor = {
        "ledger_events": before["checkpoint"]["ledger_events"],
        "ledger_tip": before["checkpoint"]["ledger_tip"],
    }
    if expected_cursor != actual_cursor:
        receipt = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "receipt",
            "session_id": manifest["session_id"],
            "campaign_id": manifest["campaign_id"],
            "session_manifest_sha256": manifest["session_manifest_sha256"],
            "request_id": request_id,
            "action_sha256": action_sha256,
            "ok": False,
            "processed_at_utc": _utc_now(),
            "result": None,
            "error": {
                "type": "CursorConflict",
                "message": "the request was based on stale status and was not executed",
            },
            "status": _public_status(before),
        }
        receipt = _authenticate_publication(
            manifest, receipt, message_type="receipt"
        )
        _atomic_write_json(trusted_receipt, receipt)
        _publish_receipt(manifest, receipt)
        _publish_status(manifest, before)
        _discard_untrusted_path(ready_path)
        return receipt
    pending_path = evaluator_run / BRIDGE_STATE
    if pending_path.exists():
        raise OperatorError("another bridge action is pending operator recovery")
    pending = {
        "schema_version": 1,
        "request_id": request_id,
        "action_sha256": action_sha256,
        "before_ledger_events": before["checkpoint"]["ledger_events"],
        "before_ledger_tip": before["checkpoint"]["ledger_tip"],
    }
    _atomic_write_json(pending_path, pending)
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    try:
        result = execute_action(
            manifest["problem_path"],
            manifest["evaluator_run"],
            action,
            evaluator_key=key,
            anchor_dir=anchor_dir,
            require_sealed=manifest["security_mode"] == SEALED,
            expected_cursor=expected_cursor,
        )
    except Exception as exc:  # Controller errors are a normal negative receipt.
        error = _public_error(exc, manifest)
    status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    receipt = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "receipt",
        "session_id": manifest["session_id"],
        "campaign_id": manifest["campaign_id"],
        "session_manifest_sha256": manifest["session_manifest_sha256"],
        "request_id": request_id,
        "action_sha256": action_sha256,
        "ok": error is None,
        "processed_at_utc": _utc_now(),
        "result": _public_action_result(result),
        "error": error,
        "status": _public_status(status),
    }
    receipt = _authenticate_publication(
        manifest, receipt, message_type="receipt"
    )
    _atomic_write_json(trusted_receipt, receipt)
    _publish_receipt(manifest, receipt)
    _publish_status(manifest, status)
    pending_path.unlink()
    _discard_untrusted_path(ready_path)
    return receipt


def _discard_untrusted_path(path: Path) -> None:
    """Remove a file/link or atomically quarantine an untrusted non-file node."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        path.unlink(missing_ok=True)
        return
    quarantine = path.with_name(
        f".rejected-{os.urandom(12).hex()}-{path.name}"
    )
    os.replace(path, quarantine)


def _reject_ready(
    ready_path: Path,
    manifest: Mapping[str, Any],
    exc: BaseException,
    *,
    key: bytes | None,
    anchor_dir: Path | None,
) -> dict[str, Any]:
    """Quarantine one malformed bridge request without terminating the server."""

    match = _REQUEST_RE.fullmatch(ready_path.name)
    if match is None:
        _discard_untrusted_path(ready_path)
        raise OperatorError(
            f"invalid request filename quarantined: {ready_path.name}"
        ) from exc
    request_id = match.group(1)
    action_sha256 = "0" * 64
    try:
        envelope = _read_json_object(ready_path, max_bytes=MAX_BRIDGE_BYTES)
        supplied = envelope.get("action_sha256")
        if isinstance(supplied, str) and re.fullmatch(r"[0-9a-f]{64}", supplied):
            action_sha256 = supplied
    except Exception:
        pass
    status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    receipt = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "receipt",
        "session_id": manifest["session_id"],
        "campaign_id": manifest["campaign_id"],
        "session_manifest_sha256": manifest["session_manifest_sha256"],
        "request_id": request_id,
        "action_sha256": action_sha256,
        "ok": False,
        "processed_at_utc": _utc_now(),
        "result": None,
        "error": {
            "type": "RequestRejected",
            "message": _public_error(exc, manifest)["message"],
        },
        "status": _public_status(status),
    }
    trusted_receipt = _trusted_receipt_path(manifest, request_id)
    if not trusted_receipt.exists():
        receipt = _authenticate_publication(
            manifest, receipt, message_type="receipt"
        )
        _atomic_write_json(trusted_receipt, receipt)
    else:
        receipt = _read_json_object(trusted_receipt, max_bytes=10_000_000)
    _publish_receipt(manifest, receipt)
    _publish_status(manifest, status)
    _discard_untrusted_path(ready_path)
    return receipt


def serve(args: argparse.Namespace) -> int:
    evaluator_run = _resolve(args.evaluator_run)
    manifest, key, anchor_dir = _load_manifest(
        evaluator_run,
        require_sealed=not bool(getattr(args, "allow_unsealed", False)),
    )
    lock_path = evaluator_run / "operator.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise OperatorError(
            "trusted operator is already running or needs stale-lock recovery"
        ) from exc
    try:
        os.write(lock_descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode("ascii"))
        os.close(lock_descriptor)
        _verify_bundle(manifest)
        _recover_pending(manifest, key=key, anchor_dir=anchor_dir)
        status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
        _publish_status(manifest, status)
        outbox = Path(manifest["agent_bundle"]) / "outbox"
        started = time.monotonic()
        while True:
            _verify_trusted_inputs(manifest)
            _verify_bundle(manifest)
            discovered = sorted(outbox.glob("*.ready.json"), key=lambda path: path.name)
            for invalid in tuple(
                path for path in discovered if _REQUEST_RE.fullmatch(path.name) is None
            ):
                _discard_untrusted_path(invalid)
                print(
                    f"quarantined invalid request filename: {invalid.name}",
                    file=sys.stderr,
                )
            ready_files = [
                path for path in discovered if _REQUEST_RE.fullmatch(path.name) is not None
            ]
            if ready_files:
                try:
                    receipt = _process_ready(
                        ready_files[0], manifest, key=key, anchor_dir=anchor_dir
                    )
                except (OperatorError, OSError, ValueError, RecursionError) as exc:
                    if (evaluator_run / BRIDGE_STATE).exists():
                        raise
                    receipt = _reject_ready(
                        ready_files[0],
                        manifest,
                        exc,
                        key=key,
                        anchor_dir=anchor_dir,
                    )
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in receipt.items()
                            if key != "authentication"
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.once:
                    return 0
                exit_on_terminal = bool(
                    getattr(args, "exit_on_terminal", False)
                    or getattr(args, "exit_on_commit", False)
                )
                if exit_on_terminal and receipt["status"]["state"].get(
                    "terminal_decision"
                ) in {"positive_commit", "negative_close"}:
                    return 0
                started = time.monotonic()
                continue
            if args.idle_timeout is not None and time.monotonic() - started >= args.idle_timeout:
                return 0
            time.sleep(args.poll_interval)
    finally:
        try:
            os.close(lock_descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def operator_status(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_run = _resolve(args.evaluator_run)
    manifest, key, anchor_dir = _load_manifest(
        evaluator_run,
        require_sealed=not bool(getattr(args, "allow_unsealed", False)),
    )
    _verify_trusted_inputs(manifest)
    _verify_bundle(manifest)
    status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    _publish_status(manifest, status)
    return status


def _result_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OperatorError(f"terminal result {field} must be an object")
    return dict(value)


def _result_id_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise OperatorError(f"terminal result {field} must be a non-empty ID list")
    result = tuple(value)
    if (
        not all(isinstance(item, str) and _ID_RE.fullmatch(item) for item in result)
        or len(set(result)) != len(result)
    ):
        raise OperatorError(f"terminal result {field} contains invalid or duplicate IDs")
    return result


def _result_evidence(
    state: Mapping[str, Any], evidence_ids: tuple[str, ...]
) -> dict[str, Any]:
    probes = _result_mapping(state.get("probes"), "state.probes")
    evaluations = _result_mapping(state.get("evaluations"), "state.evaluations")
    exported: dict[str, Any] = {}
    for evidence_id in evidence_ids:
        probe = probes.get(evidence_id)
        evaluation = evaluations.get(evidence_id)
        if probe is not None and evaluation is not None:
            raise OperatorError(f"terminal result has ambiguous evidence ID: {evidence_id}")
        if probe is not None:
            exported[evidence_id] = {
                "kind": "probe",
                "record": _result_mapping(probe, f"probe {evidence_id}"),
            }
        elif evaluation is not None:
            exported[evidence_id] = {
                "kind": "evaluation",
                "record": _result_mapping(evaluation, f"evaluation {evidence_id}"),
            }
        else:
            raise OperatorError(f"terminal result cites unknown evidence ID: {evidence_id}")
    return exported


def _positive_result(state: Mapping[str, Any]) -> dict[str, Any]:
    candidate_id = state.get("committed_candidate_id")
    if not isinstance(candidate_id, str) or not _ID_RE.fullmatch(candidate_id):
        raise OperatorError("positive terminal result has no valid committed candidate")
    candidates = _result_mapping(state.get("candidates"), "state.candidates")
    candidate = _result_mapping(
        candidates.get(candidate_id), f"committed candidate {candidate_id}"
    )
    spec = _result_mapping(candidate.get("spec"), "committed candidate spec")
    try:
        parsed = AnsatzSpec.from_dict(spec)
        semantic_hash = candidate_hash(parsed)
    except (TypeError, ValueError, KeyError) as exc:
        raise OperatorError(f"committed AnsatzSpec is invalid: {exc}") from exc

    commit_metadata = _result_mapping(
        state.get("commit_metadata"), "state.commit_metadata"
    )
    evidence_ids = _result_id_list(
        commit_metadata.get("evidence_ids"), "commit evidence_ids"
    )
    comparison = _result_mapping(
        commit_metadata.get("comparison"), "commit comparison"
    )
    trusted_evidence = _result_evidence(state, evidence_ids)
    promotions: list[tuple[str, dict[str, Any]]] = []
    for evidence_id in evidence_ids:
        item = trusted_evidence[evidence_id]
        if item["kind"] != "evaluation":
            continue
        record = _result_mapping(item["record"], f"evaluation {evidence_id}")
        if (
            record.get("candidate_id") == candidate_id
            and record.get("stage") == "promotion"
            and record.get("passed") is True
        ):
            promotions.append((evidence_id, record))
    if len(promotions) != 1:
        raise OperatorError(
            "positive terminal result must cite exactly one passed promotion for "
            "the committed candidate"
        )
    promotion_id, promotion_record = promotions[0]
    metrics = _result_mapping(
        promotion_record.get("metrics"), f"promotion {promotion_id}.metrics"
    )
    if metrics.get("valid") is not True or metrics.get("candidate_hash") != semantic_hash:
        raise OperatorError(
            "cited promotion is not a valid evaluator record for the committed semantic candidate"
        )
    binding = _result_mapping(
        metrics.get("optimized_parameter_binding"),
        f"promotion {promotion_id}.optimized_parameter_binding",
    )
    declared_names = {parameter.name for parameter in parsed.parameters}
    if set(binding) != declared_names:
        raise OperatorError(
            "promotion optimized binding does not match the committed AnsatzSpec parameters"
        )
    checked_binding: dict[str, float] = {}
    for name, value in binding.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OperatorError("promotion optimized binding values must be finite numbers")
        checked = float(value)
        if not math.isfinite(checked):
            raise OperatorError("promotion optimized binding values must be finite numbers")
        checked_binding[name] = checked
    best_energy = metrics.get("best_energy")
    if (
        isinstance(best_energy, bool)
        or not isinstance(best_energy, (int, float))
        or not math.isfinite(float(best_energy))
    ):
        raise OperatorError("passed promotion is missing a finite evaluator best energy")

    promotion = {
        "evaluation_id": promotion_id,
        "evaluation_record_sha256": _sha256_bytes(
            b"autovqe-evaluation-record-v1\0" + _canonical_bytes(promotion_record)
        ),
        "candidate_semantic_sha256": semantic_hash,
        "optimized_parameter_binding": checked_binding,
        "energy": {
            key: metrics[key]
            for key in (
                "best_energy",
                "baseline_energy",
                "energy_improvement",
                "required_energy_improvement",
                "energy_trace",
                "best_energy_trace",
            )
            if key in metrics
        },
        "optimization": {
            key: metrics[key]
            for key in ("objective_calls", "optimizer", "seed")
            if key in metrics
        },
        "resources": {
            "metrics": _result_mapping(metrics.get("metrics"), "promotion resources"),
            "eligibility": _result_mapping(
                metrics.get("resource_policy"), "promotion resource policy"
            ),
        },
        "compiler_audit": _result_mapping(metrics.get("audit"), "promotion audit"),
        "violations": list(metrics.get("violations", [])),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_semantic_sha256": semantic_hash,
        "ansatz_spec": spec,
        "promotion": promotion,
        "evidence_ids": list(evidence_ids),
        "comparison": comparison,
        "evidence": _redact_private_evaluator_values(trusted_evidence),
    }


def _negative_result(state: Mapping[str, Any]) -> dict[str, Any]:
    reason = state.get("negative_close_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise OperatorError("negative terminal result has no close reason")
    evidence_ids = _result_id_list(
        state.get("negative_close_evidence_ids"), "negative-close evidence_ids"
    )
    return {
        "reason": reason,
        "evidence_ids": list(evidence_ids),
        "evidence": _redact_private_evaluator_values(
            _result_evidence(state, evidence_ids)
        ),
    }


def _terminal_result_artifact(
    manifest: Mapping[str, Any], status: Mapping[str, Any]
) -> dict[str, Any]:
    state = _result_mapping(status.get("state"), "state")
    decision = state.get("terminal_decision")
    if decision not in {"positive_commit", "negative_close"}:
        raise OperatorError(
            "terminal result export requires a controller-accepted positive commit "
            "or negative close"
        )
    checkpoint = _result_mapping(status.get("checkpoint"), "checkpoint")
    security = _result_mapping(status.get("security"), "security")
    sealed = manifest["security_mode"] == SEALED
    dirty = bool(manifest["git_dirty"])
    if not sealed:
        trust_classification = "UNTRUSTED_LOCAL_INTEGRATION"
    else:
        trust_classification = "SEALED_PROTOCOL_VERIFIED"

    limitations = [
        "The result applies only to the recorded Hamiltonian, AnsatzSpec, and fixed evaluator protocol.",
        "A passed promotion is not a proof of a global optimum, ground-state accuracy, or generalization to other Hamiltonians.",
        "Resource counts depend on the recorded canonical and declared-backend compilation policies.",
        "documented_non_dominance is a controller-grounded rationale, not an independently computed Pareto frontier.",
        "Agent-authored energy, parameter, optimizer, and resource claims are excluded from this artifact.",
        "This artifact alone does not verify OS isolation, agent identity, network policy, model configuration, or an external holdout score and is never benchmark-grade.",
    ]
    if not sealed:
        limitations.append(
            "This local_unsealed integration artifact is untrusted and must not be reported as a scored or tamper-resistant result."
        )
    if dirty:
        limitations.append(
            "The evaluator checkout was dirty at preparation time; the source-tree hash binds the run, but the artifact is not benchmark-grade."
        )

    core = {
        "schema_version": 1,
        "artifact_type": "autovqe_terminal_result",
        "decision": decision,
        "trust": {
            "classification": trust_classification,
            "security_mode": manifest["security_mode"],
            "tamper_evident": bool(security.get("tamper_evident", False)),
            "rollback_protected": bool(security.get("rollback_protected", False)),
            # Protocol verification cannot attest the host ACL, Codex identity,
            # network policy, model configuration, or an external holdout score.
            "benchmark_grade": False,
            "benchmark_prerequisites": {
                "protocol_integrity_verified": bool(
                    sealed
                    and security.get("tamper_evident") is True
                    and security.get("rollback_protected") is True
                ),
                "source_clean": not dirty,
                "source_commit_recorded": bool(manifest.get("git_commit")),
                "os_isolation_verified": False,
                "agent_identity_verified": False,
                "network_policy_verified": False,
                "model_configuration_verified": False,
                "external_holdout_score_present": False,
            },
            "warning": security.get("warning"),
        },
        "provenance": {
            "campaign": {
                "campaign_id": manifest["campaign_id"],
                "research_mode": manifest["research_mode"],
                "model_label": manifest["model_label"],
            },
            "problem": {
                "problem_id": manifest["problem_id"],
                "raw_input_sha256": manifest["problem_file_sha256"],
            },
            "observation": {
                "content_hash": manifest["observation_hash"],
                "file_sha256": manifest["observation_file_sha256"],
            },
            "run": {
                "run_id": manifest["run_id"],
                "session_id": manifest["session_id"],
                "session_manifest_sha256": manifest["session_manifest_sha256"],
                "protocol_version": manifest["protocol_version"],
                "ledger_events": checkpoint["ledger_events"],
                "ledger_tip": checkpoint["ledger_tip"],
                "terminal_event_seq": state["last_seq"],
                "terminal_event_hash": state["last_hash"],
            },
            "source": {
                "evaluator_source_sha256": manifest["evaluator_source_sha256"],
                "git_commit": manifest["git_commit"],
                "git_dirty": dirty,
            },
        },
        "budget": {
            "total": state["total_budget"],
            "spent": state["spent_budget"],
            "remaining": state["remaining_budget"],
        },
        "result": (
            _positive_result(state)
            if decision == "positive_commit"
            else _negative_result(state)
        ),
        "limitations": limitations,
    }
    return {
        **core,
        "artifact_sha256": _sha256_bytes(
            _RESULT_ARTIFACT_DOMAIN + _canonical_bytes(core)
        ),
    }


def _lexical_absolute(value: str | Path, *, base: Path = _REPO_ROOT) -> Path:
    """Return an absolute normalized path without resolving filesystem links."""

    path = Path(os.path.expanduser(os.fspath(value)))
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(os.fspath(path)))


def _lexically_within(path: Path, root: Path) -> bool:
    try:
        normalized_path = os.path.normcase(os.path.abspath(os.fspath(path)))
        normalized_root = os.path.normcase(os.path.abspath(os.fspath(root)))
        return os.path.commonpath((normalized_path, normalized_root)) == normalized_root
    except ValueError:
        return False


def _lexically_related(left: Path, right: Path) -> bool:
    return _lexically_within(left, right) or _lexically_within(right, left)


def _link_or_reparse_metadata(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _filesystem_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
    )


def _inspect_export_parent_chain(
    parent: Path,
) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    """Validate and fingerprint every existing component of an output parent."""

    if not parent.is_absolute() or not parent.anchor:
        raise OperatorError("terminal result output parent must be absolute")
    parts = parent.parts
    current = Path(parts[0])
    chain: list[tuple[Path, tuple[int, int, int]]] = []
    for index in range(len(parts)):
        if index:
            current = current / parts[index]
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise OperatorError(
                "terminal result output parent must already exist: " + str(current)
            ) from exc
        except OSError as exc:
            raise OperatorError(
                f"cannot inspect terminal result output parent {current}: {exc}"
            ) from exc
        if _link_or_reparse_metadata(metadata):
            raise OperatorError(
                "terminal result output parent must not contain a symbolic link "
                f"or reparse point: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise OperatorError(
                f"terminal result output parent component is not a directory: {current}"
            )
        chain.append((current, _filesystem_identity(metadata)))
    return tuple(chain)


def _recheck_export_parent_chain(
    chain: tuple[tuple[Path, tuple[int, int, int]], ...]
) -> None:
    for path, expected_identity in chain:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise OperatorError(
                f"terminal result output parent changed during export: {path}"
            ) from exc
        if (
            _link_or_reparse_metadata(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
            or _filesystem_identity(metadata) != expected_identity
        ):
            raise OperatorError(
                f"terminal result output parent changed during export: {path}"
            )


def _existing_export_target(path: Path) -> os.stat_result | None:
    if not os.path.lexists(os.fspath(path)):
        return None
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise OperatorError(f"cannot inspect terminal result output {path}: {exc}") from exc
    if _link_or_reparse_metadata(metadata):
        raise OperatorError(
            "terminal result output must not be a symbolic link or reparse point"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise OperatorError("terminal result output must be a regular file")
    return metadata


def _validated_result_output(
    requested: str | Path | None,
    *,
    evaluator_run: Path,
    agent_bundle: Path,
) -> tuple[Path, tuple[tuple[Path, tuple[int, int, int]], ...]]:
    """Validate a lexical result path inside the evaluator-owned run tree."""

    trusted_root = _lexical_absolute(evaluator_run)
    output = (
        trusted_root / "final_result.json"
        if requested is None
        else _lexical_absolute(requested)
    )
    lexical_agent = _lexical_absolute(agent_bundle)
    if _lexically_related(output, lexical_agent):
        raise OperatorError(
            "terminal result output must be lexically disjoint from the agent bundle"
        )
    if output == trusted_root or not _lexically_within(output, trusted_root):
        raise OperatorError(
            "terminal result output must stay inside the evaluator-owned run directory"
        )

    chain = _inspect_export_parent_chain(output.parent)
    target_metadata = _existing_export_target(output)
    try:
        resolved_root = trusted_root.resolve(strict=True)
        resolved_parent = output.parent.resolve(strict=True)
        resolved_output = (
            output.resolve(strict=True)
            if target_metadata is not None
            else resolved_parent / output.name
        )
        resolved_agent = agent_bundle.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OperatorError(f"cannot resolve terminal result output boundary: {exc}") from exc
    if resolved_output == resolved_root or not _lexically_within(
        resolved_output, resolved_root
    ):
        raise OperatorError(
            "terminal result output resolves outside the evaluator-owned run directory"
        )
    if _lexically_related(resolved_output, resolved_agent):
        raise OperatorError(
            "terminal result output resolves into the agent-writable bundle"
        )
    _recheck_export_parent_chain(chain)
    return output, chain


def _secure_write_terminal_result(
    path: Path,
    payload: bytes,
    *,
    parent_chain: tuple[tuple[Path, tuple[int, int, int]], ...],
) -> None:
    """Create once without following the target and preserve idempotent replay."""

    before = _existing_export_target(path)
    if before is not None:
        expected_identity = _filesystem_identity(before)
        existing = _read_regular_bytes(path, max_bytes=MAX_RESULT_ARTIFACT_BYTES)
        _recheck_export_parent_chain(parent_chain)
        after = _existing_export_target(path)
        if after is None or _filesystem_identity(after) != expected_identity:
            raise OperatorError("terminal result output changed while being verified")
        if existing != payload:
            raise OperatorError(
                "terminal result output already exists with different content; "
                "choose a fresh path"
            )
        return

    _recheck_export_parent_chain(parent_chain)
    parent_descriptor: int | None = None
    target_descriptor: int | None = None
    opened_identity: tuple[int, int, int] | None = None
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        if os.name != "nt":
            parent_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            parent_descriptor = os.open(path.parent, parent_flags)
            opened_parent = os.fstat(parent_descriptor)
            if _filesystem_identity(opened_parent) != parent_chain[-1][1]:
                raise OperatorError(
                    "terminal result output parent changed before target creation"
                )
            target_descriptor = os.open(
                path.name,
                flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        else:
            # Python has no Windows dir_fd/openat. The destination is restricted
            # to the evaluator-owned tree, and the complete reparse-free parent
            # identity chain is checked immediately before and after CREATE_NEW.
            target_descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(target_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OperatorError("terminal result output is not a regular file")
        opened_identity = _filesystem_identity(opened)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(target_descriptor, remaining)
            if written <= 0:
                raise OSError("short write while exporting terminal result")
            remaining = remaining[written:]
        os.fsync(target_descriptor)
        if parent_descriptor is not None:
            os.fsync(parent_descriptor)
    except FileExistsError as exc:
        raise OperatorError(
            "terminal result output appeared during exclusive creation; retry export"
        ) from exc
    finally:
        if target_descriptor is not None:
            os.close(target_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)

    _recheck_export_parent_chain(parent_chain)
    after = _existing_export_target(path)
    if after is None or _filesystem_identity(after) != opened_identity:
        raise OperatorError("terminal result output changed during creation")
    published = _read_regular_bytes(path, max_bytes=MAX_RESULT_ARTIFACT_BYTES)
    _recheck_export_parent_chain(parent_chain)
    final = _existing_export_target(path)
    if (
        final is None
        or _filesystem_identity(final) != opened_identity
        or published != payload
    ):
        raise OperatorError("terminal result output failed post-write verification")


def export_result(args: argparse.Namespace) -> dict[str, Any]:
    evaluator_run = _resolve(args.evaluator_run)
    manifest, key, anchor_dir = _load_manifest(
        evaluator_run,
        require_sealed=not bool(getattr(args, "allow_unsealed", False)),
    )
    # Keep this explicit at the export boundary even though manifest loading
    # also validates these inputs. Provenance is meaningful only if the exact
    # evaluator source and raw problem are still the preparation-time bytes.
    _verify_trusted_inputs(manifest)
    _verify_bundle(manifest)
    status = _verified_status(manifest, key=key, anchor_dir=anchor_dir)
    artifact = _terminal_result_artifact(manifest, status)
    output, parent_chain = _validated_result_output(
        getattr(args, "output", None),
        evaluator_run=evaluator_run,
        agent_bundle=Path(manifest["agent_bundle"]),
    )
    rendered = _render_json_bytes(artifact)
    if len(rendered) > MAX_RESULT_ARTIFACT_BYTES:
        raise OperatorError("terminal result exceeds the artifact size limit")
    _secure_write_terminal_result(
        output,
        rendered,
        parent_chain=parent_chain,
    )
    return {
        "artifact": str(output),
        "artifact_sha256": artifact["artifact_sha256"],
        "decision": artifact["decision"],
        "trust_classification": artifact["trust"]["classification"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trusted AutoVQE meta-agent operator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="initialize a run and agent bundle")
    prepare_parser.add_argument("--campaign", required=True, type=Path)
    prepare_parser.add_argument(
        "--security", choices=(LOCAL_UNSEALED, SEALED), default=SEALED
    )
    prepare_parser.add_argument("--agent-bundle", type=Path)
    prepare_parser.add_argument("--evaluator-run", type=Path)
    prepare_parser.add_argument(
        "--model-label",
        help="exact Codex model/config label recorded in the campaign provenance",
    )
    prepare_parser.add_argument("--allow-dirty-evaluator", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="process agent outbox actions")
    serve_parser.add_argument("--evaluator-run", required=True, type=Path)
    serve_parser.add_argument("--poll-interval", type=float, default=0.2)
    serve_parser.add_argument("--idle-timeout", type=float)
    serve_parser.add_argument("--once", action="store_true")
    serve_parser.add_argument(
        "--exit-on-terminal",
        action="store_true",
        help="stop after an accepted positive commit or negative close",
    )
    serve_parser.add_argument(
        "--exit-on-commit",
        dest="exit_on_terminal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--allow-unsealed",
        action="store_true",
        help="explicitly permit a local_unsealed integration campaign",
    )

    status_parser = subparsers.add_parser("status", help="verify and publish evaluator status")
    status_parser.add_argument("--evaluator-run", required=True, type=Path)
    status_parser.add_argument(
        "--allow-unsealed",
        action="store_true",
        help="explicitly permit a local_unsealed integration campaign",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="export a canonical evaluator-derived terminal result",
    )
    export_parser.add_argument("--evaluator-run", required=True, type=Path)
    export_parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output JSON path inside the evaluator run "
            "(default: <evaluator-run>/final_result.json)"
        ),
    )
    export_parser.add_argument(
        "--allow-unsealed",
        action="store_true",
        help=(
            "explicitly export a local_unsealed integration result; the artifact "
            "is labeled untrusted"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_campaign(args)
        elif args.command == "serve":
            if args.poll_interval <= 0:
                raise OperatorError("poll interval must be positive")
            if args.idle_timeout is not None and args.idle_timeout <= 0:
                raise OperatorError("idle timeout must be positive")
            return serve(args)
        elif args.command == "export":
            result = export_result(args)
        else:
            result = operator_status(args)
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (OperatorError, OSError, ValueError) as exc:
        print(f"operator error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("operator stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
