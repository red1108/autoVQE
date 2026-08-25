"""Standalone file-bridge client copied into an isolated Codex bundle.

This module deliberately imports only the Python standard library. The trusted
operator validates every envelope again; client-side validation is usability,
not a trust boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
MAX_ACTION_BYTES = 1_000_000
PUBLICATION_AUTH_SCHEME = "merkle_lamport_sha256_v1"
PUBLICATION_AUTH_SLOTS = 512
_PUBLICATION_MESSAGE_DOMAIN = b"autovqe-publication-message-v1\0"
_PUBLICATION_LEAF_DOMAIN = b"autovqe-publication-lamport-leaf-v1\0"
_PUBLICATION_NODE_DOMAIN = b"autovqe-publication-merkle-node-v1\0"
_REQUEST_ID_RE = re.compile(r"^req_[0-9a-f]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ClientError(RuntimeError):
    """Raised for an invalid bundle, action, or bridge response."""


def _reject_constant(value: str) -> None:
    raise ClientError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClientError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded regular file without following a swapped leaf link."""

    try:
        before_open = path.lstat()
    except FileNotFoundError as exc:
        raise ClientError(f"missing protected file: {path}") from exc
    if stat.S_ISLNK(before_open.st_mode):
        raise ClientError(f"symbolic links are not accepted: {path}")
    if not stat.S_ISREG(before_open.st_mode):
        raise ClientError(f"protected input must be a regular file: {path}")
    if before_open.st_size > max_bytes:
        raise ClientError(
            f"protected input exceeds {max_bytes} raw bytes: {before_open.st_size}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClientError(f"cannot securely open protected input {path}: {exc}") from exc
    try:
        after_open = os.fstat(descriptor)
        if not stat.S_ISREG(after_open.st_mode):
            raise ClientError(f"protected input must be a regular file: {path}")
        if (before_open.st_dev, before_open.st_ino) != (
            after_open.st_dev,
            after_open.st_ino,
        ):
            raise ClientError(f"protected input changed while being opened: {path}")
        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = os.read(descriptor, min(65536, max_bytes + 1 - received))
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if received > max_bytes:
                raise ClientError(f"protected input exceeds {max_bytes} raw bytes")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json_object(path: Path, *, max_bytes: int = MAX_ACTION_BYTES) -> dict[str, Any]:
    raw = _read_regular_bytes(path, max_bytes=max_bytes)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ClientError(f"invalid UTF-8 JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClientError(f"{path} must contain exactly one JSON object")
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
        raise ClientError(f"value is not canonical JSON: {exc}") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent


def _load_session(root: Path) -> dict[str, Any]:
    session = _read_json_object(root / "session.json", max_bytes=100_000)
    required = {
        "schema_version",
        "protocol_version",
        "session_id",
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
        "evaluator_source_sha256",
        "observation_file_sha256",
        "publication_auth_scheme",
        "publication_auth_root",
        "publication_auth_slots",
        "manifest_sha256",
    }
    if set(session) != required:
        raise ClientError("session.json has unexpected fields")
    if session["schema_version"] != 1 or session["protocol_version"] != PROTOCOL_VERSION:
        raise ClientError("unsupported agent bundle protocol")
    supplied_hash = session["manifest_sha256"]
    if not isinstance(supplied_hash, str):
        raise ClientError("session manifest hash is invalid")
    core = {key: value for key, value in session.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_bytes(core)).hexdigest() != supplied_hash:
        raise ClientError("session.json integrity check failed")
    client_bytes = _read_regular_bytes(root / "client.py", max_bytes=MAX_ACTION_BYTES)
    if not isinstance(session["client_sha256"], str) or not hmac.compare_digest(
        hashlib.sha256(client_bytes).hexdigest(), session["client_sha256"]
    ):
        raise ClientError("bundle client integrity check failed")
    if (
        session["publication_auth_scheme"] != PUBLICATION_AUTH_SCHEME
        or session["publication_auth_slots"] != PUBLICATION_AUTH_SLOTS
        or not isinstance(session["publication_auth_root"], str)
        or not _SHA256_RE.fullmatch(session["publication_auth_root"])
    ):
        raise ClientError("session publication authentication key is invalid")
    return session


def _verify_publication(
    publication: dict[str, Any], session: dict[str, Any], *, message_type: str
) -> dict[str, Any]:
    common = {
        "schema_version",
        "protocol_version",
        "message_type",
        "session_id",
        "campaign_id",
        "session_manifest_sha256",
        "authentication",
    }
    type_fields = (
        {"published_at_utc", "publication_capacity", "status"}
        if message_type == "status"
        else {
            "request_id",
            "action_sha256",
            "ok",
            "processed_at_utc",
            "result",
            "error",
            "status",
        }
    )
    if set(publication) != common | type_fields:
        raise ClientError(f"authenticated {message_type} fields are invalid")
    if publication["schema_version"] != 1 or publication["protocol_version"] != PROTOCOL_VERSION:
        raise ClientError(f"authenticated {message_type} protocol is invalid")
    if publication["message_type"] != message_type:
        raise ClientError("publication message type mismatch")
    if publication["session_id"] != session["session_id"]:
        raise ClientError("publication belongs to a different session")
    if publication["campaign_id"] != session["campaign_id"]:
        raise ClientError("publication belongs to a different campaign")
    if publication["session_manifest_sha256"] != session["manifest_sha256"]:
        raise ClientError("publication session manifest binding mismatch")
    if message_type == "status":
        capacity = publication["publication_capacity"]
        if (
            not isinstance(capacity, dict)
            or set(capacity)
            != {"total_slots", "remaining_slots_after_publish"}
            or capacity["total_slots"] != session["publication_auth_slots"]
            or isinstance(capacity["remaining_slots_after_publish"], bool)
            or not isinstance(capacity["remaining_slots_after_publish"], int)
            or not 0
            <= capacity["remaining_slots_after_publish"]
            < capacity["total_slots"]
        ):
            raise ClientError("publication capacity telemetry is invalid")

    authentication = publication["authentication"]
    auth_fields = {
        "scheme",
        "key_index",
        "message_sha256",
        "revealed_secrets",
        "opposite_hashes",
        "merkle_proof",
    }
    if not isinstance(authentication, dict) or set(authentication) != auth_fields:
        raise ClientError("publication authentication fields are invalid")
    unsigned = {key: value for key, value in publication.items() if key != "authentication"}
    digest = hashlib.sha256(
        _PUBLICATION_MESSAGE_DOMAIN + _canonical_bytes(unsigned)
    ).digest()
    message_hash = authentication["message_sha256"]
    if (
        authentication["scheme"] != session["publication_auth_scheme"]
        or not isinstance(message_hash, str)
        or not hmac.compare_digest(message_hash, digest.hex())
    ):
        raise ClientError("publication content authentication failed")
    key_index = authentication["key_index"]
    slots = session["publication_auth_slots"]
    if (
        isinstance(key_index, bool)
        or not isinstance(key_index, int)
        or not 0 <= key_index < slots
    ):
        raise ClientError("publication authentication key index is invalid")
    if message_type == "status" and publication["publication_capacity"][
        "remaining_slots_after_publish"
    ] != slots - key_index - 1:
        raise ClientError("publication capacity does not match signature key index")
    revealed = authentication["revealed_secrets"]
    opposite = authentication["opposite_hashes"]
    proof = authentication["merkle_proof"]
    if (
        not isinstance(revealed, list)
        or len(revealed) != 256
        or not isinstance(opposite, list)
        or len(opposite) != 256
        or not isinstance(proof, list)
        or len(proof) != slots.bit_length() - 1
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in revealed)
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in opposite)
        or not all(isinstance(value, str) and _SHA256_RE.fullmatch(value) for value in proof)
    ):
        raise ClientError("publication hash-based signature is invalid")

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
    if not hmac.compare_digest(node.hex(), session["publication_auth_root"]):
        raise ClientError("publication signature is not from the trusted campaign key")
    return unsigned


def _load_action(root: Path, action_path: Path) -> tuple[dict[str, Any], bytes]:
    actions_root = (root / "actions").resolve()
    resolved_action = action_path.resolve()
    if resolved_action.parent != actions_root:
        raise ClientError("action files must be direct children of the bundle actions directory")
    action = _read_json_object(resolved_action)
    canonical = _canonical_bytes(action)
    if len(canonical) > MAX_ACTION_BYTES:
        raise ClientError("canonical action exceeds the bridge limit")
    return action, canonical


def _render(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


def show_status(root: Path) -> int:
    session = _load_session(root)
    status_path = root / "inbox" / "latest_status.json"
    signed_status = _read_json_object(status_path, max_bytes=5_000_000)
    status = _verify_publication(signed_status, session, message_type="status")
    print(_render(status))
    return 0


def show_session(root: Path) -> int:
    print(_render(_load_session(root)))
    return 0


def validate_action(root: Path, action_path: Path) -> int:
    _load_session(root)
    _, canonical = _load_action(root, action_path)
    print(
        _render(
            {
                "ok": True,
                "canonical_bytes": len(canonical),
                "action_sha256": hashlib.sha256(canonical).hexdigest(),
                "scope": "strict JSON/path/size only; evaluator semantics were not run",
            }
        )
    )
    return 0


def submit(root: Path, action_path: Path, *, timeout: float, poll_interval: float) -> int:
    session = _load_session(root)
    action, action_bytes = _load_action(root, action_path)

    outbox = root / "outbox"
    inbox = root / "inbox"
    pending = tuple(outbox.glob("*.ready.json"))
    if pending:
        raise ClientError(
            "another request is pending; inspect status/receipt before submitting again"
        )

    request_id = f"req_{secrets.token_hex(12)}"
    assert _REQUEST_ID_RE.fullmatch(request_id)
    action_sha256 = hashlib.sha256(action_bytes).hexdigest()
    signed_status = _read_json_object(
        inbox / "latest_status.json", max_bytes=5_000_000
    )
    latest_status = _verify_publication(
        signed_status, session, message_type="status"
    )
    try:
        checkpoint = latest_status["status"]["checkpoint"]
        expected_cursor = {
            "ledger_events": checkpoint["ledger_events"],
            "ledger_tip": checkpoint["ledger_tip"],
        }
    except (KeyError, TypeError) as exc:
        raise ClientError("latest status does not contain a valid evaluator cursor") from exc
    envelope = {
        "protocol_version": PROTOCOL_VERSION,
        "session_id": session["session_id"],
        "campaign_id": session["campaign_id"],
        "request_id": request_id,
        "manifest_sha256": session["manifest_sha256"],
        "expected_cursor": expected_cursor,
        "action_sha256": action_sha256,
        "action": action,
    }
    ready_path = outbox / f"{request_id}.ready.json"
    _atomic_write(ready_path, _canonical_bytes(envelope) + b"\n")

    receipt_path = inbox / f"{request_id}.receipt.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if receipt_path.exists():
            signed_receipt = _read_json_object(receipt_path, max_bytes=10_000_000)
            receipt = _verify_publication(
                signed_receipt, session, message_type="receipt"
            )
            if receipt.get("request_id") != request_id:
                raise ClientError("bridge receipt request_id mismatch")
            if receipt.get("action_sha256") != action_sha256:
                raise ClientError("bridge receipt action hash mismatch")
            print(_render(receipt))
            return 0 if receipt.get("ok") is True else 2
        time.sleep(poll_interval)
    raise ClientError(
        "timed out waiting for the trusted evaluator; run status before deciding whether to retry"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoVQE isolated agent bridge client")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="print the latest evaluator-published status")
    subparsers.add_parser("session", help="print the immutable public session manifest")
    validate_parser = subparsers.add_parser(
        "validate", help="preflight one action's strict JSON path and size only"
    )
    validate_parser.add_argument("--action", required=True, type=Path)
    submit_parser = subparsers.add_parser("submit", help="submit exactly one action")
    submit_parser.add_argument("--action", required=True, type=Path)
    submit_parser.add_argument("--timeout", type=float, default=900.0)
    submit_parser.add_argument("--poll-interval", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _bundle_root()
    try:
        if args.command == "status":
            return show_status(root)
        if args.command == "session":
            return show_session(root)
        if args.command == "validate":
            return validate_action(root, args.action)
        if args.timeout <= 0 or args.poll_interval <= 0:
            raise ClientError("timeout and poll interval must be positive")
        return submit(
            root,
            args.action,
            timeout=float(args.timeout),
            poll_interval=float(args.poll_interval),
        )
    except ClientError as exc:
        print(f"client error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
