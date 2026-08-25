from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RECORD_FIELDS = {
    "version",
    "seq",
    "type",
    "payload",
    "cost",
    "prev_hash",
    "hash",
}


class LedgerError(RuntimeError):
    """Base class for event-ledger failures."""


class LedgerFormatError(LedgerError):
    """Raised when an event is not valid canonical ledger data."""


class LedgerIntegrityError(LedgerError):
    """Raised when a ledger's sequence or hash chain is invalid."""


def _normalize_json(value: Any, *, path: str = "payload") -> Any:
    """Return a detached JSON value, rejecting ambiguous/non-finite data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LedgerFormatError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LedgerFormatError(f"{path} contains a non-string object key")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise LedgerFormatError(f"{path} contains unsupported value {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerFormatError(f"event is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _event_hash(core: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(core)).hexdigest()


def _validated_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerFormatError("event cost must be a number")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise LedgerFormatError("event cost must be finite and non-negative")
    return cost


@dataclass(frozen=True)
class LedgerEvent:
    """An immutable, content-addressed event in a JSONL hash chain."""

    seq: int
    event_type: str
    payload: Mapping[str, Any]
    cost: float
    prev_hash: str
    event_hash: str
    version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        event_type: str,
        payload: Mapping[str, Any],
        cost: float,
        prev_hash: str,
    ) -> LedgerEvent:
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise LedgerFormatError("event seq must be a non-negative integer")
        if not isinstance(event_type, str) or not _EVENT_TYPE_RE.fullmatch(event_type):
            raise LedgerFormatError(f"invalid event type: {event_type!r}")
        if not isinstance(prev_hash, str) or not _HASH_RE.fullmatch(prev_hash):
            raise LedgerFormatError("prev_hash must be a lowercase SHA-256 digest")
        if not isinstance(payload, Mapping):
            raise LedgerFormatError("event payload must be an object")

        normalized_payload = _normalize_json(payload)
        normalized_cost = _validated_cost(cost)
        core = {
            "version": SCHEMA_VERSION,
            "seq": seq,
            "type": event_type,
            "payload": normalized_payload,
            "cost": normalized_cost,
            "prev_hash": prev_hash,
        }
        return cls(
            seq=seq,
            event_type=event_type,
            payload=_freeze_json(normalized_payload),
            cost=normalized_cost,
            prev_hash=prev_hash,
            event_hash=_event_hash(core),
        )

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> LedgerEvent:
        if not isinstance(raw, Mapping):
            raise LedgerFormatError("ledger line must contain a JSON object")
        fields = set(raw)
        if fields != _RECORD_FIELDS:
            missing = sorted(_RECORD_FIELDS - fields)
            extra = sorted(fields - _RECORD_FIELDS)
            raise LedgerFormatError(f"invalid event fields: missing={missing} extra={extra}")
        if raw["version"] != SCHEMA_VERSION:
            raise LedgerFormatError(f"unsupported ledger version: {raw['version']!r}")
        supplied_hash = raw["hash"]
        if not isinstance(supplied_hash, str) or not _HASH_RE.fullmatch(supplied_hash):
            raise LedgerFormatError("hash must be a lowercase SHA-256 digest")

        event = cls.create(
            seq=raw["seq"],
            event_type=raw["type"],
            payload=raw["payload"],
            cost=raw["cost"],
            prev_hash=raw["prev_hash"],
        )
        if event.event_hash != supplied_hash:
            raise LedgerIntegrityError(
                f"event {event.seq} hash mismatch: expected {event.event_hash}, got {supplied_hash}"
            )
        return event

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seq": self.seq,
            "type": self.event_type,
            "payload": _thaw_json(self.payload),
            "cost": self.cost,
            "prev_hash": self.prev_hash,
            "hash": self.event_hash,
        }

    def to_json(self) -> str:
        return _canonical_bytes(self.to_record()).decode("utf-8")


class JsonlEventLedger:
    """Append-only JSONL storage whose events form a SHA-256 hash chain."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read_events(self) -> tuple[LedgerEvent, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise LedgerFormatError(f"ledger path is not a file: {self.path}")

        events: list[LedgerEvent] = []
        expected_prev = GENESIS_HASH
        expected_seq = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise LedgerFormatError(f"blank ledger line at {line_number}")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LedgerFormatError(f"invalid JSON at ledger line {line_number}: {exc}") from exc
                try:
                    event = LedgerEvent.from_record(raw)
                except LedgerError as exc:
                    raise type(exc)(f"ledger line {line_number}: {exc}") from exc

                if event.seq != expected_seq:
                    raise LedgerIntegrityError(
                        f"ledger line {line_number}: expected seq {expected_seq}, got {event.seq}"
                    )
                if event.prev_hash != expected_prev:
                    raise LedgerIntegrityError(
                        f"ledger line {line_number}: expected prev_hash {expected_prev}, "
                        f"got {event.prev_hash}"
                    )
                events.append(event)
                expected_seq += 1
                expected_prev = event.event_hash
        return tuple(events)

    def verify(self) -> tuple[LedgerEvent, ...]:
        return self.read_events()

    @property
    def tip_hash(self) -> str:
        events = self.read_events()
        return events[-1].event_hash if events else GENESIS_HASH

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        cost: float = 0.0,
    ) -> LedgerEvent:
        # Re-read before every append so an externally modified ledger is never
        # extended as though its old tip were still trusted.
        events = self.read_events()
        event = LedgerEvent.create(
            seq=len(events),
            event_type=event_type,
            payload=payload,
            cost=cost,
            prev_hash=events[-1].event_hash if events else GENESIS_HASH,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise LedgerIntegrityError("non-empty ledger must end with a newline before append")

        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.to_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event


__all__ = [
    "GENESIS_HASH",
    "SCHEMA_VERSION",
    "JsonlEventLedger",
    "LedgerError",
    "LedgerEvent",
    "LedgerFormatError",
    "LedgerIntegrityError",
]
