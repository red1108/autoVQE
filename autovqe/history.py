from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


HISTORY_VERSION = 1
_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RECORD_FIELDS = {"version", "seq", "type", "payload", "cost"}


class HistoryError(RuntimeError):
    """Base class for research-history failures."""


class HistoryFormatError(HistoryError):
    """Raised when a history record is not valid JSON event data."""


class HistoryIntegrityError(HistoryError):
    """Raised when history sequence numbers are not contiguous."""


def _normalize_json(value: Any, *, path: str = "payload") -> Any:
    """Return a detached JSON value, rejecting ambiguous/non-finite data."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HistoryFormatError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise HistoryFormatError(f"{path} contains a non-string object key")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise HistoryFormatError(
        f"{path} contains unsupported value {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validated_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HistoryFormatError("event cost must be a number")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise HistoryFormatError("event cost must be finite and non-negative")
    return cost


@dataclass(frozen=True)
class RunEvent:
    """One immutable event in a sequential JSONL research history."""

    seq: int
    event_type: str
    payload: Mapping[str, Any]
    cost: float
    version: int = HISTORY_VERSION

    @classmethod
    def create(
        cls,
        *,
        seq: int,
        event_type: str,
        payload: Mapping[str, Any],
        cost: float,
    ) -> "RunEvent":
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise HistoryFormatError("event seq must be a non-negative integer")
        if not isinstance(event_type, str) or not _EVENT_TYPE_RE.fullmatch(event_type):
            raise HistoryFormatError(f"invalid event type: {event_type!r}")
        if not isinstance(payload, Mapping):
            raise HistoryFormatError("event payload must be an object")

        normalized_payload = _normalize_json(payload)
        return cls(
            seq=seq,
            event_type=event_type,
            payload=_freeze_json(normalized_payload),
            cost=_validated_cost(cost),
        )

    @classmethod
    def from_record(cls, raw: Mapping[str, Any]) -> "RunEvent":
        if not isinstance(raw, Mapping):
            raise HistoryFormatError("history line must contain a JSON object")
        fields = set(raw)
        if fields != _RECORD_FIELDS:
            missing = sorted(_RECORD_FIELDS - fields)
            extra = sorted(fields - _RECORD_FIELDS)
            raise HistoryFormatError(
                f"invalid event fields: missing={missing} extra={extra}"
            )
        if raw["version"] != HISTORY_VERSION:
            raise HistoryFormatError(
                f"unsupported history version: {raw['version']!r}"
            )
        return cls.create(
            seq=raw["seq"],
            event_type=raw["type"],
            payload=raw["payload"],
            cost=raw["cost"],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seq": self.seq,
            "type": self.event_type,
            "payload": _thaw_json(self.payload),
            "cost": self.cost,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_record(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


class JsonlRunHistory:
    """Plain append-only JSONL storage with contiguous sequence numbers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read_events(self) -> tuple[RunEvent, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file():
            raise HistoryFormatError(f"history path is not a file: {self.path}")

        events: list[RunEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise HistoryFormatError(f"blank history line at {line_number}")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HistoryFormatError(
                        f"invalid JSON at history line {line_number}: {exc}"
                    ) from exc
                try:
                    event = RunEvent.from_record(raw)
                except HistoryError as exc:
                    raise type(exc)(f"history line {line_number}: {exc}") from exc
                expected_seq = len(events)
                if event.seq != expected_seq:
                    raise HistoryIntegrityError(
                        f"history line {line_number}: expected seq "
                        f"{expected_seq}, got {event.seq}"
                    )
                events.append(event)
        return tuple(events)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        cost: float = 0.0,
        expected_seq: int | None = None,
    ) -> RunEvent:
        events = self.read_events()
        next_seq = len(events)
        if expected_seq is not None and next_seq != expected_seq:
            raise HistoryIntegrityError(
                f"history changed before append: expected seq {expected_seq}, "
                f"got {next_seq}"
            )
        event = RunEvent.create(
            seq=next_seq,
            event_type=event_type,
            payload=payload,
            cost=cost,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise HistoryFormatError(
                        "non-empty history must end with a newline before append"
                    )

        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(event.to_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event


__all__ = [
    "HISTORY_VERSION",
    "HistoryError",
    "HistoryIntegrityError",
    "HistoryFormatError",
    "JsonlRunHistory",
    "RunEvent",
]
