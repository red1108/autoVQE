from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

from .ledger import GENESIS_HASH, JsonlEventLedger, LedgerEvent, LedgerIntegrityError


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TERMINAL = {"REVISED", "RETIRED"}


class ResearchError(RuntimeError):
    """Base class for research-loop failures."""


class ActionParseError(ResearchError):
    """Raised when an action does not match the typed action schema."""


class TransitionError(ResearchError):
    """Raised when an action is invalid for the current lifecycle state."""


class BudgetExceeded(ResearchError):
    """Raised when an action would overspend the campaign budget."""


class Lifecycle(str, Enum):
    PROPOSED = "PROPOSED"
    PROBED = "PROBED"
    SUPPORTED = "SUPPORTED"
    CANDIDATE = "CANDIDATE"
    AUDITED = "AUDITED"
    SMOKE = "SMOKE"
    PROMOTED = "PROMOTED"
    REVISED = "REVISED"
    RETIRED = "RETIRED"


class ActionType(str, Enum):
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    RECORD_PROBE = "record_probe"
    SUBMIT_CANDIDATE = "submit_candidate"
    RECORD_EVALUATION = "record_evaluation"
    REVISE = "revise"
    RETIRE = "retire"
    COMMIT = "commit"
    CLOSE_NEGATIVE = "close_negative"


class EntityKind(str, Enum):
    HYPOTHESIS = "hypothesis"
    CANDIDATE = "candidate"


class ProbeVerdict(str, Enum):
    SUPPORTED = "supported"
    INCONCLUSIVE = "inconclusive"
    REFUTED = "refuted"


class EvaluationStage(str, Enum):
    AUDIT = "audit"
    SMOKE = "smoke"
    PROMOTION = "promotion"


def _validated_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ActionParseError(
            f"{field_name} must match {_ID_RE.pattern!r}; got {value!r}"
        )
    return value


def _validated_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionParseError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validated_cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError("cost must be a number")
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        raise ActionParseError("cost must be finite and non-negative")
    return cost


def _validated_mapping(value: Any, field_name: str, *, nonempty: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionParseError(f"{field_name} must be an object")
    try:
        detached = json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ActionParseError(f"{field_name} must contain JSON data: {exc}") from exc
    if nonempty and not detached:
        raise ActionParseError(f"{field_name} must not be empty")
    return detached


def _validated_ids(value: Any, field_name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ActionParseError(f"{field_name} must be a list of IDs")
    validated = tuple(_validated_id(item, field_name) for item in value)
    if nonempty and not validated:
        raise ActionParseError(f"{field_name} must not be empty")
    if len(set(validated)) != len(validated):
        raise ActionParseError(f"{field_name} must not contain duplicate IDs")
    return validated


def _coerce_enum(enum_type: type[Enum], value: Any, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = [item.value for item in enum_type]
        raise ActionParseError(f"{field_name} must be one of {allowed}; got {value!r}") from exc


@dataclass(frozen=True)
class ProposeHypothesisAction:
    hypothesis_id: str
    claim: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _validated_id(self.hypothesis_id, "hypothesis_id"))
        object.__setattr__(self, "claim", _validated_mapping(self.claim, "claim", nonempty=True))
        object.__setattr__(self, "metadata", _validated_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class RecordProbeAction:
    hypothesis_id: str
    probe_id: str
    verdict: ProbeVerdict | str
    result: Mapping[str, Any]
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _validated_id(self.hypothesis_id, "hypothesis_id"))
        object.__setattr__(self, "probe_id", _validated_id(self.probe_id, "probe_id"))
        object.__setattr__(self, "verdict", _coerce_enum(ProbeVerdict, self.verdict, "verdict"))
        object.__setattr__(self, "result", _validated_mapping(self.result, "result"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class SubmitCandidateAction:
    candidate_id: str
    hypothesis_id: str
    spec: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validated_id(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "hypothesis_id", _validated_id(self.hypothesis_id, "hypothesis_id"))
        object.__setattr__(self, "spec", _validated_mapping(self.spec, "spec", nonempty=True))
        object.__setattr__(self, "metadata", _validated_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class RecordEvaluationAction:
    candidate_id: str
    evaluation_id: str
    stage: EvaluationStage | str
    passed: bool
    metrics: Mapping[str, Any]
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validated_id(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "evaluation_id", _validated_id(self.evaluation_id, "evaluation_id"))
        object.__setattr__(self, "stage", _coerce_enum(EvaluationStage, self.stage, "stage"))
        if not isinstance(self.passed, bool):
            raise ActionParseError("passed must be a boolean")
        object.__setattr__(self, "metrics", _validated_mapping(self.metrics, "metrics"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class ReviseAction:
    entity: EntityKind | str
    source_id: str
    new_id: str
    replacement: Mapping[str, Any]
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity", _coerce_enum(EntityKind, self.entity, "entity"))
        object.__setattr__(self, "source_id", _validated_id(self.source_id, "source_id"))
        object.__setattr__(self, "new_id", _validated_id(self.new_id, "new_id"))
        if self.source_id == self.new_id:
            raise ActionParseError("new_id must differ from source_id")
        object.__setattr__(
            self,
            "replacement",
            _validated_mapping(self.replacement, "replacement", nonempty=True),
        )
        object.__setattr__(self, "reason", _validated_text(self.reason, "reason"))
        object.__setattr__(self, "metadata", _validated_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class RetireAction:
    entity: EntityKind | str
    entity_id: str
    reason: str
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity", _coerce_enum(EntityKind, self.entity, "entity"))
        object.__setattr__(self, "entity_id", _validated_id(self.entity_id, "entity_id"))
        object.__setattr__(self, "reason", _validated_text(self.reason, "reason"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class CommitAction:
    candidate_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _validated_id(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "metadata", _validated_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


@dataclass(frozen=True)
class CloseNegativeAction:
    reason: str
    evidence_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _validated_text(self.reason, "reason"))
        object.__setattr__(
            self,
            "evidence_ids",
            _validated_ids(self.evidence_ids, "evidence_ids", nonempty=True),
        )
        object.__setattr__(self, "metadata", _validated_mapping(self.metadata, "metadata"))
        object.__setattr__(self, "cost", _validated_cost(self.cost))


ResearchAction = Union[
    ProposeHypothesisAction,
    RecordProbeAction,
    SubmitCandidateAction,
    RecordEvaluationAction,
    ReviseAction,
    RetireAction,
    CommitAction,
    CloseNegativeAction,
]

_ACTION_CLASSES = (
    ProposeHypothesisAction,
    RecordProbeAction,
    SubmitCandidateAction,
    RecordEvaluationAction,
    ReviseAction,
    RetireAction,
    CommitAction,
    CloseNegativeAction,
)


def _strict_fields(
    raw: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    fields = set(raw)
    missing = required - fields
    extra = fields - required - optional
    if missing or extra:
        raise ActionParseError(
            f"invalid action fields: missing={sorted(missing)} extra={sorted(extra)}"
        )


def parse_action(raw: ResearchAction | Mapping[str, Any]) -> ResearchAction:
    """Parse a strict mapping into one of the closed set of action dataclasses."""

    if isinstance(raw, _ACTION_CLASSES):
        return raw
    if not isinstance(raw, Mapping):
        raise ActionParseError("action must be a mapping or typed action")
    action_type = raw.get("type")
    try:
        kind = ActionType(action_type)
    except (TypeError, ValueError) as exc:
        raise ActionParseError(f"unknown action type: {action_type!r}") from exc

    common_optional = {"cost"}
    if kind is ActionType.PROPOSE_HYPOTHESIS:
        _strict_fields(
            raw,
            required={"type", "hypothesis_id", "claim"},
            optional=common_optional | {"metadata"},
        )
        return ProposeHypothesisAction(
            hypothesis_id=raw["hypothesis_id"],
            claim=raw["claim"],
            metadata=raw.get("metadata", {}),
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.RECORD_PROBE:
        _strict_fields(
            raw,
            required={"type", "hypothesis_id", "probe_id", "verdict", "result"},
            optional=common_optional,
        )
        return RecordProbeAction(
            hypothesis_id=raw["hypothesis_id"],
            probe_id=raw["probe_id"],
            verdict=raw["verdict"],
            result=raw["result"],
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.SUBMIT_CANDIDATE:
        _strict_fields(
            raw,
            required={"type", "candidate_id", "hypothesis_id", "spec"},
            optional=common_optional | {"metadata"},
        )
        return SubmitCandidateAction(
            candidate_id=raw["candidate_id"],
            hypothesis_id=raw["hypothesis_id"],
            spec=raw["spec"],
            metadata=raw.get("metadata", {}),
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.RECORD_EVALUATION:
        _strict_fields(
            raw,
            required={"type", "candidate_id", "evaluation_id", "stage", "passed", "metrics"},
            optional=common_optional,
        )
        return RecordEvaluationAction(
            candidate_id=raw["candidate_id"],
            evaluation_id=raw["evaluation_id"],
            stage=raw["stage"],
            passed=raw["passed"],
            metrics=raw["metrics"],
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.REVISE:
        _strict_fields(
            raw,
            required={"type", "entity", "source_id", "new_id", "replacement", "reason"},
            optional=common_optional | {"metadata"},
        )
        return ReviseAction(
            entity=raw["entity"],
            source_id=raw["source_id"],
            new_id=raw["new_id"],
            replacement=raw["replacement"],
            reason=raw["reason"],
            metadata=raw.get("metadata", {}),
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.RETIRE:
        _strict_fields(
            raw,
            required={"type", "entity", "entity_id", "reason"},
            optional=common_optional,
        )
        return RetireAction(
            entity=raw["entity"],
            entity_id=raw["entity_id"],
            reason=raw["reason"],
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.COMMIT:
        _strict_fields(
            raw,
            required={"type", "candidate_id"},
            optional=common_optional | {"metadata"},
        )
        return CommitAction(
            candidate_id=raw["candidate_id"],
            metadata=raw.get("metadata", {}),
            cost=raw.get("cost", 0.0),
        )
    if kind is ActionType.CLOSE_NEGATIVE:
        _strict_fields(
            raw,
            required={"type", "reason", "evidence_ids"},
            optional=common_optional | {"metadata"},
        )
        return CloseNegativeAction(
            reason=raw["reason"],
            evidence_ids=raw["evidence_ids"],
            metadata=raw.get("metadata", {}),
            cost=raw.get("cost", 0.0),
        )
    raise AssertionError(f"unhandled action type {kind}")


def _action_type(action: ResearchAction) -> ActionType:
    if isinstance(action, ProposeHypothesisAction):
        return ActionType.PROPOSE_HYPOTHESIS
    if isinstance(action, RecordProbeAction):
        return ActionType.RECORD_PROBE
    if isinstance(action, SubmitCandidateAction):
        return ActionType.SUBMIT_CANDIDATE
    if isinstance(action, RecordEvaluationAction):
        return ActionType.RECORD_EVALUATION
    if isinstance(action, ReviseAction):
        return ActionType.REVISE
    if isinstance(action, RetireAction):
        return ActionType.RETIRE
    if isinstance(action, CommitAction):
        return ActionType.COMMIT
    if isinstance(action, CloseNegativeAction):
        return ActionType.CLOSE_NEGATIVE
    raise TypeError(f"unsupported action {type(action).__name__}")


def action_to_mapping(action: ResearchAction) -> dict[str, Any]:
    """Return the public JSON representation used by scripted agents."""

    kind = _action_type(action)
    raw: dict[str, Any] = {"type": kind.value, "cost": action.cost}
    if isinstance(action, ProposeHypothesisAction):
        raw.update(
            hypothesis_id=action.hypothesis_id,
            claim=copy.deepcopy(dict(action.claim)),
            metadata=copy.deepcopy(dict(action.metadata)),
        )
    elif isinstance(action, RecordProbeAction):
        raw.update(
            hypothesis_id=action.hypothesis_id,
            probe_id=action.probe_id,
            verdict=action.verdict.value,
            result=copy.deepcopy(dict(action.result)),
        )
    elif isinstance(action, SubmitCandidateAction):
        raw.update(
            candidate_id=action.candidate_id,
            hypothesis_id=action.hypothesis_id,
            spec=copy.deepcopy(dict(action.spec)),
            metadata=copy.deepcopy(dict(action.metadata)),
        )
    elif isinstance(action, RecordEvaluationAction):
        raw.update(
            candidate_id=action.candidate_id,
            evaluation_id=action.evaluation_id,
            stage=action.stage.value,
            passed=action.passed,
            metrics=copy.deepcopy(dict(action.metrics)),
        )
    elif isinstance(action, ReviseAction):
        raw.update(
            entity=action.entity.value,
            source_id=action.source_id,
            new_id=action.new_id,
            replacement=copy.deepcopy(dict(action.replacement)),
            reason=action.reason,
            metadata=copy.deepcopy(dict(action.metadata)),
        )
    elif isinstance(action, RetireAction):
        raw.update(entity=action.entity.value, entity_id=action.entity_id, reason=action.reason)
    elif isinstance(action, CommitAction):
        raw.update(candidate_id=action.candidate_id, metadata=copy.deepcopy(dict(action.metadata)))
    elif isinstance(action, CloseNegativeAction):
        raw.update(
            reason=action.reason,
            evidence_ids=list(action.evidence_ids),
            metadata=copy.deepcopy(dict(action.metadata)),
        )
    return raw


def _event_payload(action: ResearchAction) -> dict[str, Any]:
    raw = action_to_mapping(action)
    raw.pop("type")
    raw.pop("cost")
    return raw


@dataclass(frozen=True)
class HypothesisRecord:
    hypothesis_id: str
    claim: Mapping[str, Any]
    metadata: Mapping[str, Any]
    status: Lifecycle = Lifecycle.PROPOSED
    parent_id: str | None = None
    probe_ids: tuple[str, ...] = ()
    revised_to: str | None = None
    retired_reason: str | None = None


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    hypothesis_id: str
    spec: Mapping[str, Any]
    metadata: Mapping[str, Any]
    status: Lifecycle = Lifecycle.CANDIDATE
    parent_id: str | None = None
    evaluation_ids: tuple[str, ...] = ()
    revised_to: str | None = None
    retired_reason: str | None = None


@dataclass(frozen=True)
class ProbeRecord:
    probe_id: str
    hypothesis_id: str
    verdict: ProbeVerdict
    result: Mapping[str, Any]
    cost: float


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    candidate_id: str
    stage: EvaluationStage
    passed: bool
    metrics: Mapping[str, Any]
    cost: float
    status_before: Lifecycle
    status_after: Lifecycle


@dataclass
class ResearchState:
    total_budget: float
    spent_budget: float = 0.0
    hypotheses: dict[str, HypothesisRecord] = field(default_factory=dict)
    candidates: dict[str, CandidateRecord] = field(default_factory=dict)
    probes: dict[str, ProbeRecord] = field(default_factory=dict)
    evaluations: dict[str, EvaluationRecord] = field(default_factory=dict)
    committed_candidate_id: str | None = None
    commit_metadata: dict[str, Any] = field(default_factory=dict)
    negative_closed: bool = False
    negative_close_reason: str | None = None
    negative_close_evidence_ids: tuple[str, ...] = ()
    negative_close_metadata: dict[str, Any] = field(default_factory=dict)
    last_seq: int = -1
    last_hash: str = GENESIS_HASH

    def __post_init__(self) -> None:
        self.total_budget = _validated_cost(self.total_budget)
        self.spent_budget = _validated_cost(self.spent_budget)
        if self.spent_budget > self.total_budget + 1e-12:
            raise BudgetExceeded(
                f"spent budget {self.spent_budget} exceeds total budget {self.total_budget}"
            )

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.total_budget - self.spent_budget)

    @property
    def committed(self) -> bool:
        return self.committed_candidate_id is not None

    @property
    def terminal(self) -> bool:
        return self.committed or self.negative_closed

    @property
    def terminal_decision(self) -> str | None:
        if self.committed:
            return "positive_commit"
        if self.negative_closed:
            return "negative_close"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "spent_budget": self.spent_budget,
            "remaining_budget": self.remaining_budget,
            "committed_candidate_id": self.committed_candidate_id,
            "commit_metadata": copy.deepcopy(self.commit_metadata),
            "negative_closed": self.negative_closed,
            "negative_close_reason": self.negative_close_reason,
            "negative_close_evidence_ids": list(self.negative_close_evidence_ids),
            "negative_close_metadata": copy.deepcopy(self.negative_close_metadata),
            "terminal_decision": self.terminal_decision,
            "last_seq": self.last_seq,
            "last_hash": self.last_hash,
            "hypotheses": {
                key: {
                    "hypothesis_id": item.hypothesis_id,
                    "claim": copy.deepcopy(dict(item.claim)),
                    "metadata": copy.deepcopy(dict(item.metadata)),
                    "status": item.status.value,
                    "parent_id": item.parent_id,
                    "probe_ids": list(item.probe_ids),
                    "revised_to": item.revised_to,
                    "retired_reason": item.retired_reason,
                }
                for key, item in sorted(self.hypotheses.items())
            },
            "candidates": {
                key: {
                    "candidate_id": item.candidate_id,
                    "hypothesis_id": item.hypothesis_id,
                    "spec": copy.deepcopy(dict(item.spec)),
                    "metadata": copy.deepcopy(dict(item.metadata)),
                    "status": item.status.value,
                    "parent_id": item.parent_id,
                    "evaluation_ids": list(item.evaluation_ids),
                    "revised_to": item.revised_to,
                    "retired_reason": item.retired_reason,
                }
                for key, item in sorted(self.candidates.items())
            },
            "probes": {
                key: {
                    "probe_id": item.probe_id,
                    "hypothesis_id": item.hypothesis_id,
                    "verdict": item.verdict.value,
                    "result": copy.deepcopy(dict(item.result)),
                    "cost": item.cost,
                }
                for key, item in sorted(self.probes.items())
            },
            "evaluations": {
                key: {
                    "evaluation_id": item.evaluation_id,
                    "candidate_id": item.candidate_id,
                    "stage": item.stage.value,
                    "passed": item.passed,
                    "metrics": copy.deepcopy(dict(item.metrics)),
                    "cost": item.cost,
                    "status_before": item.status_before.value,
                    "status_after": item.status_after.value,
                }
                for key, item in sorted(self.evaluations.items())
            },
        }


def _active_candidates(state: ResearchState, hypothesis_id: str) -> list[str]:
    return [
        candidate.candidate_id
        for candidate in state.candidates.values()
        if candidate.hypothesis_id == hypothesis_id and candidate.status.value not in _TERMINAL
    ]


def _apply_propose(state: ResearchState, action: ProposeHypothesisAction) -> None:
    if action.hypothesis_id in state.hypotheses:
        raise TransitionError(f"hypothesis already exists: {action.hypothesis_id}")
    state.hypotheses[action.hypothesis_id] = HypothesisRecord(
        hypothesis_id=action.hypothesis_id,
        claim=copy.deepcopy(dict(action.claim)),
        metadata=copy.deepcopy(dict(action.metadata)),
    )


def _apply_probe(state: ResearchState, action: RecordProbeAction) -> None:
    hypothesis = state.hypotheses.get(action.hypothesis_id)
    if hypothesis is None:
        raise TransitionError(f"unknown hypothesis: {action.hypothesis_id}")
    if action.probe_id in state.probes:
        raise TransitionError(f"probe already exists: {action.probe_id}")
    if hypothesis.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
        raise TransitionError(
            f"cannot probe hypothesis {action.hypothesis_id} in {hypothesis.status.value}"
        )

    next_status = (
        Lifecycle.SUPPORTED
        if action.verdict is ProbeVerdict.SUPPORTED
        else Lifecycle.PROBED
    )
    state.probes[action.probe_id] = ProbeRecord(
        probe_id=action.probe_id,
        hypothesis_id=action.hypothesis_id,
        verdict=action.verdict,
        result=copy.deepcopy(dict(action.result)),
        cost=action.cost,
    )
    state.hypotheses[action.hypothesis_id] = replace(
        hypothesis,
        status=next_status,
        probe_ids=hypothesis.probe_ids + (action.probe_id,),
    )


def _apply_submit_candidate(state: ResearchState, action: SubmitCandidateAction) -> None:
    if action.candidate_id in state.candidates:
        raise TransitionError(f"candidate already exists: {action.candidate_id}")
    hypothesis = state.hypotheses.get(action.hypothesis_id)
    if hypothesis is None:
        raise TransitionError(f"unknown hypothesis: {action.hypothesis_id}")
    if hypothesis.status is not Lifecycle.SUPPORTED:
        raise TransitionError(
            f"candidate requires SUPPORTED hypothesis; {action.hypothesis_id} is "
            f"{hypothesis.status.value}"
        )
    state.candidates[action.candidate_id] = CandidateRecord(
        candidate_id=action.candidate_id,
        hypothesis_id=action.hypothesis_id,
        spec=copy.deepcopy(dict(action.spec)),
        metadata=copy.deepcopy(dict(action.metadata)),
    )


def _evaluation_transition(candidate: CandidateRecord, action: RecordEvaluationAction) -> Lifecycle:
    if action.stage is EvaluationStage.AUDIT:
        allowed = {Lifecycle.CANDIDATE, Lifecycle.AUDITED}
        passed_status = Lifecycle.AUDITED
    elif action.stage is EvaluationStage.SMOKE:
        allowed = {Lifecycle.AUDITED, Lifecycle.SMOKE}
        passed_status = Lifecycle.SMOKE
    else:
        allowed = {Lifecycle.SMOKE, Lifecycle.PROMOTED}
        passed_status = Lifecycle.PROMOTED
    if candidate.status not in allowed:
        raise TransitionError(
            f"cannot run {action.stage.value} evaluation for {candidate.candidate_id} "
            f"in {candidate.status.value}"
        )
    return passed_status if action.passed else Lifecycle.RETIRED


def _apply_evaluation(state: ResearchState, action: RecordEvaluationAction) -> None:
    candidate = state.candidates.get(action.candidate_id)
    if candidate is None:
        raise TransitionError(f"unknown candidate: {action.candidate_id}")
    if action.evaluation_id in state.evaluations:
        raise TransitionError(f"evaluation already exists: {action.evaluation_id}")
    hypothesis = state.hypotheses[candidate.hypothesis_id]
    if hypothesis.status is not Lifecycle.SUPPORTED:
        raise TransitionError(
            f"candidate's hypothesis {candidate.hypothesis_id} is no longer SUPPORTED"
        )

    next_status = _evaluation_transition(candidate, action)
    state.evaluations[action.evaluation_id] = EvaluationRecord(
        evaluation_id=action.evaluation_id,
        candidate_id=action.candidate_id,
        stage=action.stage,
        passed=action.passed,
        metrics=copy.deepcopy(dict(action.metrics)),
        cost=action.cost,
        status_before=candidate.status,
        status_after=next_status,
    )
    state.candidates[action.candidate_id] = replace(
        candidate,
        status=next_status,
        evaluation_ids=candidate.evaluation_ids + (action.evaluation_id,),
        retired_reason=None if action.passed else f"failed {action.stage.value} evaluation",
    )


def _apply_revise(state: ResearchState, action: ReviseAction) -> None:
    if action.entity is EntityKind.HYPOTHESIS:
        source = state.hypotheses.get(action.source_id)
        if source is None:
            raise TransitionError(f"unknown hypothesis: {action.source_id}")
        if action.new_id in state.hypotheses:
            raise TransitionError(f"hypothesis already exists: {action.new_id}")
        if source.status is Lifecycle.REVISED:
            raise TransitionError(f"hypothesis already revised: {action.source_id}")
        active = _active_candidates(state, action.source_id)
        if active:
            raise TransitionError(
                f"retire or revise active candidates before revising hypothesis "
                f"{action.source_id}: {active}"
            )
        source_status = Lifecycle.RETIRED if source.status is Lifecycle.RETIRED else Lifecycle.REVISED
        state.hypotheses[action.source_id] = replace(
            source,
            status=source_status,
            revised_to=action.new_id,
        )
        state.hypotheses[action.new_id] = HypothesisRecord(
            hypothesis_id=action.new_id,
            claim=copy.deepcopy(dict(action.replacement)),
            metadata={
                **copy.deepcopy(dict(action.metadata)),
                "revision_reason": action.reason,
            },
            parent_id=action.source_id,
        )
        return

    source = state.candidates.get(action.source_id)
    if source is None:
        raise TransitionError(f"unknown candidate: {action.source_id}")
    if source.status is Lifecycle.PROMOTED:
        raise TransitionError(
            f"promoted candidate {action.source_id} cannot be revised; commit it"
        )
    if action.new_id in state.candidates:
        raise TransitionError(f"candidate already exists: {action.new_id}")
    if source.status is Lifecycle.REVISED:
        raise TransitionError(f"candidate already revised: {action.source_id}")
    hypothesis = state.hypotheses[source.hypothesis_id]
    if hypothesis.status is not Lifecycle.SUPPORTED:
        raise TransitionError(
            f"cannot revise candidate under {hypothesis.status.value} hypothesis"
        )
    source_status = Lifecycle.RETIRED if source.status is Lifecycle.RETIRED else Lifecycle.REVISED
    state.candidates[action.source_id] = replace(
        source,
        status=source_status,
        revised_to=action.new_id,
    )
    state.candidates[action.new_id] = CandidateRecord(
        candidate_id=action.new_id,
        hypothesis_id=source.hypothesis_id,
        spec=copy.deepcopy(dict(action.replacement)),
        metadata={
            **copy.deepcopy(dict(source.metadata)),
            **copy.deepcopy(dict(action.metadata)),
            "revision_reason": action.reason,
        },
        parent_id=action.source_id,
    )


def _apply_retire(state: ResearchState, action: RetireAction) -> None:
    if action.entity is EntityKind.HYPOTHESIS:
        item = state.hypotheses.get(action.entity_id)
        if item is None:
            raise TransitionError(f"unknown hypothesis: {action.entity_id}")
        if item.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
            raise TransitionError(
                f"hypothesis {action.entity_id} is already {item.status.value}"
            )
        active = _active_candidates(state, action.entity_id)
        if active:
            raise TransitionError(
                f"retire active candidates before hypothesis {action.entity_id}: {active}"
            )
        state.hypotheses[action.entity_id] = replace(
            item,
            status=Lifecycle.RETIRED,
            retired_reason=action.reason,
        )
        return

    item = state.candidates.get(action.entity_id)
    if item is None:
        raise TransitionError(f"unknown candidate: {action.entity_id}")
    if item.status is Lifecycle.PROMOTED:
        raise TransitionError(
            f"promoted candidate {action.entity_id} cannot be retired; commit it"
        )
    if item.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
        raise TransitionError(f"candidate {action.entity_id} is already {item.status.value}")
    state.candidates[action.entity_id] = replace(
        item,
        status=Lifecycle.RETIRED,
        retired_reason=action.reason,
    )


def _apply_commit(state: ResearchState, action: CommitAction) -> None:
    candidate = state.candidates.get(action.candidate_id)
    if candidate is None:
        raise TransitionError(f"unknown candidate: {action.candidate_id}")
    if candidate.status is not Lifecycle.PROMOTED:
        raise TransitionError(
            f"commit requires PROMOTED candidate; {action.candidate_id} is "
            f"{candidate.status.value}"
        )
    hypothesis = state.hypotheses[candidate.hypothesis_id]
    if hypothesis.status is not Lifecycle.SUPPORTED:
        raise TransitionError(
            f"commit requires a SUPPORTED parent hypothesis; got {hypothesis.status.value}"
        )
    state.committed_candidate_id = action.candidate_id
    state.commit_metadata = copy.deepcopy(dict(action.metadata))


def _apply_close_negative(state: ResearchState, action: CloseNegativeAction) -> None:
    if not state.hypotheses:
        raise TransitionError("negative close requires at least one investigated hypothesis")
    live_hypotheses = sorted(
        item.hypothesis_id
        for item in state.hypotheses.values()
        if item.status.value not in _TERMINAL
    )
    live_candidates = sorted(
        item.candidate_id
        for item in state.candidates.values()
        if item.status.value not in _TERMINAL
    )
    if live_hypotheses or live_candidates:
        raise TransitionError(
            "negative close requires every hypothesis and candidate to be terminal; "
            f"live_hypotheses={live_hypotheses} live_candidates={live_candidates}"
        )
    for evidence_id in action.evidence_ids:
        in_probes = evidence_id in state.probes
        in_evaluations = evidence_id in state.evaluations
        if in_probes and in_evaluations:
            raise TransitionError(f"ambiguous evidence ID: {evidence_id}")
        if not in_probes and not in_evaluations:
            raise TransitionError(f"unknown evidence ID: {evidence_id}")
    cited_ids = set(action.evidence_ids)
    uncovered_hypotheses: list[str] = []
    for hypothesis in state.hypotheses.values():
        cited_refutation = any(
            probe_id in cited_ids
            and state.probes[probe_id].verdict is ProbeVerdict.REFUTED
            for probe_id in hypothesis.probe_ids
        )
        cited_candidate_result = any(
            evaluation.evaluation_id in cited_ids
            and state.candidates[evaluation.candidate_id].hypothesis_id
            == hypothesis.hypothesis_id
            for evaluation in state.evaluations.values()
        )
        if not cited_refutation and not cited_candidate_result:
            uncovered_hypotheses.append(hypothesis.hypothesis_id)
    if uncovered_hypotheses:
        raise TransitionError(
            "negative close lacks refutation/evaluation evidence for hypotheses: "
            f"{sorted(uncovered_hypotheses)}"
        )
    state.negative_closed = True
    state.negative_close_reason = action.reason
    state.negative_close_evidence_ids = action.evidence_ids
    state.negative_close_metadata = copy.deepcopy(dict(action.metadata))


def apply_action(state: ResearchState, action_like: ResearchAction | Mapping[str, Any]) -> ResearchState:
    """Pure reducer used both for live dispatch and deterministic replay."""

    action = parse_action(action_like)
    if state.terminal:
        raise TransitionError("campaign is terminal; no further actions are allowed")
    projected = state.spent_budget + action.cost
    if projected > state.total_budget + 1e-12:
        raise BudgetExceeded(
            f"action costs {action.cost}, remaining budget is {state.remaining_budget}"
        )

    next_state = copy.deepcopy(state)
    if isinstance(action, ProposeHypothesisAction):
        _apply_propose(next_state, action)
    elif isinstance(action, RecordProbeAction):
        _apply_probe(next_state, action)
    elif isinstance(action, SubmitCandidateAction):
        _apply_submit_candidate(next_state, action)
    elif isinstance(action, RecordEvaluationAction):
        _apply_evaluation(next_state, action)
    elif isinstance(action, ReviseAction):
        _apply_revise(next_state, action)
    elif isinstance(action, RetireAction):
        _apply_retire(next_state, action)
    elif isinstance(action, CommitAction):
        _apply_commit(next_state, action)
    elif isinstance(action, CloseNegativeAction):
        _apply_close_negative(next_state, action)
    else:
        raise AssertionError(f"unhandled action {type(action).__name__}")
    next_state.spent_budget = projected
    return next_state


def _action_from_event(event: LedgerEvent) -> ResearchAction:
    # LedgerEvent intentionally freezes nested mappings.  Its record view is
    # the detached JSON representation expected by the action parser.
    payload = event.to_record()["payload"]
    raw = {"type": event.event_type, "cost": event.cost, **payload}
    return parse_action(raw)


def replay_events(events: Iterable[LedgerEvent], *, total_budget: float) -> ResearchState:
    """Rebuild state from an already parsed event sequence."""

    state = ResearchState(total_budget=total_budget)
    expected_seq = 0
    expected_prev = GENESIS_HASH
    for event in events:
        if event.seq != expected_seq or event.prev_hash != expected_prev:
            raise LedgerIntegrityError(
                f"invalid replay chain at seq {event.seq}: expected seq={expected_seq} "
                f"prev_hash={expected_prev}"
            )
        state = apply_action(state, _action_from_event(event))
        state.last_seq = event.seq
        state.last_hash = event.event_hash
        expected_seq += 1
        expected_prev = event.event_hash
    return state


def replay_ledger(ledger: JsonlEventLedger, *, total_budget: float) -> ResearchState:
    return replay_events(ledger.verify(), total_budget=total_budget)


class ResearchLoop:
    """A small event-sourced controller suitable for scripted or external agents."""

    def __init__(self, ledger: JsonlEventLedger | str | Path, *, total_budget: float):
        self.ledger = ledger if isinstance(ledger, JsonlEventLedger) else JsonlEventLedger(ledger)
        self._state = replay_ledger(self.ledger, total_budget=total_budget)

    @property
    def state(self) -> ResearchState:
        return copy.deepcopy(self._state)

    def dispatch(self, action_like: ResearchAction | Mapping[str, Any]) -> LedgerEvent:
        action = parse_action(action_like)
        current_events = self.ledger.verify()
        current_tip = current_events[-1].event_hash if current_events else GENESIS_HASH
        if len(current_events) != self._state.last_seq + 1 or current_tip != self._state.last_hash:
            raise LedgerIntegrityError("ledger changed since this ResearchLoop was opened")

        next_state = apply_action(self._state, action)
        event = self.ledger.append(
            _action_type(action).value,
            _event_payload(action),
            cost=action.cost,
        )
        expected_seq = self._state.last_seq + 1
        if event.seq != expected_seq or event.prev_hash != self._state.last_hash:
            raise LedgerIntegrityError("ledger tip changed during dispatch")
        next_state.last_seq = event.seq
        next_state.last_hash = event.event_hash
        self._state = next_state
        return event

    def run_script(
        self,
        actions: Iterable[ResearchAction | Mapping[str, Any]],
    ) -> ResearchState:
        for action in actions:
            self.dispatch(action)
        return self.state


__all__ = [
    "ActionParseError",
    "ActionType",
    "BudgetExceeded",
    "CandidateRecord",
    "CloseNegativeAction",
    "CommitAction",
    "EntityKind",
    "EvaluationRecord",
    "EvaluationStage",
    "HypothesisRecord",
    "Lifecycle",
    "ProbeRecord",
    "ProbeVerdict",
    "ProposeHypothesisAction",
    "RecordEvaluationAction",
    "RecordProbeAction",
    "ResearchAction",
    "ResearchError",
    "ResearchLoop",
    "ResearchState",
    "RetireAction",
    "ReviseAction",
    "SubmitCandidateAction",
    "TransitionError",
    "action_to_mapping",
    "apply_action",
    "parse_action",
    "replay_events",
    "replay_ledger",
]
