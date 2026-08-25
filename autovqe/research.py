"""Research state, transitions, and replay for AutoVQE.

The public controller validates agent requests and performs measurements. This
module records only the controller's normalized events and reconstructs the
scientific state. Plain JSON events avoid a second public action API while
preserving deterministic replay and failed branches.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .history import HistoryIntegrityError, JsonlRunHistory, RunEvent


class ResearchError(RuntimeError):
    """Base class for research-loop failures."""


class EventFormatError(ResearchError):
    """Raised when a controller event is not valid JSON event data."""


class TransitionError(ResearchError):
    """Raised when an event violates the research lifecycle."""


class BudgetExceeded(ResearchError):
    """Raised when an event would overspend the run budget."""


class Lifecycle(str, Enum):
    PROPOSED = "PROPOSED"
    READY = "READY"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CANDIDATE = "CANDIDATE"
    AUDITED = "AUDITED"
    SMOKE = "SMOKE"
    PROMOTED = "PROMOTED"
    REVISED = "REVISED"
    RETIRED = "RETIRED"


class ProbeVerdict(str, Enum):
    SUPPORTED = "supported"
    INCONCLUSIVE = "inconclusive"
    REFUTED = "refuted"


class EvaluationStage(str, Enum):
    AUDIT = "audit"
    SMOKE = "smoke"
    PROMOTION = "promotion"


COMMIT_ENERGY_TOLERANCE = 5e-4
MIN_NEGATIVE_OBJECTIVE_ACTIVITY_FRACTION = 1e-6
MIN_NEGATIVE_STRUCTURE_LINEAGES = 2
_COMPARISON_RESOURCE_NAMES = (
    "conservative_twoq_count",
    "conservative_total_gate_count",
    "conservative_depth",
)


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TERMINAL = frozenset({Lifecycle.REVISED, Lifecycle.RETIRED})
_EVENT_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "propose_hypothesis": (
        {"type", "hypothesis_id", "claim"},
        {"metadata", "cost"},
    ),
    "record_probe": (
        {"type", "hypothesis_id", "probe_id", "verdict", "result"},
        {"cost"},
    ),
    "submit_candidate": (
        {"type", "candidate_id", "hypothesis_id", "spec"},
        {"metadata", "parent_id", "symmetry_evidence_ids", "cost"},
    ),
    "record_evaluation": (
        {"type", "candidate_id", "evaluation_id", "stage", "passed", "metrics"},
        {"cost"},
    ),
    "revise": (
        {"type", "entity", "source_id", "new_id", "replacement", "reason"},
        {"metadata", "symmetry_evidence_ids", "evidence_ids", "cost"},
    ),
    "retire": (
        {"type", "entity", "entity_id", "reason"},
        {"evidence_ids", "cost"},
    ),
    "commit": (
        {"type", "candidate_id"},
        {"metadata", "cost"},
    ),
    "close_negative": (
        {"type", "reason", "evidence_ids"},
        {"metadata", "cost"},
    ),
}


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise EventFormatError(f"{field_name} must match {_ID_RE.pattern!r}")
    return value


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventFormatError(f"{field_name} must be a non-empty string")
    return value.strip()


def _cost(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventFormatError("cost must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EventFormatError("cost must be finite and non-negative")
    return result


def _json(value: Any, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EventFormatError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EventFormatError(f"{field_name} contains a non-string key")
            result[key] = _json(item, f"{field_name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    raise EventFormatError(
        f"{field_name} contains unsupported value {type(value).__name__}"
    )


def _mapping(value: Any, field_name: str, *, nonempty: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EventFormatError(f"{field_name} must be an object")
    result = _json(value, field_name)
    if nonempty and not result:
        raise EventFormatError(f"{field_name} must not be empty")
    return result


def _identifiers(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise EventFormatError(f"{field_name} must be a non-empty list")
    result = [_identifier(item, field_name) for item in value]
    if len(set(result)) != len(result):
        raise EventFormatError(f"{field_name} must not contain duplicates")
    return result


def _optional_identifiers(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EventFormatError(f"{field_name} must be a list")
    result = [_identifier(item, field_name) for item in value]
    if len(set(result)) != len(result):
        raise EventFormatError(f"{field_name} must not contain duplicates")
    return result


def normalize_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach one controller-owned event."""

    if not isinstance(raw, Mapping):
        raise EventFormatError("event must be an object")
    event_type = raw.get("type")
    if not isinstance(event_type, str) or event_type not in _EVENT_FIELDS:
        raise EventFormatError(f"unsupported event type: {event_type!r}")
    required, optional = _EVENT_FIELDS[event_type]
    missing = required - set(raw)
    extra = set(raw) - required - optional
    if missing or extra:
        raise EventFormatError(
            f"invalid {event_type} fields: missing={sorted(missing)} extra={sorted(extra)}"
        )

    event = {"type": event_type, "cost": _cost(raw.get("cost", 0.0))}
    if event_type == "propose_hypothesis":
        event.update(
            hypothesis_id=_identifier(raw["hypothesis_id"], "hypothesis_id"),
            claim=_mapping(raw["claim"], "claim", nonempty=True),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
        )
    elif event_type == "record_probe":
        try:
            verdict = ProbeVerdict(raw["verdict"])
        except (TypeError, ValueError) as exc:
            raise EventFormatError(f"invalid probe verdict: {raw['verdict']!r}") from exc
        event.update(
            hypothesis_id=_identifier(raw["hypothesis_id"], "hypothesis_id"),
            probe_id=_identifier(raw["probe_id"], "probe_id"),
            verdict=verdict.value,
            result=_mapping(raw["result"], "result"),
        )
    elif event_type == "submit_candidate":
        parent_id = raw.get("parent_id")
        event.update(
            candidate_id=_identifier(raw["candidate_id"], "candidate_id"),
            hypothesis_id=_identifier(raw["hypothesis_id"], "hypothesis_id"),
            spec=_mapping(raw["spec"], "spec", nonempty=True),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
            parent_id=None if parent_id is None else _identifier(parent_id, "parent_id"),
            symmetry_evidence_ids=_optional_identifiers(
                raw.get("symmetry_evidence_ids"), "symmetry_evidence_ids"
            ),
        )
    elif event_type == "record_evaluation":
        try:
            stage = EvaluationStage(raw["stage"])
        except (TypeError, ValueError) as exc:
            raise EventFormatError(f"invalid evaluation stage: {raw['stage']!r}") from exc
        if not isinstance(raw["passed"], bool):
            raise EventFormatError("passed must be a boolean")
        event.update(
            candidate_id=_identifier(raw["candidate_id"], "candidate_id"),
            evaluation_id=_identifier(raw["evaluation_id"], "evaluation_id"),
            stage=stage.value,
            passed=raw["passed"],
            metrics=_mapping(raw["metrics"], "metrics"),
        )
    elif event_type == "revise":
        entity = raw["entity"]
        if entity not in {"hypothesis", "candidate"}:
            raise EventFormatError("entity must be hypothesis or candidate")
        event.update(
            entity=entity,
            source_id=_identifier(raw["source_id"], "source_id"),
            new_id=_identifier(raw["new_id"], "new_id"),
            replacement=_mapping(raw["replacement"], "replacement", nonempty=True),
            reason=_text(raw["reason"], "reason"),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
            symmetry_evidence_ids=_optional_identifiers(
                raw.get("symmetry_evidence_ids"), "symmetry_evidence_ids"
            ),
            evidence_ids=_optional_identifiers(
                raw.get("evidence_ids"), "evidence_ids"
            ),
        )
    elif event_type == "retire":
        entity = raw["entity"]
        if entity not in {"hypothesis", "candidate"}:
            raise EventFormatError("entity must be hypothesis or candidate")
        event.update(
            entity=entity,
            entity_id=_identifier(raw["entity_id"], "entity_id"),
            reason=_text(raw["reason"], "reason"),
            evidence_ids=_optional_identifiers(
                raw.get("evidence_ids"), "evidence_ids"
            ),
        )
    elif event_type == "commit":
        event.update(
            candidate_id=_identifier(raw["candidate_id"], "candidate_id"),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
        )
    elif event_type == "close_negative":
        event.update(
            reason=_text(raw["reason"], "reason"),
            evidence_ids=_identifiers(raw["evidence_ids"], "evidence_ids"),
            metadata=_mapping(raw.get("metadata", {}), "metadata"),
        )
    return event


@dataclass(frozen=True)
class HypothesisRecord:
    hypothesis_id: str
    claim: Mapping[str, Any]
    metadata: Mapping[str, Any]
    status: Lifecycle
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
    symmetry_evidence_ids: tuple[str, ...] = ()
    evaluation_ids: tuple[str, ...] = ()
    revised_to: str | None = None
    retired_reason: str | None = None
    disposition_evidence_ids: tuple[str, ...] = ()


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

    def __post_init__(self) -> None:
        self.total_budget = _cost(self.total_budget)
        self.spent_budget = _cost(self.spent_budget)
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
                    "symmetry_evidence_ids": list(item.symmetry_evidence_ids),
                    "evaluation_ids": list(item.evaluation_ids),
                    "revised_to": item.revised_to,
                    "retired_reason": item.retired_reason,
                    "disposition_evidence_ids": list(
                        item.disposition_evidence_ids
                    ),
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


def _initial_hypothesis_status(claim: Mapping[str, Any]) -> Lifecycle:
    if claim.get("kind") in {"ansatz_structure", "null_control"}:
        return Lifecycle.READY
    return Lifecycle.PROPOSED


def _active_candidates(state: ResearchState, hypothesis_id: str) -> list[str]:
    return [
        candidate.candidate_id
        for candidate in state.candidates.values()
        if candidate.hypothesis_id == hypothesis_id and candidate.status not in _TERMINAL
    ]


def _validate_symmetry_evidence(
    state: ResearchState, evidence_ids: Iterable[str]
) -> tuple[str, ...]:
    result = tuple(evidence_ids)
    for evidence_id in result:
        probe = state.probes.get(evidence_id)
        if probe is None or probe.verdict is not ProbeVerdict.SUPPORTED:
            raise TransitionError(
                f"symmetry evidence must cite a supported probe: {evidence_id}"
            )
        hypothesis = state.hypotheses[probe.hypothesis_id]
        if hypothesis.claim.get("kind") != "exact_pauli_symmetry":
            raise TransitionError(
                f"symmetry evidence does not cite an exact symmetry: {evidence_id}"
            )
    return result


def comparison_point(record: EvaluationRecord) -> dict[str, Any] | None:
    """Extract one fixed-promotion evaluator point for fair comparisons."""

    policy = record.metrics.get("resource_policy")
    energy = record.metrics.get("best_energy")
    if (
        record.stage is not EvaluationStage.PROMOTION
        or record.metrics.get("valid") is not True
        or not isinstance(policy, Mapping)
        or policy.get("eligible") is not True
        or isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(float(energy))
        or not isinstance(policy.get("observed"), Mapping)
    ):
        return None
    observed = policy["observed"]
    if any(
        isinstance(observed.get(name), bool)
        or not isinstance(observed.get(name), int)
        or observed[name] < 0
        for name in _COMPARISON_RESOURCE_NAMES
    ):
        return None
    audit = record.metrics.get("audit")
    parameter_count = (
        audit.get("unique_trainable_params") if isinstance(audit, Mapping) else None
    )
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
    ):
        return None
    return {
        "candidate_id": record.candidate_id,
        "evaluation_id": record.evaluation_id,
        "stage": record.stage.value,
        "passed": record.passed,
        "best_energy": float(energy),
        "resources": {
            **{
                name: int(observed[name])
                for name in _COMPARISON_RESOURCE_NAMES
            },
            "unique_trainable_params": int(parameter_count),
        },
    }


def comparison_dominates_target(
    target: Mapping[str, Any], comparator: Mapping[str, Any]
) -> bool:
    """Return whether ``comparator`` dominates ``target`` under the local rule."""

    target_energy = target["best_energy"]
    comparator_energy = comparator["best_energy"]
    if target_energy > comparator_energy + COMMIT_ENERGY_TOLERANCE:
        return True
    if abs(target_energy - comparator_energy) > COMMIT_ENERGY_TOLERANCE:
        return False
    target_resources = target["resources"]
    comparator_resources = comparator["resources"]
    return all(
        target_resources[name] >= comparator_resources[name]
        for name in target_resources
    ) and any(
        target_resources[name] > comparator_resources[name]
        for name in target_resources
    )


def _valid_comparison_evaluation(record: EvaluationRecord) -> bool:
    return comparison_point(record) is not None


def _validate_promoted_disposition(
    state: ResearchState,
    candidate: CandidateRecord,
    evidence_ids: Iterable[str],
) -> tuple[str, ...]:
    evidence = tuple(evidence_ids)
    if candidate.status is not Lifecycle.PROMOTED:
        if evidence:
            raise TransitionError(
                "comparison evidence is only valid when disposing a promoted candidate"
            )
        return ()
    records = [state.evaluations.get(evidence_id) for evidence_id in evidence]
    if any(record is None for record in records):
        raise TransitionError("promoted disposition cites unknown evaluation evidence")
    target = [
        record
        for record in records
        if record is not None
        and record.candidate_id == candidate.candidate_id
        and record.stage is EvaluationStage.PROMOTION
        and record.passed
    ]
    comparators = [
        record
        for record in records
        if record is not None
        and record.candidate_id != candidate.candidate_id
        and state.candidates[record.candidate_id].hypothesis_id
        != candidate.hypothesis_id
        and _valid_comparison_evaluation(record)
    ]
    if len(target) != 1 or not comparators:
        raise TransitionError(
            "promoted disposition requires its passed promotion and a fair "
            "different-hypothesis comparison"
        )
    target_point = comparison_point(target[0])
    if target_point is None:
        raise TransitionError(
            "promoted disposition target lacks valid promotion comparison metrics"
        )
    if not any(
        (point := comparison_point(record)) is not None
        and comparison_dominates_target(target_point, point)
        for record in comparators
    ):
        raise TransitionError(
            "promoted disposition requires a comparator that actually dominates "
            "the target"
        )
    return evidence


def _apply_propose(state: ResearchState, event: Mapping[str, Any]) -> None:
    hypothesis_id = str(event["hypothesis_id"])
    if hypothesis_id in state.hypotheses:
        raise TransitionError(f"hypothesis already exists: {hypothesis_id}")
    claim = dict(event["claim"])
    state.hypotheses[hypothesis_id] = HypothesisRecord(
        hypothesis_id=hypothesis_id,
        claim=copy.deepcopy(claim),
        metadata=copy.deepcopy(dict(event["metadata"])),
        status=_initial_hypothesis_status(claim),
    )


def _apply_probe(state: ResearchState, event: Mapping[str, Any]) -> None:
    hypothesis_id = str(event["hypothesis_id"])
    probe_id = str(event["probe_id"])
    hypothesis = state.hypotheses.get(hypothesis_id)
    if hypothesis is None:
        raise TransitionError(f"unknown hypothesis: {hypothesis_id}")
    if probe_id in state.probes:
        raise TransitionError(f"probe already exists: {probe_id}")
    if hypothesis.status is not Lifecycle.PROPOSED:
        raise TransitionError(
            f"cannot probe hypothesis {hypothesis_id} in {hypothesis.status.value}"
        )
    verdict = ProbeVerdict(event["verdict"])
    next_status = {
        ProbeVerdict.SUPPORTED: Lifecycle.SUPPORTED,
        ProbeVerdict.REFUTED: Lifecycle.REFUTED,
        ProbeVerdict.INCONCLUSIVE: Lifecycle.INCONCLUSIVE,
    }[verdict]
    state.probes[probe_id] = ProbeRecord(
        probe_id=probe_id,
        hypothesis_id=hypothesis_id,
        verdict=verdict,
        result=copy.deepcopy(dict(event["result"])),
        cost=float(event["cost"]),
    )
    state.hypotheses[hypothesis_id] = replace(
        hypothesis,
        status=next_status,
        probe_ids=hypothesis.probe_ids + (probe_id,),
    )


def _apply_submit_candidate(state: ResearchState, event: Mapping[str, Any]) -> None:
    candidate_id = str(event["candidate_id"])
    hypothesis_id = str(event["hypothesis_id"])
    if candidate_id in state.candidates:
        raise TransitionError(f"candidate already exists: {candidate_id}")
    hypothesis = state.hypotheses.get(hypothesis_id)
    if hypothesis is None:
        raise TransitionError(f"unknown hypothesis: {hypothesis_id}")
    if hypothesis.status not in {Lifecycle.READY, Lifecycle.SUPPORTED}:
        raise TransitionError(
            f"candidate requires READY or SUPPORTED hypothesis; {hypothesis_id} is "
            f"{hypothesis.status.value}"
        )
    parent_id = event.get("parent_id")
    if parent_id is not None:
        parent = state.candidates.get(str(parent_id))
        if parent is None or parent.hypothesis_id != hypothesis_id:
            raise TransitionError("candidate parent must exist under the same hypothesis")
    symmetry_evidence_ids = _validate_symmetry_evidence(
        state, event["symmetry_evidence_ids"]
    )
    state.candidates[candidate_id] = CandidateRecord(
        candidate_id=candidate_id,
        hypothesis_id=hypothesis_id,
        spec=copy.deepcopy(dict(event["spec"])),
        metadata=copy.deepcopy(dict(event["metadata"])),
        parent_id=None if parent_id is None else str(parent_id),
        symmetry_evidence_ids=symmetry_evidence_ids,
    )


def _evaluation_transition(candidate: CandidateRecord, stage: EvaluationStage) -> Lifecycle:
    expected = {
        EvaluationStage.AUDIT: (Lifecycle.CANDIDATE, Lifecycle.AUDITED),
        EvaluationStage.SMOKE: (Lifecycle.AUDITED, Lifecycle.SMOKE),
        EvaluationStage.PROMOTION: (Lifecycle.SMOKE, Lifecycle.PROMOTED),
    }[stage]
    if candidate.status is not expected[0]:
        raise TransitionError(
            f"cannot run {stage.value} evaluation for {candidate.candidate_id} "
            f"in {candidate.status.value}"
        )
    return expected[1]


def _apply_evaluation(state: ResearchState, event: Mapping[str, Any]) -> None:
    candidate_id = str(event["candidate_id"])
    evaluation_id = str(event["evaluation_id"])
    candidate = state.candidates.get(candidate_id)
    if candidate is None:
        raise TransitionError(f"unknown candidate: {candidate_id}")
    if evaluation_id in state.evaluations:
        raise TransitionError(f"evaluation already exists: {evaluation_id}")
    hypothesis = state.hypotheses[candidate.hypothesis_id]
    if hypothesis.status not in {Lifecycle.READY, Lifecycle.SUPPORTED}:
        raise TransitionError(
            f"candidate's hypothesis {candidate.hypothesis_id} is no longer active"
        )
    stage = EvaluationStage(event["stage"])
    passed_status = _evaluation_transition(candidate, stage)
    next_status = passed_status if bool(event["passed"]) else Lifecycle.RETIRED
    state.evaluations[evaluation_id] = EvaluationRecord(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        stage=stage,
        passed=bool(event["passed"]),
        metrics=copy.deepcopy(dict(event["metrics"])),
        cost=float(event["cost"]),
        status_before=candidate.status,
        status_after=next_status,
    )
    state.candidates[candidate_id] = replace(
        candidate,
        status=next_status,
        evaluation_ids=candidate.evaluation_ids + (evaluation_id,),
        retired_reason=None if bool(event["passed"]) else f"failed {stage.value}",
    )


def _apply_revise(state: ResearchState, event: Mapping[str, Any]) -> None:
    entity = str(event["entity"])
    source_id = str(event["source_id"])
    new_id = str(event["new_id"])
    reason = str(event["reason"])
    metadata = copy.deepcopy(dict(event["metadata"]))
    if entity == "hypothesis":
        source = state.hypotheses.get(source_id)
        if source is None:
            raise TransitionError(f"unknown hypothesis: {source_id}")
        if new_id in state.hypotheses:
            raise TransitionError(f"hypothesis already exists: {new_id}")
        if source.status is Lifecycle.REVISED:
            raise TransitionError(f"hypothesis was already revised: {source_id}")
        active = _active_candidates(state, source_id)
        if active:
            raise TransitionError(f"retire active candidates before revision: {active}")
        replacement = copy.deepcopy(dict(event["replacement"]))
        state.hypotheses[source_id] = replace(
            source,
            status=Lifecycle.REVISED,
            revised_to=new_id,
        )
        state.hypotheses[new_id] = HypothesisRecord(
            hypothesis_id=new_id,
            claim=replacement,
            metadata={**metadata, "revision_reason": reason},
            status=_initial_hypothesis_status(replacement),
            parent_id=source_id,
        )
        return

    source = state.candidates.get(source_id)
    if source is None:
        raise TransitionError(f"unknown candidate: {source_id}")
    evidence_ids = _validate_promoted_disposition(
        state, source, event["evidence_ids"]
    )
    if source.status is Lifecycle.REVISED:
        raise TransitionError(f"candidate was already revised: {source_id}")
    if new_id in state.candidates:
        raise TransitionError(f"candidate already exists: {new_id}")
    hypothesis = state.hypotheses[source.hypothesis_id]
    if hypothesis.status not in {Lifecycle.READY, Lifecycle.SUPPORTED}:
        raise TransitionError("cannot revise candidate under an inactive hypothesis")
    state.candidates[source_id] = replace(
        source,
        status=Lifecycle.REVISED,
        revised_to=new_id,
        disposition_evidence_ids=evidence_ids,
    )
    state.candidates[new_id] = CandidateRecord(
        candidate_id=new_id,
        hypothesis_id=source.hypothesis_id,
        spec=copy.deepcopy(dict(event["replacement"])),
        metadata={
            **copy.deepcopy(dict(source.metadata)),
            **metadata,
            "revision_reason": reason,
        },
        parent_id=source_id,
        symmetry_evidence_ids=_validate_symmetry_evidence(
            state, event["symmetry_evidence_ids"]
        ),
    )


def _apply_retire(state: ResearchState, event: Mapping[str, Any]) -> None:
    entity = str(event["entity"])
    entity_id = str(event["entity_id"])
    reason = str(event["reason"])
    if entity == "hypothesis":
        item = state.hypotheses.get(entity_id)
        if item is None:
            raise TransitionError(f"unknown hypothesis: {entity_id}")
        if item.status in _TERMINAL:
            raise TransitionError(f"hypothesis is already terminal: {entity_id}")
        active = _active_candidates(state, entity_id)
        if active:
            raise TransitionError(f"retire active candidates first: {active}")
        state.hypotheses[entity_id] = replace(
            item,
            status=Lifecycle.RETIRED,
            retired_reason=reason,
        )
        return
    item = state.candidates.get(entity_id)
    if item is None:
        raise TransitionError(f"unknown candidate: {entity_id}")
    evidence_ids = _validate_promoted_disposition(
        state, item, event["evidence_ids"]
    )
    if item.status in _TERMINAL:
        raise TransitionError(f"candidate is already terminal: {entity_id}")
    state.candidates[entity_id] = replace(
        item,
        status=Lifecycle.RETIRED,
        retired_reason=reason,
        disposition_evidence_ids=evidence_ids,
    )


def _apply_commit(state: ResearchState, event: Mapping[str, Any]) -> None:
    candidate_id = str(event["candidate_id"])
    candidate = state.candidates.get(candidate_id)
    if candidate is None:
        raise TransitionError(f"unknown candidate: {candidate_id}")
    if candidate.status is not Lifecycle.PROMOTED:
        raise TransitionError(
            f"commit requires PROMOTED candidate; {candidate_id} is {candidate.status.value}"
        )
    state.committed_candidate_id = candidate_id
    state.commit_metadata = copy.deepcopy(dict(event["metadata"]))


def _is_valid_numerical_evaluation(record: EvaluationRecord) -> bool:
    objective_calls = record.metrics.get("objective_calls")
    return (
        record.stage in {EvaluationStage.SMOKE, EvaluationStage.PROMOTION}
        and record.metrics.get("valid") is True
        and isinstance(objective_calls, int)
        and not isinstance(objective_calls, bool)
        and objective_calls > 0
    )


def _has_objective_activity(record: EvaluationRecord) -> bool:
    span = record.metrics.get("objective_energy_span")
    active_norm = record.metrics.get("hamiltonian_active_norm")
    fraction = record.metrics.get("objective_activity_fraction")
    constant = record.metrics.get("constant_hamiltonian")
    if (
        isinstance(span, bool)
        or not isinstance(span, (int, float))
        or not math.isfinite(float(span))
        or float(span) < 0.0
        or isinstance(active_norm, bool)
        or not isinstance(active_norm, (int, float))
        or not math.isfinite(float(active_norm))
        or float(active_norm) < 0.0
        or not isinstance(constant, bool)
    ):
        return False
    span_value = float(span)
    norm_value = float(active_norm)
    if constant:
        return norm_value == 0.0 and fraction is None
    if (
        norm_value <= 0.0
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or float(fraction) < 0.0
    ):
        return False
    fraction_value = float(fraction)
    return math.isclose(
        fraction_value,
        span_value / norm_value,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) and fraction_value >= MIN_NEGATIVE_OBJECTIVE_ACTIVITY_FRACTION


def _is_numerical_falsification(record: EvaluationRecord) -> bool:
    return (
        _is_valid_numerical_evaluation(record)
        and not record.passed
        and _has_objective_activity(record)
    )


def _candidate_has_objective_activity(
    state: ResearchState, candidate_id: str
) -> bool:
    return any(
        evaluation.candidate_id == candidate_id
        and _is_valid_numerical_evaluation(evaluation)
        and _has_objective_activity(evaluation)
        for evaluation in state.evaluations.values()
    )


def _hypothesis_lineage_root(state: ResearchState, hypothesis_id: str) -> str:
    current = state.hypotheses[hypothesis_id]
    seen: set[str] = set()
    while current.parent_id is not None:
        if current.hypothesis_id in seen:
            raise TransitionError("hypothesis revision lineage contains a cycle")
        seen.add(current.hypothesis_id)
        parent = state.hypotheses.get(current.parent_id)
        if parent is None:
            raise TransitionError("hypothesis revision lineage has a missing parent")
        current = parent
    return current.hypothesis_id


def derived_negative_close_evidence(state: ResearchState) -> tuple[str, ...]:
    """Return the evaluator-owned evidence eligible for negative closure."""

    evidence_ids = {
        probe.probe_id
        for probe in state.probes.values()
        if probe.verdict is ProbeVerdict.REFUTED
    }
    evidence_ids.update(
        evaluation.evaluation_id
        for evaluation in state.evaluations.values()
        if _is_numerical_falsification(evaluation)
    )
    for candidate in state.candidates.values():
        evidence_ids.update(candidate.disposition_evidence_ids)
    return tuple(sorted(evidence_ids))


def validate_negative_close_coverage(
    state: ResearchState,
    evidence_ids: Iterable[str],
) -> dict[str, Any]:
    """Validate fixed search breadth and return its replayable summary."""

    live_hypotheses = sorted(
        item.hypothesis_id
        for item in state.hypotheses.values()
        if item.status not in _TERMINAL
    )
    live_candidates = sorted(
        item.candidate_id
        for item in state.candidates.values()
        if item.status not in _TERMINAL
    )
    if live_hypotheses or live_candidates:
        raise TransitionError(
            "negative close requires terminal branches; "
            f"live_hypotheses={live_hypotheses} live_candidates={live_candidates}"
        )

    cited = set(evidence_ids)
    for evidence_id in cited:
        if (evidence_id in state.probes) == (evidence_id in state.evaluations):
            raise TransitionError(f"unknown or ambiguous evidence ID: {evidence_id}")

    substantive_lineages = {
        _hypothesis_lineage_root(state, hypothesis.hypothesis_id)
        for hypothesis in state.hypotheses.values()
        if hypothesis.claim.get("kind") != "null_control"
    }
    if not substantive_lineages:
        raise TransitionError(
            "negative close requires a tested non-control hypothesis"
        )

    failed_evaluations = [
        evaluation
        for evaluation in state.evaluations.values()
        if evaluation.evaluation_id in cited
        and _is_numerical_falsification(evaluation)
    ]
    dominated_candidates = [
        candidate
        for candidate in state.candidates.values()
        if candidate.disposition_evidence_ids
        and set(candidate.disposition_evidence_ids) <= cited
        and _candidate_has_objective_activity(state, candidate.candidate_id)
    ]

    covered_lineages = {
        _hypothesis_lineage_root(
            state, state.probes[evidence_id].hypothesis_id
        )
        for evidence_id in cited
        if evidence_id in state.probes
        and state.probes[evidence_id].verdict is ProbeVerdict.REFUTED
    }
    covered_lineages.update(
        _hypothesis_lineage_root(
            state,
            state.candidates[evaluation.candidate_id].hypothesis_id,
        )
        for evaluation in failed_evaluations
    )
    covered_lineages.update(
        _hypothesis_lineage_root(state, candidate.hypothesis_id)
        for candidate in dominated_candidates
    )
    covered_lineages &= substantive_lineages
    uncovered = sorted(substantive_lineages - covered_lineages)
    if uncovered:
        flat_lineages = {
            _hypothesis_lineage_root(
                state,
                state.candidates[evaluation.candidate_id].hypothesis_id,
            )
            for evaluation in state.evaluations.values()
            if _is_valid_numerical_evaluation(evaluation)
            and not evaluation.passed
            and not _has_objective_activity(evaluation)
        }
        flat_uncovered = sorted(set(uncovered) & flat_lineages)
        if flat_uncovered:
            raise TransitionError(
                "negative close does not count flat or unverified objective activity "
                f"for lineages: {flat_uncovered}"
            )
        raise TransitionError(
            f"negative close lacks evidence for lineages: {uncovered}"
        )

    numerical_candidate_ids = {
        evaluation.candidate_id for evaluation in failed_evaluations
    }
    numerical_candidate_ids.update(
        candidate.candidate_id for candidate in dominated_candidates
    )
    numerical_candidate_ids = {
        candidate_id
        for candidate_id in numerical_candidate_ids
        if state.hypotheses[
            state.candidates[candidate_id].hypothesis_id
        ].claim.get("kind")
        != "null_control"
    }
    structure_candidate_ids = {
        candidate_id
        for candidate_id in numerical_candidate_ids
        if state.hypotheses[
            state.candidates[candidate_id].hypothesis_id
        ].claim.get("kind")
        == "ansatz_structure"
    }
    structure_lineages = {
        _hypothesis_lineage_root(
            state, state.candidates[candidate_id].hypothesis_id
        )
        for candidate_id in structure_candidate_ids
    }
    promotion_candidate_ids = {
        evaluation.candidate_id
        for evaluation in failed_evaluations
        if evaluation.stage is EvaluationStage.PROMOTION
        and evaluation.candidate_id in numerical_candidate_ids
    }
    promotion_candidate_ids.update(
        candidate.candidate_id
        for candidate in dominated_candidates
        if candidate.candidate_id in numerical_candidate_ids
    )
    feedback_revision_ids = {
        candidate_id
        for candidate_id in numerical_candidate_ids
        if state.candidates[candidate_id].parent_id in numerical_candidate_ids
    }
    if (
        not promotion_candidate_ids
        and len(structure_lineages) < MIN_NEGATIVE_STRUCTURE_LINEAGES
    ):
        raise TransitionError(
            "negative close requires objective-active numerical failures across at "
            f"least {MIN_NEGATIVE_STRUCTURE_LINEAGES} independent ansatz_structure "
            "lineages, or objective-active promotion-depth evidence; "
            f"found {len(structure_lineages)} structure lineages"
        )

    return {
        "covered_lineages": sorted(covered_lineages),
        "feedback_revision_ids": sorted(feedback_revision_ids),
        "numerical_candidate_ids": sorted(numerical_candidate_ids),
        "promotion_candidate_ids": sorted(promotion_candidate_ids),
        "search_mode": (
            "promotion_depth"
            if promotion_candidate_ids
            else "structural_breadth"
        ),
        "structure_lineage_ids": sorted(structure_lineages),
    }


def _apply_close_negative(state: ResearchState, event: Mapping[str, Any]) -> None:
    evidence_ids = tuple(event["evidence_ids"])
    coverage = validate_negative_close_coverage(state, evidence_ids)
    metadata = copy.deepcopy(dict(event["metadata"]))
    if "coverage" in metadata and metadata["coverage"] != coverage:
        raise TransitionError("negative close coverage summary does not match state")
    state.negative_closed = True
    state.negative_close_reason = str(event["reason"])
    state.negative_close_evidence_ids = evidence_ids
    state.negative_close_metadata = metadata


_APPLIERS = {
    "propose_hypothesis": _apply_propose,
    "record_probe": _apply_probe,
    "submit_candidate": _apply_submit_candidate,
    "record_evaluation": _apply_evaluation,
    "revise": _apply_revise,
    "retire": _apply_retire,
    "commit": _apply_commit,
    "close_negative": _apply_close_negative,
}


def apply_event(state: ResearchState, event_like: Mapping[str, Any]) -> ResearchState:
    """Apply one normalized controller event without mutating ``state``."""

    event = normalize_event(event_like)
    if state.terminal:
        raise TransitionError("research run is terminal; no further events are allowed")
    projected = state.spent_budget + float(event["cost"])
    if projected > state.total_budget + 1e-12:
        raise BudgetExceeded(
            f"event costs {event['cost']}, remaining budget is {state.remaining_budget}"
        )
    next_state = copy.deepcopy(state)
    _APPLIERS[str(event["type"])](next_state, event)
    next_state.spent_budget = projected
    return next_state


def _event_mapping(event: RunEvent) -> dict[str, Any]:
    payload = _json(event.payload, "payload")
    return {
        "type": event.event_type,
        **payload,
        "cost": event.cost,
    }


def replay_events(events: Iterable[RunEvent], *, total_budget: float) -> ResearchState:
    state = ResearchState(total_budget=total_budget)
    for expected_seq, event in enumerate(events):
        if event.seq != expected_seq:
            raise HistoryIntegrityError(
                f"expected history seq {expected_seq}, got {event.seq}"
            )
        state = apply_event(state, _event_mapping(event))
        state.last_seq = event.seq
    return state


def replay_history(history: JsonlRunHistory, *, total_budget: float) -> ResearchState:
    return replay_events(history.read_events(), total_budget=total_budget)


class ResearchLoop:
    """Append normalized controller events and replay branch state."""

    def __init__(
        self,
        history: JsonlRunHistory | str | Path,
        *,
        total_budget: float,
    ):
        self.history = (
            history if isinstance(history, JsonlRunHistory) else JsonlRunHistory(history)
        )
        self._state = replay_history(self.history, total_budget=total_budget)

    @property
    def state(self) -> ResearchState:
        return copy.deepcopy(self._state)

    def dispatch(self, event_like: Mapping[str, Any]) -> RunEvent:
        event = normalize_event(event_like)
        current_state = replay_history(
            self.history,
            total_budget=self._state.total_budget,
        )
        if current_state.to_dict() != self._state.to_dict():
            raise HistoryIntegrityError(
                "research history changed since this ResearchLoop was opened"
            )
        next_state = apply_event(self._state, event)
        expected_seq = self._state.last_seq + 1
        payload = {
            key: value for key, value in event.items() if key not in {"type", "cost"}
        }
        recorded = self.history.append(
            str(event["type"]),
            payload,
            cost=float(event["cost"]),
            expected_seq=expected_seq,
        )
        next_state.last_seq = recorded.seq
        self._state = next_state
        return recorded

    def run_script(self, events: Iterable[Mapping[str, Any]]) -> ResearchState:
        for event in events:
            self.dispatch(event)
        return self.state


__all__ = [
    "BudgetExceeded",
    "CandidateRecord",
    "COMMIT_ENERGY_TOLERANCE",
    "MIN_NEGATIVE_OBJECTIVE_ACTIVITY_FRACTION",
    "MIN_NEGATIVE_STRUCTURE_LINEAGES",
    "EvaluationRecord",
    "EvaluationStage",
    "EventFormatError",
    "HypothesisRecord",
    "Lifecycle",
    "ProbeRecord",
    "ProbeVerdict",
    "ResearchError",
    "ResearchLoop",
    "ResearchState",
    "TransitionError",
    "apply_event",
    "comparison_dominates_target",
    "comparison_point",
    "derived_negative_close_evidence",
    "normalize_event",
    "replay_events",
    "replay_history",
    "validate_negative_close_coverage",
]
