"""Lean closed research loop for AutoVQE.

The external agent supplies hypotheses and typed ansatz specifications.  This
module owns probes, evaluations, resource measurements, stage progression and
terminal decisions. One append-only JSONL stream is the complete run state;
there is no separate history or controller event model.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .ansatz import AnsatzIRValidationError, AnsatzSpec, pauli_label
from .evaluator import (
    EvaluationProtocol,
    audit_public_candidate,
    candidate_identity,
    evaluate_public_problem,
)
from .problem import (
    PublicProblem,
    canonical_data,
    decode_json_object,
    load_problem_document,
    observe_problem,
)
from .probes import (
    EXACT_SYMMETRY_TOLERANCE,
    ProbeValidationError,
    generator_from_recipe,
    initial_state_circuit,
    initial_state_moments,
    operation_symmetry_residuals,
    run_public_probe,
    validate_special_operation_relevance,
)


RUN_FILE = "run.json"
PROBLEM_FILE = "problem.json"
HISTORY_FILE = "events.jsonl"
RUN_SCHEMA_VERSION = 2

MAX_BUDGET = 100.0
MAX_EVENTS = 200
MAX_EXTERNAL_ACTION_BYTES = 1_000_000
MAX_ACTIVE_HYPOTHESES = 3
MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS = 2
MAX_CANDIDATE_OPERATIONS = 256
MAX_CANDIDATE_PARAMETERS = 128
MAX_PARAMETER_FANOUT = 64
RESOURCE_LIMITS = {
    "twoq_count": 512,
    "total_gate_count": 2048,
    "depth": 1024,
}
EVALUATION_LIFECYCLE = (
    ("audit", "AUDITED"),
    ("smoke", "SMOKE"),
    ("promotion", "PROMOTED"),
)
PASSED_STAGE_STATUS = dict(EVALUATION_LIFECYCLE)
EVALUATION_PROTOCOLS = {
    "smoke": (2.0, EvaluationProtocol(max_evals=32, restarts=1, seed=7)),
    "promotion": (6.0, EvaluationProtocol(max_evals=96, restarts=3, seed=997)),
}
COMMIT_ENERGY_TOLERANCE = 5e-4
MIN_ENERGY_IMPROVEMENT = 1e-6

TERMINAL_STATUSES = {"REVISED", "RETIRED"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NEW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
class ResearchError(RuntimeError):
    """Raised for an invalid run, action, or lifecycle transition."""


def _response(
    result: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "result": dict(result),
        "state_summary": _compact_state(state),
    }


def _finite(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0) or number < 0:
        qualifier = "positive " if positive else "non-negative "
        raise ResearchError(f"{field} must be a finite {qualifier}number")
    return number


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ResearchError(f"{field} must match {_ID.pattern!r}")
    return value


def _new_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _NEW_ID.fullmatch(value):
        raise ResearchError(f"{field} must match {_NEW_ID.pattern!r}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchError(f"{field} must be an object")
    try:
        result = copy.deepcopy(dict(value))
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResearchError(f"{field} must contain finite JSON data: {exc}") from exc
    return result


def _strict(
    action: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(action)
    extra = set(action) - required - optional
    if missing or extra:
        raise ResearchError(
            f"invalid external action fields: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )


def _preregistered(record: Mapping[str, Any]) -> bool:
    return any(
        isinstance(record.get(key), str) and record[key].strip()
        for key in ("prediction", "falsifier")
    )


def _family_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchError(f"required file is missing: {path}")
    try:
        return decode_json_object(path.read_text(encoding="utf-8-sig"), path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json(value) + "\n", encoding="utf-8")


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file():
        raise ResearchError(f"history path is not a file: {path}")
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ResearchError(f"blank history line at {line_number}")
        try:
            event = decode_json_object(line, f"{path}:{line_number}")
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc
        if set(event) != {"seq", "type", "cost", "payload"}:
            raise ResearchError(f"invalid event fields at history line {line_number}")
        if event["seq"] != len(events):
            raise ResearchError(
                f"non-contiguous history at line {line_number}: {event['seq']}"
            )
        _text(event["type"], "event type")
        _finite(event["cost"], "event cost")
        _mapping(event["payload"], "event payload")
        events.append(event)
    return events


def _initial_state(total_budget: float) -> dict[str, Any]:
    return {
        "total_budget": total_budget,
        "spent_budget": 0.0,
        "last_seq": -1,
        "hypotheses": {},
        "probes": {},
        "candidates": {},
        "evaluations": {},
        "terminal": None,
    }


def _hypothesis_record(
    payload: Mapping[str, Any], parent_id: str | None
) -> dict[str, Any]:
    return {
        "family": payload["family"],
        "prediction": payload.get("prediction"),
        "falsifier": payload.get("falsifier"),
        "status": "READY",
        "parent_id": parent_id,
    }


def _record_evaluation(
    state: dict[str, Any], candidate_id: str, evaluation: Mapping[str, Any]
) -> str:
    stage = evaluation["stage"]
    if stage not in PASSED_STAGE_STATUS:
        raise ResearchError(f"unsupported evaluation stage: {stage!r}")
    state["evaluations"][evaluation["evaluation_id"]] = {
        "candidate_id": candidate_id,
        "stage": stage,
        "passed": evaluation["passed"],
        "metrics": evaluation["metrics"],
    }
    return PASSED_STAGE_STATUS[stage] if evaluation["passed"] else "RETIRED"


def _apply_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    if state["terminal"] is not None:
        raise ResearchError("history continues after a terminal decision")
    cost = _finite(event["cost"], "event cost")
    if state["spent_budget"] + cost > state["total_budget"] + 1e-12:
        raise ResearchError("history exceeds its research budget")
    payload = _mapping(event["payload"], "event payload")
    kind = event["type"]
    hypotheses = state["hypotheses"]
    candidates = state["candidates"]

    if kind == "propose_hypothesis":
        hypotheses[payload["hypothesis_id"]] = _hypothesis_record(payload, None)
    elif kind == "record_symmetry_probe":
        state["probes"][payload["probe_id"]] = {
            "generator": payload["generator"],
            "verdict": payload["verdict"],
            "result": payload["result"],
        }
    elif kind == "submit_candidate":
        audit = payload["audit"]
        candidate_id = payload["candidate_id"]
        candidates[candidate_id] = {
            "hypothesis_id": payload["hypothesis_id"],
            "spec": payload["spec"],
            "status": _record_evaluation(
                state,
                candidate_id,
                {**audit, "stage": "audit"},
            ),
            "symmetry_evidence_ids": payload.get("symmetry_evidence_ids", []),
        }
    elif kind == "record_evaluation":
        candidate = candidates[payload["candidate_id"]]
        candidate["status"] = _record_evaluation(
            state, payload["candidate_id"], payload
        )
    elif kind == "revise_hypothesis":
        source_id, new_id = payload["source_id"], payload["new_id"]
        source = hypotheses[source_id]
        source["status"] = "REVISED"
        hypotheses[new_id] = _hypothesis_record(payload, source_id)
    elif kind in {"retire_hypothesis", "retire_candidate"}:
        collection = hypotheses if kind == "retire_hypothesis" else candidates
        collection[payload["entity_id"]]["status"] = "RETIRED"
    elif kind == "commit":
        state["terminal"] = {"decision": "positive_commit", **payload}
    elif kind == "close_negative":
        state["terminal"] = {"decision": "negative_close", **payload}
    else:
        raise ResearchError(f"unsupported event type in history: {kind!r}")

    state["spent_budget"] += cost
    state["last_seq"] = int(event["seq"])


def _replay(run_dir: Path, total_budget: float) -> dict[str, Any]:
    state = _initial_state(total_budget)
    for event in _read_events(run_dir / HISTORY_FILE):
        _apply_event(state, event)
    return state


def _append(
    run_dir: Path,
    state: Mapping[str, Any],
    event_type: str,
    payload: Mapping[str, Any],
    cost: float,
) -> dict[str, Any]:
    path = run_dir / HISTORY_FILE
    events = _read_events(path)
    if len(events) != state["last_seq"] + 1:
        raise ResearchError("history changed before append")
    record = {
        "seq": len(events),
        "type": _text(event_type, "event type"),
        "cost": _finite(cost, "event cost"),
        "payload": _mapping(payload, "event payload"),
    }
    projected = copy.deepcopy(dict(state))
    _apply_event(projected, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return projected


def _next_stage(status: str) -> str | None:
    for index, (_, reached_status) in enumerate(EVALUATION_LIFECYCLE[:-1]):
        if reached_status == status:
            return EVALUATION_LIFECYCLE[index + 1][0]
    return None


def _candidate_evaluations(
    state: Mapping[str, Any], candidate_id: str
) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (evaluation_id, evaluation)
        for evaluation_id, evaluation in state["evaluations"].items()
        if evaluation["candidate_id"] == candidate_id
    ]


def _resource_violations(resources: Mapping[str, Any]) -> list[str]:
    return [
        f"{name}={resources.get(name)!r} exceeds or violates limit {limit}"
        for name, limit in RESOURCE_LIMITS.items()
        if isinstance(resources.get(name), bool)
        or not isinstance(resources.get(name), int)
        or resources[name] < 0
        or resources[name] > limit
    ]


def _comparison_point(
    state: Mapping[str, Any], evaluation_id: str, evaluation: Mapping[str, Any]
) -> dict[str, Any] | None:
    metrics = evaluation["metrics"]
    energy = metrics.get("best_energy")
    resources = metrics.get("resources")
    if (
        evaluation["stage"] != "promotion"
        or evaluation["passed"] is not True
        or metrics.get("valid") is not True
        or isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(float(energy))
        or not isinstance(resources, Mapping)
        or _resource_violations(resources)
    ):
        return None
    candidate = state["candidates"][evaluation["candidate_id"]]
    return {
        "candidate_id": evaluation["candidate_id"],
        "root_id": _structure_root(state, candidate["hypothesis_id"]),
        "evaluation_id": evaluation_id,
        "best_energy": float(energy),
        "resources": dict(resources),
    }


def _structure_root(state: Mapping[str, Any], hypothesis_id: str) -> str | None:
    hypothesis = state["hypotheses"].get(hypothesis_id)
    if hypothesis is None:
        return None
    root = hypothesis_id
    parent_id = hypothesis.get("parent_id")
    while parent_id is not None:
        parent = state["hypotheses"].get(parent_id)
        if parent is None:
            break
        root = parent_id
        parent_id = parent.get("parent_id")
    return root


def _promotion_points(state: Mapping[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    points = []
    for evaluation_id, evaluation in state["evaluations"].items():
        if evaluation["candidate_id"] == candidate_id:
            continue
        point = _comparison_point(state, evaluation_id, evaluation)
        if point is not None:
            points.append(point)
    return sorted(points, key=lambda item: item["candidate_id"])


def _comparators(
    state: Mapping[str, Any],
    candidate_id: str,
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_id = state["candidates"][candidate_id]["hypothesis_id"]
    primary_root = _structure_root(state, primary_id)
    if primary_root is None:
        return []
    return [
        point
        for point in points
        if point["root_id"] != primary_root
    ]


def _dominates(target: Mapping[str, Any], comparator: Mapping[str, Any]) -> bool:
    if target["best_energy"] > comparator["best_energy"] + COMMIT_ENERGY_TOLERANCE:
        return True
    if abs(target["best_energy"] - comparator["best_energy"]) > COMMIT_ENERGY_TOLERANCE:
        return False
    return all(
        target["resources"][name] >= comparator["resources"][name]
        for name in target["resources"]
    ) and any(
        target["resources"][name] > comparator["resources"][name]
        for name in target["resources"]
    )


def _positive_decision(state: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = state["candidates"].get(candidate_id)
    if candidate is None or candidate["status"] != "PROMOTED":
        raise ResearchError("commit requires a passed promotion")
    if not _preregistered(state["hypotheses"][candidate["hypothesis_id"]]):
        raise ResearchError("commit requires a preregistered prediction or falsifier")
    promotions = [
        item
        for item in _candidate_evaluations(state, candidate_id)
        if item[1]["stage"] == "promotion" and item[1]["passed"]
    ]
    if len(promotions) != 1:
        raise ResearchError("commit requires exactly one passed promotion")
    target = _comparison_point(state, *promotions[0])
    comparisons = _promotion_points(state, candidate_id)
    fair = _comparators(state, candidate_id, comparisons)
    if target is None or not fair:
        raise ResearchError("commit requires evaluator-owned target and comparator evidence")
    dominator = next((item for item in comparisons if _dominates(target, item)), None)
    if dominator:
        raise ResearchError(f"target is dominated by comparator {dominator['candidate_id']}")
    evidence_ids = [promotions[0][0], *(item["evaluation_id"] for item in comparisons)]
    return {
        "candidate_id": candidate_id,
        "promotion_evaluation_id": promotions[0][0],
        "evidence_ids": evidence_ids,
        "comparison": {
            "energy_tolerance": COMMIT_ENERGY_TOLERANCE,
            "target": target,
            "evaluations": comparisons,
            "fair_comparator_evaluation_ids": [item["evaluation_id"] for item in fair],
        },
    }


def _negative_decision(state: Mapping[str, Any]) -> dict[str, Any]:
    live = [
        key
        for collection in (state["hypotheses"], state["candidates"])
        for key, value in collection.items()
        if value["status"] not in TERMINAL_STATUSES
    ]
    if live:
        raise ResearchError(f"negative close requires terminal branches: {live}")
    failed = [
        (key, value)
        for key, value in state["evaluations"].items()
        if value["stage"] in {"smoke", "promotion"}
        and not value["passed"]
        and value["metrics"].get("objective_calls", 0) > 0
        and value["metrics"].get("objective_activity_fraction", 0.0) >= 1e-6
    ]
    roots = {
        _structure_root(
            state,
            state["candidates"][value["candidate_id"]]["hypothesis_id"],
        )
        for _, value in failed
    }
    promotion_depth = any(value["stage"] == "promotion" for _, value in failed)
    if not promotion_depth and len(roots) < 2:
        raise ResearchError(
            "negative close needs an objective-active promotion failure or two "
            "independent failed structure lineages"
        )
    return {
        "evidence_ids": [key for key, _ in failed],
        "coverage": {
            "search_mode": "promotion_depth" if promotion_depth else "structural_breadth",
            "structure_lineage_ids": sorted(root for root in roots if root is not None),
        },
    }


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    hypotheses = {
        hypothesis_id: {
            "family": record["family"],
            "status": record["status"],
            "next_action": "submit_candidate" if record["status"] == "READY" else None,
            **({"parent_id": record["parent_id"]} if record.get("parent_id") else {}),
        }
        for hypothesis_id, record in sorted(state["hypotheses"].items())
    }
    candidates: dict[str, Any] = {}
    for candidate_id, record in sorted(state["candidates"].items()):
        status = record["status"]
        if status == "PROMOTED":
            next_action = (
                "commit_or_dispose_after_comparison"
                if _comparators(
                    state, candidate_id, _promotion_points(state, candidate_id)
                )
                else "evaluate_different_hypothesis:promotion"
            )
        else:
            stage = _next_stage(status)
            next_action = f"evaluate_candidate:{stage}" if stage else None
        evaluations = _candidate_evaluations(state, candidate_id)
        summary = {
            "hypothesis_id": record["hypothesis_id"],
            "status": status,
            "next_action": next_action,
            **(
                {
                    "latest_evaluation": {
                        "evaluation_id": evaluations[-1][0],
                        "stage": evaluations[-1][1]["stage"],
                        "passed": evaluations[-1][1]["passed"],
                        **evaluations[-1][1]["metrics"],
                    }
                }
                if evaluations
                else {}
            ),
        }
        if record["symmetry_evidence_ids"]:
            summary["symmetry_evidence_ids"] = list(record["symmetry_evidence_ids"])
        candidates[candidate_id] = summary
    return {
        "terminal_decision": state["terminal"]["decision"] if state["terminal"] else None,
        "budget": {
            "spent": state["spent_budget"],
            "remaining": state["total_budget"] - state["spent_budget"],
            "total": state["total_budget"],
        },
        "hypotheses": hypotheses,
        "symmetry_probes": {
            probe_id: {
                "verdict": record["verdict"],
                "metrics": record["result"]["metrics"],
            }
            for probe_id, record in sorted(state["probes"].items())
        },
        "candidates": candidates,
    }


class ResearchController:
    """Single-layer action validator, evaluator, and event writer."""

    def __init__(self, problem: PublicProblem, run_dir: str | Path, *, total_budget: float):
        self.problem = problem
        self.run_dir = Path(run_dir)
        self.total_budget = total_budget

    @property
    def state(self) -> dict[str, Any]:
        return _replay(self.run_dir, self.total_budget)

    def _capacity(self, state: Mapping[str, Any], cost: float, *, events: int = 1, terminal: bool = False) -> None:
        reserve = 0 if terminal else 1
        if state["last_seq"] + 1 + events + reserve > MAX_EVENTS:
            raise ResearchError(f"research run reached {MAX_EVENTS} event cap")
        if state["spent_budget"] + cost > self.total_budget + 1e-12:
            raise ResearchError(
                f"action costs {cost}, remaining budget is "
                f"{self.total_budget - state['spent_budget']}"
            )

    @staticmethod
    def _hypothesis_fields(action: Mapping[str, Any]) -> dict[str, Any]:
        fields = {
            "family": _text(action["family"], "family"),
            **{
                key: _text(action[key], key)
                for key in ("prediction", "falsifier")
                if key in action
            },
        }
        if not _preregistered(fields):
            raise ResearchError(
                "structure hypothesis must preregister a prediction or falsifier"
            )
        return fields

    def _unique_structure_family(
        self,
        state: Mapping[str, Any],
        family: str,
        *,
        allowed_root: str | None = None,
    ) -> None:
        family_key = _family_key(family)
        duplicates = [
            hypothesis_id
            for hypothesis_id, record in state["hypotheses"].items()
            if _family_key(str(record["family"])) == family_key
            and (
                allowed_root is None
                or _structure_root(state, hypothesis_id) != allowed_root
            )
        ]
        if duplicates:
            raise ResearchError(
                f"ansatz_structure family duplicates existing hypothesis: {duplicates}"
            )

    def _propose(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(
            action,
            required={"type", "hypothesis_id", "family"},
            optional={"prediction", "falsifier"},
        )
        state = self.state
        hypothesis_id = _new_identifier(action["hypothesis_id"], "hypothesis_id")
        if hypothesis_id in state["hypotheses"]:
            raise ResearchError(f"hypothesis already exists: {hypothesis_id}")
        active = sum(
            record["status"] not in TERMINAL_STATUSES
            for record in state["hypotheses"].values()
        )
        if active >= MAX_ACTIVE_HYPOTHESES:
            raise ResearchError(f"at most {MAX_ACTIVE_HYPOTHESES} hypotheses may be active")
        fields = self._hypothesis_fields(action)
        self._unique_structure_family(state, fields["family"])
        self._capacity(state, 0.1)
        new_state = _append(self.run_dir,
            state,
            "propose_hypothesis",
            {"hypothesis_id": hypothesis_id, **fields},
            0.1,
        )
        return _response(
            {"accepted": True, "hypothesis_id": hypothesis_id},
            new_state,
        )

    def _probe(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(
            action,
            required={"type", "probe_id", "generator"},
            optional=set(),
        )
        state = self.state
        probe_id = _new_identifier(action["probe_id"], "probe_id")
        if probe_id in state["probes"]:
            raise ResearchError(f"symmetry probe already exists: {probe_id}")
        recipe = _mapping(action["generator"], "generator")
        if any(record["generator"] == recipe for record in state["probes"].values()):
            raise ResearchError("this symmetry generator was already probed")
        request = {"type": "normalized_commutator", "generator": recipe}
        try:
            measured = run_public_probe(self.problem, request)
            cost = measured.cost_units
            self._capacity(state, cost)
        except Exception as exc:
            raise ResearchError(f"probe failed: {exc}") from exc
        passed = measured.probe_type == "normalized_commutator" and bool(
            measured.metrics.get("exact", False)
        )
        payload = {
            "probe_id": probe_id,
            "generator": recipe,
            "verdict": "supported" if passed else "refuted",
            "result": measured.to_dict(),
        }
        new_state = _append(self.run_dir, state, "record_symmetry_probe", payload, cost)
        return _response(
            {**measured.to_dict(), "probe_id": probe_id, "passed": passed},
            new_state,
        )

    def _symmetry_evidence(self, state: Mapping[str, Any], raw: Any) -> list[str]:
        if raw is None:
            evidence: list[Any] = []
        elif isinstance(raw, list):
            evidence = raw
        else:
            raise ResearchError("symmetry_evidence_ids must be a list")
        values = [_identifier(item, "symmetry_evidence_ids") for item in evidence]
        result = sorted(set(values))
        for evidence_id in result:
            probe = state["probes"].get(evidence_id)
            if probe is None or probe["verdict"] != "supported":
                raise ResearchError(f"symmetry evidence must cite a supported probe: {evidence_id}")
        return result

    def _unique_candidate(
        self,
        state: Mapping[str, Any],
        spec: Mapping[str, Any],
    ) -> None:
        try:
            identity = candidate_identity(spec)
        except Exception as exc:
            raise ResearchError(f"invalid candidate: {exc}") from exc
        duplicates = [
            candidate_id
            for candidate_id, record in state["candidates"].items()
            if candidate_identity(record["spec"]) == identity
        ]
        if duplicates:
            raise ResearchError(f"candidate is semantically equivalent to existing {duplicates}")

    def _submit(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(
            action,
            required={"type", "candidate_id", "hypothesis_id", "spec"},
            optional={"symmetry_evidence_ids"},
        )
        state = self.state
        candidate_id = _new_identifier(action["candidate_id"], "candidate_id")
        hypothesis_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        if candidate_id in state["candidates"]:
            raise ResearchError(f"candidate already exists: {candidate_id}")
        hypothesis = state["hypotheses"].get(hypothesis_id)
        if hypothesis is None or hypothesis["status"] != "READY":
            raise ResearchError("candidate requires a READY structure hypothesis")
        try:
            spec = AnsatzSpec.from_dict(_mapping(action["spec"], "spec")).to_dict()
        except Exception as exc:
            raise ResearchError(f"invalid candidate: {exc}") from exc
        self._unique_candidate(state, spec)
        evidence = self._symmetry_evidence(state, action.get("symmetry_evidence_ids"))
        active = sum(
            record["hypothesis_id"] == hypothesis_id
            and record["status"] not in TERMINAL_STATUSES
            for record in state["candidates"].values()
        )
        if active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
            raise ResearchError("too many active candidates under this hypothesis")
        candidate_payload = {
            "candidate_id": candidate_id,
            "hypothesis_id": hypothesis_id,
            "spec": spec,
            "symmetry_evidence_ids": evidence,
        }
        self._capacity(state, 0.35)
        passed, metrics = self._audit(state, spec, evidence)
        evaluation_id = f"evaluation:{candidate_id}:audit"
        candidate_payload["audit"] = {
            "evaluation_id": evaluation_id,
            "passed": bool(passed),
            "metrics": metrics,
        }
        new_state = _append(self.run_dir, state, "submit_candidate", candidate_payload, 0.35)
        return _response(
            {
                "accepted": True,
                "candidate_id": candidate_id,
                "audit_evaluation_id": evaluation_id,
                "audit_passed": bool(passed),
                **metrics,
            },
            new_state,
        )

    def _audit(
        self,
        state: Mapping[str, Any],
        spec: Mapping[str, Any],
        evidence_ids: list[str],
    ) -> tuple[bool, dict[str, Any]]:
        try:
            parsed = AnsatzSpec.from_dict(spec)
            if parsed.num_qubits != self.problem.num_qubits:
                raise ResearchError("candidate num_qubits must match the problem")
            resource = audit_public_candidate(self.problem, spec)
            if not resource.valid:
                raise ResearchError("; ".join(resource.violations))
            audit = resource.audit
            if audit["operations"] > MAX_CANDIDATE_OPERATIONS:
                raise ResearchError("candidate exceeds operation cap")
            if audit["unique_trainable_params"] > MAX_CANDIDATE_PARAMETERS:
                raise ResearchError("candidate exceeds parameter cap")
            if audit["operations"] <= 0 or audit["unique_trainable_params"] <= 0:
                raise ResearchError("candidate needs an operation and a trainable parameter")
            fanout = {
                name: count
                for name, count in audit["parameter_occurrences"].items()
                if count > MAX_PARAMETER_FANOUT
            }
            if fanout:
                raise ResearchError(f"parameter fan-out exceeds {MAX_PARAMETER_FANOUT}: {fanout}")

            labels = {term.pauli for term in self.problem.pauli_terms}
            for operation in parsed.operations:
                if operation.gate == "PauliRotation" and len(operation.qubits) > 2:
                    assert operation.pauli is not None
                    if pauli_label(
                        self.problem.num_qubits, operation.qubits, operation.pauli
                    ) not in labels:
                        raise ResearchError(
                            "PauliRotation above locality 2 must be a Hamiltonian term"
                        )
            special = [
                (index, operation)
                for index, operation in enumerate(parsed.operations)
                if operation.gate in {"XYExchange", "IsotropicExchange"}
            ]
            if special and not evidence_ids:
                raise ResearchError("conservation gates require supported symmetry evidence")
            symmetry_audit: dict[str, Any] | None = None
            if evidence_ids:
                constraints: dict[str, Any] = {}
                charges: dict[str, Any] = {}
                prepared = initial_state_circuit(self.problem)
                for evidence_id in evidence_ids:
                    probe = state["probes"][evidence_id]
                    charge = generator_from_recipe(
                        self.problem.num_qubits, probe["generator"]
                    )
                    charges[evidence_id] = charge
                    residuals = operation_symmetry_residuals(
                        self.problem.num_qubits, parsed.operations, charge
                    )
                    max_residual = max(residuals, default=0.0)
                    if max_residual > EXACT_SYMMETRY_TOLERANCE:
                        raise ResearchError(
                            f"candidate breaks cited symmetry {evidence_id}: {max_residual:.3e}"
                        )
                    mean, variance = initial_state_moments(prepared, charge)
                    if variance > EXACT_SYMMETRY_TOLERANCE:
                        raise ResearchError(
                            f"initial state has no definite sector for {evidence_id}"
                        )
                    constraints[evidence_id] = {
                        "hamiltonian_residual": probe["result"]["metrics"]["residual"],
                        "max_operation_residual": max_residual,
                        "initial_state_mean": mean,
                        "initial_state_variance": variance,
                    }
                relevance = []
                for index, operation in special:
                    relevant: list[str] = []
                    failures: dict[str, str] = {}
                    for evidence_id, charge in charges.items():
                        try:
                            validate_special_operation_relevance(
                                self.problem.num_qubits,
                                operation,
                                charge,
                                symmetry_residual=constraints[evidence_id]["hamiltonian_residual"],
                                sector_variance=constraints[evidence_id]["initial_state_variance"],
                            )
                            relevant.append(evidence_id)
                        except ProbeValidationError as exc:
                            failures[evidence_id] = str(exc)
                    if not relevant:
                        raise ResearchError(
                            f"special gate {index} has no relevant cited symmetry: {failures}"
                        )
                    relevance.append(
                        {
                            "operation_index": index,
                            "gate": operation.gate,
                            "evidence_ids": relevant,
                        }
                    )
                symmetry_audit = {
                    "constraints": constraints,
                    "special_operation_relevance": relevance,
                }

            violations = _resource_violations(resource.resources)
            passed = not violations
            result: dict[str, Any] = {
                "valid": passed,
                "audit": audit,
                "resources": resource.resources,
                "violations": violations,
            }
            if symmetry_audit is not None:
                result["symmetry_audit"] = symmetry_audit
            return passed, result
        except (ResearchError, ProbeValidationError, AnsatzIRValidationError) as exc:
            return False, {"valid": False, "violations": [f"{type(exc).__name__}: {exc}"]}
        except Exception as exc:
            raise ResearchError(f"candidate audit infrastructure failed: {exc}") from exc

    def _evaluate(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(action, required={"type", "candidate_id"})
        state = self.state
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        candidate = state["candidates"].get(candidate_id)
        if candidate is None:
            raise ResearchError(f"unknown candidate: {candidate_id}")
        stage = _next_stage(candidate["status"])
        if stage is None:
            raise ResearchError(f"candidate has no next evaluation in {candidate['status']}")
        evaluation_id = f"evaluation:{candidate_id}:{stage}"
        cost, protocol = EVALUATION_PROTOCOLS[stage]
        comparison_points = _promotion_points(state, candidate_id)
        if stage == "promotion" and not _comparators(
            state, candidate_id, comparison_points
        ):
            candidate_root = _structure_root(state, candidate["hypothesis_id"])
            ready = [
                item_id
                for item_id, item in state["candidates"].items()
                if item_id != candidate_id
                and item["status"] == "SMOKE"
                and _structure_root(state, item["hypothesis_id"]) != candidate_root
            ]
            if not ready:
                raise ResearchError(
                    "promotion requires a candidate from a different structure root "
                    "that passed smoke"
                )
            self._capacity(state, 2 * cost, events=2)
        else:
            self._capacity(state, cost)
        evaluation = evaluate_public_problem(
            self.problem, candidate["spec"], protocol=protocol
        )
        if not evaluation.valid:
            raise ResearchError(
                f"optimizer failed without evidence: {list(evaluation.violations)}"
            )
        metrics = evaluation.to_dict()
        metrics.pop("optimized_parameter_binding", None)
        baseline = evaluation.baseline_energy
        improvement = (
            None
            if evaluation.best_energy is None or baseline is None
            else baseline - evaluation.best_energy
        )
        if baseline is None:
            raise ResearchError("evaluator omitted the baseline energy")
        threshold = max(MIN_ENERGY_IMPROVEMENT, MIN_ENERGY_IMPROVEMENT * abs(baseline))
        resource_violations = _resource_violations(evaluation.resources)
        metrics.update(
            energy_improvement=improvement,
            required_energy_improvement=threshold,
            violations=[*evaluation.violations, *resource_violations],
        )
        passed = bool(
            not resource_violations
            and improvement is not None
            and improvement >= threshold
        )
        if passed and stage == "promotion":
            smoke = [
                item["metrics"].get("best_energy")
                for _, item in _candidate_evaluations(state, candidate_id)
                if item["stage"] == "smoke" and item["passed"]
            ]
            passed = bool(
                smoke
                and evaluation.best_energy is not None
                and evaluation.best_energy <= min(smoke) + COMMIT_ENERGY_TOLERANCE
            )
        event_payload = {
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "stage": stage,
            "passed": bool(passed),
            "metrics": metrics,
        }
        new_state = _append(self.run_dir, state, "record_evaluation", event_payload, cost)
        return _response(
            {
                "candidate_id": candidate_id,
                "evaluation_id": evaluation_id,
                "stage": stage,
                "passed": bool(passed),
                **metrics,
            },
            new_state,
        )

    def _disposition_evidence(self, state: Mapping[str, Any], candidate_id: str) -> list[str]:
        target_records = [
            (evaluation_id, record)
            for evaluation_id, record in _candidate_evaluations(state, candidate_id)
            if record["stage"] == "promotion" and record["passed"]
        ]
        if len(target_records) != 1:
            raise ResearchError("promoted candidate lacks one passed promotion")
        target = _comparison_point(state, *target_records[0])
        dominators = [
            item
            for item in _promotion_points(state, candidate_id)
            if target is not None and _dominates(target, item)
        ]
        if not dominators:
            raise ResearchError(
                "promoted candidate may be disposed only after a dominating promotion"
            )
        return [target_records[0][0], *(item["evaluation_id"] for item in dominators)]

    def _revise_hypothesis(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(
            action,
            required={"type", "source_id", "new_id", "family", "reason"},
            optional={"prediction", "falsifier"},
        )
        state = self.state
        source_id = _identifier(action["source_id"], "source_id")
        new_id = _new_identifier(action["new_id"], "new_id")
        reason = _text(action["reason"], "reason")
        source = state["hypotheses"].get(source_id)
        if source is None or source["status"] == "REVISED":
            raise ResearchError("unknown or already revised hypothesis")
        if new_id in state["hypotheses"]:
            raise ResearchError(f"hypothesis already exists: {new_id}")
        active = [
            item_id
            for item_id, item in state["candidates"].items()
            if item["hypothesis_id"] == source_id
            and item["status"] not in TERMINAL_STATUSES
        ]
        if active:
            raise ResearchError(f"retire active candidates before revision: {active}")
        fields = self._hypothesis_fields(action)
        root = _structure_root(state, source_id)
        self._unique_structure_family(
            state, fields["family"], allowed_root=root
        )
        other_active = sum(
            key != source_id and value["status"] not in TERMINAL_STATUSES
            for key, value in state["hypotheses"].items()
        )
        if other_active >= MAX_ACTIVE_HYPOTHESES:
            raise ResearchError("hypothesis revision would exceed the active cap")
        self._capacity(state, 0.1)
        new_state = _append(self.run_dir,
            state,
            "revise_hypothesis",
            {
                "source_id": source_id,
                "new_id": new_id,
                **fields,
                "reason": reason,
            },
            0.1,
        )
        return _response(
            {"accepted": True, "new_id": new_id},
            new_state,
        )

    def _retire_hypothesis(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(action, required={"type", "hypothesis_id", "reason"})
        state = self.state
        entity_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        record = state["hypotheses"].get(entity_id)
        if record is None or record["status"] in TERMINAL_STATUSES:
            raise ResearchError("unknown or already terminal hypothesis")
        live = [
            key
            for key, candidate in state["candidates"].items()
            if candidate["hypothesis_id"] == entity_id
            and candidate["status"] not in TERMINAL_STATUSES
        ]
        if live:
            raise ResearchError(f"retire active candidates first: {live}")
        self._capacity(state, 0.0)
        new_state = _append(self.run_dir,
            state,
            "retire_hypothesis",
            {
                "entity_id": entity_id,
                "reason": _text(action["reason"], "reason"),
            },
            0.0,
        )
        return _response({"accepted": True}, new_state)

    def _retire_candidate(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(action, required={"type", "candidate_id", "reason"})
        state = self.state
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        record = state["candidates"].get(candidate_id)
        if record is None or record["status"] in TERMINAL_STATUSES:
            raise ResearchError("unknown or already terminal candidate")
        evidence_ids = (
            self._disposition_evidence(state, candidate_id)
            if record["status"] == "PROMOTED"
            else []
        )
        self._capacity(state, 0.0)
        new_state = _append(self.run_dir,
            state,
            "retire_candidate",
            {
                "entity_id": candidate_id,
                "reason": _text(action["reason"], "reason"),
                "evidence_ids": evidence_ids,
            },
            0.0,
        )
        return _response(
            {"accepted": True, "evidence_ids": evidence_ids},
            new_state,
        )

    def _commit(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(action, required={"type", "candidate_id"})
        state = self.state
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        decision = _positive_decision(state, candidate_id)
        self._capacity(state, 0.0, terminal=True)
        new_state = _append(
            self.run_dir, state, "commit", {"candidate_id": candidate_id}, 0.0
        )
        return _response({"accepted": True, **decision}, new_state)

    def _close_negative(self, action: Mapping[str, Any]) -> dict[str, Any]:
        _strict(action, required={"type", "reason"})
        state = self.state
        decision = _negative_decision(state)
        self._capacity(state, 0.0, terminal=True)
        reason = _text(action["reason"], "reason")
        new_state = _append(
            self.run_dir, state, "close_negative", {"reason": reason}, 0.0
        )
        return _response({"accepted": True, **decision}, new_state)

    def dispatch_external(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(action, Mapping):
            raise ResearchError("external action must be an object")
        try:
            size = len(json.dumps(action, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ResearchError(f"action must contain finite JSON: {exc}") from exc
        if size > MAX_EXTERNAL_ACTION_BYTES:
            raise ResearchError("external action exceeds size limit")
        state = self.state
        if state["terminal"] is not None:
            raise ResearchError("research run is terminal")
        action_type = action.get("type")
        if action_type in {"record_symmetry_probe", "record_evaluation"}:
            raise ResearchError(f"{action_type} is evaluator-owned")
        handler = {
            "propose_hypothesis": self._propose,
            "request_symmetry_probe": self._probe,
            "submit_candidate": self._submit,
            "evaluate_candidate": self._evaluate,
            "revise_hypothesis": self._revise_hypothesis,
            "retire_hypothesis": self._retire_hypothesis,
            "retire_candidate": self._retire_candidate,
            "commit": self._commit,
            "close_negative": self._close_negative,
        }.get(action_type)
        if handler is None:
            raise ResearchError(f"unsupported external action type: {action_type!r}")
        return handler(action)


def _context(run_dir: Path) -> dict[str, Any]:
    context = _read_json(run_dir / RUN_FILE)
    if set(context) != {"schema_version", "total_budget"}:
        raise ResearchError("invalid run context fields")
    if context["schema_version"] != RUN_SCHEMA_VERSION:
        raise ResearchError(f"unsupported run schema: {context['schema_version']!r}")
    context["total_budget"] = _validated_total_budget(context["total_budget"])
    return context


def _validated_total_budget(value: Any) -> float:
    budget = _finite(value, "total_budget", positive=True)
    if budget > MAX_BUDGET:
        raise ResearchError(f"total_budget cannot exceed {MAX_BUDGET}")
    return budget


def initialize_run(
    problem_path: str | Path,
    run_dir: str | Path,
    *,
    total_budget: float,
) -> dict[str, Any]:
    destination = Path(run_dir)
    budget = _validated_total_budget(total_budget)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ResearchError(f"research run already exists: {destination}")
    try:
        problem, document = load_problem_document(Path(problem_path).resolve())
    except Exception as exc:
        raise ResearchError(f"cannot load problem: {exc}") from exc
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / RUN_FILE, {"schema_version": RUN_SCHEMA_VERSION, "total_budget": budget})
    _write_json(destination / PROBLEM_FILE, document)
    return {
        "run_dir": str(destination),
        "observation": canonical_data(observe_problem(problem)),
        "state": _compact_state(_initial_state(budget)),
    }


def load_controller(run_dir: str | Path) -> ResearchController:
    directory = Path(run_dir)
    context = _context(directory)
    try:
        problem, _ = load_problem_document(directory / PROBLEM_FILE)
    except Exception as exc:
        raise ResearchError(f"cannot load run-local problem snapshot: {exc}") from exc
    return ResearchController(problem, directory, total_budget=context["total_budget"])


def execute_action(run_dir: str | Path, action: Mapping[str, Any]) -> dict[str, Any]:
    return load_controller(run_dir).dispatch_external(action)


def execute_action_file(run_dir: str | Path, action_path: str | Path) -> dict[str, Any]:
    path = Path(action_path)
    if not path.is_file():
        raise ResearchError(f"action must be a regular JSON file: {path}")
    if path.stat().st_size > MAX_EXTERNAL_ACTION_BYTES:
        raise ResearchError("action file exceeds size limit")
    return execute_action(run_dir, _read_json(path))


def run_status(run_dir: str | Path) -> dict[str, Any]:
    controller = load_controller(run_dir)
    state = controller.state
    return {
        "run_dir": str(Path(run_dir)),
        "events": state["last_seq"] + 1,
        "state": _compact_state(state),
    }


def _evidence(state: Mapping[str, Any], evidence_ids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evidence_id in evidence_ids:
        if evidence_id in state["probes"]:
            result[evidence_id] = {"kind": "probe", **state["probes"][evidence_id]}
        elif evidence_id in state["evaluations"]:
            result[evidence_id] = {
                "kind": "evaluation",
                **state["evaluations"][evidence_id],
            }
        else:
            raise ResearchError(f"terminal decision cites unknown evidence: {evidence_id}")
    return result


def run_result(run_dir: str | Path) -> dict[str, Any]:
    controller = load_controller(run_dir)
    state = controller.state
    if state["terminal"] is None:
        raise ResearchError("research run is not terminal")
    terminal = state["terminal"]
    budget = {"spent": state["spent_budget"], "total": state["total_budget"]}
    note = "No independent reference score was provided; exact ground-state accuracy is not claimed."
    if terminal["decision"] == "negative_close":
        decision = _negative_decision(state)
        evidence_ids = list(decision["evidence_ids"])
        return {
            "decision": "negative_close",
            "reason": terminal["reason"],
            "coverage": decision["coverage"],
            "evidence_ids": evidence_ids,
            "evidence": _evidence(state, evidence_ids),
            "branches": _compact_state(state),
            "budget": budget,
            "scope": "This closes only the investigated branches under the local AutoVQE rule.",
            "reference_score": None,
            "reference_score_note": note,
        }

    candidate_id = terminal["candidate_id"]
    decision = _positive_decision(state, candidate_id)
    candidate = state["candidates"][candidate_id]
    evaluation_id = decision["promotion_evaluation_id"]
    promotion = state["evaluations"].get(evaluation_id)
    if promotion is None or not promotion["passed"] or promotion["stage"] != "promotion":
        raise ResearchError("terminal commit lacks its passed promotion")
    metrics = promotion["metrics"]
    resources, audit, energy = metrics.get("resources"), metrics.get("audit"), metrics.get("best_energy")
    if not isinstance(resources, Mapping) or not isinstance(audit, Mapping) or not isinstance(energy, (int, float)):
        raise ResearchError("terminal promotion lacks evaluator-owned results")
    replay = evaluate_public_problem(
        controller.problem,
        candidate["spec"],
        protocol=EVALUATION_PROTOCOLS["promotion"][1],
    )
    binding = replay.optimized_parameter_binding
    if (
        not replay.valid
        or replay.best_energy is None
        or not isinstance(binding, Mapping)
        or replay.resources != resources
        or not math.isclose(
            float(replay.best_energy),
            float(energy),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ResearchError("terminal promotion replay does not match recorded evidence")
    evidence_ids = list(decision["evidence_ids"])
    return {
        "decision": "positive_commit",
        "candidate_id": candidate_id,
        "ansatz": candidate["spec"],
        "energy": float(energy),
        "optimized_parameters": dict(binding),
        "resources": dict(resources),
        "audit": dict(audit),
        "promotion_evaluation_id": evaluation_id,
        "evidence_ids": evidence_ids,
        "evidence": _evidence(state, evidence_ids),
        "comparison": decision["comparison"],
        "budget": budget,
        "scope": "This proves only the recorded local AutoVQE promotion rule.",
        "reference_score": None,
        "reference_score_note": note,
    }


def render_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
