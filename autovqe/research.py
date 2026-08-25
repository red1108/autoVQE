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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .ansatz import AnsatzIRValidationError, AnsatzSpec, compile_ansatz
from .evaluator import (
    EvaluationProtocol,
    audit_public_candidate,
    candidate_identity,
    evaluate_public_problem,
)
from .problem import (
    PublicProblem,
    canonical_data,
    hamiltonian_from_problem,
    load_problem_document,
    observe_problem,
)
from .probes import (
    EXACT_SYMMETRY_TOLERANCE,
    ProbeValidationError,
    algebraic_probe_cost_units,
    energy_from_circuit,
    generator_from_recipe,
    initial_state_circuit,
    initial_state_moments,
    operation_symmetry_residuals,
    run_public_probe,
    validate_special_operation_relevance,
    validate_symmetry_generator,
)


RUN_FILE = "run.json"
PROBLEM_FILE = "problem.json"
HISTORY_FILE = "events.jsonl"
RUN_SCHEMA_VERSION = 1

MAX_BUDGET = 100.0
MAX_EVENTS = 200
MAX_EXTERNAL_ACTION_BYTES = 1_000_000
MAX_ACTIVE_HYPOTHESES = 3
MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS = 2
MAX_CANDIDATE_OPERATIONS = 256
MAX_CANDIDATE_PARAMETERS = 128
MAX_CANDIDATE_SPEC_NODES = 4096
MAX_PARAMETER_FANOUT = 64
MAX_TWOQ_GATES = 512
MAX_TOTAL_GATES = 2048
MAX_DEPTH = 1024

SMOKE_PROTOCOL = EvaluationProtocol(max_evals=32, restarts=1, seed=7)
PROMOTION_PROTOCOL = EvaluationProtocol(max_evals=96, restarts=3, seed=997)
COMMIT_ENERGY_TOLERANCE = 5e-4
MIN_ENERGY_IMPROVEMENT = 1e-6

EXACT_SYMMETRY = "exact_pauli_symmetry"
STRUCTURE = "ansatz_structure"
NULL_CONTROL = "null_control"
TERMINAL_STATUSES = {"REVISED", "RETIRED"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NEW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_HIDDEN = {
    "optimized_parameter_binding",
    "best_values",
    "energy_trace",
    "best_energy_trace",
}


class ResearchError(RuntimeError):
    """Raised for an invalid run, action, or lifecycle transition."""


@dataclass(frozen=True)
class StepResult:
    action_type: str
    result: dict[str, Any]
    state_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clone(value: Any) -> Any:
    return copy.deepcopy(canonical_data(value))


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
        result = _clone(dict(value))
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


def _metadata(value: Any, *, allowed: set[str]) -> dict[str, str]:
    data = _mapping(value, "metadata")
    extra = set(data) - allowed
    if extra:
        raise ResearchError(f"metadata contains unsupported fields: {sorted(extra)}")
    return {key: _text(item, f"metadata.{key}") for key, item in data.items()}


def _preregistered(metadata: Mapping[str, Any]) -> bool:
    return any(
        isinstance(metadata.get(key), str) and metadata[key].strip()
        for key in ("prediction", "falsifier")
    )


def _family_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(text: str, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ResearchError(f"non-finite JSON number in {source}: {value}")
            ),
        )
    except ResearchError:
        raise
    except json.JSONDecodeError as exc:
        raise ResearchError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchError(f"{source} must contain one JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchError(f"required file is missing: {path}")
    try:
        return _decode(path.read_text(encoding="utf-8-sig"), path)
    except (OSError, UnicodeError) as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        _clone(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public(item)
            for key, item in value.items()
            if str(key) not in _HIDDEN
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return copy.deepcopy(value)


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
        event = _decode(line, path)
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
        "terminal_decision": None,
        "commit": None,
        "negative_close": None,
    }


def _apply_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    if state["terminal_decision"] is not None:
        raise ResearchError("history continues after a terminal decision")
    cost = _finite(event["cost"], "event cost")
    if state["spent_budget"] + cost > state["total_budget"] + 1e-12:
        raise ResearchError("history exceeds its research budget")
    payload = _mapping(event["payload"], "event payload")
    kind = event["type"]
    hypotheses = state["hypotheses"]
    candidates = state["candidates"]

    if kind == "propose_hypothesis":
        claim = payload["claim"]
        hypotheses[payload["hypothesis_id"]] = {
            "claim": claim,
            "metadata": payload.get("metadata", {}),
            "status": "PROPOSED" if claim["kind"] == EXACT_SYMMETRY else "READY",
            "parent_id": None,
            "probe_ids": [],
            "revised_to": None,
            "retired_reason": None,
        }
    elif kind == "record_probe":
        state["probes"][payload["probe_id"]] = {
            "hypothesis_id": payload["hypothesis_id"],
            "verdict": payload["verdict"],
            "result": payload["result"],
            "cost": cost,
        }
        hypothesis = hypotheses[payload["hypothesis_id"]]
        hypothesis["probe_ids"].append(payload["probe_id"])
        hypothesis["status"] = {
            "supported": "SUPPORTED",
            "refuted": "REFUTED",
            "inconclusive": "INCONCLUSIVE",
        }[payload["verdict"]]
    elif kind == "submit_candidate":
        candidates[payload["candidate_id"]] = {
            "hypothesis_id": payload["hypothesis_id"],
            "spec": payload["spec"],
            "metadata": payload.get("metadata", {}),
            "status": "CANDIDATE",
            "parent_id": payload.get("parent_id"),
            "symmetry_evidence_ids": payload.get("symmetry_evidence_ids", []),
            "evaluation_ids": [],
            "revised_to": None,
            "retired_reason": None,
            "disposition_evidence_ids": [],
        }
    elif kind == "record_evaluation":
        state["evaluations"][payload["evaluation_id"]] = {
            "candidate_id": payload["candidate_id"],
            "stage": payload["stage"],
            "passed": payload["passed"],
            "metrics": payload["metrics"],
            "cost": cost,
        }
        candidate = candidates[payload["candidate_id"]]
        candidate["evaluation_ids"].append(payload["evaluation_id"])
        candidate["status"] = (
            {"audit": "AUDITED", "smoke": "SMOKE", "promotion": "PROMOTED"}[
                payload["stage"]
            ]
            if payload["passed"]
            else "RETIRED"
        )
        if not payload["passed"]:
            candidate["retired_reason"] = f"failed {payload['stage']}"
    elif kind == "revise":
        source_id, new_id = payload["source_id"], payload["new_id"]
        if payload["entity"] == "hypothesis":
            source = hypotheses[source_id]
            source["status"], source["revised_to"] = "REVISED", new_id
            replacement = payload["replacement"]
            hypotheses[new_id] = {
                "claim": replacement,
                "metadata": {**payload.get("metadata", {}), "revision_reason": payload["reason"]},
                "status": "PROPOSED" if replacement["kind"] == EXACT_SYMMETRY else "READY",
                "parent_id": source_id,
                "probe_ids": [],
                "revised_to": None,
                "retired_reason": None,
            }
        else:
            source = candidates[source_id]
            source["status"], source["revised_to"] = "REVISED", new_id
            source["disposition_evidence_ids"] = payload.get("evidence_ids", [])
            candidates[new_id] = {
                "hypothesis_id": source["hypothesis_id"],
                "spec": payload["replacement"],
                "metadata": {
                    **source["metadata"],
                    **payload.get("metadata", {}),
                    "revision_reason": payload["reason"],
                },
                "status": "CANDIDATE",
                "parent_id": source_id,
                "symmetry_evidence_ids": payload.get("symmetry_evidence_ids", []),
                "evaluation_ids": [],
                "revised_to": None,
                "retired_reason": None,
                "disposition_evidence_ids": [],
            }
    elif kind == "retire":
        collection = hypotheses if payload["entity"] == "hypothesis" else candidates
        item = collection[payload["entity_id"]]
        item["status"] = "RETIRED"
        item["retired_reason"] = payload["reason"]
        if payload["entity"] == "candidate":
            item["disposition_evidence_ids"] = payload.get("evidence_ids", [])
    elif kind == "commit":
        state["terminal_decision"] = "positive_commit"
        state["commit"] = payload
    elif kind == "close_negative":
        state["terminal_decision"] = "negative_close"
        state["negative_close"] = payload
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
    *,
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
    # Apply once before persistence so invalid internal transitions cannot be written.
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
    return {"CANDIDATE": "audit", "AUDITED": "smoke", "SMOKE": "promotion"}.get(
        status
    )


def _resource_policy(metrics: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = ("template", "audit_worst", "canonical_template", "canonical_audit_worst")
    suffixes = {
        "twoq_count": MAX_TWOQ_GATES,
        "total_gate_count": MAX_TOTAL_GATES,
        "depth": MAX_DEPTH,
    }
    inputs: dict[str, int] = {}
    for prefix in prefixes:
        for suffix, limit in suffixes.items():
            name = f"{prefix}_{suffix}"
            value = metrics.get(name, limit + 1)
            inputs[name] = (
                int(value)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                else limit + 1
            )
    observed = {
        "conservative_twoq_count": max(
            value for name, value in inputs.items() if name.endswith("_twoq_count")
        ),
        "conservative_total_gate_count": max(
            value for name, value in inputs.items() if name.endswith("_total_gate_count")
        ),
        "conservative_depth": max(
            value for name, value in inputs.items() if name.endswith("_depth")
        ),
    }
    limits = {
        "conservative_twoq_count": MAX_TWOQ_GATES,
        "conservative_total_gate_count": MAX_TOTAL_GATES,
        "conservative_depth": MAX_DEPTH,
    }
    violations = [
        f"{name}={observed[name]} exceeds {limit}"
        for name, limit in limits.items()
        if observed[name] > limit
    ]
    return {
        "eligible": not violations,
        "observed": observed,
        "limits": limits,
        "violations": violations,
    }


def _comparison_point(state: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = evaluation["metrics"]
    policy = metrics.get("resource_policy")
    energy = metrics.get("best_energy")
    if (
        evaluation["stage"] != "promotion"
        or evaluation["passed"] is not True
        or metrics.get("valid") is not True
        or not isinstance(policy, Mapping)
        or policy.get("eligible") is not True
        or isinstance(energy, bool)
        or not isinstance(energy, (int, float))
        or not math.isfinite(float(energy))
        or not isinstance(policy.get("observed"), Mapping)
    ):
        return None
    candidate = state["candidates"][evaluation["candidate_id"]]
    audit = metrics.get("audit")
    parameter_count = audit.get("unique_trainable_params") if isinstance(audit, Mapping) else None
    resources = dict(policy["observed"])
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
        return None
    resources["unique_trainable_params"] = parameter_count
    return {
        "candidate_id": evaluation["candidate_id"],
        "hypothesis_id": candidate["hypothesis_id"],
        "evaluation_id": next(
            key for key, value in state["evaluations"].items() if value is evaluation
        ),
        "stage": "promotion",
        "passed": bool(evaluation["passed"]),
        "best_energy": float(energy),
        "resources": resources,
    }


def _structure_root(state: Mapping[str, Any], hypothesis_id: str) -> str | None:
    hypothesis = state["hypotheses"].get(hypothesis_id)
    if hypothesis is None or hypothesis["claim"].get("kind") != STRUCTURE:
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
    for evaluation in state["evaluations"].values():
        if evaluation["candidate_id"] == candidate_id:
            continue
        other = state["candidates"][evaluation["candidate_id"]]
        if _structure_root(state, other["hypothesis_id"]) is None:
            continue
        point = _comparison_point(state, evaluation)
        if point is not None:
            points.append(point)
    return sorted(points, key=lambda item: item["candidate_id"])


def _comparators(state: Mapping[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    primary_id = state["candidates"][candidate_id]["hypothesis_id"]
    primary_root = _structure_root(state, primary_id)
    if primary_root is None:
        return []
    return [
        point
        for point in _promotion_points(state, candidate_id)
        if _structure_root(state, point["hypothesis_id"]) != primary_root
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


def _compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    hypotheses: dict[str, Any] = {}
    for hypothesis_id, record in sorted(state["hypotheses"].items()):
        kind = record["claim"].get("kind")
        if kind == EXACT_SYMMETRY and record["status"] == "SUPPORTED":
            next_action = "cite_probe_from_ansatz_structure_or_retire"
        else:
            next_action = {
                "PROPOSED": "request_probe",
                "READY": "submit_candidate",
                "REFUTED": "revise_or_retire",
                "INCONCLUSIVE": "revise_or_retire",
            }.get(record["status"])
        summary: dict[str, Any] = {
            "kind": kind,
            "status": record["status"],
            "next_action": next_action,
        }
        if record["probe_ids"]:
            probe_id = record["probe_ids"][-1]
            probe = state["probes"][probe_id]
            summary["latest_probe"] = {
                "probe_id": probe_id,
                "verdict": probe["verdict"],
                "cost": probe["cost"],
                **{
                    key: _public(probe["result"][key])
                    for key in ("probe_type", "metrics", "valid", "violations")
                    if key in probe["result"]
                },
            }
        for key in ("parent_id", "revised_to", "retired_reason"):
            if record.get(key) is not None:
                summary[key] = record[key]
        hypotheses[hypothesis_id] = summary

    candidates: dict[str, Any] = {}
    for candidate_id, record in sorted(state["candidates"].items()):
        status = record["status"]
        if status == "PROMOTED":
            next_action = (
                "commit_or_dispose_after_comparison"
                if _comparators(state, candidate_id)
                else "evaluate_different_hypothesis:promotion"
            )
        else:
            stage = _next_stage(status)
            next_action = f"evaluate_candidate:{stage}" if stage else None
        summary = {
            "hypothesis_id": record["hypothesis_id"],
            "status": status,
            "next_action": next_action,
        }
        if record["evaluation_ids"]:
            evaluation_id = record["evaluation_ids"][-1]
            evaluation = state["evaluations"][evaluation_id]
            details = evaluation["metrics"]
            summary["latest_evaluation"] = {
                "evaluation_id": evaluation_id,
                "stage": evaluation["stage"],
                "passed": evaluation["passed"],
                "cost": evaluation["cost"],
                "summary": {
                    key: _public(details[key])
                    for key in (
                        "valid",
                        "best_energy",
                        "objective_calls",
                        "trace_summary",
                        "objective_energy_span",
                        "hamiltonian_active_norm",
                        "objective_activity_fraction",
                        "constant_hamiltonian",
                        "baseline_energy",
                        "energy_improvement",
                        "required_energy_improvement",
                        "promotion_blocked_reason",
                        "resource_policy",
                        "violations",
                    )
                    if key in details
                },
            }
        if record["symmetry_evidence_ids"]:
            summary["symmetry_evidence_ids"] = list(record["symmetry_evidence_ids"])
        for key in ("parent_id", "revised_to", "retired_reason"):
            if record.get(key) is not None:
                summary[key] = record[key]
        candidates[candidate_id] = summary
    return {
        "terminal_decision": state["terminal_decision"],
        "budget": {
            "spent": state["spent_budget"],
            "remaining": state["total_budget"] - state["spent_budget"],
            "total": state["total_budget"],
        },
        "hypotheses": hypotheses,
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

    def _emit(self, state: Mapping[str, Any], action_type: str, payload: Mapping[str, Any], cost: float) -> StepResult:
        new_state = _append(self.run_dir, state, action_type, payload, cost=cost)
        return StepResult(action_type, _public(payload), _compact_state(new_state))

    def _claim(self, raw: Any) -> dict[str, Any]:
        claim = _mapping(raw, "claim")
        kind = claim.get("kind")
        if kind == EXACT_SYMMETRY:
            if set(claim) != {"kind", "generator"}:
                raise ResearchError("exact_pauli_symmetry requires kind and generator")
            recipe = _mapping(claim["generator"], "claim.generator")
            try:
                generator = generator_from_recipe(self.problem.num_qubits, recipe)
                validate_symmetry_generator(hamiltonian_from_problem(self.problem), generator)
            except Exception as exc:
                raise ResearchError(f"invalid symmetry generator: {exc}") from exc
            return {"kind": kind, "generator": recipe}
        if kind == STRUCTURE:
            if set(claim) != {"kind", "family"}:
                raise ResearchError("ansatz_structure requires kind and family")
            return {"kind": kind, "family": _text(claim["family"], "claim.family")}
        if kind == NULL_CONTROL and set(claim) == {"kind"}:
            return {"kind": kind}
        raise ResearchError(
            "claim.kind must be exact_pauli_symmetry, ansatz_structure, or null_control"
        )

    def _unique_structure_claim(
        self, state: Mapping[str, Any], claim: Mapping[str, Any]
    ) -> None:
        if claim.get("kind") != STRUCTURE:
            return
        family = _family_key(str(claim["family"]))
        duplicates = [
            hypothesis_id
            for hypothesis_id, record in state["hypotheses"].items()
            if record["claim"].get("kind") == STRUCTURE
            and _family_key(str(record["claim"]["family"])) == family
        ]
        if duplicates:
            raise ResearchError(
                f"ansatz_structure family duplicates existing hypothesis: {duplicates}"
            )

    def _propose(self, action: Mapping[str, Any]) -> StepResult:
        _strict(action, required={"type", "hypothesis_id", "claim"}, optional={"metadata"})
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
        claim = self._claim(action["claim"])
        self._unique_structure_claim(state, claim)
        metadata = _metadata(
            action.get("metadata", {}),
            allowed={"rationale", "prediction", "falsifier"},
        )
        self._capacity(state, 0.1)
        result = self._emit(
            state,
            "propose_hypothesis",
            {"hypothesis_id": hypothesis_id, "claim": claim, "metadata": metadata},
            0.1,
        )
        return StepResult(
            result.action_type,
            {
                "accepted": True,
                "claim_kind": claim["kind"],
                "requires_probe": claim["kind"] == EXACT_SYMMETRY,
            },
            result.state_summary,
        )

    def _probe(self, action: Mapping[str, Any]) -> StepResult:
        _strict(action, required={"type", "hypothesis_id"})
        state = self.state
        hypothesis_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        hypothesis = state["hypotheses"].get(hypothesis_id)
        if hypothesis is None:
            raise ResearchError(f"unknown hypothesis: {hypothesis_id}")
        if hypothesis["status"] != "PROPOSED" or hypothesis["claim"]["kind"] != EXACT_SYMMETRY:
            raise ResearchError("only a proposed exact_pauli_symmetry can be probed")
        request = {
            "type": "normalized_commutator",
            "generator": hypothesis["claim"]["generator"],
        }
        try:
            cost = algebraic_probe_cost_units(hamiltonian_from_problem(self.problem), request)
            self._capacity(state, cost)
            measured = run_public_probe(self.problem, request)
        except Exception as exc:
            raise ResearchError(f"probe failed: {exc}") from exc
        passed = measured.probe_type == "normalized_commutator" and bool(
            measured.metrics.get("exact", False)
        )
        probe_id = f"probe:{hypothesis_id}"
        payload = {
            "hypothesis_id": hypothesis_id,
            "probe_id": probe_id,
            "verdict": "supported" if passed else "refuted",
            "result": measured.to_dict(),
        }
        emitted = self._emit(state, "record_probe", payload, cost)
        return StepResult(
            "request_probe",
            {**_public(measured.to_dict()), "probe_id": probe_id, "passed": passed},
            emitted.state_summary,
        )

    def _symmetry_evidence(self, state: Mapping[str, Any], raw: Any, hypothesis_id: str) -> list[str]:
        if raw is None:
            evidence: list[Any] = []
        elif isinstance(raw, list):
            evidence = raw
        else:
            raise ResearchError("symmetry_evidence_ids must be a list")
        values = [_identifier(item, "symmetry_evidence_ids") for item in evidence]
        primary = state["hypotheses"][hypothesis_id]
        if primary["claim"]["kind"] == EXACT_SYMMETRY and primary["status"] == "SUPPORTED":
            values.extend(primary["probe_ids"])
        result = sorted(set(values))
        for evidence_id in result:
            probe = state["probes"].get(evidence_id)
            if probe is None or probe["verdict"] != "supported":
                raise ResearchError(f"symmetry evidence must cite a supported probe: {evidence_id}")
            source = state["hypotheses"][probe["hypothesis_id"]]
            if source["claim"]["kind"] != EXACT_SYMMETRY:
                raise ResearchError(f"symmetry evidence is not an exact symmetry: {evidence_id}")
        return result

    def _candidate_metadata(self, raw: Any, kind: str) -> dict[str, Any]:
        metadata = _metadata(raw, allowed={"rationale", "prediction", "falsifier"})
        if kind != NULL_CONTROL and not _preregistered(metadata):
            raise ResearchError("promotable candidate must preregister a prediction or falsifier")
        return metadata

    def _unique_candidate(
        self,
        state: Mapping[str, Any],
        spec: Mapping[str, Any],
        repair_source: str | None = None,
        symmetry_evidence_ids: tuple[str, ...] = (),
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
        if duplicates == [repair_source]:
            source = state["candidates"][repair_source]
            evaluations = [state["evaluations"][item] for item in source["evaluation_ids"]]
            if evaluations and all(item["stage"] == "audit" for item in evaluations) and any(
                not item["passed"] for item in evaluations
            ):
                return
            source_macros = {
                operation["macro"] for operation in source["spec"]["operations"]
            }
            replacement_macros = {
                operation["macro"] for operation in spec["operations"]
            }
            special = {"XYExchange", "IsotropicExchange"}
            added_evidence = set(symmetry_evidence_ids) - set(
                source["symmetry_evidence_ids"]
            )
            if replacement_macros & special and not source_macros & special and added_evidence:
                # The semantic family is unchanged, but supported conservation
                # evidence now permits a cheaper trusted implementation.
                return
        if duplicates:
            raise ResearchError(f"candidate is semantically equivalent to existing {duplicates}")

    def _submit(self, action: Mapping[str, Any]) -> StepResult:
        _strict(
            action,
            required={"type", "candidate_id", "hypothesis_id", "spec"},
            optional={"metadata", "symmetry_evidence_ids"},
        )
        state = self.state
        candidate_id = _new_identifier(action["candidate_id"], "candidate_id")
        hypothesis_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        if candidate_id in state["candidates"]:
            raise ResearchError(f"candidate already exists: {candidate_id}")
        hypothesis = state["hypotheses"].get(hypothesis_id)
        if hypothesis is None or hypothesis["status"] not in {"READY", "SUPPORTED"}:
            raise ResearchError("candidate requires a READY or SUPPORTED hypothesis")
        kind = hypothesis["claim"]["kind"]
        if kind not in {STRUCTURE, NULL_CONTROL}:
            raise ResearchError(
                "candidate primary hypothesis must be ansatz_structure or null_control; "
                "cite exact symmetry through symmetry_evidence_ids"
            )
        try:
            spec = AnsatzSpec.from_dict(_mapping(action["spec"], "spec")).to_dict()
        except Exception as exc:
            raise ResearchError(f"invalid candidate: {exc}") from exc
        if kind == NULL_CONTROL:
            parsed = AnsatzSpec.from_dict(spec)
            if parsed.parameters or parsed.operations:
                raise ResearchError("null_control must be a typed no-op")
        self._unique_candidate(state, spec)
        evidence = self._symmetry_evidence(
            state, action.get("symmetry_evidence_ids"), hypothesis_id
        )
        metadata = self._candidate_metadata(
            action.get("metadata", {}), kind
        )
        active = sum(
            record["hypothesis_id"] == hypothesis_id
            and record["status"] not in TERMINAL_STATUSES
            for record in state["candidates"].values()
        )
        if active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
            raise ResearchError("too many active candidates under this hypothesis")
        self._capacity(state, 0.1)
        emitted = self._emit(
            state,
            "submit_candidate",
            {
                "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id,
                "spec": spec,
                "metadata": metadata,
                "parent_id": None,
                "symmetry_evidence_ids": evidence,
            },
            0.1,
        )
        return StepResult("submit_candidate", {"accepted": True}, emitted.state_summary)

    def _audit(self, state: Mapping[str, Any], candidate_id: str) -> tuple[bool, dict[str, Any]]:
        candidate = state["candidates"][candidate_id]
        hypothesis = state["hypotheses"][candidate["hypothesis_id"]]
        is_control = hypothesis["claim"]["kind"] == NULL_CONTROL
        try:
            parsed = AnsatzSpec.from_dict(candidate["spec"])
            if parsed.num_qubits != self.problem.num_qubits:
                raise ResearchError("candidate num_qubits must match the problem")
            compiled = compile_ansatz(parsed)
            audit = compiled.audit
            if audit["operations"] > MAX_CANDIDATE_OPERATIONS:
                raise ResearchError("candidate exceeds operation cap")
            if audit["unique_trainable_params"] > MAX_CANDIDATE_PARAMETERS:
                raise ResearchError("candidate exceeds parameter cap")
            if audit["spec_nodes"] > MAX_CANDIDATE_SPEC_NODES:
                raise ResearchError("candidate exceeds spec-node cap")
            if not is_control and (
                audit["operations"] <= 0 or audit["unique_trainable_params"] <= 0
            ):
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
                if operation.macro == "PauliRotation" and len(operation.qubits) > 2:
                    label = ["I"] * self.problem.num_qubits
                    for qubit, letter in zip(
                        operation.qubits, operation.options["pauli"], strict=True
                    ):
                        label[self.problem.num_qubits - qubit - 1] = letter
                    if "".join(label) not in labels:
                        raise ResearchError(
                            "PauliRotation above locality 2 must be a Hamiltonian term"
                        )
            allowed_scales = {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}
            bad_literals = [
                dict(item)
                for item in audit["fixed_literals"]
                if item["role"] != "scale"
                or not any(
                    math.isclose(item["value"], value, abs_tol=1e-12)
                    for value in allowed_scales
                )
            ]
            if bad_literals:
                raise ResearchError(f"unapproved fixed numeric literals: {bad_literals}")

            special = [
                (index, operation)
                for index, operation in enumerate(parsed.operations)
                if operation.macro in {"XYExchange", "IsotropicExchange"}
            ]
            evidence_ids = candidate["symmetry_evidence_ids"]
            if special and not evidence_ids:
                raise ResearchError("conservation gates require supported symmetry evidence")
            symmetry_audit: dict[str, Any] | None = None
            if evidence_ids:
                constraints: dict[str, Any] = {}
                charges: dict[str, Any] = {}
                zero = compiled.circuit.assign_parameters(
                    {parameter: 0.0 for parameter in compiled.parameters.values()},
                    inplace=False,
                )
                prepared = initial_state_circuit(self.problem)
                prepared.compose(zero, inplace=True)
                for evidence_id in evidence_ids:
                    probe = state["probes"][evidence_id]
                    source = state["hypotheses"][probe["hypothesis_id"]]
                    charge = generator_from_recipe(
                        self.problem.num_qubits, source["claim"]["generator"]
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
                        "hypothesis_id": probe["hypothesis_id"],
                        "hamiltonian_residual": probe["result"]["metrics"]["residual"],
                        "max_operation_residual": max_residual,
                        "initial_state_mean": mean,
                        "initial_state_variance": variance,
                    }
                relevance = []
                for index, operation in special:
                    relevant: dict[str, Any] = {}
                    failures: dict[str, str] = {}
                    for evidence_id, charge in charges.items():
                        try:
                            values = validate_special_operation_relevance(
                                self.problem.num_qubits,
                                operation,
                                charge,
                                symmetry_residual=constraints[evidence_id]["hamiltonian_residual"],
                                sector_variance=constraints[evidence_id]["initial_state_variance"],
                            )
                            relevant[evidence_id] = {
                                "touching_charge_norm": values[0],
                                "relevant_charge_fraction": values[1],
                                "residual": values[2],
                                "conditioned_symmetry_residual": values[3],
                                "conditioned_sector_variance": values[4],
                            }
                        except ProbeValidationError as exc:
                            failures[evidence_id] = str(exc)
                    if not relevant:
                        raise ResearchError(
                            f"special gate {index} has no relevant cited symmetry: {failures}"
                        )
                    relevance.append(
                        {"operation_index": index, "macro": operation.macro, "constraints": relevant}
                    )
                symmetry_audit = {
                    "constraints": constraints,
                    "special_operation_relevance": relevance,
                }

            resource = audit_public_candidate(self.problem, candidate["spec"])
            policy = _resource_policy(resource.metrics)
            violations = list(resource.violations) + list(policy["violations"])
            passed = bool(resource.valid and policy["eligible"] and not violations)
            result: dict[str, Any] = {
                "valid": passed,
                "audit": _clone(audit),
                "metrics": dict(resource.metrics),
                "resource_policy": policy,
                "violations": violations,
            }
            if symmetry_audit is not None:
                result["symmetry_audit"] = symmetry_audit
            return passed, result
        except (ResearchError, ProbeValidationError, AnsatzIRValidationError) as exc:
            return False, {"valid": False, "violations": [f"{type(exc).__name__}: {exc}"]}
        except Exception as exc:
            raise ResearchError(f"candidate audit infrastructure failed: {exc}") from exc

    def _baseline(self, spec: Mapping[str, Any]) -> float:
        compiled = compile_ansatz(spec)
        zero = compiled.circuit.assign_parameters(
            {parameter: 0.0 for parameter in compiled.parameters.values()}, inplace=False
        )
        prepared = initial_state_circuit(self.problem)
        prepared.compose(zero, inplace=True)
        return energy_from_circuit(prepared, hamiltonian_from_problem(self.problem))

    def _evaluate(self, action: Mapping[str, Any]) -> StepResult:
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
        if stage == "audit":
            cost = 0.25
            self._capacity(state, cost)
            passed, metrics = self._audit(state, candidate_id)
            public_metrics = metrics
        else:
            cost = 2.0 if stage == "smoke" else 6.0
            candidate_kind = state["hypotheses"][candidate["hypothesis_id"]]["claim"][
                "kind"
            ]
            if (
                stage == "promotion"
                and candidate_kind == STRUCTURE
                and not _comparators(state, candidate_id)
            ):
                candidate_root = _structure_root(state, candidate["hypothesis_id"])
                ready = [
                    item_id
                    for item_id, item in state["candidates"].items()
                    if item_id != candidate_id
                    and item["status"] == "SMOKE"
                    and state["hypotheses"][item["hypothesis_id"]]["claim"]["kind"]
                    == STRUCTURE
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
            protocol = SMOKE_PROTOCOL if stage == "smoke" else PROMOTION_PROTOCOL
            baseline = self._baseline(candidate["spec"])
            evaluation = evaluate_public_problem(
                self.problem, candidate["spec"], protocol=protocol
            ).result
            if not evaluation.valid:
                raise ResearchError(
                    f"optimizer failed without evidence: {list(evaluation.violations)}"
                )
            metrics = evaluation.to_dict()
            metrics.pop("optimized_parameter_binding", None)
            policy = _resource_policy(evaluation.metrics)
            improvement = (
                None if evaluation.best_energy is None else baseline - evaluation.best_energy
            )
            threshold = max(MIN_ENERGY_IMPROVEMENT, MIN_ENERGY_IMPROVEMENT * abs(baseline))
            metrics.update(
                resource_policy=policy,
                baseline_energy=baseline,
                energy_improvement=improvement,
                required_energy_improvement=threshold,
            )
            passed = bool(policy["eligible"] and improvement is not None and improvement >= threshold)
            hypothesis = state["hypotheses"][candidate["hypothesis_id"]]
            if stage == "smoke" and hypothesis["claim"]["kind"] == NULL_CONTROL:
                passed = bool(evaluation.valid and policy["eligible"])
            if stage == "promotion" and hypothesis["claim"]["kind"] == NULL_CONTROL:
                passed = False
                metrics["promotion_blocked_reason"] = "null_control cannot be promoted"
            if passed and stage == "promotion":
                smoke = [
                    state["evaluations"][item]["metrics"].get("best_energy")
                    for item in candidate["evaluation_ids"]
                    if state["evaluations"][item]["stage"] == "smoke"
                    and state["evaluations"][item]["passed"]
                ]
                passed = bool(
                    smoke
                    and evaluation.best_energy is not None
                    and evaluation.best_energy <= min(smoke) + COMMIT_ENERGY_TOLERANCE
                )
            public_metrics = _public(metrics)
        event_payload = {
            "candidate_id": candidate_id,
            "evaluation_id": evaluation_id,
            "stage": stage,
            "passed": bool(passed),
            "metrics": metrics,
        }
        emitted = self._emit(state, "record_evaluation", event_payload, cost)
        return StepResult(
            "evaluate_candidate",
            {
                "candidate_id": candidate_id,
                "evaluation_id": evaluation_id,
                "stage": stage,
                "passed": bool(passed),
                **public_metrics,
            },
            emitted.state_summary,
        )

    def _disposition_evidence(self, state: Mapping[str, Any], candidate_id: str) -> list[str]:
        target_records = [
            (evaluation_id, record)
            for evaluation_id, record in state["evaluations"].items()
            if record["candidate_id"] == candidate_id
            and record["stage"] == "promotion"
            and record["passed"]
        ]
        if len(target_records) != 1:
            raise ResearchError("promoted candidate lacks one passed promotion")
        target = _comparison_point(state, target_records[0][1])
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

    def _revise(self, action: Mapping[str, Any]) -> StepResult:
        _strict(
            action,
            required={"type", "entity", "source_id", "new_id", "replacement", "reason"},
            optional={"metadata", "symmetry_evidence_ids"},
        )
        state = self.state
        entity = action["entity"]
        source_id = _identifier(action["source_id"], "source_id")
        new_id = _new_identifier(action["new_id"], "new_id")
        reason = _text(action["reason"], "reason")
        evidence_ids: list[str] = []
        symmetry_ids: list[str] = []
        if entity == "hypothesis":
            if "symmetry_evidence_ids" in action:
                raise ResearchError("symmetry_evidence_ids applies only to candidates")
            source = state["hypotheses"].get(source_id)
            if source is None or source["status"] == "REVISED":
                raise ResearchError("unknown or already revised hypothesis")
            if new_id in state["hypotheses"]:
                raise ResearchError(f"hypothesis already exists: {new_id}")
            active = [
                item_id
                for item_id, item in state["candidates"].items()
                if item["hypothesis_id"] == source_id and item["status"] not in TERMINAL_STATUSES
            ]
            if active:
                raise ResearchError(f"retire active candidates before revision: {active}")
            replacement = self._claim(action["replacement"])
            self._unique_structure_claim(state, replacement)
            metadata = _metadata(
                action.get("metadata", {}), allowed={"rationale", "prediction", "falsifier"}
            )
            other_active = sum(
                key != source_id and value["status"] not in TERMINAL_STATUSES
                for key, value in state["hypotheses"].items()
            )
            if other_active >= MAX_ACTIVE_HYPOTHESES:
                raise ResearchError("hypothesis revision would exceed the active cap")
        elif entity == "candidate":
            source = state["candidates"].get(source_id)
            if source is None or source["status"] == "REVISED":
                raise ResearchError("unknown or already revised candidate")
            if new_id in state["candidates"]:
                raise ResearchError(f"candidate already exists: {new_id}")
            hypothesis = state["hypotheses"][source["hypothesis_id"]]
            if hypothesis["status"] not in {"READY", "SUPPORTED"}:
                raise ResearchError("cannot revise under an inactive hypothesis")
            if hypothesis["claim"]["kind"] not in {STRUCTURE, NULL_CONTROL}:
                raise ResearchError(
                    "candidate primary hypothesis must be ansatz_structure or null_control"
                )
            try:
                replacement = AnsatzSpec.from_dict(
                    _mapping(action["replacement"], "replacement")
                ).to_dict()
            except Exception as exc:
                raise ResearchError(f"invalid candidate revision: {exc}") from exc
            symmetry_ids = (
                self._symmetry_evidence(
                    state, action["symmetry_evidence_ids"], source["hypothesis_id"]
                )
                if "symmetry_evidence_ids" in action
                else list(source["symmetry_evidence_ids"])
            )
            metadata = self._candidate_metadata(
                action.get("metadata", {}), hypothesis["claim"]["kind"]
            )
            self._unique_candidate(
                state,
                replacement,
                source_id,
                tuple(symmetry_ids),
            )
            other_active = sum(
                key != source_id
                and value["hypothesis_id"] == source["hypothesis_id"]
                and value["status"] not in TERMINAL_STATUSES
                for key, value in state["candidates"].items()
            )
            if other_active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
                raise ResearchError("candidate revision would exceed the active cap")
            if source["status"] == "PROMOTED":
                evidence_ids = self._disposition_evidence(state, source_id)
        else:
            raise ResearchError("entity must be hypothesis or candidate")
        self._capacity(state, 0.1)
        emitted = self._emit(
            state,
            "revise",
            {
                "entity": entity,
                "source_id": source_id,
                "new_id": new_id,
                "replacement": replacement,
                "reason": reason,
                "metadata": metadata,
                "symmetry_evidence_ids": symmetry_ids,
                "evidence_ids": evidence_ids,
            },
            0.1,
        )
        return StepResult("revise", {"accepted": True, "new_id": new_id}, emitted.state_summary)

    def _retire(self, action: Mapping[str, Any]) -> StepResult:
        _strict(action, required={"type", "entity", "entity_id", "reason"})
        state = self.state
        entity = action["entity"]
        if entity not in {"hypothesis", "candidate"}:
            raise ResearchError("entity must be hypothesis or candidate")
        entity_id = _identifier(action["entity_id"], "entity_id")
        collection = state["hypotheses"] if entity == "hypothesis" else state["candidates"]
        record = collection.get(entity_id)
        if record is None or record["status"] in TERMINAL_STATUSES:
            raise ResearchError("unknown or already terminal entity")
        if entity == "hypothesis":
            live = [
                key
                for key, candidate in state["candidates"].items()
                if candidate["hypothesis_id"] == entity_id
                and candidate["status"] not in TERMINAL_STATUSES
            ]
            if live:
                raise ResearchError(f"retire active candidates first: {live}")
            evidence_ids = []
        else:
            evidence_ids = (
                self._disposition_evidence(state, entity_id)
                if record["status"] == "PROMOTED"
                else []
            )
        self._capacity(state, 0.0)
        emitted = self._emit(
            state,
            "retire",
            {
                "entity": entity,
                "entity_id": entity_id,
                "reason": _text(action["reason"], "reason"),
                "evidence_ids": evidence_ids,
            },
            0.0,
        )
        return StepResult("retire", {"accepted": True}, emitted.state_summary)

    def _commit(self, action: Mapping[str, Any]) -> StepResult:
        _strict(action, required={"type", "candidate_id"})
        state = self.state
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        candidate = state["candidates"].get(candidate_id)
        if candidate is None or candidate["status"] != "PROMOTED":
            raise ResearchError("commit requires a passed promotion")
        if _structure_root(state, candidate["hypothesis_id"]) is None:
            raise ResearchError("commit target must belong to an ansatz_structure root")
        if not _preregistered(candidate["metadata"]):
            raise ResearchError("commit requires a preregistered prediction or falsifier")
        promotions = [
            (key, value)
            for key, value in state["evaluations"].items()
            if value["candidate_id"] == candidate_id
            and value["stage"] == "promotion"
            and value["passed"]
        ]
        if len(promotions) != 1:
            raise ResearchError("commit requires exactly one passed promotion")
        target = _comparison_point(state, promotions[0][1])
        fair_comparators = _comparators(state, candidate_id)
        comparison_points = _promotion_points(state, candidate_id)
        if target is None or not fair_comparators:
            raise ResearchError("commit requires evaluator-owned target and comparator evidence")
        for comparator in comparison_points:
            if _dominates(target, comparator):
                raise ResearchError(
                    f"target is dominated by comparator {comparator['candidate_id']}"
                )
        evidence_ids = [
            promotions[0][0],
            *(item["evaluation_id"] for item in comparison_points),
        ]
        comparison = {
            "mode": "evaluated_competitor",
            "energy_tolerance": COMMIT_ENERGY_TOLERANCE,
            "target": target,
            "evaluations": comparison_points,
            "fair_comparator_evaluation_ids": [
                item["evaluation_id"] for item in fair_comparators
            ],
        }
        self._capacity(state, 0.0, terminal=True)
        emitted = self._emit(
            state,
            "commit",
            {
                "candidate_id": candidate_id,
                "evidence_ids": evidence_ids,
                "promotion_evaluation_id": promotions[0][0],
                "comparison": comparison,
            },
            0.0,
        )
        return StepResult(
            "commit",
            {
                "accepted": True,
                "candidate_id": candidate_id,
                "evidence_ids": evidence_ids,
                "promotion_evaluation_id": promotions[0][0],
                "comparison": comparison,
            },
            emitted.state_summary,
        )

    def _close_negative(self, action: Mapping[str, Any]) -> StepResult:
        _strict(action, required={"type", "reason"})
        state = self.state
        live_h = [key for key, value in state["hypotheses"].items() if value["status"] not in TERMINAL_STATUSES]
        live_c = [key for key, value in state["candidates"].items() if value["status"] not in TERMINAL_STATUSES]
        if live_h or live_c:
            raise ResearchError(f"negative close requires terminal branches: {live_h + live_c}")
        evidence: set[str] = {
            key for key, value in state["probes"].items() if value["verdict"] == "refuted"
        }
        failed = []
        for key, value in state["evaluations"].items():
            candidate = state["candidates"][value["candidate_id"]]
            if _structure_root(state, candidate["hypothesis_id"]) is None:
                continue
            metrics = value["metrics"]
            active = metrics.get("objective_activity_fraction")
            if (
                value["stage"] in {"smoke", "promotion"}
                and not value["passed"]
                and isinstance(metrics.get("objective_calls"), int)
                and metrics["objective_calls"] > 0
                and isinstance(active, (int, float))
                and not isinstance(active, bool)
                and active >= 1e-6
            ):
                evidence.add(key)
                failed.append(value)
        roots: set[str] = set()
        for value in failed:
            hypothesis_id = state["candidates"][value["candidate_id"]]["hypothesis_id"]
            root = _structure_root(state, hypothesis_id)
            if root is not None:
                roots.add(root)
        if not evidence or (not any(item["stage"] == "promotion" for item in failed) and len(roots) < 2):
            raise ResearchError(
                "negative close needs an objective-active promotion failure or two "
                "independent failed structure lineages"
            )
        coverage = {
            "search_mode": "promotion_depth" if any(item["stage"] == "promotion" for item in failed) else "structural_breadth",
            "structure_lineage_ids": sorted(roots),
        }
        self._capacity(state, 0.0, terminal=True)
        emitted = self._emit(
            state,
            "close_negative",
            {
                "reason": _text(action["reason"], "reason"),
                "evidence_ids": sorted(evidence),
                "coverage": coverage,
            },
            0.0,
        )
        return StepResult(
            "close_negative",
            {"accepted": True, "evidence_ids": sorted(evidence), "coverage": coverage},
            emitted.state_summary,
        )

    def dispatch_external(self, action: Mapping[str, Any]) -> StepResult:
        if not isinstance(action, Mapping):
            raise ResearchError("external action must be an object")
        try:
            size = len(json.dumps(action, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ResearchError(f"action must contain finite JSON: {exc}") from exc
        if size > MAX_EXTERNAL_ACTION_BYTES:
            raise ResearchError("external action exceeds size limit")
        state = self.state
        if state["terminal_decision"] is not None:
            raise ResearchError("research run is terminal")
        action_type = action.get("type")
        if action_type in {"record_probe", "record_evaluation"}:
            raise ResearchError(f"{action_type} is evaluator-owned")
        handler = {
            "propose_hypothesis": self._propose,
            "request_probe": self._probe,
            "submit_candidate": self._submit,
            "evaluate_candidate": self._evaluate,
            "revise": self._revise,
            "retire": self._retire,
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
    budget = _finite(context["total_budget"], "total_budget", positive=True)
    if budget > MAX_BUDGET:
        raise ResearchError(f"total_budget cannot exceed {MAX_BUDGET}")
    context["total_budget"] = budget
    return context


def initialize_run(
    problem_path: str | Path,
    run_dir: str | Path,
    *,
    total_budget: float,
) -> dict[str, Any]:
    destination = Path(run_dir)
    budget = _finite(total_budget, "total_budget", positive=True)
    if budget > MAX_BUDGET:
        raise ResearchError(f"total_budget cannot exceed {MAX_BUDGET}")
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
        "observation": _clone(observe_problem(problem)),
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
    return _public(load_controller(run_dir).dispatch_external(action).to_dict())


def execute_action_file(run_dir: str | Path, action_path: str | Path) -> dict[str, Any]:
    path = Path(action_path)
    if not path.is_file():
        raise ResearchError(f"action must be a regular JSON file: {path}")
    if path.stat().st_size > MAX_EXTERNAL_ACTION_BYTES:
        raise ResearchError("action file exceeds size limit")
    return execute_action(run_dir, _read_json(path))


def compact_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return _public(_compact_state(state))


def run_status(run_dir: str | Path, *, full: bool = False) -> dict[str, Any]:
    controller = load_controller(run_dir)
    state = controller.state
    return _public(
        {
            "run_dir": str(Path(run_dir)),
            "events": state["last_seq"] + 1,
            "state": _clone(state) if full else _compact_state(state),
        }
    )


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
    return _public(result)


def run_result(run_dir: str | Path) -> dict[str, Any]:
    controller = load_controller(run_dir)
    state = controller.state
    if state["terminal_decision"] is None:
        raise ResearchError("research run is not terminal")
    budget = {"spent": state["spent_budget"], "total": state["total_budget"]}
    note = "No independent reference score was provided; exact ground-state accuracy is not claimed."
    if state["terminal_decision"] == "negative_close":
        decision = state["negative_close"]
        evidence_ids = list(decision["evidence_ids"])
        return {
            "decision": "negative_close",
            "reason": decision["reason"],
            "coverage": _clone(decision["coverage"]),
            "evidence_ids": evidence_ids,
            "evidence": _evidence(state, evidence_ids),
            "branches": _compact_state(state),
            "budget": budget,
            "scope": "This closes only the investigated branches under the local AutoVQE rule.",
            "reference_score": None,
            "reference_score_note": note,
        }

    decision = state["commit"]
    candidate_id = decision["candidate_id"]
    candidate = state["candidates"][candidate_id]
    evaluation_id = decision["promotion_evaluation_id"]
    promotion = state["evaluations"].get(evaluation_id)
    if promotion is None or not promotion["passed"] or promotion["stage"] != "promotion":
        raise ResearchError("terminal commit lacks its passed promotion")
    metrics = promotion["metrics"]
    resources, audit, energy = metrics.get("metrics"), metrics.get("audit"), metrics.get("best_energy")
    if not isinstance(resources, Mapping) or not isinstance(audit, Mapping) or not isinstance(energy, (int, float)):
        raise ResearchError("terminal promotion lacks evaluator-owned results")
    replay = evaluate_public_problem(
        controller.problem,
        candidate["spec"],
        protocol=PROMOTION_PROTOCOL,
    ).result
    binding = replay.optimized_parameter_binding
    if (
        not replay.valid
        or replay.best_energy is None
        or not isinstance(binding, Mapping)
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
        "ansatz": _clone(candidate["spec"]),
        "energy": float(energy),
        "optimized_parameters": _clone(binding),
        "resources": _clone(resources),
        "audit": _clone(audit),
        "promotion_evaluation_id": evaluation_id,
        "evidence_ids": evidence_ids,
        "evidence": _evidence(state, evidence_ids),
        "comparison": _clone(decision["comparison"]),
        "budget": budget,
        "scope": "This proves only the recorded local AutoVQE promotion rule.",
        "reference_score": None,
        "reference_score_note": note,
    }


def render_json(value: Any) -> str:
    return json.dumps(_clone(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


__all__ = [
    "HISTORY_FILE",
    "PROBLEM_FILE",
    "RUN_FILE",
    "ResearchError",
    "execute_action",
    "execute_action_file",
    "initialize_run",
    "render_json",
    "run_result",
    "run_status",
]
