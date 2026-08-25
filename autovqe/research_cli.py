"""Small local interface for AutoVQE's closed research loop.

The command layer deliberately does only four things: create a run, apply one
agent action, show current state, and report a terminal result. Scientific
validation belongs to :mod:`autovqe.controller`; this module only persists the
run. Optimizer bindings are materialized only for an accepted terminal result.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_data
from .controller import (
    MAX_EXTERNAL_ACTION_BYTES,
    PROMOTION_EVALUATION_PROTOCOL,
    ResearchController,
)
from .evaluator import evaluate_public_problem
from .history import JsonlRunHistory
from .observations import observe_problem
from .problem import load_problem_document
from .research import (
    EvaluationStage,
    Lifecycle,
    ResearchLoop,
    ResearchState,
    comparison_point,
)


RUN_FILE = "run.json"
PROBLEM_FILE = "problem.json"
OBSERVATION_FILE = "observation.json"
HISTORY_FILE = "events.jsonl"
RUN_SCHEMA_VERSION = 1
_RUN_FIELDS = {"schema_version", "problem_path", "total_budget"}
_PRETERMINAL_HIDDEN_KEYS = {
    "optimized_parameter_binding",
    "best_values",
    "energy_trace",
    "best_energy_trace",
}


class ResearchCliError(RuntimeError):
    """Raised when a local research run or action file is invalid."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResearchCliError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ResearchCliError(f"non-finite JSON number is not allowed: {value}")


def _decode_json(text: str, *, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ResearchCliError:
        raise
    except json.JSONDecodeError as exc:
        raise ResearchCliError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchCliError(f"{source} must contain one JSON object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchCliError(f"required file is missing: {path}")
    try:
        return _decode_json(path.read_text(encoding="utf-8-sig"), source=path)
    except OSError as exc:
        raise ResearchCliError(f"cannot read {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def _read_action_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchCliError(f"action must be a regular JSON file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ResearchCliError(f"cannot inspect action file {path}: {exc}") from exc
    if size > MAX_EXTERNAL_ACTION_BYTES:
        raise ResearchCliError(
            f"action file exceeds {MAX_EXTERNAL_ACTION_BYTES} bytes: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ResearchCliError(f"cannot read action file {path}: {exc}") from exc
    if len(text.encode("utf-8")) > MAX_EXTERNAL_ACTION_BYTES:
        raise ResearchCliError(
            f"action file exceeds {MAX_EXTERNAL_ACTION_BYTES} bytes: {path}"
        )
    return _decode_json(text, source=path)


def _validated_budget(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchCliError("total_budget must be a number")
    budget = float(value)
    if not math.isfinite(budget) or budget <= 0:
        raise ResearchCliError("total_budget must be finite and positive")
    return budget


def _run_path(run_dir: Path) -> Path:
    return run_dir / RUN_FILE


def _observation_path(run_dir: Path) -> Path:
    return run_dir / OBSERVATION_FILE


def _problem_path(run_dir: Path) -> Path:
    return run_dir / PROBLEM_FILE


def _history_path(run_dir: Path) -> Path:
    return run_dir / HISTORY_FILE


def _load_context(run_dir: Path) -> dict[str, Any]:
    context = _read_json(_run_path(run_dir))
    if set(context) != _RUN_FIELDS:
        raise ResearchCliError(
            "invalid run fields: "
            f"missing={sorted(_RUN_FIELDS - set(context))} "
            f"extra={sorted(set(context) - _RUN_FIELDS)}"
        )
    if context["schema_version"] != RUN_SCHEMA_VERSION:
        raise ResearchCliError(
            f"unsupported research run schema: {context['schema_version']!r}"
        )
    problem_path = context["problem_path"]
    if not isinstance(problem_path, str) or not problem_path:
        raise ResearchCliError("run problem_path must be a non-empty string")
    context["total_budget"] = _validated_budget(context["total_budget"])
    return context


def _load_problem(run_dir: Path, context: Mapping[str, Any]):
    problem_path = Path(str(context["problem_path"]))
    try:
        problem, current = load_problem_document(problem_path)
    except Exception as exc:
        raise ResearchCliError(f"cannot load run problem {problem_path}: {exc}") from exc
    expected = _read_json(_problem_path(run_dir))
    if expected != current:
        raise ResearchCliError(
            "the Hamiltonian or its public constraints changed after research init"
        )
    return problem


def _preterminal_view(value: Any) -> Any:
    """Remove optimizer bindings from ordinary agent-facing command output."""

    if isinstance(value, Mapping):
        return {
            str(key): _preterminal_view(item)
            for key, item in value.items()
            if str(key) not in _PRETERMINAL_HIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_preterminal_view(item) for item in value]
    return copy.deepcopy(value)


def _next_candidate_action(state: ResearchState, candidate_id: str) -> str | None:
    record = state.candidates[candidate_id]
    if record.status is Lifecycle.PROMOTED:
        fair = any(
            evaluation.candidate_id != candidate_id
            and state.candidates[evaluation.candidate_id].hypothesis_id
            != record.hypothesis_id
            and comparison_point(evaluation) is not None
            for evaluation in state.evaluations.values()
        )
        return (
            "commit_or_dispose_after_comparison"
            if fair
            else "evaluate_different_hypothesis:promotion"
        )
    if record.status is Lifecycle.RETIRED:
        branch = state.hypotheses[record.hypothesis_id]
        return (
            "revise"
            if branch.status in {Lifecycle.READY, Lifecycle.SUPPORTED}
            else None
        )
    return {
        Lifecycle.CANDIDATE: "evaluate_candidate:audit",
        Lifecycle.AUDITED: "evaluate_candidate:smoke",
        Lifecycle.SMOKE: "evaluate_candidate:promotion",
    }.get(record.status)


def _audit_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    occurrences = value.get("parameter_occurrences", {})
    occurrence_total = (
        sum(int(item) for item in occurrences.values())
        if isinstance(occurrences, Mapping)
        else None
    )
    return {
        "unique_trainable_params": value.get("unique_trainable_params"),
        "parameter_occurrences": occurrence_total,
        "operations": value.get("operations"),
        "logical_macros": copy.deepcopy(value.get("logical_macros", {})),
        "spec_nodes": value.get("spec_nodes"),
    }


def _resource_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        "eligible": value.get("eligible"),
        "observed": copy.deepcopy(value.get("observed", {})),
        "violations": copy.deepcopy(value.get("violations", [])),
    }


def _symmetry_audit_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    summary: dict[str, Any] = {}
    constraints = value.get("constraints")
    if isinstance(constraints, Mapping):
        summary["constraints"] = copy.deepcopy(dict(constraints))

    relevance = value.get("special_operation_relevance")
    if isinstance(relevance, (list, tuple)):
        macro_counts: dict[str, int] = {}
        evidence_ids: set[str] = set()
        max_residual = 0.0
        for item in relevance:
            if not isinstance(item, Mapping):
                continue
            macro = item.get("macro")
            if isinstance(macro, str):
                macro_counts[macro] = macro_counts.get(macro, 0) + 1
            checks = item.get("constraints")
            if not isinstance(checks, Mapping):
                continue
            evidence_ids.update(str(key) for key in checks)
            for check in checks.values():
                if not isinstance(check, Mapping):
                    continue
                residual = check.get("residual")
                if isinstance(residual, (int, float)) and not isinstance(residual, bool):
                    max_residual = max(max_residual, float(residual))
        summary["special_operations"] = {
            "count": len(relevance),
            "macro_counts": macro_counts,
            "evidence_ids": sorted(evidence_ids),
            "max_residual": max_residual,
        }

    for key, item in value.items():
        if key not in {"constraints", "special_operation_relevance"} and isinstance(
            item, (str, int, float, bool)
        ):
            summary[str(key)] = copy.deepcopy(item)
    return summary


def _evaluation_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: copy.deepcopy(metrics[key])
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
            "violations",
        )
        if key in metrics
    }
    return summary


def _audit_record_summary(record: Any) -> dict[str, Any]:
    metrics = record.metrics
    summary: dict[str, Any] = {
        "evaluation_id": record.evaluation_id,
        "passed": record.passed,
    }
    for key in ("valid", "violations"):
        if key in metrics:
            summary[key] = copy.deepcopy(metrics[key])
    symmetry = _symmetry_audit_summary(metrics.get("symmetry_audit"))
    if symmetry is not None:
        summary["symmetry"] = symmetry
    circuit = _audit_summary(metrics.get("audit"))
    if circuit is not None:
        summary["circuit"] = circuit
    resources = _resource_summary(metrics.get("resource_policy"))
    if resources is not None:
        summary["resources"] = resources
    return summary


def _latest_evaluation_summary(record: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "evaluation_id": record.evaluation_id,
        "stage": record.stage.value,
        "passed": record.passed,
        "cost": record.cost,
    }
    if record.stage is not EvaluationStage.AUDIT:
        details = _evaluation_summary(record.metrics)
        if details:
            summary["summary"] = details
    return summary


def _latest_probe_summary(record: Any) -> dict[str, Any]:
    result = record.result
    summary: dict[str, Any] = {
        "probe_id": record.probe_id,
        "verdict": record.verdict.value,
        "cost": record.cost,
    }
    for key in ("probe_type", "metrics", "valid", "violations"):
        if key in result:
            summary[key] = copy.deepcopy(result[key])
    return summary


def compact_state(state: ResearchState) -> dict[str, Any]:
    """Return one bounded decision summary per research branch."""

    hypotheses: dict[str, Any] = {}
    for hypothesis_id, record in sorted(state.hypotheses.items()):
        summary: dict[str, Any] = {
            "kind": record.claim.get("kind"),
            "status": record.status.value,
            "next_action": {
                "PROPOSED": "request_probe",
                "READY": "submit_candidate",
                "SUPPORTED": "submit_candidate",
                "REFUTED": "revise_or_retire",
                "INCONCLUSIVE": "revise_or_retire",
            }.get(record.status.value),
        }
        if record.probe_ids:
            latest_probe = state.probes.get(record.probe_ids[-1])
            if latest_probe is not None:
                summary["latest_probe"] = _latest_probe_summary(latest_probe)
        for key, value in (
            ("parent_id", record.parent_id),
            ("revised_to", record.revised_to),
            ("retired_reason", record.retired_reason),
        ):
            if value is not None:
                summary[key] = value
        hypotheses[hypothesis_id] = summary

    candidates: dict[str, Any] = {}
    for candidate_id, record in sorted(state.candidates.items()):
        summary = {
            "hypothesis_id": record.hypothesis_id,
            "status": record.status.value,
            "next_action": _next_candidate_action(state, candidate_id),
        }
        evaluations = [
            state.evaluations[evaluation_id]
            for evaluation_id in record.evaluation_ids
            if evaluation_id in state.evaluations
        ]
        audit_record = next(
            (
                evaluation
                for evaluation in evaluations
                if evaluation.stage is EvaluationStage.AUDIT
            ),
            None,
        )
        if audit_record is not None:
            summary["audit_summary"] = _audit_record_summary(audit_record)
        if evaluations:
            summary["latest_evaluation"] = _latest_evaluation_summary(evaluations[-1])
        if record.symmetry_evidence_ids:
            summary["symmetry_evidence_ids"] = list(record.symmetry_evidence_ids)
        if record.disposition_evidence_ids:
            summary["disposition_evidence_ids"] = list(
                record.disposition_evidence_ids
            )
        for key, value in (
            ("parent_id", record.parent_id),
            ("revised_to", record.revised_to),
            ("retired_reason", record.retired_reason),
        ):
            if value is not None:
                summary[key] = value
        candidates[candidate_id] = summary

    return {
        "terminal_decision": state.terminal_decision,
        "budget": {
            "spent": state.spent_budget,
            "remaining": state.remaining_budget,
            "total": state.total_budget,
        },
        "hypotheses": hypotheses,
        "candidates": candidates,
    }


def initialize_run(
    problem_path: str | Path,
    run_dir: str | Path,
    *,
    total_budget: float,
) -> dict[str, Any]:
    destination = Path(run_dir)
    budget = _validated_budget(total_budget)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ResearchCliError(f"research run already exists: {destination}")

    source = Path(problem_path).resolve()
    try:
        problem, document = load_problem_document(source)
    except Exception as exc:
        raise ResearchCliError(f"cannot load problem {source}: {exc}") from exc
    observation = observe_problem(problem)

    destination.mkdir(parents=True, exist_ok=True)
    context = {
        "schema_version": RUN_SCHEMA_VERSION,
        "problem_path": str(source),
        "total_budget": budget,
    }
    _write_json(_run_path(destination), context)
    _write_json(_problem_path(destination), document)
    _write_json(_observation_path(destination), observation)

    history = JsonlRunHistory(_history_path(destination))
    history.read_events()
    state = ResearchLoop(history, total_budget=budget).state
    return _preterminal_view(
        {
            "run_dir": str(destination),
            "observation": observation,
            "state": compact_state(state),
        }
    )


def load_controller(run_dir: str | Path) -> ResearchController:
    directory = Path(run_dir)
    context = _load_context(directory)
    problem = _load_problem(directory, context)
    return ResearchController(
        problem,
        _history_path(directory),
        total_budget=float(context["total_budget"]),
    )


def execute_action(
    run_dir: str | Path,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        raise ResearchCliError("action must be a JSON object")
    result = load_controller(run_dir).dispatch_external(action).to_dict()
    return _preterminal_view(result)


def execute_action_file(
    run_dir: str | Path,
    action_path: str | Path,
) -> dict[str, Any]:
    return execute_action(run_dir, _read_action_file(Path(action_path)))


def run_status(run_dir: str | Path, *, full: bool = False) -> dict[str, Any]:
    directory = Path(run_dir)
    context = _load_context(directory)
    _load_problem(directory, context)
    history = JsonlRunHistory(_history_path(directory))
    state = ResearchLoop(
        history,
        total_budget=float(context["total_budget"]),
    ).state
    return _preterminal_view(
        {
            "run_dir": str(directory),
            "events": len(history.read_events()),
            "state": state.to_dict() if full else compact_state(state),
        }
    )


def _cited_evidence(
    state: ResearchState,
    evidence_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Return evaluator-owned records cited by a terminal decision."""

    state_data = state.to_dict()
    records: dict[str, Any] = {}
    for evidence_id in evidence_ids:
        if evidence_id in state_data["probes"]:
            records[evidence_id] = {
                "kind": "probe",
                **state_data["probes"][evidence_id],
            }
        elif evidence_id in state_data["evaluations"]:
            records[evidence_id] = {
                "kind": "evaluation",
                **state_data["evaluations"][evidence_id],
            }
        else:
            raise ResearchCliError(
                f"terminal decision cites unknown evidence: {evidence_id}"
            )
    return _preterminal_view(records)


def run_result(run_dir: str | Path) -> dict[str, Any]:
    """Return the final scientific result after an accepted terminal decision."""

    controller = load_controller(run_dir)
    state = controller.state
    if not state.terminal:
        raise ResearchCliError(
            "research run is not terminal; commit a promoted candidate or close negative"
        )

    if state.negative_closed:
        cited = tuple(state.negative_close_evidence_ids)
        return {
            "decision": "negative_close",
            "reason": state.negative_close_reason,
            "coverage": copy.deepcopy(
                state.negative_close_metadata.get("coverage", {})
            ),
            "evidence_ids": list(cited),
            "evidence": _cited_evidence(state, cited),
            "branches": compact_state(state),
            "budget": {
                "spent": state.spent_budget,
                "total": state.total_budget,
            },
            "scope": (
                "This closes only the recorded investigated branches under the "
                "local AutoVQE rule; it does not prove that no useful ansatz exists."
            ),
            "reference_score": None,
            "reference_score_note": (
                "No independent reference score was provided; this result cannot "
                "claim exact ground-state accuracy."
            ),
        }

    candidate_id = state.committed_candidate_id
    if candidate_id is None:
        raise ResearchCliError("terminal run has no recorded decision")
    candidate = state.candidates[candidate_id]
    cited = tuple(state.commit_metadata.get("evidence_ids", ()))
    promotions = [
        state.evaluations[evidence_id]
        for evidence_id in cited
        if evidence_id in state.evaluations
        and state.evaluations[evidence_id].candidate_id == candidate_id
        and state.evaluations[evidence_id].stage is EvaluationStage.PROMOTION
        and state.evaluations[evidence_id].passed
    ]
    if len(promotions) != 1:
        raise ResearchCliError(
            "committed candidate must cite exactly one passed promotion evaluation"
        )
    promotion = promotions[0]
    comparison = state.commit_metadata.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ResearchCliError("terminal commit is missing its recorded comparison")
    metrics = copy.deepcopy(dict(promotion.metrics))
    metrics.pop("optimized_parameter_binding", None)
    energy = metrics.get("best_energy")
    resources = metrics.get("metrics")
    audit = metrics.get("audit")
    if energy is None or not isinstance(resources, Mapping) or not isinstance(audit, Mapping):
        raise ResearchCliError("terminal promotion is missing evaluator results")

    try:
        materialized = evaluate_public_problem(
            controller.problem,
            candidate.spec,
            protocol=PROMOTION_EVALUATION_PROTOCOL,
        ).result
    except Exception as exc:
        raise ResearchCliError(
            f"terminal promotion parameters could not be reproduced: {exc}"
        ) from exc
    parameters = materialized.optimized_parameter_binding
    if (
        not materialized.valid
        or materialized.best_energy is None
        or not isinstance(parameters, Mapping)
    ):
        raise ResearchCliError(
            "terminal promotion parameters could not be reproduced by the fixed evaluator"
        )
    if not math.isclose(
        float(materialized.best_energy),
        float(energy),
        rel_tol=1e-10,
        abs_tol=1e-9,
    ):
        raise ResearchCliError(
            "terminal promotion no longer reproduces its recorded evaluator energy"
        )
    if canonical_data(materialized.audit) != canonical_data(audit) or canonical_data(
        materialized.metrics
    ) != canonical_data(resources):
        raise ResearchCliError(
            "terminal promotion no longer reproduces its recorded resource audit"
        )

    scope = (
        "This is the result of the recorded local AutoVQE promotion rule, "
        "not proof of the exact ground state or cross-problem generalization."
    )

    return {
        "decision": "positive_commit",
        "candidate_id": candidate_id,
        "ansatz": canonical_data(candidate.spec),
        "energy": float(energy),
        "optimized_parameters": canonical_data(parameters),
        "resources": canonical_data(resources),
        "audit": canonical_data(audit),
        "promotion_evaluation_id": promotion.evaluation_id,
        "evidence_ids": list(cited),
        "evidence": _cited_evidence(state, cited),
        "comparison": canonical_data(comparison),
        "budget": {
            "spent": state.spent_budget,
            "total": state.total_budget,
        },
        "scope": scope,
        "reference_score": None,
        "reference_score_note": (
            "No independent reference score was provided; this result cannot claim "
            "exact ground-state accuracy."
        ),
    }


def render_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = [
    "HISTORY_FILE",
    "OBSERVATION_FILE",
    "PROBLEM_FILE",
    "RUN_FILE",
    "ResearchCliError",
    "compact_state",
    "execute_action",
    "execute_action_file",
    "initialize_run",
    "load_controller",
    "render_json",
    "run_result",
    "run_status",
]
