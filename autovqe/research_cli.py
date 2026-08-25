"""Small local interface for AutoVQE's closed research loop.

The command layer deliberately does only four things: create a run, apply one
agent action, show current state, and report a terminal result. Scientific
validation belongs to :mod:`autovqe.controller`; this module only persists the
run and keeps pre-terminal optimizer bindings out of normal command output.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import prepare
from .contracts import canonical_data
from .controller import MAX_EXTERNAL_ACTION_BYTES, ResearchController
from .history import JsonlRunHistory
from .observations import adapt_prepare_problem
from .research import EvaluationStage, ResearchLoop, ResearchState


RUN_FILE = "run.json"
OBSERVATION_FILE = "observation.json"
HISTORY_FILE = "events.jsonl"
RUN_SCHEMA_VERSION = 1
_RUN_FIELDS = {"schema_version", "problem_path", "total_budget"}
_PRETERMINAL_PRIVATE_KEYS = {
    "optimized_parameter_binding",
    "best_values",
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
        return _decode_json(path.read_text(encoding="utf-8"), source=path)
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
        text = path.read_text(encoding="utf-8")
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


def _load_problem_views(run_dir: Path, context: Mapping[str, Any]):
    problem_path = Path(str(context["problem_path"]))
    try:
        prepared = prepare.load_problem(problem_path)
    except Exception as exc:
        raise ResearchCliError(f"cannot load run problem {problem_path}: {exc}") from exc
    views = adapt_prepare_problem(prepared)
    expected = _read_json(_observation_path(run_dir))
    current = canonical_data(views.observation_bundle)
    if expected != current:
        raise ResearchCliError(
            "the Hamiltonian or its public constraints changed after research init"
        )
    return views


def _preterminal_view(value: Any) -> Any:
    """Remove optimizer bindings from ordinary agent-facing command output."""

    if isinstance(value, Mapping):
        return {
            str(key): _preterminal_view(item)
            for key, item in value.items()
            if str(key) not in _PRETERMINAL_PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_preterminal_view(item) for item in value]
    return copy.deepcopy(value)


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
        prepared = prepare.load_problem(source)
    except Exception as exc:
        raise ResearchCliError(f"cannot load problem {source}: {exc}") from exc
    views = adapt_prepare_problem(prepared)

    destination.mkdir(parents=True, exist_ok=True)
    context = {
        "schema_version": RUN_SCHEMA_VERSION,
        "problem_path": str(source),
        "total_budget": budget,
    }
    _write_json(_run_path(destination), context)
    _write_json(_observation_path(destination), views.observation_bundle)

    history = JsonlRunHistory(_history_path(destination))
    history.read_events()
    state = ResearchLoop(history, total_budget=budget).state
    return _preterminal_view(
        {
            "run_dir": str(destination),
            "observation": views.observation_bundle,
            "state": state.to_dict(),
        }
    )


def load_controller(run_dir: str | Path) -> ResearchController:
    directory = Path(run_dir)
    context = _load_context(directory)
    views = _load_problem_views(directory, context)
    return ResearchController(
        views.public_problem,
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


def run_status(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    context = _load_context(directory)
    _load_problem_views(directory, context)
    history = JsonlRunHistory(_history_path(directory))
    state = ResearchLoop(
        history,
        total_budget=float(context["total_budget"]),
    ).state
    return _preterminal_view(
        {
            "run_dir": str(directory),
            "events": len(history.read_events()),
            "state": state.to_dict(),
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

    scope = (
        "This is the result of the recorded local AutoVQE promotion rule, "
        "not proof of the exact ground state or cross-problem generalization."
    )
    if state.negative_closed:
        cited = tuple(state.negative_close_evidence_ids)
        return {
            "decision": "negative_close",
            "reason": state.negative_close_reason,
            "evidence_ids": list(cited),
            "evidence": _cited_evidence(state, cited),
            "branches": {
                "hypotheses": canonical_data(state.hypotheses),
                "candidates": canonical_data(state.candidates),
            },
            "budget": {
                "spent": state.spent_budget,
                "total": state.total_budget,
            },
            "scope": scope,
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
    parameters = metrics.pop("optimized_parameter_binding", None)
    if not isinstance(parameters, Mapping):
        raise ResearchCliError("terminal promotion is missing optimized parameters")
    energy = metrics.get("best_energy")
    resources = metrics.get("metrics")
    audit = metrics.get("audit")
    if energy is None or not isinstance(resources, Mapping) or not isinstance(audit, Mapping):
        raise ResearchCliError("terminal promotion is missing evaluator results")

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
    "RUN_FILE",
    "ResearchCliError",
    "execute_action",
    "execute_action_file",
    "initialize_run",
    "load_controller",
    "render_json",
    "run_result",
    "run_status",
]
