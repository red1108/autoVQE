"""Closed hypothesis, probe, audit, evaluation, and decision loop."""
from __future__ import annotations
import copy
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping
from .ansatz import AnsatzIRValidationError, AnsatzSpec, pauli_label
from .evaluator import EvaluationProtocol, audit_public_candidate, candidate_identity, evaluate_public_problem
from .problem import PublicProblem, canonical_data, decode_json_object, load_problem_document, observe_problem
from .probes import EXACT_SYMMETRY_TOLERANCE, ProbeValidationError, generator_from_recipe, initial_state_circuit, initial_state_moments, operation_symmetry_residuals, run_public_probe, validate_special_operation_relevance

RUN_FILE, PROBLEM_FILE, HISTORY_FILE, RUN_SCHEMA_VERSION = "run.json", "problem.json", "events.jsonl", 2
MAX_BUDGET, MAX_EVENTS, MAX_EXTERNAL_ACTION_BYTES = 100.0, 200, 1_000_000
MAX_ACTIVE_HYPOTHESES, MAX_ACTIVE_CANDIDATES = 3, 2
MAX_OPERATIONS, MAX_PARAMETERS, MAX_FANOUT = 256, 128, 64
RESOURCE_LIMITS = {"twoq_count": 512, "total_gate_count": 2048, "depth": 1024}
STAGE_STATUS = {"audit": "AUDITED", "smoke": "SMOKE", "promotion": "PROMOTED"}
NEXT_STAGE = {"AUDITED": "smoke", "SMOKE": "promotion"}
PROTOCOLS = {"smoke": (2.0, EvaluationProtocol(max_evals=32, restarts=1, seed=7)), "promotion": (6.0, EvaluationProtocol(max_evals=96, restarts=3, seed=997))}
TERMINAL, ENERGY_TOLERANCE, MIN_IMPROVEMENT = {"REVISED", "RETIRED"}, 5e-4, 1e-6
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
NEW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
SCHEMAS = {
    "propose_hypothesis": ({"type", "hypothesis_id", "family"}, {"prediction", "falsifier"}),
    "request_symmetry_probe": ({"type", "probe_id", "generator"}, set()),
    "submit_candidate": ({"type", "candidate_id", "hypothesis_id", "spec"}, {"symmetry_evidence_ids"}),
    "evaluate_candidate": ({"type", "candidate_id"}, set()),
    "revise_hypothesis": ({"type", "source_id", "new_id", "family", "reason"}, {"prediction", "falsifier"}),
    "retire_hypothesis": ({"type", "hypothesis_id", "reason"}, set()),
    "retire_candidate": ({"type", "candidate_id", "reason"}, set()),
    "commit": ({"type", "candidate_id"}, set()),
    "close_negative": ({"type", "reason"}, set()),
}

class ResearchError(RuntimeError):
    pass

def render_json(value: Any) -> str:
    return json.dumps(canonical_data(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)

def _finite(value: Any, field: str, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or positive and value == 0
    ):
        raise ResearchError(f"{field} must be a finite {'positive' if positive else 'non-negative'} number")
    return float(value)

def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchError(f"{field} must be a non-empty string")
    return value.strip()

def _id(value: Any, field: str, new: bool = False) -> str:
    pattern = NEW_ID if new else ID
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ResearchError(f"{field} must match {pattern.pattern!r}")
    return value

def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ResearchError(f"{field} keys must be strings")
    try:
        value = copy.deepcopy(dict(value))
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResearchError(f"{field} must contain finite JSON data: {exc}") from exc
    return value

def _strict(action: Mapping[str, Any], required: set[str], optional: set[str]) -> None:
    missing, extra = required - set(action), set(action) - required - optional
    if missing or extra:
        raise ResearchError(
            f"invalid external action fields: missing={sorted(missing)} extra={sorted(extra)}"
        )

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchError(f"required file is missing: {path}")
    try:
        return decode_json_object(path.read_text(encoding="utf-8-sig"), path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc

def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file():
        raise ResearchError(f"history path is not a file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ResearchError(f"cannot read {path}: {exc}") from exc
    result = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ResearchError(f"blank history line at {number}")
        try:
            event = decode_json_object(line, f"{path}:{number}")
        except ValueError as exc:
            raise ResearchError(str(exc)) from exc
        if set(event) != {"seq", "type", "cost", "payload"} or event["seq"] != len(result):
            raise ResearchError(f"invalid or non-contiguous history at line {number}")
        _text(event["type"], "event type")
        _finite(event["cost"], "event cost")
        _mapping(event["payload"], "event payload")
        result.append(event)
    return result

def _state(budget: float) -> dict[str, Any]:
    return {"total_budget": budget, "spent_budget": 0.0, "last_seq": -1, "hypotheses": {}, "probes": {}, "candidates": {}, "evaluations": {}, "terminal": None}

def _store_evaluation(state: dict[str, Any], candidate_id: str, payload: Mapping[str, Any]) -> str:
    stage = payload["stage"]
    if stage not in STAGE_STATUS: raise ResearchError(f"unsupported evaluation stage: {stage!r}")
    state["evaluations"][payload["evaluation_id"]] = {"candidate_id": candidate_id, "stage": stage, "passed": payload["passed"], "metrics": payload["metrics"]}
    return STAGE_STATUS[stage] if payload["passed"] else "RETIRED"

def _apply(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    if state["terminal"] is not None: raise ResearchError("history continues after a terminal decision")
    cost = _finite(event["cost"], "event cost")
    payload = _mapping(event["payload"], "event payload")
    kind = event["type"]
    if state["spent_budget"] + cost > state["total_budget"] + 1e-12:
        raise ResearchError("history exceeds its research budget")
    hypotheses, candidates = state["hypotheses"], state["candidates"]
    if kind == "propose_hypothesis":
        hypotheses[payload["hypothesis_id"]] = {"family": payload["family"], "prediction": payload.get("prediction"), "falsifier": payload.get("falsifier"), "status": "READY", "parent_id": None}
    elif kind == "record_symmetry_probe":
        state["probes"][payload["probe_id"]] = {key: payload[key] for key in ("generator", "verdict", "result")}
    elif kind == "submit_candidate":
        candidate_id, audit = payload["candidate_id"], payload["audit"]
        candidates[candidate_id] = {"hypothesis_id": payload["hypothesis_id"], "spec": payload["spec"], "status": _store_evaluation(state, candidate_id, {**audit, "stage": "audit"}), "symmetry_evidence_ids": payload.get("symmetry_evidence_ids", [])}
    elif kind == "record_evaluation":
        candidates[payload["candidate_id"]]["status"] = _store_evaluation(state, payload["candidate_id"], payload)
    elif kind == "revise_hypothesis":
        hypotheses[payload["source_id"]]["status"] = "REVISED"
        hypotheses[payload["new_id"]] = {"family": payload["family"], "prediction": payload.get("prediction"), "falsifier": payload.get("falsifier"), "status": "READY", "parent_id": payload["source_id"]}
    elif kind in {"retire_hypothesis", "retire_candidate"}:
        (hypotheses if kind == "retire_hypothesis" else candidates)[payload["entity_id"]]["status"] = "RETIRED"
    elif kind in {"commit", "close_negative"}:
        state["terminal"] = {"decision": "positive_commit" if kind == "commit" else "negative_close", **payload}
    else:
        raise ResearchError(f"unsupported event type in history: {kind!r}")
    state["spent_budget"] += cost
    state["last_seq"] = int(event["seq"])

def _replay(run_dir: Path, budget: float) -> dict[str, Any]:
    state = _state(budget)
    for event in _events(run_dir / HISTORY_FILE):
        _apply(state, event)
    return state

def _append(run_dir: Path, state: Mapping[str, Any], kind: str, payload: Mapping[str, Any], cost: float) -> dict[str, Any]:
    path, events = run_dir / HISTORY_FILE, _events(run_dir / HISTORY_FILE)
    if len(events) != state["last_seq"] + 1: raise ResearchError("history changed before append")
    record = {"seq": len(events), "type": _text(kind, "event type"), "cost": _finite(cost, "event cost"), "payload": _mapping(payload, "event payload")}
    projected = copy.deepcopy(dict(state))
    _apply(projected, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return projected

def _evals(state: Mapping[str, Any], candidate_id: str) -> list[tuple[str, Mapping[str, Any]]]:
    return [(key, value) for key, value in state["evaluations"].items() if value["candidate_id"] == candidate_id]

def _root(state: Mapping[str, Any], hypothesis_id: str) -> str | None:
    if hypothesis_id not in state["hypotheses"]: return None
    root = hypothesis_id
    while state["hypotheses"][root].get("parent_id") is not None: root = state["hypotheses"][root]["parent_id"]
    return root

def _resource_errors(resources: Mapping[str, Any]) -> list[str]:
    return [f"{name}={resources.get(name)!r} exceeds or violates limit {limit}" for name, limit in RESOURCE_LIMITS.items() if type(resources.get(name)) is not int or not 0 <= resources[name] <= limit]

def _point(state: Mapping[str, Any], evaluation_id: str, evaluation: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics, energy, resources = evaluation["metrics"], evaluation["metrics"].get("best_energy"), evaluation["metrics"].get("resources")
    if evaluation["stage"] != "promotion" or evaluation["passed"] is not True or metrics.get("valid") is not True or isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy) or not isinstance(resources, Mapping) or _resource_errors(resources): return None
    candidate = state["candidates"][evaluation["candidate_id"]]
    return {"candidate_id": evaluation["candidate_id"], "root_id": _root(state, candidate["hypothesis_id"]), "evaluation_id": evaluation_id, "best_energy": float(energy), "resources": dict(resources)}

def _points(state: Mapping[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    points = [_point(state, key, value) for key, value in state["evaluations"].items() if value["candidate_id"] != candidate_id]
    return sorted((point for point in points if point is not None), key=lambda point: point["candidate_id"])

def _fair(state: Mapping[str, Any], candidate_id: str, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    root = _root(state, state["candidates"][candidate_id]["hypothesis_id"])
    return [point for point in points if point["root_id"] != root] if root is not None else []

def _dominates(target: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    if target["best_energy"] > other["best_energy"] + ENERGY_TOLERANCE: return True
    if abs(target["best_energy"] - other["best_energy"]) > ENERGY_TOLERANCE: return False
    return all(target["resources"][key] >= other["resources"][key] for key in target["resources"]) and any(target["resources"][key] > other["resources"][key] for key in target["resources"])

def _positive(state: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    candidate = state["candidates"].get(candidate_id)
    if candidate is None or candidate["status"] != "PROMOTED": raise ResearchError("commit requires a passed promotion")
    hypothesis = state["hypotheses"][candidate["hypothesis_id"]]
    if not any(isinstance(hypothesis.get(key), str) and hypothesis[key].strip() for key in ("prediction", "falsifier")): raise ResearchError("commit requires a preregistered prediction or falsifier")
    promotions = [item for item in _evals(state, candidate_id) if item[1]["stage"] == "promotion" and item[1]["passed"]]
    if len(promotions) != 1: raise ResearchError("commit requires exactly one passed promotion")
    target, comparisons = _point(state, *promotions[0]), _points(state, candidate_id)
    fair = _fair(state, candidate_id, comparisons)
    if target is None or not fair: raise ResearchError("commit requires evaluator-owned target and comparator evidence")
    dominator = next((item for item in comparisons if _dominates(target, item)), None)
    if dominator: raise ResearchError(f"target is dominated by comparator {dominator['candidate_id']}")
    evidence = [promotions[0][0], *(item["evaluation_id"] for item in comparisons)]
    comparison = {"energy_tolerance": ENERGY_TOLERANCE, "target": target, "evaluations": comparisons, "fair_comparator_evaluation_ids": [item["evaluation_id"] for item in fair]}
    return {"candidate_id": candidate_id, "promotion_evaluation_id": promotions[0][0], "evidence_ids": evidence, "comparison": comparison}

def _negative(state: Mapping[str, Any]) -> dict[str, Any]:
    live = [key for collection in (state["hypotheses"], state["candidates"]) for key, value in collection.items() if value["status"] not in TERMINAL]
    if live: raise ResearchError(f"negative close requires terminal branches: {live}")
    failed = [(key, value) for key, value in state["evaluations"].items() if value["stage"] in {"smoke", "promotion"} and not value["passed"] and value["metrics"].get("objective_calls", 0) > 0 and value["metrics"].get("objective_activity_fraction", 0) >= 1e-6]
    roots = {_root(state, state["candidates"][value["candidate_id"]]["hypothesis_id"]) for _, value in failed}
    deep = any(value["stage"] == "promotion" for _, value in failed)
    if not deep and len(roots) < 2: raise ResearchError("negative close needs an objective-active promotion failure or two independent failed structure lineages")
    return {"evidence_ids": [key for key, _ in failed], "coverage": {"search_mode": "promotion_depth" if deep else "structural_breadth", "structure_lineage_ids": sorted(root for root in roots if root is not None)}}

def _summary(state: Mapping[str, Any]) -> dict[str, Any]:
    hypotheses = {key: {"family": value["family"], "status": value["status"], "next_action": "submit_candidate" if value["status"] == "READY" else None, **({"parent_id": value["parent_id"]} if value.get("parent_id") else {})} for key, value in sorted(state["hypotheses"].items())}
    candidates = {}
    for key, value in sorted(state["candidates"].items()):
        status, evaluations = value["status"], _evals(state, key)
        if status == "PROMOTED": next_action = "commit_or_dispose_after_comparison" if _fair(state, key, _points(state, key)) else "evaluate_different_hypothesis:promotion"
        else: next_action = f"evaluate_candidate:{NEXT_STAGE[status]}" if status in NEXT_STAGE else None
        item = {"hypothesis_id": value["hypothesis_id"], "status": status, "next_action": next_action}
        if evaluations: item["latest_evaluation"] = {"evaluation_id": evaluations[-1][0], "stage": evaluations[-1][1]["stage"], "passed": evaluations[-1][1]["passed"], **evaluations[-1][1]["metrics"]}
        if value["symmetry_evidence_ids"]: item["symmetry_evidence_ids"] = list(value["symmetry_evidence_ids"])
        candidates[key] = item
    probes = {key: {"verdict": value["verdict"], "metrics": value["result"]["metrics"]} for key, value in sorted(state["probes"].items())}
    return {"terminal_decision": state["terminal"]["decision"] if state["terminal"] else None, "budget": {"spent": state["spent_budget"], "remaining": state["total_budget"] - state["spent_budget"], "total": state["total_budget"]}, "hypotheses": hypotheses, "symmetry_probes": probes, "candidates": candidates}

class ResearchController:
    def __init__(self, problem: PublicProblem, run_dir: str | Path, *, total_budget: float):
        self.problem, self.run_dir, self.total_budget = problem, Path(run_dir), total_budget

    @property
    def state(self) -> dict[str, Any]:
        return _replay(self.run_dir, self.total_budget)

    def _capacity(self, state: Mapping[str, Any], cost: float, events: int = 1, terminal: bool = False) -> None:
        if state["last_seq"] + 1 + events + (0 if terminal else 1) > MAX_EVENTS:
            raise ResearchError(f"research run reached {MAX_EVENTS} event cap")
        if state["spent_budget"] + cost > self.total_budget + 1e-12:
            remaining = self.total_budget - state["spent_budget"]
            raise ResearchError(f"action costs {cost}, remaining budget is {remaining}")

    def _emit(self, state: Mapping[str, Any], kind: str, payload: Mapping[str, Any], result: Mapping[str, Any], cost: float, *, capacity_cost: float | None = None, capacity_events: int = 1, terminal: bool = False) -> dict[str, Any]:
        self._capacity(state, cost if capacity_cost is None else capacity_cost, capacity_events, terminal)
        new_state = _append(self.run_dir, state, kind, payload, cost)
        return {"result": dict(result), "state_summary": _summary(new_state)}

    @staticmethod
    def _fields(action: Mapping[str, Any]) -> dict[str, Any]:
        fields = {"family": _text(action["family"], "family"), **{key: _text(action[key], key) for key in ("prediction", "falsifier") if key in action}}
        if not any(key in fields for key in ("prediction", "falsifier")):
            raise ResearchError("structure hypothesis must preregister a prediction or falsifier")
        return fields

    @staticmethod
    def _unique_family(state: Mapping[str, Any], family: str, allowed_root: str | None = None) -> None:
        key = " ".join(family.split()).casefold()
        duplicates = [item_id for item_id, item in state["hypotheses"].items() if " ".join(str(item["family"]).split()).casefold() == key and (allowed_root is None or _root(state, item_id) != allowed_root)]
        if duplicates: raise ResearchError(f"ansatz_structure family duplicates existing hypothesis: {duplicates}")

    def _hypothesis(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state, fields = self.state, self._fields(action)
        if action["type"] == "propose_hypothesis":
            new_id = _id(action["hypothesis_id"], "hypothesis_id", True)
            if new_id in state["hypotheses"]: raise ResearchError(f"hypothesis already exists: {new_id}")
            if sum(item["status"] not in TERMINAL for item in state["hypotheses"].values()) >= MAX_ACTIVE_HYPOTHESES: raise ResearchError(f"at most {MAX_ACTIVE_HYPOTHESES} hypotheses may be active")
            self._unique_family(state, fields["family"]); payload = {"hypothesis_id": new_id, **fields}; result = {"accepted": True, "hypothesis_id": new_id}; kind = "propose_hypothesis"
        else:
            source_id, new_id = _id(action["source_id"], "source_id"), _id(action["new_id"], "new_id", True)
            source = state["hypotheses"].get(source_id)
            if source is None or source["status"] == "REVISED": raise ResearchError("unknown or already revised hypothesis")
            if new_id in state["hypotheses"]: raise ResearchError(f"hypothesis already exists: {new_id}")
            active = [key for key, item in state["candidates"].items() if item["hypothesis_id"] == source_id and item["status"] not in TERMINAL]
            if active: raise ResearchError(f"retire active candidates before revision: {active}")
            self._unique_family(state, fields["family"], _root(state, source_id))
            if sum(key != source_id and item["status"] not in TERMINAL for key, item in state["hypotheses"].items()) >= MAX_ACTIVE_HYPOTHESES: raise ResearchError("hypothesis revision would exceed the active cap")
            payload = {"source_id": source_id, "new_id": new_id, **fields, "reason": _text(action["reason"], "reason")}; result = {"accepted": True, "new_id": new_id}; kind = "revise_hypothesis"
        return self._emit(state, kind, payload, result, 0.1)

    def _probe(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state, probe_id, recipe = self.state, _id(action["probe_id"], "probe_id", True), _mapping(action["generator"], "generator")
        if probe_id in state["probes"]: raise ResearchError(f"symmetry probe already exists: {probe_id}")
        if any(item["generator"] == recipe for item in state["probes"].values()): raise ResearchError("this symmetry generator was already probed")
        try: measured = run_public_probe(self.problem, {"type": "normalized_commutator", "generator": recipe})
        except Exception as exc: raise ResearchError(f"probe failed: {exc}") from exc
        passed = bool(measured.metrics.get("exact", False)); payload = {"probe_id": probe_id, "generator": recipe, "verdict": "supported" if passed else "refuted", "result": measured.to_dict()}
        return self._emit(state, "record_symmetry_probe", payload, {**measured.to_dict(), "probe_id": probe_id, "passed": passed}, measured.cost_units)

    @staticmethod
    def _evidence(state: Mapping[str, Any], raw: Any) -> list[str]:
        if raw is None: raw = []
        if not isinstance(raw, list): raise ResearchError("symmetry_evidence_ids must be a list")
        evidence = sorted({_id(item, "symmetry_evidence_ids") for item in raw})
        if any(item not in state["probes"] or state["probes"][item]["verdict"] != "supported" for item in evidence): raise ResearchError("symmetry evidence must cite supported probes")
        return evidence

    def _audit(self, state: Mapping[str, Any], spec: Mapping[str, Any], evidence: list[str]) -> tuple[bool, dict[str, Any]]:
        try:
            parsed = AnsatzSpec.from_dict(spec)
            if parsed.num_qubits != self.problem.num_qubits: raise ResearchError("candidate num_qubits must match the problem")
            resource = audit_public_candidate(self.problem, parsed)
            if not resource.valid: raise ResearchError("; ".join(resource.violations))
            audit = resource.audit
            if (
                not 0 < audit["operations"] <= MAX_OPERATIONS
                or not 0 < audit["unique_trainable_params"] <= MAX_PARAMETERS
            ):
                raise ResearchError("candidate needs bounded operations and parameters")
            fanout = {key: value for key, value in audit["parameter_occurrences"].items() if value > MAX_FANOUT}
            if fanout: raise ResearchError(f"parameter fan-out exceeds {MAX_FANOUT}: {fanout}")
            labels = {term.pauli for term in self.problem.pauli_terms}
            for operation in parsed.operations:
                is_untrusted_long_rotation = (
                    operation.gate == "PauliRotation"
                    and len(operation.qubits) > 2
                    and pauli_label(
                        self.problem.num_qubits, operation.qubits, operation.pauli or ""
                    ) not in labels
                )
                if is_untrusted_long_rotation:
                    raise ResearchError(
                        "PauliRotation above locality 2 must be a Hamiltonian term"
                    )
            special = [(index, operation) for index, operation in enumerate(parsed.operations) if operation.gate in {"XYExchange", "IsotropicExchange"}]
            if special and not evidence: raise ResearchError("conservation gates require supported symmetry evidence")
            charges, prepared = [], initial_state_circuit(self.problem)
            for evidence_id in evidence:
                probe = state["probes"][evidence_id]; charge = generator_from_recipe(self.problem.num_qubits, probe["generator"])
                residual = max(operation_symmetry_residuals(self.problem.num_qubits, parsed.operations, charge), default=0.0)
                if residual > EXACT_SYMMETRY_TOLERANCE: raise ResearchError(f"candidate breaks cited symmetry {evidence_id}: {residual:.3e}")
                _, variance = initial_state_moments(prepared, charge)
                if variance > EXACT_SYMMETRY_TOLERANCE: raise ResearchError(f"initial state has no definite sector for {evidence_id}")
                charges.append((charge, probe["result"]["metrics"]["residual"], variance))
            for index, operation in special:
                relevant = False
                failures = {}
                for evidence_id, (charge, residual, variance) in zip(evidence, charges, strict=True):
                    try:
                        validate_special_operation_relevance(
                            self.problem.num_qubits,
                            operation,
                            charge,
                            symmetry_residual=residual,
                            sector_variance=variance,
                        )
                        relevant = True
                        break
                    except ProbeValidationError as exc:
                        failures[evidence_id] = str(exc)
                if not relevant:
                    raise ResearchError(
                        f"special gate {index} has no relevant cited symmetry: {failures}"
                    )
            violations = _resource_errors(resource.resources)
            passed = not violations
            return passed, {"valid": passed, "audit": audit, "resources": resource.resources, "violations": violations}
        except (ResearchError, ProbeValidationError, AnsatzIRValidationError) as exc:
            return False, {"valid": False, "violations": [f"{type(exc).__name__}: {exc}"]}
        except Exception as exc:
            raise ResearchError(f"candidate audit infrastructure failed: {exc}") from exc

    def _submit(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state, candidate_id, hypothesis_id = self.state, _id(action["candidate_id"], "candidate_id", True), _id(action["hypothesis_id"], "hypothesis_id")
        if candidate_id in state["candidates"]: raise ResearchError(f"candidate already exists: {candidate_id}")
        if hypothesis_id not in state["hypotheses"] or state["hypotheses"][hypothesis_id]["status"] != "READY": raise ResearchError("candidate requires a READY structure hypothesis")
        try: spec = AnsatzSpec.from_dict(_mapping(action["spec"], "spec")).to_dict(); identity = candidate_identity(spec)
        except Exception as exc: raise ResearchError(f"invalid candidate: {exc}") from exc
        duplicates = [key for key, item in state["candidates"].items() if candidate_identity(item["spec"]) == identity]
        if duplicates: raise ResearchError(f"candidate is semantically equivalent to existing {duplicates}")
        if sum(item["hypothesis_id"] == hypothesis_id and item["status"] not in TERMINAL for item in state["candidates"].values()) >= MAX_ACTIVE_CANDIDATES: raise ResearchError("too many active candidates under this hypothesis")
        evidence = self._evidence(state, action.get("symmetry_evidence_ids"))
        self._capacity(state, 0.35)
        passed, metrics = self._audit(state, spec, evidence)
        evaluation_id = f"evaluation:{candidate_id}:audit"; audit = {"evaluation_id": evaluation_id, "passed": passed, "metrics": metrics}
        payload = {"candidate_id": candidate_id, "hypothesis_id": hypothesis_id, "spec": spec, "symmetry_evidence_ids": evidence, "audit": audit}
        result = {"accepted": True, "candidate_id": candidate_id, "audit_evaluation_id": evaluation_id, "audit_passed": passed, **metrics}
        return self._emit(state, "submit_candidate", payload, result, 0.35)

    def _evaluate(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state, candidate_id = self.state, _id(action["candidate_id"], "candidate_id")
        candidate = state["candidates"].get(candidate_id)
        if candidate is None: raise ResearchError(f"unknown candidate: {candidate_id}")
        stage = NEXT_STAGE.get(candidate["status"])
        if stage is None: raise ResearchError(f"candidate has no next evaluation in {candidate['status']}")
        cost, protocol = PROTOCOLS[stage]; capacity_cost, capacity_events = cost, 1
        points = _points(state, candidate_id)
        if stage == "promotion" and not _fair(state, candidate_id, points):
            root = _root(state, candidate["hypothesis_id"])
            ready = [key for key, item in state["candidates"].items() if key != candidate_id and item["status"] == "SMOKE" and _root(state, item["hypothesis_id"]) != root]
            if not ready: raise ResearchError("promotion requires a candidate from a different structure root that passed smoke")
            capacity_cost, capacity_events = 2 * cost, 2
        self._capacity(state, capacity_cost, capacity_events)
        evaluation = evaluate_public_problem(self.problem, candidate["spec"], protocol=protocol)
        if not evaluation.valid: raise ResearchError(f"optimizer failed without evidence: {list(evaluation.violations)}")
        metrics = evaluation.to_dict(); metrics.pop("optimized_parameter_binding", None)
        if evaluation.baseline_energy is None: raise ResearchError("evaluator omitted the baseline energy")
        improvement = None if evaluation.best_energy is None else evaluation.baseline_energy - evaluation.best_energy
        threshold = max(MIN_IMPROVEMENT, MIN_IMPROVEMENT * abs(evaluation.baseline_energy))
        violations = _resource_errors(evaluation.resources)
        metrics.update(energy_improvement=improvement, required_energy_improvement=threshold, violations=[*evaluation.violations, *violations])
        passed = bool(not violations and improvement is not None and improvement >= threshold)
        if passed and stage == "promotion":
            smoke = [item["metrics"].get("best_energy") for _, item in _evals(state, candidate_id) if item["stage"] == "smoke" and item["passed"]]
            passed = bool(smoke and evaluation.best_energy is not None and evaluation.best_energy <= min(smoke) + ENERGY_TOLERANCE)
        evaluation_id = f"evaluation:{candidate_id}:{stage}"; payload = {"candidate_id": candidate_id, "evaluation_id": evaluation_id, "stage": stage, "passed": passed, "metrics": metrics}
        result = {key: payload[key] for key in ("candidate_id", "evaluation_id", "stage", "passed")}
        return self._emit(state, "record_evaluation", payload, {**result, **metrics}, cost, capacity_cost=capacity_cost, capacity_events=capacity_events)

    @staticmethod
    def _disposition(state: Mapping[str, Any], candidate_id: str) -> list[str]:
        records = [item for item in _evals(state, candidate_id) if item[1]["stage"] == "promotion" and item[1]["passed"]]
        if len(records) != 1: raise ResearchError("promoted candidate lacks one passed promotion")
        target = _point(state, *records[0]); dominators = [item for item in _points(state, candidate_id) if target is not None and _dominates(target, item)]
        if not dominators: raise ResearchError("promoted candidate may be disposed only after a dominating promotion")
        return [records[0][0], *(item["evaluation_id"] for item in dominators)]

    def _retire(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state, candidate_action = self.state, action["type"] == "retire_candidate"
        field, collection = ("candidate_id", state["candidates"]) if candidate_action else ("hypothesis_id", state["hypotheses"])
        entity_id = _id(action[field], field); record = collection.get(entity_id)
        if record is None or record["status"] in TERMINAL: raise ResearchError(f"unknown or already terminal {'candidate' if candidate_action else 'hypothesis'}")
        if candidate_action: evidence = self._disposition(state, entity_id) if record["status"] == "PROMOTED" else []
        else:
            live = [key for key, item in state["candidates"].items() if item["hypothesis_id"] == entity_id and item["status"] not in TERMINAL]
            if live: raise ResearchError(f"retire active candidates first: {live}")
            evidence = []
        payload = {"entity_id": entity_id, "reason": _text(action["reason"], "reason"), **({"evidence_ids": evidence} if candidate_action else {})}
        return self._emit(state, action["type"], payload, {"accepted": True, **({"evidence_ids": evidence} if candidate_action else {})}, 0.0)

    def _terminate(self, action: Mapping[str, Any]) -> dict[str, Any]:
        state = self.state
        if action["type"] == "commit":
            candidate_id = _id(action["candidate_id"], "candidate_id")
            decision = _positive(state, candidate_id)
            payload = {"candidate_id": candidate_id}
            kind = "commit"
        else:
            decision = _negative(state)
            payload = {"reason": _text(action["reason"], "reason")}
            kind = "close_negative"
        return self._emit(state, kind, payload, {"accepted": True, **decision}, 0.0, terminal=True)

    def dispatch_external(self, action: Mapping[str, Any]) -> dict[str, Any]:
        action = _mapping(action, "external action")
        if len(json.dumps(action, allow_nan=False).encode()) > MAX_EXTERNAL_ACTION_BYTES: raise ResearchError("external action exceeds size limit")
        if self.state["terminal"] is not None: raise ResearchError("research run is terminal")
        kind = action.get("type")
        if kind in {"record_symmetry_probe", "record_evaluation"}: raise ResearchError(f"{kind} is evaluator-owned")
        if kind not in SCHEMAS: raise ResearchError(f"unsupported external action type: {kind!r}")
        _strict(action, *SCHEMAS[kind])
        if kind in {"propose_hypothesis", "revise_hypothesis"}: return self._hypothesis(action)
        if kind == "request_symmetry_probe": return self._probe(action)
        if kind == "submit_candidate": return self._submit(action)
        if kind == "evaluate_candidate": return self._evaluate(action)
        if kind.startswith("retire_"): return self._retire(action)
        return self._terminate(action)

def _budget(value: Any) -> float:
    value = _finite(value, "total_budget", True)
    if value > MAX_BUDGET: raise ResearchError(f"total_budget cannot exceed {MAX_BUDGET}")
    return value

def initialize_run(problem_path: str | Path, run_dir: str | Path, *, total_budget: float) -> dict[str, Any]:
    destination, budget = Path(run_dir), _budget(total_budget)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())): raise ResearchError(f"research run already exists: {destination}")
    try: problem, document = load_problem_document(Path(problem_path).resolve())
    except Exception as exc: raise ResearchError(f"cannot load problem: {exc}") from exc
    destination.mkdir(parents=True, exist_ok=True)
    (destination / RUN_FILE).write_text(render_json({"schema_version": RUN_SCHEMA_VERSION, "total_budget": budget}) + "\n", encoding="utf-8")
    (destination / PROBLEM_FILE).write_text(render_json(document) + "\n", encoding="utf-8")
    return {"run_dir": str(destination), "observation": canonical_data(observe_problem(problem)), "state": _summary(_state(budget))}

def load_controller(run_dir: str | Path) -> ResearchController:
    directory, context = Path(run_dir), _read_json(Path(run_dir) / RUN_FILE)
    if set(context) != {"schema_version", "total_budget"} or context["schema_version"] != RUN_SCHEMA_VERSION: raise ResearchError("invalid or unsupported run context")
    try: problem, _ = load_problem_document(directory / PROBLEM_FILE)
    except Exception as exc: raise ResearchError(f"cannot load run-local problem snapshot: {exc}") from exc
    return ResearchController(problem, directory, total_budget=_budget(context["total_budget"]))

def execute_action(run_dir: str | Path, action: Mapping[str, Any]) -> dict[str, Any]: return load_controller(run_dir).dispatch_external(action)

def execute_action_file(run_dir: str | Path, action_path: str | Path) -> dict[str, Any]:
    path = Path(action_path)
    if not path.is_file(): raise ResearchError(f"action must be a regular JSON file: {path}")
    if path.stat().st_size > MAX_EXTERNAL_ACTION_BYTES: raise ResearchError("action file exceeds size limit")
    return execute_action(run_dir, _read_json(path))

def run_status(run_dir: str | Path) -> dict[str, Any]:
    controller = load_controller(run_dir); state = controller.state
    return {"run_dir": str(Path(run_dir)), "events": state["last_seq"] + 1, "state": _summary(state)}

def _evidence(state: Mapping[str, Any], ids: list[str]) -> dict[str, Any]:
    result = {}
    for evidence_id in ids:
        if evidence_id in state["probes"]: result[evidence_id] = {"kind": "probe", **state["probes"][evidence_id]}
        elif evidence_id in state["evaluations"]: result[evidence_id] = {"kind": "evaluation", **state["evaluations"][evidence_id]}
        else: raise ResearchError(f"terminal decision cites unknown evidence: {evidence_id}")
    return result

def run_result(run_dir: str | Path) -> dict[str, Any]:
    controller = load_controller(run_dir); state = controller.state; terminal = state["terminal"]
    if terminal is None: raise ResearchError("research run is not terminal")
    budget, note = {"spent": state["spent_budget"], "total": state["total_budget"]}, "No independent reference score was provided; exact ground-state accuracy is not claimed."
    if terminal["decision"] == "negative_close":
        decision = _negative(state); ids = decision["evidence_ids"]
        return {"decision": "negative_close", "reason": terminal["reason"], "coverage": decision["coverage"], "evidence_ids": ids, "evidence": _evidence(state, ids), "branches": _summary(state), "budget": budget, "scope": "This closes only the investigated branches under the local AutoVQE rule.", "reference_score": None, "reference_score_note": note}
    candidate_id = terminal["candidate_id"]; decision = _positive(state, candidate_id); candidate = state["candidates"][candidate_id]
    evaluation_id = decision["promotion_evaluation_id"]; metrics = state["evaluations"][evaluation_id]["metrics"]
    resources, audit, energy = metrics.get("resources"), metrics.get("audit"), metrics.get("best_energy")
    if not isinstance(resources, Mapping) or not isinstance(audit, Mapping) or isinstance(energy, bool) or not isinstance(energy, (int, float)): raise ResearchError("terminal promotion lacks evaluator-owned results")
    replay = evaluate_public_problem(controller.problem, candidate["spec"], protocol=PROTOCOLS["promotion"][1]); binding = replay.optimized_parameter_binding
    if not replay.valid or replay.best_energy is None or not isinstance(binding, Mapping) or replay.resources != resources or not math.isclose(replay.best_energy, energy, rel_tol=0, abs_tol=1e-10): raise ResearchError("terminal promotion replay does not match recorded evidence")
    ids = decision["evidence_ids"]
    return {"decision": "positive_commit", "candidate_id": candidate_id, "ansatz": candidate["spec"], "energy": float(energy), "optimized_parameters": dict(binding), "resources": dict(resources), "audit": dict(audit), "promotion_evaluation_id": evaluation_id, "evidence_ids": ids, "evidence": _evidence(state, ids), "comparison": decision["comparison"], "budget": budget, "scope": "This proves only the recorded local AutoVQE promotion rule.", "reference_score": None, "reference_score_note": note}
