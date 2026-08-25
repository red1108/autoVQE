"""Trusted, compact controller for the AutoVQE research loop.

The external agent proposes scientific ideas and typed circuits. The
controller chooses deterministic evidence identifiers, fixes the evaluation
stage, performs every measurement, and records the resulting evidence.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .ansatz_ir import AnsatzIRValidationError, AnsatzSpec
from .compiler import compile_ansatz
from .contracts import PublicProblem
from .evaluator import (
    EvaluationProtocol,
    audit_public_candidate,
    candidate_identity,
    evaluate_public_problem,
)
from .problem import hamiltonian_from_problem
from .probes import (
    EXACT_SYMMETRY_TOLERANCE,
    ProbeValidationError,
    ProbeResult,
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
from .research import (
    COMMIT_ENERGY_TOLERANCE,
    EvaluationStage,
    Lifecycle,
    ProbeVerdict,
    ResearchLoop,
    ResearchState,
    TransitionError,
    comparison_dominates_target,
    comparison_point,
    derived_negative_close_evidence,
    validate_negative_close_coverage,
)


class ControllerError(RuntimeError):
    """Raised when an external action violates the controller contract."""


MAX_CANDIDATE_OPERATIONS = 256
MAX_CANDIDATE_PARAMETERS = 128
MAX_CANDIDATE_SPEC_NODES = 4096
MAX_PARAMETER_FANOUT = 64
MAX_CANONICAL_TWOQ_GATES = 512
MAX_CANONICAL_TOTAL_GATES = 2048
MAX_CANONICAL_DEPTH = 1024
MAX_ACTIVE_HYPOTHESES = 3
MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS = 2
MAX_EXTERNAL_ACTION_BYTES = 1_000_000
MAX_HISTORY_EVENTS = 200
MIN_SMOKE_ENERGY_IMPROVEMENT = 1e-6
SMOKE_EVALUATION_PROTOCOL = EvaluationProtocol(max_evals=32, restarts=1, seed=7)
PROMOTION_EVALUATION_PROTOCOL = EvaluationProtocol(
    max_evals=96, restarts=3, seed=997
)

EXACT_SYMMETRY_CLAIM = "exact_pauli_symmetry"
STRUCTURE_CLAIM = "ansatz_structure"
NULL_CONTROL_CLAIM = "null_control"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NEW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


@dataclass(frozen=True)
class StepResult:
    """Compact response to one external action."""

    action_type: str
    result: dict[str, Any]
    state_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_action(
    action: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(action)
    extra = set(action) - required - optional
    if missing or extra:
        raise ControllerError(
            f"invalid external action fields: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ControllerError(f"{field} must match {_ID_RE.pattern!r}")
    return value


def _new_identifier(value: Any, field: str) -> str:
    """Validate agent-created IDs while leaving room for controller prefixes."""

    if not isinstance(value, str) or not _NEW_ID_RE.fullmatch(value):
        raise ControllerError(f"{field} must match {_NEW_ID_RE.pattern!r}")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControllerError(f"{field} must be an object")
    return copy.deepcopy(dict(value))


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControllerError(f"{field} must be a non-empty string")
    return value.strip()


def _text_metadata(
    value: Any,
    field: str,
    *,
    allowed: set[str],
) -> dict[str, str]:
    metadata = _mapping(value, field)
    extra = set(metadata) - allowed
    if extra:
        raise ControllerError(f"{field} contains unsupported fields: {sorted(extra)}")
    return {key: _text(item, f"{field}.{key}") for key, item in metadata.items()}


def _preregistered(metadata: Mapping[str, Any]) -> bool:
    return any(
        isinstance(metadata.get(field), str) and str(metadata[field]).strip()
        for field in ("prediction", "falsifier")
    )


def _probe_passed(result: ProbeResult) -> bool:
    return result.probe_type == "normalized_commutator" and bool(
        result.metrics.get("exact", False)
    )


def _next_stage(status: Lifecycle) -> EvaluationStage | None:
    return {
        Lifecycle.CANDIDATE: EvaluationStage.AUDIT,
        Lifecycle.AUDITED: EvaluationStage.SMOKE,
        Lifecycle.SMOKE: EvaluationStage.PROMOTION,
    }.get(status)


def _has_fair_comparator(state: ResearchState, candidate_id: str) -> bool:
    candidate = state.candidates[candidate_id]
    return any(
        record.candidate_id != candidate_id
        and state.candidates[record.candidate_id].hypothesis_id
        != candidate.hypothesis_id
        and comparison_point(record) is not None
        for record in state.evaluations.values()
    )


def _state_summary(state: ResearchState) -> dict[str, Any]:
    hypotheses = {}
    for hypothesis_id, record in sorted(state.hypotheses.items()):
        next_action = {
            Lifecycle.PROPOSED: "request_probe",
            Lifecycle.READY: "submit_candidate",
            Lifecycle.SUPPORTED: "submit_candidate",
            Lifecycle.REFUTED: "revise_or_retire",
            Lifecycle.INCONCLUSIVE: "revise_or_retire",
        }.get(record.status)
        hypotheses[hypothesis_id] = {
            "status": record.status.value,
            "next_action": next_action,
        }
    candidates: dict[str, Any] = {}
    for candidate_id, record in sorted(state.candidates.items()):
        stage = _next_stage(record.status)
        branch = state.hypotheses[record.hypothesis_id]
        if record.status is Lifecycle.PROMOTED:
            next_action = (
                "commit_or_dispose_after_comparison"
                if _has_fair_comparator(state, candidate_id)
                else "evaluate_different_hypothesis:promotion"
            )
        elif record.status is Lifecycle.RETIRED:
            next_action = (
                "revise"
                if branch.status in {Lifecycle.READY, Lifecycle.SUPPORTED}
                else None
            )
        else:
            next_action = (
                f"evaluate_candidate:{stage.value}" if stage is not None else None
            )
        candidates[candidate_id] = {
            "hypothesis_id": record.hypothesis_id,
            "status": record.status.value,
            "next_action": next_action,
        }
    return {
        "budget": {
            "spent": state.spent_budget,
            "remaining": state.remaining_budget,
            "total": state.total_budget,
        },
        "last_seq": state.last_seq,
        "terminal_decision": state.terminal_decision,
        "hypotheses": hypotheses,
        "candidates": candidates,
    }


def _resource_eligibility(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fallbacks = {
        **{
            f"{prefix}_twoq_count": MAX_CANONICAL_TWOQ_GATES + 1
            for prefix in (
                "template",
                "audit_worst",
                "canonical_template",
                "canonical_audit_worst",
            )
        },
        **{
            f"{prefix}_total_gate_count": MAX_CANONICAL_TOTAL_GATES + 1
            for prefix in (
                "template",
                "audit_worst",
                "canonical_template",
                "canonical_audit_worst",
            )
        },
        **{
            f"{prefix}_depth": MAX_CANONICAL_DEPTH + 1
            for prefix in (
                "template",
                "audit_worst",
                "canonical_template",
                "canonical_audit_worst",
            )
        },
    }
    inputs: dict[str, int] = {}
    for name, fallback in fallbacks.items():
        value = metrics.get(name, fallback)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            value = fallback
        inputs[name] = int(value)
    observed = {
        "conservative_twoq_count": max(
            inputs[name] for name in inputs if name.endswith("_twoq_count")
        ),
        "conservative_total_gate_count": max(
            inputs[name]
            for name in inputs
            if name.endswith("_total_gate_count")
        ),
        "conservative_depth": max(
            inputs[name] for name in inputs if name.endswith("_depth")
        ),
    }
    limits = {
        "conservative_twoq_count": MAX_CANONICAL_TWOQ_GATES,
        "conservative_total_gate_count": MAX_CANONICAL_TOTAL_GATES,
        "conservative_depth": MAX_CANONICAL_DEPTH,
    }
    violations = [
        f"{name}={observed[name]} exceeds {limit}"
        for name, limit in limits.items()
        if observed[name] > limit
    ]
    return {
        "eligible": not violations,
        "inputs": inputs,
        "observed": observed,
        "limits": limits,
        "violations": violations,
    }


class ResearchController:
    """Validate agent actions and own every probe and evaluation result."""

    def __init__(
        self,
        problem: PublicProblem,
        history: str | Path,
        *,
        total_budget: float = 100.0,
    ):
        self.problem = problem
        self.loop = ResearchLoop(history, total_budget=total_budget)

    @property
    def state(self) -> ResearchState:
        return self.loop.state

    def _result(self, action_type: str, result: Mapping[str, Any]) -> StepResult:
        return StepResult(
            action_type=action_type,
            result=copy.deepcopy(dict(result)),
            state_summary=_state_summary(self.loop.state),
        )

    def _ensure_capacity(
        self,
        *,
        cost: float,
        events: int = 1,
        terminal: bool = False,
    ) -> None:
        state = self.loop.state
        terminal_reserve = 0 if terminal else 1
        if state.last_seq + 1 + events + terminal_reserve > MAX_HISTORY_EVENTS:
            suffix = "" if terminal else " while reserving one terminal event"
            raise ControllerError(
                f"research run reached {MAX_HISTORY_EVENTS} event cap{suffix}"
            )
        if cost > state.remaining_budget + 1e-12:
            raise ControllerError(
                f"action costs {cost}, remaining budget is {state.remaining_budget}"
            )

    def _validated_claim(self, value: Any) -> dict[str, Any]:
        claim = _mapping(value, "claim")
        kind = claim.get("kind")
        if kind == EXACT_SYMMETRY_CLAIM:
            if set(claim) != {"kind", "generator"}:
                raise ControllerError(
                    "exact_pauli_symmetry claim fields must be exactly kind and generator"
                )
            recipe = _mapping(claim["generator"], "claim.generator")
            try:
                generator = generator_from_recipe(self.problem.num_qubits, recipe)
                validate_symmetry_generator(
                    hamiltonian_from_problem(self.problem), generator
                )
            except Exception as exc:
                raise ControllerError(f"invalid symmetry generator: {exc}") from exc
            return {"kind": kind, "generator": recipe}
        if kind == STRUCTURE_CLAIM:
            if set(claim) != {"kind", "family"}:
                raise ControllerError(
                    "ansatz_structure claim fields must be exactly kind and family"
                )
            return {"kind": kind, "family": _text(claim["family"], "claim.family")}
        if kind == NULL_CONTROL_CLAIM:
            if set(claim) != {"kind"}:
                raise ControllerError("null_control claim contains unsupported fields")
            return {"kind": kind}
        raise ControllerError(
            "claim.kind must be exact_pauli_symmetry, ansatz_structure, or null_control"
        )

    def _propose_hypothesis(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(
            action,
            required={"type", "hypothesis_id", "claim"},
            optional={"metadata"},
        )
        hypothesis_id = _new_identifier(action["hypothesis_id"], "hypothesis_id")
        if hypothesis_id in self.loop.state.hypotheses:
            raise ControllerError(f"hypothesis already exists: {hypothesis_id}")
        self._ensure_capacity(cost=0.1)
        claim = self._validated_claim(action["claim"])
        active = sum(
            record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
            for record in self.loop.state.hypotheses.values()
        )
        if active >= MAX_ACTIVE_HYPOTHESES:
            raise ControllerError(
                f"at most {MAX_ACTIVE_HYPOTHESES} hypotheses may be active"
            )
        self.loop.dispatch(
            {
                "type": "propose_hypothesis",
                "hypothesis_id": hypothesis_id,
                "claim": claim,
                "metadata": _text_metadata(
                    action.get("metadata", {}),
                    "metadata",
                    allowed={"rationale", "prediction", "falsifier"},
                ),
                "cost": 0.1,
            }
        )
        requires_probe = claim["kind"] == EXACT_SYMMETRY_CLAIM
        return self._result(
            "propose_hypothesis",
            {
                "accepted": True,
                "claim_kind": claim["kind"],
                "requires_probe": requires_probe,
            },
        )

    def _request_probe(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(action, required={"type", "hypothesis_id"})
        hypothesis_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        hypothesis = self.loop.state.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ControllerError(f"unknown hypothesis: {hypothesis_id}")
        if hypothesis.status is not Lifecycle.PROPOSED:
            raise ControllerError(
                f"cannot probe hypothesis {hypothesis_id} in {hypothesis.status.value}"
            )
        if hypothesis.claim.get("kind") != EXACT_SYMMETRY_CLAIM:
            raise ControllerError("only exact_pauli_symmetry claims require probes")
        probe_id = f"probe:{hypothesis_id}"
        if probe_id in self.loop.state.probes:
            raise ControllerError(f"probe already exists: {probe_id}")
        request = {
            "type": "normalized_commutator",
            "generator": copy.deepcopy(hypothesis.claim["generator"]),
        }
        try:
            probe_cost = algebraic_probe_cost_units(
                hamiltonian_from_problem(self.problem), request
            )
        except Exception as exc:
            raise ControllerError(f"probe preflight failed: {exc}") from exc
        self._ensure_capacity(cost=probe_cost)
        result = run_public_probe(self.problem, request)
        if not math.isclose(result.cost_units, probe_cost, rel_tol=0.0, abs_tol=1e-12):
            raise ControllerError("probe preflight and result costs disagree")
        passed = _probe_passed(result)
        payload = result.to_dict()
        payload.update({"probe_id": probe_id, "passed": passed})
        self.loop.dispatch(
            {
                "type": "record_probe",
                "hypothesis_id": hypothesis_id,
                "probe_id": probe_id,
                "verdict": (
                    ProbeVerdict.SUPPORTED.value
                    if passed
                    else ProbeVerdict.REFUTED.value
                ),
                "result": result.to_dict(),
                "cost": result.cost_units,
            }
        )
        return self._result("request_probe", payload)

    def _ensure_unique_candidate(
        self,
        spec: Mapping[str, Any],
        *,
        representation_repair_source: str | None = None,
    ) -> None:
        try:
            identity = candidate_identity(spec)
        except Exception as exc:
            raise ControllerError(f"invalid candidate: {exc}") from exc
        duplicates = sorted(
            record.candidate_id
            for record in self.loop.state.candidates.values()
            if candidate_identity(record.spec) == identity
        )
        if duplicates == [representation_repair_source]:
            source = self.loop.state.candidates[representation_repair_source]
            evaluations = [
                self.loop.state.evaluations[evaluation_id]
                for evaluation_id in source.evaluation_ids
            ]
            if (
                evaluations
                and all(
                    evaluation.stage is EvaluationStage.AUDIT
                    for evaluation in evaluations
                )
                and any(not evaluation.passed for evaluation in evaluations)
            ):
                # No optimizer call was made. Permit one representation repair
                # when equivalent shorthand is required to pass an atomic audit.
                return
        if duplicates:
            raise ControllerError(
                "candidate is semantically equivalent to an existing candidate: "
                f"{duplicates}"
            )

    @staticmethod
    def _enforcement(kind: str, *, preserves_symmetry: bool = False) -> str:
        if kind == NULL_CONTROL_CLAIM:
            return "diagnostic"
        if preserves_symmetry:
            return "preserve"
        return {
            EXACT_SYMMETRY_CLAIM: "unconstrained",
            STRUCTURE_CLAIM: "unconstrained",
        }[kind]

    def _validated_symmetry_evidence_ids(
        self,
        raw: Any,
        *,
        primary_hypothesis_id: str,
    ) -> list[str]:
        if raw is None:
            values: list[Any] = []
        elif isinstance(raw, list):
            values = list(raw)
        else:
            raise ControllerError("symmetry_evidence_ids must be a list")
        evidence_ids = [
            _identifier(value, "symmetry_evidence_ids") for value in values
        ]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ControllerError("symmetry_evidence_ids must not contain duplicates")

        primary = self.loop.state.hypotheses[primary_hypothesis_id]
        if (
            primary.claim.get("kind") == EXACT_SYMMETRY_CLAIM
            and primary.status is Lifecycle.SUPPORTED
        ):
            evidence_ids.extend(primary.probe_ids)

        result: list[str] = []
        for evidence_id in sorted(set(evidence_ids)):
            probe = self.loop.state.probes.get(evidence_id)
            if probe is None or probe.verdict is not ProbeVerdict.SUPPORTED:
                raise ControllerError(
                    "symmetry_evidence_ids must cite evaluator-supported probes: "
                    f"{evidence_id}"
                )
            hypothesis = self.loop.state.hypotheses[probe.hypothesis_id]
            if hypothesis.claim.get("kind") != EXACT_SYMMETRY_CLAIM:
                raise ControllerError(
                    f"symmetry evidence is not an exact symmetry: {evidence_id}"
                )
            result.append(evidence_id)
        return result

    def _candidate_metadata(
        self,
        hypothesis_kind: str,
        raw: Any,
        *,
        preserves_symmetry: bool = False,
        revision: bool = False,
    ) -> dict[str, Any]:
        metadata = _text_metadata(
            raw,
            "metadata",
            allowed={"prediction", "falsifier", "rationale"},
        )
        expected = self._enforcement(
            hypothesis_kind, preserves_symmetry=preserves_symmetry
        )
        metadata["enforcement"] = expected
        if hypothesis_kind != NULL_CONTROL_CLAIM and not _preregistered(metadata):
            noun = "revision" if revision else "candidate"
            raise ControllerError(
                f"promotable {noun} metadata must preregister a non-empty "
                "prediction or falsifier"
            )
        return metadata

    def _submit_candidate(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(
            action,
            required={"type", "candidate_id", "hypothesis_id", "spec"},
            optional={"metadata", "symmetry_evidence_ids"},
        )
        candidate_id = _new_identifier(action["candidate_id"], "candidate_id")
        hypothesis_id = _identifier(action["hypothesis_id"], "hypothesis_id")
        if candidate_id in self.loop.state.candidates:
            raise ControllerError(f"candidate already exists: {candidate_id}")
        hypothesis = self.loop.state.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ControllerError(f"unknown hypothesis: {hypothesis_id}")
        if hypothesis.status not in {Lifecycle.READY, Lifecycle.SUPPORTED}:
            raise ControllerError(
                f"candidate requires READY or SUPPORTED hypothesis; "
                f"{hypothesis_id} is {hypothesis.status.value}"
            )
        spec = _mapping(action["spec"], "spec")
        try:
            parsed = AnsatzSpec.from_dict(spec)
        except Exception as exc:
            raise ControllerError(f"invalid candidate: {exc}") from exc
        if hypothesis.claim["kind"] == NULL_CONTROL_CLAIM and (
            parsed.parameters or parsed.operations
        ):
            raise ControllerError(
                "null_control candidate must be a typed no-op with no parameters "
                "or operations"
            )
        spec = parsed.to_dict()
        self._ensure_unique_candidate(spec)
        symmetry_evidence_ids = self._validated_symmetry_evidence_ids(
            action.get("symmetry_evidence_ids"),
            primary_hypothesis_id=hypothesis_id,
        )
        metadata = self._candidate_metadata(
            str(hypothesis.claim["kind"]),
            action.get("metadata", {}),
            preserves_symmetry=bool(symmetry_evidence_ids),
        )
        active = sum(
            record.hypothesis_id == hypothesis_id
            and record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
            for record in self.loop.state.candidates.values()
        )
        if active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
            raise ControllerError(
                f"at most {MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS} candidates may be "
                "active per hypothesis"
            )
        self._ensure_capacity(cost=0.1)
        self.loop.dispatch(
            {
                "type": "submit_candidate",
                "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id,
                "spec": spec,
                "metadata": metadata,
                "symmetry_evidence_ids": symmetry_evidence_ids,
                "cost": 0.1,
            }
        )
        return self._result("submit_candidate", {"accepted": True})

    def _compile_audit(self, candidate_id: str) -> tuple[bool, dict[str, Any]]:
        candidate = self.loop.state.candidates[candidate_id]
        hypothesis = self.loop.state.hypotheses[candidate.hypothesis_id]
        is_null_control = hypothesis.claim.get("kind") == NULL_CONTROL_CLAIM
        try:
            parsed = AnsatzSpec.from_dict(candidate.spec)
            if parsed.num_qubits != self.problem.num_qubits:
                raise ControllerError("candidate num_qubits must match the problem")
            compiled = compile_ansatz(parsed)
            audit = compiled.audit
            if audit.operations > MAX_CANDIDATE_OPERATIONS:
                raise ControllerError(
                    f"candidate exceeds operation cap {MAX_CANDIDATE_OPERATIONS}"
                )
            if audit.unique_trainable_params > MAX_CANDIDATE_PARAMETERS:
                raise ControllerError(
                    f"candidate exceeds parameter cap {MAX_CANDIDATE_PARAMETERS}"
                )
            if audit.spec_nodes > MAX_CANDIDATE_SPEC_NODES:
                raise ControllerError(
                    f"candidate exceeds spec-node cap {MAX_CANDIDATE_SPEC_NODES}"
                )
            if not is_null_control and audit.operations <= 0:
                raise ControllerError("candidate requires at least one operation")
            if not is_null_control and audit.unique_trainable_params <= 0:
                raise ControllerError(
                    "candidate requires at least one trainable parameter"
                )
            excessive_fanout = {
                name: occurrences
                for name, occurrences in audit.parameter_occurrences.items()
                if occurrences > MAX_PARAMETER_FANOUT
            }
            if excessive_fanout:
                raise ControllerError(
                    f"parameter fan-out exceeds {MAX_PARAMETER_FANOUT}: "
                    f"{excessive_fanout}"
                )

            hamiltonian_labels = {term.pauli for term in self.problem.pauli_terms}
            for operation in parsed.operations:
                if operation.macro != "PauliRotation" or len(operation.qubits) <= 2:
                    continue
                local_pauli = operation.options["pauli"]
                label = ["I"] * self.problem.num_qubits
                for qubit, letter in zip(operation.qubits, local_pauli, strict=True):
                    label[self.problem.num_qubits - qubit - 1] = letter
                if "".join(label) not in hamiltonian_labels:
                    raise ControllerError(
                        "PauliRotation above locality 2 must be a Hamiltonian term"
                    )

            allowed_scales = {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}
            unapproved_literals = [
                literal.to_dict()
                for literal in audit.fixed_literals
                if literal.role != "scale"
                or not any(
                    math.isclose(
                        literal.value, allowed, rel_tol=0.0, abs_tol=1e-12
                    )
                    for allowed in allowed_scales
                )
            ]
            if unapproved_literals:
                raise ControllerError(
                    f"candidate contains unapproved fixed numeric literals: "
                    f"{unapproved_literals}"
                )

            hypothesis = self.loop.state.hypotheses[candidate.hypothesis_id]
            claim_kind = str(hypothesis.claim["kind"])
            conservation_macros = sorted(
                {
                    operation.macro
                    for operation in parsed.operations
                    if operation.macro in {"XYExchange", "IsotropicExchange"}
                }
            )
            if conservation_macros and not candidate.symmetry_evidence_ids:
                raise ControllerError(
                    "XYExchange and IsotropicExchange require a SUPPORTED "
                    "exact_pauli_symmetry evidence probe"
                )
            if candidate.metadata.get("enforcement") != self._enforcement(
                claim_kind,
                preserves_symmetry=bool(candidate.symmetry_evidence_ids),
            ):
                raise ControllerError("candidate enforcement metadata is inconsistent")

            symmetry_audit: dict[str, Any] | None = None
            if candidate.symmetry_evidence_ids:
                charges: dict[str, Any] = {}
                constraints: dict[str, Any] = {}
                zero = compiled.circuit.assign_parameters(
                    {parameter: 0.0 for parameter in compiled.parameters.values()},
                    inplace=False,
                )
                prepared = initial_state_circuit(self.problem)
                prepared.compose(zero, inplace=True)
                for evidence_id in candidate.symmetry_evidence_ids:
                    probe = self.loop.state.probes[evidence_id]
                    symmetry_hypothesis = self.loop.state.hypotheses[
                        probe.hypothesis_id
                    ]
                    recipe = symmetry_hypothesis.claim.get("generator")
                    if not isinstance(recipe, Mapping):
                        raise ControllerError(
                            "exact symmetry evidence requires a generator recipe"
                        )
                    charge = generator_from_recipe(self.problem.num_qubits, recipe)
                    charges[evidence_id] = charge
                    residuals = operation_symmetry_residuals(
                        self.problem.num_qubits, parsed.operations, charge
                    )
                    max_residual = max(residuals, default=0.0)
                    if max_residual > EXACT_SYMMETRY_TOLERANCE:
                        raise ControllerError(
                            "candidate operation breaks a cited exact symmetry: "
                            f"evidence={evidence_id} residual={max_residual:.3e}"
                        )
                    sector_mean, sector_variance = initial_state_moments(
                        prepared, charge
                    )
                    if sector_variance > EXACT_SYMMETRY_TOLERANCE:
                        raise ControllerError(
                            "initial state is not in a definite sector of cited "
                            f"symmetry {evidence_id}"
                        )
                    constraints[evidence_id] = {
                        "hypothesis_id": probe.hypothesis_id,
                        "hamiltonian_residual": probe.result["metrics"]["residual"],
                        "max_operation_residual": max_residual,
                        "initial_state_mean": sector_mean,
                        "initial_state_variance": sector_variance,
                    }

                relevance: list[dict[str, Any]] = []
                for operation_index, operation in enumerate(parsed.operations):
                    if operation.macro not in {
                        "XYExchange",
                        "IsotropicExchange",
                    }:
                        continue
                    relevant_constraints: dict[str, Any] = {}
                    relevance_failures: dict[str, str] = {}
                    for evidence_id, charge in charges.items():
                        try:
                            (
                                touching_norm,
                                relevant_fraction,
                                residual,
                                conditioned_symmetry_residual,
                                conditioned_sector_variance,
                            ) = (
                                validate_special_operation_relevance(
                                    self.problem.num_qubits,
                                    operation,
                                    charge,
                                    symmetry_residual=constraints[evidence_id][
                                        "hamiltonian_residual"
                                    ],
                                    sector_variance=constraints[evidence_id][
                                        "initial_state_variance"
                                    ],
                                )
                            )
                        except ProbeValidationError as exc:
                            relevance_failures[evidence_id] = str(exc)
                            continue
                        relevant_constraints[evidence_id] = {
                            "touching_charge_norm": touching_norm,
                            "relevant_charge_fraction": relevant_fraction,
                            "residual": residual,
                            "conditioned_symmetry_residual": (
                                conditioned_symmetry_residual
                            ),
                            "conditioned_sector_variance": (
                                conditioned_sector_variance
                            ),
                        }
                    if not relevant_constraints:
                        raise ControllerError(
                            "special conservation gate has no relevant cited symmetry "
                            f"on operation {operation_index}: {relevance_failures}"
                        )
                    relevance.append(
                        {
                            "operation_index": operation_index,
                            "macro": operation.macro,
                            "constraints": relevant_constraints,
                        }
                    )
                symmetry_audit = {
                    "constraints": constraints,
                    "special_operation_relevance": relevance,
                }

            resource = audit_public_candidate(self.problem, parsed)
            policy = _resource_eligibility(resource.metrics)
            violations = list(resource.violations) + list(policy["violations"])
            passed = bool(resource.valid and policy["eligible"] and not violations)
            result: dict[str, Any] = {
                "valid": passed,
                "audit": audit.to_dict(),
                "metrics": dict(resource.metrics),
                "resource_policy": policy,
                "violations": violations,
            }
            if symmetry_audit is not None:
                result["symmetry_audit"] = symmetry_audit
            return passed, result
        except (ControllerError, ProbeValidationError, AnsatzIRValidationError) as exc:
            return False, {
                "valid": False,
                "violations": [f"{type(exc).__name__}: {exc}"],
            }
        except Exception as exc:
            raise ControllerError(
                f"candidate audit infrastructure failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _baseline_energy(self, spec: Mapping[str, Any]) -> float:
        compiled = compile_ansatz(spec)
        zero = compiled.circuit.assign_parameters(
            {parameter: 0.0 for parameter in compiled.parameters.values()},
            inplace=False,
        )
        prepared = initial_state_circuit(self.problem)
        prepared.compose(zero, inplace=True)
        return energy_from_circuit(prepared, hamiltonian_from_problem(self.problem))

    @staticmethod
    def _compact_evaluation(result: Any) -> dict[str, Any]:
        return {
            "valid": bool(result.valid),
            "best_energy": result.best_energy,
            "trace_summary": [list(point) for point in result.trace_summary],
            "objective_calls": result.objective_calls,
            "objective_energy_span": result.objective_energy_span,
            "hamiltonian_active_norm": result.hamiltonian_active_norm,
            "objective_activity_fraction": result.objective_activity_fraction,
            "constant_hamiltonian": result.constant_hamiltonian,
            "optimizer": result.optimizer,
            "seed": result.seed,
            "audit": copy.deepcopy(dict(result.audit)),
            "metrics": copy.deepcopy(dict(result.metrics)),
            "violations": list(result.violations),
        }

    def _fair_comparators(self, candidate_id: str) -> list[dict[str, Any]]:
        target = self.loop.state.candidates[candidate_id]
        points: list[dict[str, Any]] = []
        for record in self.loop.state.evaluations.values():
            if record.candidate_id == candidate_id:
                continue
            comparator = self.loop.state.candidates[record.candidate_id]
            if comparator.hypothesis_id == target.hypothesis_id:
                continue
            point = comparison_point(record)
            if point is not None:
                points.append(point)
        return sorted(points, key=lambda point: point["candidate_id"])

    def _pending_promotion_comparisons(self) -> list[str]:
        return sorted(
            candidate.candidate_id
            for candidate in self.loop.state.candidates.values()
            if candidate.status is Lifecycle.PROMOTED
            and not self._fair_comparators(candidate.candidate_id)
        )

    def _promoted_disposition_evidence(self, candidate_id: str) -> list[str]:
        target_records = [
            record
            for record in self.loop.state.evaluations.values()
            if record.candidate_id == candidate_id
            and record.stage is EvaluationStage.PROMOTION
            and record.passed
        ]
        if len(target_records) != 1:
            raise ControllerError(
                "promoted candidate lacks one passed promotion evaluation"
            )
        target = comparison_point(target_records[0])
        if target is None:
            raise ControllerError(
                "promoted candidate lacks evaluator-owned comparison coordinates"
            )
        dominators = [
            comparator
            for comparator in self._fair_comparators(candidate_id)
            if comparison_dominates_target(target, comparator)
        ]
        if not dominators:
            raise ControllerError(
                "promoted candidate can be revised or retired only after a fair "
                "different-hypothesis promotion comparison dominates it"
            )
        return [
            target_records[0].evaluation_id,
            *(comparator["evaluation_id"] for comparator in dominators),
        ]

    def _evaluate_candidate(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(action, required={"type", "candidate_id"})
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        candidate = self.loop.state.candidates.get(candidate_id)
        if candidate is None:
            raise ControllerError(f"unknown candidate: {candidate_id}")
        stage = _next_stage(candidate.status)
        if stage is None:
            raise ControllerError(
                f"candidate {candidate_id} has no next evaluation in "
                f"{candidate.status.value}"
            )
        evaluation_id = f"evaluation:{candidate_id}:{stage.value}"
        if evaluation_id in self.loop.state.evaluations:
            raise ControllerError(f"evaluation already exists: {evaluation_id}")

        if stage is EvaluationStage.AUDIT:
            cost = 0.25
            self._ensure_capacity(cost=cost)
            passed, metrics = self._compile_audit(candidate_id)
            event_metrics = metrics
        else:
            if stage is EvaluationStage.SMOKE:
                protocol = SMOKE_EVALUATION_PROTOCOL
                cost = 2.0
            else:
                protocol = PROMOTION_EVALUATION_PROTOCOL
                cost = 6.0
                if not self._fair_comparators(candidate_id):
                    ready_comparators = sorted(
                        record.candidate_id
                        for record in self.loop.state.candidates.values()
                        if record.candidate_id != candidate_id
                        and record.hypothesis_id != candidate.hypothesis_id
                        and record.status is Lifecycle.SMOKE
                    )
                    if not ready_comparators:
                        raise ControllerError(
                            "promotion requires a different-hypothesis competitor or "
                            "control that already passed smoke"
                        )
                    # The first promotion must leave enough budget to run the
                    # already-smoked comparator at the same fixed protocol.
                    self._ensure_capacity(cost=2.0 * cost, events=2)
                else:
                    self._ensure_capacity(cost=cost)
            if stage is EvaluationStage.SMOKE:
                self._ensure_capacity(cost=cost)
            try:
                baseline = self._baseline_energy(candidate.spec)
            except Exception as exc:
                raise ControllerError(
                    f"baseline evaluation failed before candidate optimization: {exc}"
                ) from exc
            run = evaluate_public_problem(self.problem, candidate.spec, protocol=protocol)
            evaluation = run.result
            if not evaluation.valid:
                raise ControllerError(
                    "candidate optimizer failed without producing scientific evidence: "
                    f"{list(evaluation.violations)}"
                )
            metrics = self._compact_evaluation(evaluation)
            event_metrics = evaluation.to_dict()
            event_metrics.pop("optimized_parameter_binding", None)
            policy = _resource_eligibility(evaluation.metrics)
            metrics["resource_policy"] = policy
            event_metrics["resource_policy"] = copy.deepcopy(policy)
            metrics["baseline_energy"] = baseline
            event_metrics["baseline_energy"] = baseline
            improvement = (
                None
                if evaluation.best_energy is None
                else baseline - evaluation.best_energy
            )
            metrics["energy_improvement"] = improvement
            event_metrics["energy_improvement"] = improvement
            threshold = max(
                MIN_SMOKE_ENERGY_IMPROVEMENT,
                MIN_SMOKE_ENERGY_IMPROVEMENT * abs(baseline),
            )
            metrics["required_energy_improvement"] = threshold
            event_metrics["required_energy_improvement"] = threshold
            passed = bool(
                evaluation.valid
                and policy["eligible"]
                and improvement is not None
                and improvement >= threshold
            )
            hypothesis = self.loop.state.hypotheses[candidate.hypothesis_id]
            if (
                stage is EvaluationStage.SMOKE
                and hypothesis.claim.get("kind") == NULL_CONTROL_CLAIM
            ):
                # A control must reach the same fixed promotion-budget
                # comparison even though it is not expected to improve itself.
                passed = bool(evaluation.valid and policy["eligible"])
            if (
                stage is EvaluationStage.PROMOTION
                and hypothesis.claim.get("kind") == NULL_CONTROL_CLAIM
            ):
                passed = False
                metrics["promotion_blocked_reason"] = (
                    "null_control candidates are diagnostic and cannot be promoted"
                )
                event_metrics["promotion_blocked_reason"] = metrics[
                    "promotion_blocked_reason"
                ]
            if passed and stage is EvaluationStage.PROMOTION:
                smoke_energies = [
                    record.metrics.get("best_energy")
                    for record in self.loop.state.evaluations.values()
                    if record.candidate_id == candidate_id
                    and record.stage is EvaluationStage.SMOKE
                    and record.passed
                ]
                smoke_energies = [
                    float(value) for value in smoke_energies if value is not None
                ]
                passed = bool(
                    smoke_energies
                    and evaluation.best_energy is not None
                    and evaluation.best_energy
                    <= min(smoke_energies) + 5e-4
                )

        self.loop.dispatch(
            {
                "type": "record_evaluation",
                "candidate_id": candidate_id,
                "evaluation_id": evaluation_id,
                "stage": stage.value,
                "passed": bool(passed),
                "metrics": event_metrics,
                "cost": cost,
            }
        )
        return self._result(
            "evaluate_candidate",
            {
                "candidate_id": candidate_id,
                "evaluation_id": evaluation_id,
                "stage": stage.value,
                "passed": bool(passed),
                **metrics,
            },
        )

    def _revise(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(
            action,
            required={"type", "entity", "source_id", "new_id", "replacement", "reason"},
            optional={"metadata", "symmetry_evidence_ids"},
        )
        entity = action["entity"]
        source_id = _identifier(action["source_id"], "source_id")
        new_id = _new_identifier(action["new_id"], "new_id")
        reason = _text(action["reason"], "reason")
        replacement = _mapping(action["replacement"], "replacement")
        metadata = _mapping(action.get("metadata", {}), "metadata")
        if entity == "hypothesis":
            if "symmetry_evidence_ids" in action:
                raise ControllerError(
                    "symmetry_evidence_ids applies only to candidate revisions"
                )
            source = self.loop.state.hypotheses.get(source_id)
            if source is None:
                raise ControllerError(f"unknown hypothesis: {source_id}")
            if source.status is Lifecycle.REVISED:
                raise ControllerError(f"hypothesis is already revised: {source_id}")
            metadata = _text_metadata(
                metadata,
                "metadata",
                allowed={"rationale", "prediction", "falsifier"},
            )
            replacement = self._validated_claim(replacement)
            active = sum(
                record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
                for record in self.loop.state.hypotheses.values()
                if record.hypothesis_id != source_id
            )
            if active >= MAX_ACTIVE_HYPOTHESES:
                raise ControllerError("hypothesis revision would exceed the active cap")
        elif entity == "candidate":
            source = self.loop.state.candidates.get(source_id)
            if source is None:
                raise ControllerError(f"unknown candidate: {source_id}")
            if source.status is Lifecycle.REVISED:
                raise ControllerError(
                    f"candidate {source_id} cannot be revised in {source.status.value}"
                )
            hypothesis = self.loop.state.hypotheses[source.hypothesis_id]
            try:
                parsed = AnsatzSpec.from_dict(replacement)
            except Exception as exc:
                raise ControllerError(f"invalid candidate revision: {exc}") from exc
            if hypothesis.claim["kind"] == NULL_CONTROL_CLAIM and (
                parsed.parameters or parsed.operations
            ):
                raise ControllerError(
                    "null_control candidate revision must be a typed no-op with no "
                    "parameters or operations"
                )
            replacement = parsed.to_dict()
            symmetry_evidence_ids = (
                self._validated_symmetry_evidence_ids(
                    action["symmetry_evidence_ids"],
                    primary_hypothesis_id=source.hypothesis_id,
                )
                if "symmetry_evidence_ids" in action
                else list(source.symmetry_evidence_ids)
            )
            metadata = self._candidate_metadata(
                str(hypothesis.claim["kind"]),
                metadata,
                preserves_symmetry=bool(symmetry_evidence_ids),
                revision=True,
            )
            self._ensure_unique_candidate(
                replacement,
                representation_repair_source=source_id,
            )
            active = sum(
                record.hypothesis_id == source.hypothesis_id
                and record.candidate_id != source_id
                and record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
                for record in self.loop.state.candidates.values()
            )
            if active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
                raise ControllerError("candidate revision would exceed the active cap")
        else:
            raise ControllerError("entity must be hypothesis or candidate")
        self._ensure_capacity(cost=0.1)
        evidence_ids = (
            self._promoted_disposition_evidence(source_id)
            if entity == "candidate"
            and self.loop.state.candidates[source_id].status is Lifecycle.PROMOTED
            else []
        )
        self.loop.dispatch(
            {
                "type": "revise",
                "entity": entity,
                "source_id": source_id,
                "new_id": new_id,
                "replacement": replacement,
                "reason": reason,
                "metadata": metadata,
                "symmetry_evidence_ids": (
                    symmetry_evidence_ids if entity == "candidate" else []
                ),
                "evidence_ids": evidence_ids,
                "cost": 0.1,
            }
        )
        return self._result("revise", {"accepted": True, "new_id": new_id})

    def _commit(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(action, required={"type", "candidate_id"})
        candidate_id = _identifier(action["candidate_id"], "candidate_id")
        candidate = self.loop.state.candidates.get(candidate_id)
        if candidate is None:
            raise ControllerError(f"unknown candidate: {candidate_id}")
        if candidate.status is not Lifecycle.PROMOTED:
            raise ControllerError("commit requires a passed promotion")
        if not _preregistered(candidate.metadata):
            raise ControllerError(
                "commit requires a preregistered prediction or falsifier"
            )
        promotions = sorted(
            (record.evaluation_id, record)
            for record in self.loop.state.evaluations.values()
            if record.candidate_id == candidate_id
            and record.stage is EvaluationStage.PROMOTION
            and record.passed
        )
        if len(promotions) != 1:
            raise ControllerError("commit requires exactly one passed promotion result")
        target = comparison_point(promotions[0][1])
        if target is None:
            raise ControllerError(
                "passed promotion lacks evaluator-owned energy/resource evidence"
            )
        comparators = self._fair_comparators(candidate_id)
        if not comparators:
            raise ControllerError(
                "commit requires a different-hypothesis competitor or control "
                "evaluated with the same promotion protocol"
            )
        for comparator in comparators:
            if not comparison_dominates_target(target, comparator):
                continue
            if target["best_energy"] > (
                comparator["best_energy"] + COMMIT_ENERGY_TOLERANCE
            ):
                raise ControllerError(
                    "target promotion is energetically worse than evaluated "
                    f"comparator {comparator['candidate_id']}"
                )
            raise ControllerError(
                "target promotion is Pareto-dominated by evaluated comparator "
                f"{comparator['candidate_id']}"
            )
        evidence_ids = [
            promotions[0][0], *(item["evaluation_id"] for item in comparators)
        ]
        metadata = {
            "evidence_ids": evidence_ids,
            "promotion_evaluation_id": promotions[0][0],
            "comparison": {
                "mode": "evaluated_competitor",
                "energy_tolerance": COMMIT_ENERGY_TOLERANCE,
                "target": target,
                "evaluations": comparators,
            },
        }
        self._ensure_capacity(cost=0.0, terminal=True)
        self.loop.dispatch(
            {
                "type": "commit",
                "candidate_id": candidate_id,
                "metadata": metadata,
                "cost": 0.0,
            }
        )
        return self._result(
            "commit", {"accepted": True, "candidate_id": candidate_id, **metadata}
        )

    def _close_negative(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(action, required={"type", "reason"})
        reason = _text(action["reason"], "reason")
        state = self.loop.state
        ordered_evidence_ids = list(derived_negative_close_evidence(state))
        try:
            coverage = validate_negative_close_coverage(
                state, ordered_evidence_ids
            )
        except TransitionError as exc:
            raise ControllerError(str(exc)) from exc
        self._ensure_capacity(cost=0.0, terminal=True)
        self.loop.dispatch(
            {
                "type": "close_negative",
                "reason": reason,
                "evidence_ids": ordered_evidence_ids,
                "metadata": {"coverage": coverage},
                "cost": 0.0,
            }
        )
        return self._result(
            "close_negative",
            {
                "accepted": True,
                "evidence_ids": ordered_evidence_ids,
                "coverage": coverage,
            },
        )

    def _retire(self, action: Mapping[str, Any]) -> StepResult:
        _strict_action(
            action, required={"type", "entity", "entity_id", "reason"}
        )
        entity = action["entity"]
        if entity not in {"hypothesis", "candidate"}:
            raise ControllerError("entity must be hypothesis or candidate")
        entity_id = _identifier(action["entity_id"], "entity_id")
        reason = _text(action["reason"], "reason")
        evidence_ids: list[str] = []
        if entity == "candidate":
            candidate = self.loop.state.candidates.get(entity_id)
            if candidate is not None and candidate.status is Lifecycle.PROMOTED:
                evidence_ids = self._promoted_disposition_evidence(entity_id)
        self._ensure_capacity(cost=0.0)
        self.loop.dispatch(
            {
                "type": "retire",
                "entity": entity,
                "entity_id": entity_id,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "cost": 0.0,
            }
        )
        return self._result("retire", {"accepted": True})

    def dispatch_external(self, action: Mapping[str, Any]) -> StepResult:
        if not isinstance(action, Mapping):
            raise ControllerError("external action must be an object")
        try:
            encoded_size = len(
                json.dumps(
                    action, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise ControllerError(
                f"external action must contain finite JSON data: {exc}"
            ) from exc
        if encoded_size > MAX_EXTERNAL_ACTION_BYTES:
            raise ControllerError(
                f"external action exceeds {MAX_EXTERNAL_ACTION_BYTES} byte cap"
            )
        if self.loop.state.terminal:
            raise ControllerError("research run is terminal; no further actions are allowed")
        action_type = action.get("type")
        if not isinstance(action_type, str):
            raise ControllerError("external action requires a string type")
        pending_comparisons = self._pending_promotion_comparisons()
        if pending_comparisons:
            candidate = self.loop.state.candidates.get(action.get("candidate_id"))
            if not (
                action_type == "evaluate_candidate"
                and candidate is not None
                and candidate.status is Lifecycle.SMOKE
                and any(
                    candidate.hypothesis_id
                    != self.loop.state.candidates[target_id].hypothesis_id
                    for target_id in pending_comparisons
                )
            ):
                raise ControllerError(
                    "a promoted candidate is waiting for its reserved fair comparison; "
                    "next evaluate a different-hypothesis SMOKE candidate at promotion: "
                    f"{pending_comparisons}"
                )
        if action_type in {"record_probe", "record_evaluation"}:
            raise ControllerError(f"{action_type} is evaluator-owned")
        handlers = {
            "propose_hypothesis": self._propose_hypothesis,
            "request_probe": self._request_probe,
            "submit_candidate": self._submit_candidate,
            "evaluate_candidate": self._evaluate_candidate,
            "revise": self._revise,
            "retire": self._retire,
            "commit": self._commit,
            "close_negative": self._close_negative,
        }
        handler = handlers.get(action_type)
        if handler is None:
            raise ControllerError(f"unsupported external action type: {action_type!r}")
        return handler(action)


__all__ = ["ControllerError", "ResearchController", "StepResult"]
