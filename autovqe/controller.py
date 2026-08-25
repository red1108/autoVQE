from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .ansatz_ir import AnsatzSpec
from .compiler import compile_ansatz
from .contracts import PublicProblem, assert_agent_safe
from .evaluator import (
    EvaluationProtocol,
    candidate_identity,
    evaluate_public_problem,
    hamiltonian_from_public,
)
from .probes import (
    EXACT_SYMMETRY_TOLERANCE,
    ProbeReceipt,
    algebraic_probe_cost_units,
    energy_from_circuit,
    generator_from_recipe,
    operation_symmetry_residuals,
    reference_moments,
    run_public_probe,
    validate_symmetry_generator,
)
from .research import (
    EvaluationStage,
    Lifecycle,
    ResearchLoop,
    ResearchState,
)


class ControllerError(RuntimeError):
    """Raised when an external action violates the trusted controller contract."""


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

EXACT_SYMMETRY_CLAIM = "exact_pauli_symmetry"
STRUCTURE_CLAIM = "ansatz_structure"
NULL_CONTROL_CLAIM = "null_control"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class ControllerReceipt:
    action_type: str
    result: dict[str, Any]
    state: dict[str, Any]

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
            f"invalid external action fields: missing={sorted(missing)} extra={sorted(extra)}"
        )


def _nonempty_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ControllerError(f"{field} must match {_ID_RE.pattern!r}")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControllerError(f"{field} must be an object")
    return copy.deepcopy(dict(value))


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControllerError(f"{field} must be a non-empty string")
    return value.strip()


def _id_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ControllerError(f"{field} must be a non-empty list of IDs")
    identifiers = tuple(_nonempty_id(item, field) for item in value)
    if len(set(identifiers)) != len(identifiers):
        raise ControllerError(f"{field} must not contain duplicate IDs")
    return identifiers


def _preregistered_fields(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        field
        for field in ("prediction", "falsifier")
        if isinstance(metadata.get(field), str) and str(metadata[field]).strip()
    )


def _probe_passed(receipt: ProbeReceipt) -> bool:
    if receipt.probe_type == "normalized_commutator":
        return bool(receipt.metrics.get("exact", False))
    return False


def _required_probe_ids(claim: Mapping[str, Any]) -> tuple[str, ...]:
    raw = claim.get("required_probe_ids", ())
    if raw in (None, ()):
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise ControllerError("claim.required_probe_ids must be a list of IDs")
    return tuple(raw)


def _admission_probe_id(hypothesis_id: str) -> str:
    return f"admission:{hypothesis_id}"


def _resource_eligibility(metrics: Mapping[str, Any]) -> dict[str, Any]:
    raw = {
        name: int(metrics.get(name, fallback))
        for name, fallback in {
            "canonical_template_twoq_count": MAX_CANONICAL_TWOQ_GATES + 1,
            "canonical_generic_worst_twoq_count": MAX_CANONICAL_TWOQ_GATES + 1,
            "canonical_template_total_gate_count": MAX_CANONICAL_TOTAL_GATES + 1,
            "canonical_generic_worst_total_gate_count": MAX_CANONICAL_TOTAL_GATES + 1,
            "canonical_template_depth": MAX_CANONICAL_DEPTH + 1,
            "canonical_generic_worst_depth": MAX_CANONICAL_DEPTH + 1,
        }.items()
    }
    # A candidate must fit both its symbolic template and all evaluator-owned
    # generic bindings.  Taking the max prevents a circuit from becoming
    # artificially cheap only at known numeric cancellation points.
    observed = {
        "canonical_conservative_twoq_count": max(
            raw["canonical_template_twoq_count"],
            raw["canonical_generic_worst_twoq_count"],
        ),
        "canonical_conservative_total_gate_count": max(
            raw["canonical_template_total_gate_count"],
            raw["canonical_generic_worst_total_gate_count"],
        ),
        "canonical_conservative_depth": max(
            raw["canonical_template_depth"],
            raw["canonical_generic_worst_depth"],
        ),
    }
    limits = {
        "canonical_conservative_twoq_count": MAX_CANONICAL_TWOQ_GATES,
        "canonical_conservative_total_gate_count": MAX_CANONICAL_TOTAL_GATES,
        "canonical_conservative_depth": MAX_CANONICAL_DEPTH,
    }
    violations = [
        f"{name}={observed[name]} exceeds {limit}"
        for name, limit in limits.items()
        if observed[name] > limit
    ]
    return {
        "eligible": not violations,
        "inputs": raw,
        "observed": observed,
        "limits": limits,
        "violations": violations,
    }


class ResearchController:
    """Validate agent actions and run evaluator-owned measurements.

    External agents can request probes and evaluations, but cannot append
    ``record_probe`` or ``record_evaluation`` events themselves.
    """

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

    def _receipt(
        self,
        action_type: str,
        events: list[Any],
        result: Mapping[str, Any],
    ) -> ControllerReceipt:
        return ControllerReceipt(
            action_type=action_type,
            result=copy.deepcopy(dict(result)),
            state=self.loop.state.to_dict(),
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
            detail = "" if terminal else " while reserving one terminal-decision event"
            raise ControllerError(
                f"research run reached {MAX_HISTORY_EVENTS} event cap{detail}"
            )
        if cost > state.remaining_budget + 1e-12:
            raise ControllerError(
                f"action costs {cost}, remaining budget is {state.remaining_budget}"
            )

    def _validated_claim(self, value: Any) -> dict[str, Any]:
        claim = _mapping(value, "claim")
        kind = claim.get("kind")
        if kind == EXACT_SYMMETRY_CLAIM:
            missing = {"kind", "generator"} - set(claim)
            extra = set(claim) - {"kind", "generator"}
            if missing or extra:
                raise ControllerError(
                    "exact_pauli_symmetry claim fields must be exactly kind and generator"
                )
            generator_recipe = _mapping(claim["generator"], "claim.generator")
            try:
                hamiltonian = hamiltonian_from_public(self.problem)
                generator = generator_from_recipe(
                    self.problem.num_qubits, generator_recipe
                )
                # This rejects vacuous, non-Hermitian, and H-copy generators,
                # but deliberately does not assume that the commutator is zero.
                validate_symmetry_generator(hamiltonian, generator)
            except Exception as exc:
                raise ControllerError(f"invalid symmetry generator: {exc}") from exc
            return {"kind": kind, "generator": generator_recipe}

        if kind == STRUCTURE_CLAIM:
            missing = {"kind", "family"} - set(claim)
            extra = set(claim) - {"kind", "family"}
            if missing or extra:
                raise ControllerError(
                    "ansatz_structure claim fields must be exactly kind and family"
                )
            family = claim["family"]
            if not isinstance(family, str) or not family.strip():
                raise ControllerError("claim.family must be a non-empty string")
            return {"kind": kind, "family": family.strip()}

        if kind == NULL_CONTROL_CLAIM:
            if set(claim) != {"kind"}:
                raise ControllerError("null_control claim contains unsupported fields")
            return {"kind": kind}

        raise ControllerError(
            "claim.kind must be exact_pauli_symmetry, ansatz_structure, or null_control"
        )

    def _admit_non_algebraic_hypothesis(
        self,
        hypothesis_id: str,
        kind: str,
    ) -> Any:
        return self.loop.dispatch(
            {
                "type": "record_probe",
                "hypothesis_id": hypothesis_id,
                "probe_id": _admission_probe_id(hypothesis_id),
                "verdict": "supported",
                "result": {
                    "controller_passed": True,
                    "admission": "non_algebraic_design_hypothesis",
                    "claim_kind": kind,
                    "algebraic_certificate": False,
                },
                "cost": 0.0,
            }
        )

    def _propose_hypothesis(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "hypothesis_id", "claim"},
            optional={"metadata", "cost"},
        )
        hypothesis_id = _nonempty_id(action["hypothesis_id"], "hypothesis_id")
        if hypothesis_id in self.loop.state.hypotheses:
            raise ControllerError(f"hypothesis already exists: {hypothesis_id}")
        raw_claim = _mapping(action["claim"], "claim")
        raw_kind = raw_claim.get("kind")
        auto_admit = raw_kind in {STRUCTURE_CLAIM, NULL_CONTROL_CLAIM}
        # Reserve evaluator-owned budget/event capacity before algebraic
        # generator validation, which may construct sparse operator products.
        self._ensure_capacity(cost=0.1, events=2 if auto_admit else 1)
        claim = self._validated_claim(raw_claim)
        active = sum(
            record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
            for record in self.loop.state.hypotheses.values()
        )
        if active >= MAX_ACTIVE_HYPOTHESES:
            raise ControllerError(
                f"at most {MAX_ACTIVE_HYPOTHESES} hypotheses may be active"
            )
        auto_admit = claim["kind"] in {STRUCTURE_CLAIM, NULL_CONTROL_CLAIM}
        sanitized = {
            "type": "propose_hypothesis",
            "hypothesis_id": hypothesis_id,
            "claim": claim,
            "metadata": _mapping(action.get("metadata", {}), "metadata"),
            "cost": 0.1,
        }
        events = [self.loop.dispatch(sanitized)]
        if auto_admit:
            events.append(self._admit_non_algebraic_hypothesis(hypothesis_id, claim["kind"]))
        return self._receipt(
            "propose_hypothesis",
            events,
            {
                "accepted": True,
                "claim_kind": claim["kind"],
                "requires_algebraic_probe": not auto_admit,
            },
        )

    def _ensure_unique_candidate(self, spec: Mapping[str, Any]) -> None:
        identity = candidate_identity(spec)
        duplicates = sorted(
            record.candidate_id
            for record in self.loop.state.candidates.values()
            if candidate_identity(record.spec) == identity
        )
        if duplicates:
            raise ControllerError(
                "candidate is semantically equivalent to an existing candidate; "
                f"cosmetic renaming/re-layering is not a new experiment: {duplicates}"
            )

    def _submit_candidate(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "candidate_id", "hypothesis_id", "spec"},
            optional={"metadata", "cost"},
        )
        candidate_id = _nonempty_id(action["candidate_id"], "candidate_id")
        hypothesis_id = _nonempty_id(action["hypothesis_id"], "hypothesis_id")
        hypothesis = self.loop.state.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ControllerError(f"unknown hypothesis: {hypothesis_id}")
        spec = _mapping(action["spec"], "spec")
        self._ensure_unique_candidate(spec)
        metadata = _mapping(action.get("metadata", {}), "metadata")
        expected_enforcement = {
            EXACT_SYMMETRY_CLAIM: "preserve",
            STRUCTURE_CLAIM: "unconstrained",
            NULL_CONTROL_CLAIM: "diagnostic",
        }[str(hypothesis.claim["kind"])]
        if metadata.get("enforcement") != expected_enforcement:
            raise ControllerError(
                f"{hypothesis.claim['kind']} candidate requires "
                f"metadata.enforcement={expected_enforcement!r}"
            )
        if (
            hypothesis.claim["kind"] != NULL_CONTROL_CLAIM
            and not _preregistered_fields(metadata)
        ):
            raise ControllerError(
                "promotable candidate metadata must preregister a non-empty "
                "prediction or falsifier before submission"
            )
        active = sum(
            record.hypothesis_id == hypothesis_id
            and record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
            for record in self.loop.state.candidates.values()
        )
        if active >= MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
            raise ControllerError(
                "at most "
                f"{MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS} candidates may be active "
                "per hypothesis"
            )
        self._ensure_capacity(cost=0.1)
        event = self.loop.dispatch(
            {
                "type": "submit_candidate",
                "candidate_id": candidate_id,
                "hypothesis_id": hypothesis_id,
                "spec": spec,
                "metadata": metadata,
                "cost": 0.1,
            }
        )
        return self._receipt(
            "submit_candidate",
            [event],
            {"accepted": True},
        )

    def _revise(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "entity", "source_id", "new_id", "replacement", "reason"},
            optional={"metadata", "cost"},
        )
        entity = action["entity"]
        source_id = _nonempty_id(action["source_id"], "source_id")
        new_id = _nonempty_id(action["new_id"], "new_id")
        replacement = _mapping(action["replacement"], "replacement")
        reason = action["reason"]
        revision_metadata = _mapping(action.get("metadata", {}), "metadata")
        if not isinstance(reason, str) or not reason.strip():
            raise ControllerError("reason must be a non-empty string")

        auto_admit = False
        capacity_checked = False
        if entity == "hypothesis":
            source = self.loop.state.hypotheses.get(source_id)
            if source is None:
                raise ControllerError(f"unknown hypothesis: {source_id}")
            raw_kind = replacement.get("kind")
            auto_admit = raw_kind in {STRUCTURE_CLAIM, NULL_CONTROL_CLAIM}
            self._ensure_capacity(cost=0.1, events=2 if auto_admit else 1)
            capacity_checked = True
            replacement = self._validated_claim(replacement)
            auto_admit = replacement["kind"] in {
                STRUCTURE_CLAIM,
                NULL_CONTROL_CLAIM,
            }
            active = sum(
                record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
                for record in self.loop.state.hypotheses.values()
            )
            if source.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
                active += 1
            if active > MAX_ACTIVE_HYPOTHESES:
                raise ControllerError(
                    f"at most {MAX_ACTIVE_HYPOTHESES} hypotheses may be active"
                )
        elif entity == "candidate":
            source = self.loop.state.candidates.get(source_id)
            if source is None:
                raise ControllerError(f"unknown candidate: {source_id}")
            if source.status is Lifecycle.PROMOTED:
                raise ControllerError(
                    f"promoted candidate {source_id} cannot be revised; commit it"
                )
            hypothesis = self.loop.state.hypotheses[source.hypothesis_id]
            expected_enforcement = {
                EXACT_SYMMETRY_CLAIM: "preserve",
                STRUCTURE_CLAIM: "unconstrained",
                NULL_CONTROL_CLAIM: "diagnostic",
            }[str(hypothesis.claim["kind"])]
            supplied_enforcement = revision_metadata.get("enforcement")
            if (
                supplied_enforcement is not None
                and supplied_enforcement != expected_enforcement
            ):
                raise ControllerError(
                    f"{hypothesis.claim['kind']} candidate revision requires "
                    f"metadata.enforcement={expected_enforcement!r} when supplied"
                )
            revision_metadata["enforcement"] = expected_enforcement
            if (
                hypothesis.claim["kind"] != NULL_CONTROL_CLAIM
                and not _preregistered_fields(revision_metadata)
            ):
                raise ControllerError(
                    "promotable candidate revision must preregister a new non-empty "
                    "prediction or falsifier in metadata"
                )
            self._ensure_unique_candidate(replacement)
            active = sum(
                record.hypothesis_id == source.hypothesis_id
                and record.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
                for record in self.loop.state.candidates.values()
            )
            if source.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
                active += 1
            if active > MAX_ACTIVE_CANDIDATES_PER_HYPOTHESIS:
                raise ControllerError(
                    "candidate revision would exceed the active-candidate cap"
                )
        else:
            raise ControllerError("entity must be hypothesis or candidate")

        if not capacity_checked:
            self._ensure_capacity(cost=0.1, events=1)
        event = self.loop.dispatch(
            {
                "type": "revise",
                "entity": entity,
                "source_id": source_id,
                "new_id": new_id,
                "replacement": replacement,
                "reason": reason.strip(),
                "metadata": revision_metadata,
                "cost": 0.1,
            }
        )
        events = [event]
        if auto_admit:
            events.append(
                self._admit_non_algebraic_hypothesis(new_id, str(replacement["kind"]))
            )
        return self._receipt(
            "revise",
            events,
            {
                "accepted": True,
                "auto_admitted": auto_admit,
            },
        )

    def _resolve_evidence_ids(self, evidence_ids: tuple[str, ...]) -> None:
        state = self.loop.state
        for evidence_id in evidence_ids:
            in_probes = evidence_id in state.probes
            in_evaluations = evidence_id in state.evaluations
            if in_probes and in_evaluations:
                raise ControllerError(f"ambiguous evidence ID: {evidence_id}")
            if not in_probes and not in_evaluations:
                raise ControllerError(f"unknown evidence ID: {evidence_id}")

    def _commit(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "candidate_id", "evidence_ids", "comparison"},
            optional={"metadata", "cost"},
        )
        candidate_id = _nonempty_id(action["candidate_id"], "candidate_id")
        state = self.loop.state
        candidate = state.candidates.get(candidate_id)
        if candidate is None:
            raise ControllerError(f"unknown candidate: {candidate_id}")
        if candidate.status is not Lifecycle.PROMOTED:
            raise ControllerError(
                f"commit requires PROMOTED candidate; {candidate_id} is "
                f"{candidate.status.value}"
            )

        preregistered_fields = _preregistered_fields(candidate.metadata)
        if not preregistered_fields:
            raise ControllerError(
                "commit requires candidate metadata to preregister a non-empty "
                "prediction or falsifier before evaluation"
            )

        evidence_ids = _id_list(action["evidence_ids"], "evidence_ids")
        self._resolve_evidence_ids(evidence_ids)
        promotion_ids = {
            evaluation.evaluation_id
            for evaluation in state.evaluations.values()
            if evaluation.candidate_id == candidate_id
            and evaluation.stage is EvaluationStage.PROMOTION
            and evaluation.passed
        }
        if not promotion_ids.intersection(evidence_ids):
            raise ControllerError(
                "commit evidence_ids must include the candidate's passed promotion "
                "evaluation"
            )

        comparison = _mapping(action["comparison"], "comparison")
        mode = comparison.get("mode")
        if mode == "evaluated_competitor":
            if set(comparison) != {"mode", "candidate_id", "evidence_ids"}:
                raise ControllerError(
                    "evaluated_competitor comparison fields must be exactly "
                    "mode, candidate_id, and evidence_ids"
                )
            competitor_id = _nonempty_id(
                comparison["candidate_id"], "comparison.candidate_id"
            )
            if competitor_id == candidate_id:
                raise ControllerError("comparison candidate must differ from commit candidate")
            if competitor_id not in state.candidates:
                raise ControllerError(f"unknown comparison candidate: {competitor_id}")
            comparison_ids = _id_list(
                comparison["evidence_ids"], "comparison.evidence_ids"
            )
            if not set(comparison_ids).issubset(evidence_ids):
                raise ControllerError(
                    "comparison evidence_ids must also appear in commit evidence_ids"
                )
            comparison_evaluations = [
                state.evaluations.get(evidence_id) for evidence_id in comparison_ids
            ]
            if any(
                record is None or record.candidate_id != competitor_id
                for record in comparison_evaluations
            ):
                raise ControllerError(
                    "evaluated competitor evidence must be evaluator records for that candidate"
                )
            if not any(
                record is not None
                and record.stage in {EvaluationStage.SMOKE, EvaluationStage.PROMOTION}
                for record in comparison_evaluations
            ):
                raise ControllerError(
                    "evaluated competitor requires at least one smoke or promotion result"
                )
            sanitized_comparison = {
                "mode": mode,
                "candidate_id": competitor_id,
                "evidence_ids": list(comparison_ids),
            }
        elif mode == "documented_non_dominance":
            if set(comparison) != {"mode", "reason", "evidence_ids"}:
                raise ControllerError(
                    "documented_non_dominance comparison fields must be exactly "
                    "mode, reason, and evidence_ids"
                )
            reason = _nonempty_text(comparison["reason"], "comparison.reason")
            comparison_ids = _id_list(
                comparison["evidence_ids"], "comparison.evidence_ids"
            )
            if not set(comparison_ids).issubset(evidence_ids):
                raise ControllerError(
                    "comparison evidence_ids must also appear in commit evidence_ids"
                )
            if not any(evidence_id in promotion_ids for evidence_id in comparison_ids):
                raise ControllerError(
                    "documented non-dominance must cite the passed promotion evaluation"
                )
            sanitized_comparison = {
                "mode": mode,
                "reason": reason,
                "evidence_ids": list(comparison_ids),
            }
        else:
            raise ControllerError(
                "comparison.mode must be evaluated_competitor or "
                "documented_non_dominance"
            )

        metadata = _mapping(action.get("metadata", {}), "metadata")
        metadata.update(
            {
                "evidence_ids": list(evidence_ids),
                "comparison": sanitized_comparison,
                "preregistered_fields": list(preregistered_fields),
            }
        )
        self._ensure_capacity(cost=0.0, terminal=True)
        event = self.loop.dispatch(
            {
                "type": "commit",
                "candidate_id": candidate_id,
                "metadata": metadata,
                "cost": 0.0,
            }
        )
        return self._receipt("commit", [event], {"accepted": True})

    def _close_negative(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "reason", "evidence_ids"},
            optional={"metadata", "cost"},
        )
        reason = _nonempty_text(action["reason"], "reason")
        evidence_ids = _id_list(action["evidence_ids"], "evidence_ids")
        self._resolve_evidence_ids(evidence_ids)
        state = self.loop.state
        live_hypotheses = sorted(
            item.hypothesis_id
            for item in state.hypotheses.values()
            if item.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
        )
        live_candidates = sorted(
            item.candidate_id
            for item in state.candidates.values()
            if item.status not in {Lifecycle.REVISED, Lifecycle.RETIRED}
        )
        if live_hypotheses or live_candidates:
            raise ControllerError(
                "negative close requires every hypothesis and candidate to be terminal; "
                f"live_hypotheses={live_hypotheses} live_candidates={live_candidates}"
            )
        substantive = any(
            evidence_id in state.evaluations
            or (
                evidence_id in state.probes
                and state.probes[evidence_id].result.get("admission")
                != "non_algebraic_design_hypothesis"
            )
            for evidence_id in evidence_ids
        )
        if not substantive:
            raise ControllerError(
                "negative close requires at least one evaluator result or substantive probe; "
                "an automatic non-algebraic admission alone is insufficient"
            )
        cited_ids = set(evidence_ids)
        uncovered_hypotheses: list[str] = []
        for hypothesis in state.hypotheses.values():
            cited_refutation = any(
                probe_id in cited_ids
                and state.probes[probe_id].verdict.value == "refuted"
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
            raise ControllerError(
                "negative close lacks refutation/evaluation evidence for hypotheses: "
                f"{sorted(uncovered_hypotheses)}"
            )
        self._ensure_capacity(cost=0.0, terminal=True)
        event = self.loop.dispatch(
            {
                "type": "close_negative",
                "reason": reason,
                "evidence_ids": list(evidence_ids),
                "metadata": _mapping(action.get("metadata", {}), "metadata"),
                "cost": 0.0,
            }
        )
        return self._receipt("close_negative", [event], {"accepted": True})

    def dispatch_external(self, action: Mapping[str, Any]) -> ControllerReceipt:
        if not isinstance(action, Mapping):
            raise ControllerError("external action must be an object")
        try:
            assert_agent_safe(action)
        except (TypeError, ValueError) as exc:
            raise ControllerError(str(exc)) from exc
        try:
            encoded_size = len(
                json.dumps(action, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
        except (TypeError, ValueError) as exc:
            raise ControllerError(f"external action must contain JSON data: {exc}") from exc
        if encoded_size > MAX_EXTERNAL_ACTION_BYTES:
            raise ControllerError(
                f"external action exceeds {MAX_EXTERNAL_ACTION_BYTES} byte cap"
            )
        if self.loop.state.terminal:
            raise ControllerError("research run is terminal; no further actions are allowed")
        action_type = action.get("type")
        if not isinstance(action_type, str):
            raise ControllerError("external action requires a string type")
        if action_type in {"record_probe", "record_evaluation"}:
            raise ControllerError(f"{action_type} is evaluator-owned and cannot be submitted")

        if action_type == "request_probe":
            return self._request_probe(action)
        if action_type == "evaluate_candidate":
            return self._evaluate_candidate(action)
        if action_type == "propose_hypothesis":
            return self._propose_hypothesis(action)
        if action_type == "submit_candidate":
            return self._submit_candidate(action)
        if action_type == "revise":
            return self._revise(action)
        if action_type == "commit":
            return self._commit(action)
        if action_type == "close_negative":
            return self._close_negative(action)

        if action_type == "retire":
            sanitized = dict(action)
            sanitized["cost"] = 0.0
            if sanitized.get("entity") == "candidate":
                candidate_id = sanitized.get("entity_id")
                candidate = self.loop.state.candidates.get(candidate_id)
                if candidate is not None and candidate.status is Lifecycle.PROMOTED:
                    raise ControllerError(
                        f"promoted candidate {candidate_id} cannot be retired; commit it"
                    )
            self._ensure_capacity(cost=float(sanitized["cost"]))
            event = self.loop.dispatch(sanitized)
            return self._receipt(action_type, [event], {"accepted": True})
        raise ControllerError(f"unsupported external action type: {action_type!r}")

    def _request_probe(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "hypothesis_id", "probe_id", "probe"},
        )
        hypothesis_id = _nonempty_id(action["hypothesis_id"], "hypothesis_id")
        probe_id = _nonempty_id(action["probe_id"], "probe_id")
        probe_request = _mapping(action["probe"], "probe")
        hypothesis = self.loop.state.hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ControllerError(f"unknown hypothesis: {hypothesis_id}")
        if hypothesis.status in {Lifecycle.REVISED, Lifecycle.RETIRED}:
            raise ControllerError(
                f"cannot probe hypothesis {hypothesis_id} in {hypothesis.status.value}"
            )
        if probe_id in self.loop.state.probes:
            raise ControllerError(f"probe already exists: {probe_id}")
        if hypothesis.probe_ids:
            raise ControllerError(
                "exact_pauli_symmetry has one fixed deterministic commutator probe; "
                f"existing evidence={list(hypothesis.probe_ids)}"
            )
        if hypothesis.claim.get("kind") != EXACT_SYMMETRY_CLAIM:
            raise ControllerError(
                "only exact_pauli_symmetry claims accept algebraic probe requests"
            )
        if set(probe_request) != {"type", "generator"}:
            raise ControllerError(
                "symmetry probe fields must be exactly type and generator"
            )
        if probe_request.get("type") != "normalized_commutator":
            raise ControllerError(
                "exact_pauli_symmetry requires a normalized_commutator probe"
            )
        claimed_generator = hypothesis.claim["generator"]
        if probe_request.get("generator") != claimed_generator:
            raise ControllerError("probe generator must match the hypothesis generator")
        # Preflight the bounded sparse work before executing the commutator,
        # and reserve exactly the complexity-derived amount recorded below.
        probe_cost = algebraic_probe_cost_units(
            hamiltonian_from_public(self.problem), probe_request
        )
        self._ensure_capacity(cost=probe_cost)

        receipt = run_public_probe(self.problem, probe_request)
        if not math.isclose(
            receipt.cost_units, probe_cost, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ControllerError("probe preflight and receipt costs disagree")
        passed = _probe_passed(receipt)
        if passed:
            verdict = "supported"
        else:
            verdict = "refuted"

        result = receipt.to_dict()
        result["controller_passed"] = passed
        event = self.loop.dispatch(
            {
                "type": "record_probe",
                "hypothesis_id": hypothesis_id,
                "probe_id": probe_id,
                "verdict": verdict,
                "result": result,
                "cost": receipt.cost_units,
            }
        )
        return self._receipt("request_probe", [event], result)

    def _audit_candidate(self, candidate_id: str) -> tuple[bool, dict[str, Any]]:
        candidate = self.loop.state.candidates[candidate_id]
        try:
            compiled = compile_ansatz(candidate.spec)
            parsed_spec = AnsatzSpec.from_dict(candidate.spec)
            if parsed_spec.num_qubits != self.problem.num_qubits:
                raise ControllerError(
                    "candidate num_qubits must match the public problem"
                )
            if compiled.audit.operations > MAX_CANDIDATE_OPERATIONS:
                raise ControllerError(
                    f"candidate exceeds operation cap {MAX_CANDIDATE_OPERATIONS}"
                )
            if compiled.audit.unique_trainable_params > MAX_CANDIDATE_PARAMETERS:
                raise ControllerError(
                    f"candidate exceeds parameter cap {MAX_CANDIDATE_PARAMETERS}"
                )
            if compiled.audit.spec_nodes > MAX_CANDIDATE_SPEC_NODES:
                raise ControllerError(
                    f"candidate exceeds spec-node cap {MAX_CANDIDATE_SPEC_NODES}"
                )
            if compiled.audit.operations <= 0:
                raise ControllerError("promotable candidate requires at least one operation")
            if compiled.audit.unique_trainable_params <= 0:
                raise ControllerError(
                    "promotable candidate requires at least one trainable parameter"
                )
            excessive_fanout = {
                name: occurrences
                for name, occurrences in compiled.audit.parameter_occurrences.items()
                if occurrences > MAX_PARAMETER_FANOUT
            }
            if excessive_fanout:
                raise ControllerError(
                    f"parameter fan-out exceeds {MAX_PARAMETER_FANOUT}: "
                    f"{excessive_fanout}"
                )

            occupation = self.problem.reference.occupation
            expected_reference_qubits = tuple(
                index for index, bit in enumerate(occupation or ()) if bit
            )
            if expected_reference_qubits:
                if (
                    parsed_spec.reference is None
                    or parsed_spec.reference.macro != "X"
                    or parsed_spec.reference.qubits != expected_reference_qubits
                ):
                    raise ControllerError(
                        "candidate reference must exactly match the evaluator-owned "
                        "public computational-basis reference"
                    )
            elif parsed_spec.reference is not None:
                raise ControllerError(
                    "candidate cannot introduce a reference preparation when the "
                    "public problem declares none"
                )

            hamiltonian_labels = {term.pauli for term in self.problem.pauli_terms}
            for operation in parsed_spec.operations:
                if operation.macro != "PauliRotation" or len(operation.qubits) <= 2:
                    continue
                local_pauli = operation.options["pauli"]
                full_label = ["I"] * self.problem.num_qubits
                for qubit, letter in zip(operation.qubits, local_pauli, strict=True):
                    full_label[self.problem.num_qubits - qubit - 1] = letter
                if "".join(full_label) not in hamiltonian_labels:
                    raise ControllerError(
                        "PauliRotation above locality 2 must be a declared Hamiltonian term"
                    )

            allowed_scales = {-2.0, -1.0, -0.5, 0.5, 1.0, 2.0}
            unapproved_literals = [
                literal.to_dict()
                for literal in compiled.audit.fixed_literals
                if literal.role != "scale"
                or not any(
                    math.isclose(literal.value, allowed, rel_tol=0.0, abs_tol=1e-12)
                    for allowed in allowed_scales
                )
            ]
            if unapproved_literals:
                raise ControllerError(
                    "candidate contains unapproved fixed numeric literals: "
                    f"{unapproved_literals}"
                )

            hypothesis = self.loop.state.hypotheses[candidate.hypothesis_id]
            claim_kind = hypothesis.claim.get("kind")
            conservation_macros = sorted(
                {
                    operation.macro
                    for operation in parsed_spec.operations
                    if operation.macro in {"XYExchange", "IsotropicExchange"}
                }
            )
            if conservation_macros and (
                claim_kind != EXACT_SYMMETRY_CLAIM
                or hypothesis.status is not Lifecycle.SUPPORTED
            ):
                raise ControllerError(
                    "XYExchange and IsotropicExchange require a controller-SUPPORTED "
                    "exact_pauli_symmetry parent; macro registry availability or a "
                    f"family name is not evidence (used={conservation_macros})"
                )
            required_enforcement = {
                EXACT_SYMMETRY_CLAIM: "preserve",
                STRUCTURE_CLAIM: "unconstrained",
                NULL_CONTROL_CLAIM: "diagnostic",
            }.get(str(claim_kind))
            enforcement = candidate.metadata.get("enforcement")
            if enforcement != required_enforcement:
                raise ControllerError(
                    f"{claim_kind} candidate requires "
                    f"metadata.enforcement={required_enforcement!r}"
                )
            symmetry_audit: dict[str, Any] | None = None
            if claim_kind == EXACT_SYMMETRY_CLAIM:
                generator_recipe = hypothesis.claim.get("generator")
                if not isinstance(generator_recipe, Mapping):
                    raise ControllerError(
                        "symmetry-preserving candidate requires a machine-readable hypothesis generator"
                    )
                charge = generator_from_recipe(self.problem.num_qubits, generator_recipe)
                residuals = operation_symmetry_residuals(
                    self.problem.num_qubits,
                    parsed_spec.operations,
                    charge,
                )
                max_residual = max(residuals, default=0.0)
                if max_residual > EXACT_SYMMETRY_TOLERANCE:
                    raise ControllerError(
                        "candidate operation breaks its claimed exact symmetry: "
                        f"residual={max_residual:.3e}"
                    )
                zero_circuit = compiled.circuit.assign_parameters(
                    {parameter: 0.0 for parameter in compiled.parameters.values()},
                    inplace=False,
                )
                reference_mean, reference_variance = reference_moments(
                    zero_circuit,
                    charge,
                )
                if reference_variance > EXACT_SYMMETRY_TOLERANCE:
                    raise ControllerError(
                        "candidate reference is not in a definite sector of the claimed symmetry"
                    )
                symmetry_audit = {
                    "max_operation_residual": max_residual,
                    "reference_mean": reference_mean,
                    "reference_variance": reference_variance,
                }
            metrics = {
                "audit": compiled.audit.to_dict(),
                "valid": True,
            }
            if symmetry_audit is not None:
                metrics["symmetry_audit"] = symmetry_audit
            return True, metrics
        except Exception as exc:
            return False, {
                "valid": False,
                "violations": [f"{type(exc).__name__}: {exc}"],
            }

    def _candidate_baseline_energy(self, spec: Mapping[str, Any]) -> float:
        compiled = compile_ansatz(spec)
        zero_mapping = {parameter: 0.0 for parameter in compiled.parameters.values()}
        zero_circuit = compiled.circuit.assign_parameters(zero_mapping, inplace=False)
        return energy_from_circuit(zero_circuit, hamiltonian_from_public(self.problem))

    def _evaluate_candidate(self, action: Mapping[str, Any]) -> ControllerReceipt:
        _strict_action(
            action,
            required={"type", "candidate_id", "evaluation_id", "stage"},
        )
        candidate_id = _nonempty_id(action["candidate_id"], "candidate_id")
        evaluation_id = _nonempty_id(action["evaluation_id"], "evaluation_id")
        try:
            stage = EvaluationStage(action["stage"])
        except (TypeError, ValueError) as exc:
            raise ControllerError("stage must be audit, smoke, or promotion") from exc
        candidate = self.loop.state.candidates.get(candidate_id)
        if candidate is None:
            raise ControllerError(f"unknown candidate: {candidate_id}")
        if evaluation_id in self.loop.state.evaluations:
            raise ControllerError(f"evaluation already exists: {evaluation_id}")
        prior_stage_ids = [
            evaluation.evaluation_id
            for evaluation in self.loop.state.evaluations.values()
            if evaluation.candidate_id == candidate_id and evaluation.stage is stage
        ]
        if prior_stage_ids:
            raise ControllerError(
                f"candidate {candidate_id} already has fixed {stage.value} evidence: "
                f"{prior_stage_ids}"
            )
        allowed_statuses = {
            EvaluationStage.AUDIT: {Lifecycle.CANDIDATE, Lifecycle.AUDITED},
            EvaluationStage.SMOKE: {Lifecycle.AUDITED, Lifecycle.SMOKE},
            EvaluationStage.PROMOTION: {Lifecycle.SMOKE, Lifecycle.PROMOTED},
        }
        if candidate.status not in allowed_statuses[stage]:
            raise ControllerError(
                f"cannot run {stage.value} for {candidate_id} in {candidate.status.value}"
            )

        if stage is EvaluationStage.AUDIT:
            cost = 0.25
            self._ensure_capacity(cost=cost)
            passed, metrics = self._audit_candidate(candidate_id)
        else:
            if stage is EvaluationStage.SMOKE:
                protocol = EvaluationProtocol(
                    optimizer="cobyla", max_evals=32, restarts=1, seed=7
                )
                cost = 2.0
            else:
                protocol = EvaluationProtocol(
                    optimizer="cobyla", max_evals=96, restarts=3, seed=997
                )
                cost = 6.0
            self._ensure_capacity(cost=cost)
            private_result = evaluate_public_problem(
                self.problem,
                candidate.spec,
                protocol=protocol,
            )
            metrics = private_result.receipt.to_dict()
            resource_policy = _resource_eligibility(metrics.get("metrics", {}))
            metrics["resource_policy"] = resource_policy
            baseline_energy: float | None = None
            try:
                baseline_energy = self._candidate_baseline_energy(candidate.spec)
            except Exception:
                pass
            metrics["baseline_energy"] = baseline_energy
            if baseline_energy is not None and private_result.receipt.best_energy is not None:
                metrics["energy_improvement"] = (
                    baseline_energy - private_result.receipt.best_energy
                )
            improvement_threshold = (
                None
                if baseline_energy is None
                else max(
                    MIN_SMOKE_ENERGY_IMPROVEMENT,
                    MIN_SMOKE_ENERGY_IMPROVEMENT * abs(baseline_energy),
                )
            )
            metrics["required_energy_improvement"] = improvement_threshold
            energy_improvement = metrics.get("energy_improvement")
            passed = bool(
                private_result.receipt.valid
                and resource_policy["eligible"]
                and improvement_threshold is not None
                and energy_improvement is not None
                and float(energy_improvement) >= improvement_threshold
            )
            hypothesis = self.loop.state.hypotheses[candidate.hypothesis_id]
            if stage is EvaluationStage.PROMOTION and hypothesis.claim.get("kind") == NULL_CONTROL_CLAIM:
                passed = False
                metrics["promotion_blocked_reason"] = (
                    "null_control candidates are diagnostic and cannot be promoted"
                )
            if passed and stage is EvaluationStage.PROMOTION:
                smoke_energies = [
                    record.metrics.get("best_energy")
                    for record in self.loop.state.evaluations.values()
                    if record.candidate_id == candidate_id
                    and record.stage is EvaluationStage.SMOKE
                    and record.passed
                ]
                smoke_energies = [float(value) for value in smoke_energies if value is not None]
                best_energy = private_result.receipt.best_energy
                passed = bool(
                    smoke_energies
                    and best_energy is not None
                    and best_energy <= min(smoke_energies) + 5e-4
                )

        event = self.loop.dispatch(
            {
                "type": "record_evaluation",
                "candidate_id": candidate_id,
                "evaluation_id": evaluation_id,
                "stage": stage.value,
                "passed": bool(passed),
                "metrics": metrics,
                "cost": cost,
            }
        )
        return self._receipt("evaluate_candidate", [event], metrics)
