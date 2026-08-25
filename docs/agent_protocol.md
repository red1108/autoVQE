# AutoVQE Research Protocol

This document defines the action language used by `autovqe.harness research`.
The protocol separates two responsibilities:

- the agent proposes hypotheses, circuit structure, and branch decisions;
- AutoVQE performs probes, compilation, optimization, and measurement.

Only evaluator-produced values count as evidence.

## 1. Start from the supplied problem

Inspect the Hamiltonian before proposing a circuit:

```bash
uv run python -m autovqe.harness inspect \
  --problem user_problem/hamiltonian.json \
  --json
```

The observation includes the real-coefficient Pauli Hamiltonian, qubit count,
declared encoding and sector metadata, optional computational-basis preparation,
backend facts, locality counts, Pauli counts, and support-graph edges. It does
not provide ansatz recommendations or an exact reference answer.

Initialize one research run:

```bash
uv run python -m autovqe.harness research init \
  --problem user_problem/hamiltonian.json \
  --run-dir .autovqe-runtime/research/run-001 \
  --budget 100
```

The problem associated with a run cannot be replaced later. Keep the input
file unchanged for the full run.

## 2. Submit one action at a time

Store one JSON object in a file and submit it:

```bash
uv run python -m autovqe.harness research step \
  --run-dir .autovqe-runtime/research/run-001 \
  --action action.json
```

Unknown fields and unknown action types are rejected. Identifiers must match
`[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}` and be unique where required. The controller
sets evaluation costs; an agent-supplied `cost` is not authoritative.

| Action | Required fields | Optional fields | Purpose |
| --- | --- | --- | --- |
| `propose_hypothesis` | `hypothesis_id`, `claim` | `metadata` | Open a falsifiable branch |
| `request_probe` | `hypothesis_id`, `probe_id`, `probe` | none | Measure an exact symmetry claim |
| `submit_candidate` | `candidate_id`, `hypothesis_id`, `spec` | `metadata` | Propose a typed circuit |
| `evaluate_candidate` | `candidate_id`, `evaluation_id`, `stage` | none | Request `audit`, `smoke`, or `promotion` |
| `revise` | `entity`, `source_id`, `new_id`, `replacement`, `reason` | `metadata` | Replace a hypothesis or candidate while preserving lineage |
| `retire` | `entity`, `entity_id`, `reason` | none | End a disproven or unproductive branch |
| `commit` | `candidate_id`, `evidence_ids`, `comparison` | `metadata` | Request a positive terminal decision |
| `close_negative` | `reason`, `evidence_ids` | `metadata` | Request a grounded negative terminal decision |

`record_probe` and `record_evaluation` are evaluator-only actions. An external
action cannot create its own evidence.

The outer limits are 1,000,000 serialized bytes per action, 200 recorded
events, three live hypotheses, and two live candidates under one hypothesis.
A nonterminal step must leave room for a final decision.

## 3. Hypothesis types

The accepted claims are intentionally closed.

### Exact Pauli symmetry

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "u1-number",
  "claim": {
    "kind": "exact_pauli_symmetry",
    "generator": {
      "type": "global_pauli_sum",
      "pauli": "Z",
      "selector": "all"
    }
  }
}
```

This branch remains proposed until a matching normalized-commutator probe is
run. A normalized residual at most `1e-10` supports the claim; a larger value
refutes it.

Generator recipes use one of these forms:

- `pauli_sum`: one to 256 unique full-register Pauli terms;
- `global_pauli_sum`: one single-site Pauli over all qubits;
- `orbit_pauli_sum`: translates one single-site seed over all qubits.

A generator must be finite, Hermitian, nontrivial, and distinct from a copy of
the Hamiltonian. Its active norm must be at least `1e-8`.

### Ansatz structure

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "local-hva",
  "claim": {
    "kind": "ansatz_structure",
    "family": "ordered rotations from selected Hamiltonian terms"
  }
}
```

This records a structural experiment, not an algebraic certificate. It is
admitted immediately so that its candidate can be tested.

### Null control

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "shallow-control",
  "claim": {"kind": "null_control"}
}
```

A control candidate is diagnostic and cannot pass promotion.

## 4. Typed ansatz rules

Every candidate uses `AnsatzSpec` version 1. It declares:

- `name` and `num_qubits`;
- unique scalar parameters;
- an optional `X` reference preparation;
- ordered operations, optionally grouped into display layers.

Each variational operation has `macro`, distinct in-range `qubits`, an `angle`
expression, and any allowlisted options. An angle is affine in declared
parameters. Compilation rejects undeclared or unused parameters,
constant-only variational gates, nonzero offsets, opaque circuit data, unknown
fields, custom gates, matrices, callables, optimizer settings, seeds, and
candidate-provided initial values.

The operation allowlist is:

| Operation | Meaning | Applicability |
| --- | --- | --- |
| `PauliRotation` | `exp(-i angle P)` for an active local Pauli word | General; locality above two must match a Hamiltonian term |
| `XYExchange` | `exp(-i angle (XX + YY))` | Requires a supported exact symmetry parent |
| `IsotropicExchange` | `exp(-i angle (XX + YY + ZZ))` | Requires a supported exact symmetry parent |
| `X` | Computational-basis preparation | Reference preparation only |

Every variational operation is the identity at angle zero. Trainable affine
coefficients are limited to `{-2, -1, -0.5, 0.5, 1, 2}`, and fixed offsets are
forbidden. An ansatz may contain at most 256 logical operations, 128 unique
parameters, 4096 representation nodes, and 64 occurrences of one parameter.

The evaluator assigns a semantic candidate fingerprint after removing display
names, layer-only grouping, parameter spelling, and declaration order. It also
folds adjacent rotations with the same generator. A previously tested physical
family cannot obtain another optimization attempt through cosmetic rewriting.

Candidate metadata must declare its intended enforcement:

- `preserve` for an exact-symmetry branch;
- `unconstrained` for a structural branch;
- `diagnostic` for a null control.

Every non-control candidate must also preregister a nonempty `prediction` or
`falsifier`. A revision supplies a fresh statement before seeing the new
candidate's evaluation.

## 5. Symmetry-preservation audit

A supported commutator result is necessary but not sufficient for using a
conservation-oriented operation. During audit, AutoVQE also checks:

1. every instantiated operation generator commutes with the claimed charge;
2. the zero-parameter reference lies in a definite charge sector;
3. the candidate uses the declared reference preparation exactly;
4. the candidate did not opt out of its required enforcement mode.

This prevents a suggestive operation name or Hamiltonian pattern from being
treated as evidence.

## 6. Evaluation stages

Stages run in order: `audit`, `smoke`, `promotion`.

| Stage | Evaluator behavior | Cost units |
| --- | --- | ---: |
| Hypothesis proposal | Validate and record the claim | 0.10 |
| Exact commutator probe | Compute normalized residual | complexity-derived |
| Candidate submission or revision | Validate structure and lineage | 0.10 |
| Audit | Compile, enforce limits, check reference and invariants | 0.25 |
| Smoke | COBYLA, at most 32 objective calls, one restart, seed 7 | 2.00 |
| Promotion | COBYLA, at most 96 objective calls, three restarts, seed 997 | 6.00 |
| Retirement or terminal decision | Validate the transition | 0.00 |

For smoke and promotion, AutoVQE derives:

- optimized energy and parameter binding;
- objective-call traces;
- unique parameter and parameter-occurrence counts;
- logical operations and compiled gate counts;
- two-qubit gate count and depth;
- declared-backend and canonical resource views;
- invariant checks and pass/fail status.

Canonical resource eligibility uses the worst value across the symbolic circuit
and three deterministic generic nonzero bindings. The limits are 512 two-qubit
gates, 2048 total gates, and depth 1024. This prevents parameter sharing or
special-value cancellation from hiding circuit cost.

Smoke and promotion must improve over the evaluator-computed zero-angle
baseline by at least `max(1e-6, 1e-6 * abs(baseline_energy))`. Promotion also
requires a passed smoke and an energy no worse than the best passed smoke plus
`5e-4`.

## 7. Lifecycle and branch decisions

Candidate states progress as follows:

```text
CANDIDATE -> AUDITED -> SMOKE -> PROMOTED
     |          |          |
     +----------+----------+-> RETIRED on failed evaluation
```

Revision creates a new linked identifier; it never overwrites the prior branch.
A hypothesis cannot be retired while it has a live child. A promoted candidate
cannot be revised or retired, so a successful result cannot be erased to force
a negative ending.

A `commit` requires:

- a promoted candidate;
- its passed promotion evidence;
- its preregistered prediction or falsifier;
- either an evaluated competitor or a documented non-dominance comparison.

A `close_negative` requires every investigated hypothesis and candidate to be
revised or retired. Its citations must include substantive probe or evaluation
evidence and cover every investigated hypothesis. Retirement alone is not
enough.

## 8. Status and final result

```bash
uv run python -m autovqe.harness research status \
  --run-dir .autovqe-runtime/research/run-001
```

Status reconstructs the current branch state from recorded actions and shows
the budget, hypotheses, candidates, probes, evaluations, and terminal decision.

After a terminal decision:

```bash
uv run python -m autovqe.harness research result \
  --run-dir .autovqe-runtime/research/run-001
```

For `positive_commit`, report the committed candidate, its typed specification,
the promotion energy, optimized parameters, resource measurements, and cited
comparison. For `negative_close`, report the explored branches, failed tests,
and cited closure evidence. Never replace evaluator values with candidate
metadata, and never describe local promotion as an exact or general result
without a separate evaluation.
