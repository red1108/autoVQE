# AutoVQE Agent Protocol

This is the operating contract for the typed research path. The agent receives
facts, not an ansatz label: it observes the public Hamiltonian, proposes a
falsifiable structural hypothesis, asks the trusted controller to test it, and
submits a candidate only after the hypothesis is supported.

The CLI currently executes one JSON action per invocation. It does not call an
LLM or manage an agent process. The tracked `meta_agent/` operator can expose
this interface to an isolated Codex bundle through a narrow inbox/outbox
bridge; its evaluator side remains outside the agent workspace.

## 1. Establish the safe view

```bash
git status --short --branch
uv run python -m autovqe.harness check
uv run python -m autovqe.harness inspect \
  --problem <problem.json> \
  --agent-json
```

The `--agent-json` payload is an `ObservationBundle`:

- `PublicProblem`: content-derived `problem_id`, Pauli terms, allowlisted
  encoding and declared sector metadata, an optional computational-basis
  preparation hint, and backend basis/connectivity;
- `StructuralObservation`: term/locality/Pauli counts, two-body support edges,
  and whether coefficients are complex.

It deliberately omits `reference_energy`, `reference_state`, fixture/model
classification, ansatz recommendations, and candidate lists. The preparation
hint under `public_problem.reference` is allowed public input; it is not an
exact evaluator reference.

Internally, `PrivateEvaluationContext` can hold the exact reference energy and
state next to the same `PublicProblem`. It must not cross the agent boundary.
The current research controller evaluates the public Hamiltonian locally; this
contract is not yet an external hidden-evaluation service.

## 2. Initialize a run

For local development only:

```bash
uv run python -m autovqe.harness research init \
  --problem <problem.json> \
  --run-dir research_run \
  --budget 20
```

This creates `security_mode: "local_unsealed"`. It is not a trust boundary when
the action-producing agent can write the workspace. Initialization remains
unsealed by default, but every later local `step` or `status` must opt in with
`--allow-unsealed`; operational APIs fail closed to sealed mode by default.

For a sealed deployment, an evaluator/orchestrator outside the agent boundary
holds a secret of at least 16 bytes and an absolute monotonic-anchor directory
outside the run directory, agent-writable workspace, and Git worktree. The
anchor and run may not contain one another:

```bash
export AUTOVQE_EVALUATOR_KEY='<evaluator-held secret, at least 16 bytes>'
export AUTOVQE_EVALUATOR_ANCHOR_DIR='/evaluator-owned/autovqe-anchors'
uv run python -m autovqe.harness research init \
  --problem <problem.json> \
  --run-dir research_run \
  --budget 20 \
  --sealed
```

Never put the evaluator key in an action, prompt, run directory, or
agent-readable environment. Do not place the anchor directory inside the run
directory or give the action-producing agent write access to it.

The run directory contains:

| Artifact | Meaning |
| --- | --- |
| `context.json` | Run schema, problem ID, observation hash, budget, security mode, optional context HMAC |
| `observation.json` | The exact safe bundle presented to the agent |
| `checkpoint.json` | Context hash, ledger event count/tip, security mode, optional checkpoint HMAC |
| `events.jsonl` | Local-unsealed ledger, or a verified mirror of the sealed authoritative ledger; created on the first accepted event |

A schema-v3 sealed context also contains an HMAC-signed random 128-bit
`run_id`. The evaluator-owned directory contains three persistent artifact
paths, all deliberately outside the run directory:

| Evaluator-owned artifact | Meaning |
| --- | --- |
| `autovqe-<run_id>.anchor.json` | HMAC-authenticated current context/count/tip |
| `autovqe-<run_id>.events.jsonl` | Authoritative sealed hash chain; materialized on the first accepted event |
| `autovqe-path-<sha256(abs-run-path)>.registry.json` | HMAC-authenticated absolute-run-path to `run_id`/context binding |

The external anchor lock file normally exists only while a sealed mutation is
in progress. It serializes the complete verification, dispatch, authoritative
append, checkpoint, mirror, and anchor update. A process crash can leave a
stale lock that requires trusted operator recovery.

Reinitializing an existing run is rejected. Every later `step` reloads the
problem and rejects a problem ID or observation hash mismatch. In sealed mode,
the context/checkpoint HMACs, path registry, authoritative ledger, external
anchored head, and run-directory mirror must all verify.

## 3. External action schema

Save exactly one action object to a JSON file and execute it:

```bash
uv run python -m autovqe.harness research step \
  --problem <problem.json> \
  --run-dir research_run \
  --action <action.json>
```

Sealed verification and both evaluator-held environment variables are required
by default. `--require-sealed` is an optional explicit spelling of that
default. Only a `local_unsealed` development run may use
`--allow-unsealed`:

```bash
uv run python -m autovqe.harness research step \
  --problem <problem.json> \
  --run-dir research_run \
  --action <action.json> \
  --allow-unsealed
```

The Python `load_controller`, `execute_action`, `execute_action_file`, and
`run_status` APIs likewise default `require_sealed=True`; development callers
must deliberately pass `require_sealed=False`.

Unknown action types and unknown fields are rejected. IDs must match
`[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`.

| Type | Required fields | Optional fields | Preconditions |
| --- | --- | --- | --- |
| `propose_hypothesis` | `hypothesis_id`, nonempty `claim` | `metadata`, ignored `cost` | New hypothesis ID |
| `request_probe` | `hypothesis_id`, `probe_id`, `probe` | none | Exact-symmetry hypothesis with no prior fixed probe |
| `submit_candidate` | `candidate_id`, `hypothesis_id`, nonempty `spec` | `metadata`, ignored `cost` | New ID; parent `SUPPORTED`; enforcement plus non-control prediction/falsifier metadata |
| `evaluate_candidate` | `candidate_id`, `evaluation_id`, `stage` | none | Correct next stage and no prior result for that fixed stage |
| `revise` | `entity`, `source_id`, `new_id`, `replacement`, `reason` | `metadata`, ignored `cost` | Source can be revised; candidate replacement preregisters fresh prediction/falsifier |
| `retire` | `entity`, `entity_id`, `reason` | ignored `cost` | Existing nonterminal entity |
| `commit` | `candidate_id`, `evidence_ids`, `comparison` | `metadata`, ignored `cost` | Candidate is `PROMOTED` with preregistered prediction/falsifier and grounded comparison evidence |
| `close_negative` | `reason`, `evidence_ids` | `metadata`, ignored `cost` | All hypotheses/candidates are terminal and evidence is substantive |

For `revise`, `entity` is `hypothesis` or `candidate`. `replacement` is a new
closed-schema claim for a hypothesis and a new `AnsatzSpec` mapping for a
candidate. A revised exact-symmetry hypothesis starts at `PROPOSED`; revised
structure/control claims receive the same non-algebraic admission marker and
become `SUPPORTED`. A revised candidate starts at `CANDIDATE`.
For a non-control candidate revision, `metadata` must contain a new nonempty
`prediction` or `falsifier`; the controller carries forward the required
enforcement mode.

The lower-level ledger action types `record_probe` and `record_evaluation` are
evaluator-owned and are rejected at the external controller. Agent-supplied
`cost` values are not authoritative; omit them. The controller replaces them
with fixed costs: `0.10` for propose/submit/revise, zero for
retire/commit/close-negative, and the evaluator receipt cost for
probes/evaluations.

The current outer limits are 1,000,000 serialized bytes per action, 200 ledger
events, three active hypotheses, and two active candidates per hypothesis.
Every nonterminal action must leave one event slot for `commit` or
`close_negative`.

## 4. Observation -> hypothesis -> probe

Treat symmetry and structure as hypotheses, even when sector metadata declares
them. The external controller currently accepts exactly three claim schemas:

| `claim.kind` | Exact claim fields | Admission/evidence |
| --- | --- | --- |
| `exact_pauli_symmetry` | `kind`, `generator` | Stays `PROPOSED` until the matching commutator probe passes |
| `ansatz_structure` | `kind`, nonempty `family` | Controller adds a zero-cost admission marker with `algebraic_certificate: false` |
| `null_control` | `kind` only | Controller adds the same non-algebraic admission marker; its diagnostic candidate cannot be promoted |

Unknown kinds or extra claim fields are rejected. A symmetry claim looks like:

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "parity",
  "claim": {
    "kind": "exact_pauli_symmetry",
    "generator": {
      "type": "pauli_sum",
      "terms": [
        { "pauli": "ZZ", "coeff": 1.0 }
      ]
    }
  },
  "metadata": {
    "rationale": "Every observed term should commute with global Z parity."
  }
}
```

The generator is validated when proposed: it must be a finite Hermitian Pauli
observable with active norm at least `1e-8` and cannot be a zero/identity
operator or a trivial copy of the Hamiltonian.

Request the exact Hamiltonian commutator probe:

```json
{
  "type": "request_probe",
  "hypothesis_id": "parity",
  "probe_id": "comm_zz",
  "probe": {
    "type": "normalized_commutator",
    "generator": {
      "type": "pauli_sum",
      "terms": [
        { "pauli": "ZZ", "coeff": 1.0 }
      ]
    }
  }
}
```

The current generator recipes are:

- `pauli_sum` has exactly `type` and `terms`. It accepts 1–256 unique,
  full-width Pauli labels. Each term has exactly `pauli` and optional `coeff`;
  the coefficient must be a finite real JSON number (not a boolean) with
  `abs(coeff) <= 1e6`.
- `global_pauli_sum` has exactly `type`, `pauli`, and optional `selector`;
  `orbit_pauli_sum` uses `seed` instead of `pauli`. Both currently accept only
  `X`/`Y`/`Z` and `selector: "all_sites"` and are limited to at most 256
  generated terms/qubits.

For an `exact_pauli_symmetry` claim, `request_probe` accepts exactly
`type: "normalized_commutator"` and the identical generator recipe stored in
the claim. Residual at most `1e-10` moves the hypothesis to `SUPPORTED`; a
nonzero result records a `refuted` verdict and leaves it `PROBED`.

`reference_moments` exists as an evaluator primitive but is not an external
`research step` probe. For a symmetry candidate, the audit computes normalized
reference moments internally from the candidate's required public reference
preparation. Normalization by squared generator norm prevents an epsilon-scaled
generator from manufacturing a sector pass.

`ansatz_structure` and `null_control` are admitted for bounded design/control
experiments, not certified as algebraic facts. Their controller-authored
admission marker is visible in the ledger and explicitly says
`algebraic_certificate: false`.

Approximate-symmetry, group/point-group/permutation, and fermionic-algebra
probes are not implemented.

## 5. Submit a typed candidate

Only a `SUPPORTED` hypothesis can own a candidate. A minimal candidate action
looks like:

```json
{
  "type": "submit_candidate",
  "candidate_id": "parity_xx_1",
  "hypothesis_id": "parity",
  "spec": {
    "version": 1,
    "name": "parity_xx_1",
    "num_qubits": 2,
    "parameters": [
      { "name": "theta" }
    ],
    "reference": {
      "macro": "X",
      "qubits": [0]
    },
    "layers": [
      {
        "name": "exchange",
        "operations": [
          {
            "macro": "PauliRotation",
            "qubits": [0, 1],
            "parameters": {
              "angle": { "parameter": "theta" }
            },
            "options": {
              "pauli": "XX"
            }
          }
        ]
      }
    ]
  },
  "metadata": {
    "prediction": "Preserve the probed parity sector with one trainable move.",
    "enforcement": "preserve"
  }
}
```

`AnsatzSpec` accepts only JSON structure, declared parameters, an optional
trusted reference macro, and ordered trusted operations. The operation macro
allowlist is `PauliRotation`, `XYExchange`, and `IsotropicExchange`; `X` is
reference-only. There is no candidate-provided Python, Qiskit instruction,
matrix, optimizer, or initial angle vector. Energy/resource claims in metadata
are non-authoritative, and adding those fields to the strict IR is rejected.
Public backend basis gates affect lowering/counting only and cannot authorize a
new logical macro.

Allowlisting does not itself authorize a conservation claim. `XYExchange` and
`IsotropicExchange` pass controller audit only under a controller-`SUPPORTED`
`exact_pauli_symmetry` parent, and then each operation must still commute with
the probed charge. A macro name, registry entry, Hamiltonian pattern, or
auto-admitted structure/control family is not evidence.

Angle expressions are affine. Parameter sharing is allowed, but the audit
reports both unique parameter names and every occurrence. All declared
parameters must be used; constant-only variational operations, fixed offsets,
unknown fields/macros, invalid supports, and opaque values fail compilation.

Candidate metadata must match its claim kind:

- `exact_pauli_symmetry` requires `"enforcement": "preserve"`;
- `ansatz_structure` requires `"enforcement": "unconstrained"`;
- `null_control` requires `"enforcement": "diagnostic"`.

Every non-control candidate's immutable submission metadata must also
preregister a nonempty `prediction` or `falsifier`; it cannot be supplied after
its evaluations are known. A candidate revision preregisters a fresh value for
the replacement before its new stages run.

Candidate hashes identify the semantic variational family. The evaluator
alpha-normalizes parameter names/order and excludes ansatz names, layer labels,
and layer-only grouping while retaining the ordered physical operation
sequence, parameter incidence, reference, qubits, macros, options, and affine
coefficients. A submit/revise action is rejected if that semantic hash already
exists in the campaign, even under another ID or a retired branch. This blocks
fresh optimizer and generic-resource trials obtained only by cosmetic renaming.

For `preserve`, audit checks the exact commutator of every trusted operation
generator with the claim's charge and the normalized charge variance of the
explicit zero-parameter reference. Audit also requires the reference
preparation to exactly match the public occupation hint (and forbids adding one
when no hint is declared), and requires at least one operation and one
trainable parameter.

## 6. Audit, smoke, promote

Request stages in order with a new evaluation ID each time:

```json
{
  "type": "evaluate_candidate",
  "candidate_id": "parity_xx_1",
  "evaluation_id": "parity_xx_1_audit",
  "stage": "audit"
}
```

Valid stage values are `audit`, `smoke`, and `promotion`.
Each fixed probe and each fixed candidate stage may run only once. Giving the
same deterministic experiment a new ID is rejected rather than counted as new
evidence.

| Stage | Evaluator-owned behavior | Cost |
| --- | --- | ---: |
| `audit` | Trusted compile; derived provenance, global complexity/locality limits, and requested exact-symmetry enforcement | 0.25 |
| `smoke` | COBYLA; at most 32 objective calls; one restart; seed 7 | 2.00 |
| `promotion` | COBYLA; at most 96 calls; three restarts; seed 997 | 6.00 |

The audit permits parameter coefficients only from
`{-2, -1, -0.5, 0.5, 1, 2}` and rejects fixed offsets/numeric options. Energy,
objective calls, optimizer/seed, resource metrics, and the compile audit are
evaluator receipts. A candidate cannot submit its own values for them.

The global audit caps are 256 logical operations, 128 unique trainable
parameters, 4096 IR nodes, and 64 occurrences (fan-out) for any one parameter.
A Pauli rotation above locality two must be a declared Hamiltonian term.

Smoke/promotion resource eligibility uses a fixed candidate-independent
canonical target: basis `{rz, sx, x, cx}`, all-to-all connectivity, and
transpiler optimization level 1. The evaluator transpiles the symbolic
template and samples three deterministic, candidate-specific generic parameter
bindings, retaining the worst generic result. For each metric the controller
then computes `canonical_conservative_* = max(canonical_template_*,
canonical_generic_worst_*)`. It requires:

- `canonical_conservative_twoq_count <= 512`;
- `canonical_conservative_total_gate_count <= 2048`;
- `canonical_conservative_depth <= 1024`.

Receipts include declared-backend `template_*`, `generic_worst_*`, and
`final_*` metrics plus `canonical_template_*`,
`canonical_generic_worst_*`, and `canonical_final_*`. The older `generic_*`
keys remain aliases of declared-backend generic-worst metrics. The receipt's
controller-owned `resource_policy` exposes the `canonical_conservative_*`
values, inputs, limits, and violations. Only that conservative triple controls
current resource eligibility; the final view remains audit evidence.

Smoke and promotion must improve over the candidate's zero-parameter baseline
by at least `max(1e-6, 1e-6 * abs(baseline_energy))`. Promotion also requires a
prior passed smoke and an energy no worse than the best smoke energy plus
`5e-4`. `null_control` promotion is always blocked. A passed promotion does not
mean that the candidate won a Pareto frontier, generalized to hidden
Hamiltonians, or met a reference-energy tolerance.

Any failed evaluation moves the candidate to `RETIRED`. Use `revise` to create
a linked replacement with a new ID, or retire the branch with a concrete
reason.

## 7. Commit or falsify

Commit only a promoted candidate:

```json
{
  "type": "commit",
  "candidate_id": "parity_xx_1",
  "evidence_ids": ["parity_xx_1_promotion"],
  "comparison": {
    "mode": "documented_non_dominance",
    "reason": "No evaluated candidate dominates this promotion under the recorded energy and resource receipts.",
    "evidence_ids": ["parity_xx_1_promotion"]
  },
  "metadata": {
    "decision": "accepted under the current local promotion protocol"
  }
}
```

`commit` stores the selected candidate ID and prevents all further events in
that run. The candidate metadata, fixed at submission, must contain a nonempty
`prediction` or `falsifier`; commit evidence must include its passed promotion.
The comparison is either an `evaluated_competitor` naming a different candidate
and at least one of its smoke/promotion evaluation IDs, or
`documented_non_dominance` with a nonempty reason and the selected promotion
ID. All comparison IDs also appear in the top-level evidence list. Commit does
not export code or modify `train.py`.

If evidence contradicts the claim, preserve that result:

```json
{
  "type": "retire",
  "entity": "hypothesis",
  "entity_id": "parity",
  "reason": "The requested commutator probe was nonzero."
}
```

An active candidate must be retired/revised before its parent hypothesis can be
retired/revised.

Retirement alone does not close a negative campaign. Once every hypothesis and
candidate is `REVISED` or `RETIRED`, submit `close_negative` with a nonempty
reason and existing probe/evaluation IDs. At least one citation must be a
substantive exact probe or evaluator result; an automatic non-algebraic
admission marker alone is insufficient. Each hypothesis must be covered by a
cited refuting probe or a cited evaluation of one of its candidates. The accepted event sets
`terminal_decision` to `negative_close` and prevents further actions.

## 8. Verify and replay

```bash
uv run python -m autovqe.harness research status \
  --run-dir research_run
```

`status` is sealed by default. For local development use `--allow-unsealed`.
It verifies context/checkpoint bindings, sequence numbers, each event hash,
every `prev_hash`, and, for a sealed run, the path registry, authoritative
external ledger, anchored head, and workspace mirror. It then replays the
authoritative ledger through the same typed transition reducer. It reports
security mode, checkpoint, budget, hypotheses, candidates, probes,
evaluations, positive/negative terminal state, and the chain tip.

In schema-v3 `sealed` mode, the context HMAC binds its random 128-bit `run_id`,
schema, problem ID, observation hash, budget, and mode. The checkpoint HMAC
binds the context hash plus ledger event count/tip. The path registry pins the
absolute run path to that signed run/context. The evaluator-owned
`autovqe-<run_id>.events.jsonl` is authoritative; the run-directory ledger is
only an event-hash-equivalent verified mirror. The anchor HMAC records the same
context and current external-ledger count/tip.

One exclusive external lock spans pre-dispatch verification, trusted dispatch
and append, mirror/checkpoint publication, and anchor advance. Advancement
checks that the old anchored tip occurs at the old event-count prefix in the
new authoritative chain. It therefore rejects both restoration of an older
valid pair and a longer non-descendant fork; the signed registry rejects
run-path swapping. Cross-file crashes fail closed instead of accepting a fork
and can require trusted operator recovery. The CLI reports
`rollback_protected: true` only when the HMAC secret and all evaluator-owned
state remain outside agent control. This is not a digital signature, remote
anchoring service, or hidden-evaluation service.

In `local_unsealed` mode, hashes catch inconsistent/casual edits but an agent
with filesystem write access can rebuild the context, checkpoint, and whole
chain. The CLI reports `tamper_evident: false` for that threat model.

## 9. Lifecycle

```text
hypothesis: PROPOSED -> SUPPORTED (probe pass)
                   \-> PROBED (probe refutation)
                   \-> REVISED | RETIRED

candidate:  CANDIDATE -> AUDITED -> SMOKE -> PROMOTED -> evidence-gated commit
                     \-> RETIRED
                     \-> REVISED (new candidate starts at CANDIDATE)

all branches REVISED/RETIRED + grounded evidence -> close_negative closes run
```

For the current external symmetry probe, `PROBED` means a recorded refutation;
it never advances from `PROBED` to `SUPPORTED`. Inspect the stored verdict
rather than inferring support from the lifecycle name alone. Structure/control
admission moves directly to `SUPPORTED` through a controller-authored marker.
A promoted candidate cannot be revised or retired; it remains live until
commit and therefore blocks negative close.

## 10. Reporting

For a research run, report:

- problem ID and observation hash;
- hypothesis claim plus exact probe receipt or non-algebraic admission marker;
- candidate ID plus evaluator-derived candidate hash;
- audit, smoke, and promotion evaluation IDs;
- best energy and objective-call trace from evaluator receipts;
- declared-backend and canonical template/generic-worst/final gate counts and
  depth, plus the resource-policy eligibility result;
- unique trainable parameters and parameter occurrence counts;
- declared enforcement mode and any exact symmetry-audit residual/reference
  moments;
- security mode, checkpoint-bound ledger tip/event count, spent/remaining
  budget, and controller-accepted commit/negative-close decision;
- limitations of the evidence.

Do not call promotion a hidden benchmark result. The external hidden service,
automatic agent adapter, holdout/generalization suite, Pareto scorer, anytime
score, approximate/group/fermionic probe families, and constraint-specific
macro profiles remain future work.

## Legacy compatibility path

`inspect` without `--agent-json`, `campaign`, `benchmark`, and `solve` retain the
older recommendation-driven workflow around `train.py`. `solve` can compare to
a reference energy and is useful for regression compatibility. It is separate
from the safe typed research protocol and must not be used as if it were a
hidden evaluator receipt.
