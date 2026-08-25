# AutoVQE Research Protocol

This is the JSON action contract for `uv run autovqe research`. The agent
proposes hypotheses, circuits, and branch decisions. The controller derives
probe definitions, evaluation stages, identifiers, optimization settings, and
all measured values.

## Start a run

```text
uv run autovqe inspect --problem user_problem/hamiltonian.json --json
uv run autovqe research init --problem user_problem/hamiltonian.json --run-dir .autovqe-runtime/research/run-001 --budget 100
```

The problem is immutable for the run. Its `initial_state_hint`, when present,
is applied by the evaluator before the variational circuit. It does not appear
inside the candidate specification.

Submit one JSON action per step:

```text
uv run autovqe research step --run-dir .autovqe-runtime/research/run-001 --action .autovqe-runtime/actions/next.json
```

Unknown fields and action types are rejected. Agent-created identifiers match
`[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}`; the 96-character limit leaves room for
controller-owned evidence prefixes. The controller owns action costs and
evidence identifiers.

## External actions

| Action | Required fields | Optional fields | Effect |
| --- | --- | --- | --- |
| `propose_hypothesis` | `hypothesis_id`, `claim` | `metadata` | Open a falsifiable branch |
| `request_probe` | `hypothesis_id` | none | Test the exact generator already recorded in the claim |
| `submit_candidate` | `candidate_id`, `hypothesis_id`, `spec` | `metadata`, `symmetry_evidence_ids` | Register a typed circuit and optional supported symmetry certificates |
| `evaluate_candidate` | `candidate_id` | none | Run the next valid stage: audit, smoke, or promotion |
| `revise` | `entity`, `source_id`, `new_id`, `replacement`, `reason` | `metadata`, candidate-only `symmetry_evidence_ids` | Create a linked replacement while preserving the failed branch |
| `retire` | `entity`, `entity_id`, `reason` | none | End a disproven or unproductive branch |
| `commit` | `candidate_id` | none | Request a positive terminal decision using controller-derived evidence |
| `close_negative` | `reason` | none | Request grounded closure after every branch was explicitly revised or retired |

`record_probe` and `record_evaluation` are internal events. External actions
cannot supply a probe result, energy, optimized values, parameter binding,
resource count, evaluation identifier, stage, optimizer, seed, or cost.

## Hypotheses

### Exact Pauli symmetry

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "total-z",
  "claim": {
    "kind": "exact_pauli_symmetry",
    "generator": {
      "type": "global_pauli_sum",
      "pauli": "Z",
      "selector": "all_sites"
    }
  }
}
```

Then request its deterministic commutator probe:

```json
{"type": "request_probe", "hypothesis_id": "total-z"}
```

The controller derives the generator and probe identifier from the claim. A
normalized commutator residual at most `1e-10` supports the branch; otherwise
it is refuted. A supported Hamiltonian-level symmetry is necessary, but does
not by itself certify a candidate circuit.

Generator recipes are closed and bounded:

- `pauli_sum`: one to 256 unique full-register Pauli terms;
- `global_pauli_sum`: one `X`, `Y`, or `Z` on every site, with
  `selector: "all_sites"`;
- `orbit_pauli_sum`: translates one single-site seed over all sites, also with
  `selector: "all_sites"`.

A custom Pauli sum uses unique full-register labels and optional real
coefficients with absolute value at most `1e6`:

```json
{"type": "pauli_sum", "terms": [
  {"pauli": "XXII", "coeff": 1.0},
  {"pauli": "YYII"}
]}
```

An orbit recipe uses a single-site `X`, `Y`, or `Z` seed; `selector` defaults
to `all_sites` when omitted:

```json
{"type": "orbit_pauli_sum", "seed": "Z", "selector": "all_sites"}
```

Generators must be finite, Hermitian, nontrivial, and not a scaled copy of the
Hamiltonian. The controller checks bounded probe cost before computing the
commutator.

### Structural hypothesis

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "selected-hva",
  "claim": {
    "kind": "ansatz_structure",
    "family": "ordered rotations from selected Hamiltonian terms"
  }
}
```

This is a testable design idea, not an algebraic certificate. It is ready for a
candidate without a commutator request.

### Null control

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "shallow-control",
  "claim": {"kind": "null_control"}
}
```

A null control uses an empty `AnsatzSpec` with no parameters or operations. It
is allowed through smoke so it can receive the same promotion-budget
evaluation as a target, but it remains diagnostic and cannot become a commit
target.

## `AnsatzSpec`

Version 1 has a single flat operation sequence:

```json
{
  "version": 1,
  "name": "two-qubit-xx",
  "num_qubits": 2,
  "parameters": ["theta"],
  "operations": [
    {
      "macro": "PauliRotation",
      "qubits": [0, 1],
      "parameters": {
        "angle": {"parameter": "theta"}
      },
      "options": {"pauli": "XX"}
    }
  ]
}
```

The only variational macros are:

| Macro | Meaning | Admission rule |
| --- | --- | --- |
| `PauliRotation` | `exp(-i angle P)` | General; locality above two must match a Hamiltonian term |
| `XYExchange` | `exp[-i angle(XX + YY)]` | Relevant supported symmetry evidence plus operation-preservation audit |
| `IsotropicExchange` | `exp[-i angle(XX + YY + ZZ)]` | Relevant supported symmetry evidence plus operation-preservation audit |

There is no `X` candidate macro or candidate preparation field. `operations`
is the only circuit sequence. Unknown macros, custom matrices, executable
payloads, opaque instructions, and candidate optimizer settings are rejected.

Angles are affine expressions over declared scalar parameters. Every
variational angle must depend on a parameter and equal zero when all parameters
are zero. All declared parameters must be used and linearly independent in the
operation-angle coordinates. Allowed trainable
coefficients are `{-2, -1, -0.5, 0.5, 1, 2}`; fixed angle offsets are
forbidden.

Example submission:

```json
{
  "type": "submit_candidate",
  "candidate_id": "xx-1",
  "hypothesis_id": "selected-hva",
  "spec": {
    "version": 1,
    "name": "two-qubit-xx",
    "num_qubits": 2,
    "parameters": ["theta"],
    "operations": [
      {
        "macro": "PauliRotation",
        "qubits": [0, 1],
        "parameters": {"angle": {"parameter": "theta"}},
        "options": {"pauli": "XX"}
      }
    ]
  },
  "metadata": {
    "prediction": "selected XX motion should beat the zero-angle baseline"
  }
}
```

Symmetry is composable evidence, not the candidate's whole research identity.
Keep the primary `hypothesis_id` for the structural branch and optionally cite
supported probe IDs separately:

```json
{
  "type": "submit_candidate",
  "candidate_id": "exchange-1",
  "hypothesis_id": "ladder-ordering",
  "symmetry_evidence_ids": ["probe:total-z"],
  "spec": {"version": 1, "name": "exchange", "num_qubits": 2,
           "parameters": ["theta"], "operations": [
             {"macro": "XYExchange", "qubits": [0, 1],
              "parameters": {"angle": {"parameter": "theta"}},
              "options": {}}
           ]},
  "metadata": {"prediction": "the exchange ordering beats its baseline"}
}
```

Every cited probe must be evaluator-supported. Audit verifies that every
operation preserves every cited charge and that the initial state has a
definite sector for each one. Each special exchange operation must also touch
at least `1e-3` of the active norm of one cited charge and preserve that
overlapping charge. A disjoint spectator or epsilon-weighted touching term is
not sufficient. The controller also conditions the Hamiltonian residual by
that fraction and the sector variance by its square before the charge can
unlock a special gate.

The controller derives enforcement from the branch and cited evidence:

- `preserve` whenever supported symmetry evidence is cited;
- `unconstrained` for `ansatz_structure`;
- `diagnostic` for `null_control`.

A promotable candidate preregisters a nonempty `prediction` or `falsifier`.
If present, candidate metadata is limited to text `prediction`, `falsifier`,
and `rationale` fields. It does not become evaluator evidence.

## Audit and automatic stage progression

Use the same action at each stage:

```json
{"type": "evaluate_candidate", "candidate_id": "xx-1"}
```

The controller infers the next stage from candidate state:

```text
CANDIDATE --evaluate--> AUDITED --evaluate--> SMOKE --evaluate--> PROMOTED
     |                     |                    |
     +---------------------+--------------------+-> RETIRED on failure
```

| Stage | Fixed evaluator behavior | Cost units |
| --- | --- | ---: |
| Hypothesis proposal | Validate and record claim | 0.10 |
| Exact commutator | Bounded normalized residual | complexity-derived |
| Candidate submission/revision | Validate representation and lineage | 0.10 |
| Audit | Compile, check literals and symmetry, measure resources | 0.25 |
| Smoke | COBYLA, at most 32 calls, one restart, seed 7 | 2.00 |
| Promotion | COBYLA, at most 96 calls, three restarts, seed 997 | 6.00 |

Audit happens before optimization. It checks representation limits, parameter
fan-out, Hamiltonian-term locality, nonzero audit-binding resources, and
the applicable symmetry rules. For `preserve`, the evaluator also prepares the
problem's initial state, verifies it lies in a definite sector of every cited
charge, and checks every operation generator against every charge.

The current caps are 256 logical operations, 128 unique parameters, 4096
representation nodes, 64 occurrences of one parameter, 512 two-qubit gates,
2048 total gates, and depth 1024. The gate/depth limits use the worst of
backend-routed and canonical template/nonzero-binding transpilation. Nonzero
audit bindings prevent zero or special parameter values from hiding circuit
cost.

Smoke and promotion must improve on the evaluator-computed zero-angle baseline
by at least `max(1e-6, 1e-6 * abs(baseline_energy))`. Promotion also requires a
passed smoke result and must finish no more than `5e-4` above its best passed
smoke energy.

Before the first promotion, a candidate from a different primary hypothesis
must already have passed smoke, and enough budget must remain to evaluate both
at promotion fidelity. Once the first candidate is promoted, that reserved
comparison is the next required action, so intervening actions cannot consume
its budget. Commit comparisons use only those fixed-protocol
promotion records. They compare energy and conservative two-qubit/total/depth
resources plus evaluator-counted unique trainable parameters. Same-hypothesis
implementation variants cannot be the sole scientific comparator.

## Failure, revision, and closure

Failed evaluations remain in history. A motivated revision creates a new ID
and links it to the old branch; it does not overwrite or retry the same
physical circuit under a cosmetic name.

```json
{
  "type": "revise",
  "entity": "candidate",
  "source_id": "xx-1",
  "new_id": "xx-2",
  "replacement": {
    "version": 1,
    "name": "single-qubit-y",
    "num_qubits": 2,
    "parameters": ["theta"],
    "operations": [
      {
        "macro": "PauliRotation",
        "qubits": [0],
        "parameters": {"angle": {"parameter": "theta"}},
        "options": {"pauli": "Y"}
      }
    ]
  },
  "reason": "the failed audit showed excessive resource growth",
  "metadata": {
    "falsifier": "the smaller circuit still fails to beat its zero-angle baseline"
  }
}
```

Use `retire` when evidence ends a branch:

```json
{"type": "retire", "entity": "candidate", "entity_id": "xx-1",
 "reason": "failed smoke and no motivated revision remains"}
```

A positive ending requires a promoted candidate, its promotion result, and a
different-hypothesis competitor or null control evaluated with the same
promotion protocol. The controller discovers and cites that evidence; the
agent does not manufacture a comparison record:

```json
{"type": "commit", "candidate_id": "xx-2"}
```

Before negative closure, explicitly revise or retire every open hypothesis and
candidate. For every non-control hypothesis, grounding must be a refuted probe,
a valid failed smoke/promotion run with objective calls, or a promoted candidate
retired after a fair dominating comparison. A passed audit or prose retirement
reason does not count. The controller also enforces a fixed search floor:
each counted failed run must have evaluator-observed objective activity of at
least `(max(E)-min(E))/||H_nonidentity||2 = 1e-6`. Closure needs either an
objective-active failed/dominated promotion, or objective-active failures in
two independent `ansatz_structure` root lineages. Hypothesis revisions remain
in their original root lineage. The explicit exception is a constant
Hamiltonian, for which no circuit can create objective activity. Null controls,
compile audits, refuted probes, and flat phase-only circuits do not count
toward the floor. A promoted candidate may be revised or retired only after an
evaluator-proven dominating comparison. The controller derives evidence
coverage; it does not retire branches for the agent:

```json
{"type": "close_negative", "reason": "all tested families failed their preregistered checks"}
```

## Status and result

```text
uv run autovqe research status --run-dir .autovqe-runtime/research/run-001
uv run autovqe research status --run-dir .autovqe-runtime/research/run-001 --full
```

Default status is compact: budget, branch states, latest evidence summaries,
and suggested next actions. `--full` returns complete branch records for
debugging or detailed review, but optimized bindings remain reserved for the
terminal result.

After `positive_commit` or `negative_close`:

```text
uv run autovqe research result --run-dir .autovqe-runtime/research/run-001
```

Only the final result should be used to report an optimized binding. The result
command reproduces the committed fixed promotion once to materialize that
binding; it is not stored in preterminal run history. A local
promotion proves the fixed within-run rule, not exact accuracy or
generalization. If no independent reference score was supplied after the
research cycle, say so explicitly.
