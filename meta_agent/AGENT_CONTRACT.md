# AutoVQE Meta-Agent Contract

This is an example-free operating summary for the generated Codex workspace.
`docs/agent_protocol.md` in the trusted source repository is the normative
implementation contract. The generated client is the only permitted interface
to that implementation.

## Trust and observation boundary

The agent may read only the generated bundle and its `ObservationBundle`. The
observation contains a content-derived problem ID, real-coefficient public
Pauli Hamiltonian, encoding and declared sector metadata, an optional
computational-basis preparation hint, declared backend facts, and mechanically
derived structural counts and support edges. It excludes exact reference
energy/state, private vectors, fixture or model classification, recommendations,
and candidate lists.

Pauli labels use Qiskit's little-endian display convention: the rightmost
character acts on qubit zero. An operation-local Pauli word is paired
left-to-right with its explicit `qubits` list; this is distinct from display
order in a full-width label. The preparation-hint tuple index is the qubit
index.

The agent must not access the trusted source tree, raw input problem, evaluator
run directory, authoritative or mirrored ledger, checkpoint, path registry,
private evaluation context, HMAC key, anchor directory, operator environment,
or bridge internals. All state reads and writes go through the generated
client. One client submission contains exactly one JSON action object.
`client.py validate --action actions/<file>.json` may be used before submission;
it checks only strict JSON, direct-child path, and size rules. A successful
local validation is not controller acceptance or scientific evidence.

## External actions

Unknown action types and unknown fields are rejected. Every identifier must
match `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}` and be unique in the scope required
by the lifecycle.

| Type | Required fields | Optional fields | Preconditions |
| --- | --- | --- | --- |
| `propose_hypothesis` | `hypothesis_id`, nonempty `claim` | `metadata`, ignored `cost` | New hypothesis ID and active-hypothesis capacity |
| `request_probe` | `hypothesis_id`, `probe_id`, `probe` | none | New probe ID; live exact-symmetry hypothesis with no prior fixed probe |
| `submit_candidate` | `candidate_id`, `hypothesis_id`, nonempty `spec` | `metadata`, ignored `cost` | New candidate ID; parent `SUPPORTED`; required enforcement and non-control prediction/falsifier metadata |
| `evaluate_candidate` | `candidate_id`, `evaluation_id`, `stage` | none | New evaluation ID, correct next lifecycle stage, and no prior result for that fixed stage |
| `revise` | `entity`, `source_id`, `new_id`, `replacement`, nonempty `reason` | `metadata`, ignored `cost` | Revisable source and capacity; promotable candidate revisions require a fresh prediction/falsifier |
| `retire` | `entity`, `entity_id`, nonempty `reason` | ignored `cost` | Existing nonterminal entity; no live child blocks a hypothesis retirement |
| `commit` | `candidate_id`, nonempty `evidence_ids`, `comparison` | `metadata`, ignored `cost` | Candidate is `PROMOTED`, has a preregistered prediction/falsifier, and cites its passed promotion plus a comparison basis |
| `close_negative` | nonempty `reason`, nonempty `evidence_ids` | `metadata`, ignored `cost` | Every hypothesis/candidate is `REVISED` or `RETIRED`; cited IDs exist and include a substantive probe or evaluation |

For `revise`, `entity` is `hypothesis` or `candidate`. A hypothesis replacement
is a complete new claim; a candidate replacement is a complete new
`AnsatzSpec`. IDs are immutable, so every replacement requires a new ID.
Candidate revision metadata is recorded before the replacement is evaluated;
it must contain a new nonempty `prediction` or `falsifier` for every non-control
replacement. The controller preserves the parent-required enforcement value.
`record_probe` and `record_evaluation` are evaluator-owned and cannot be
submitted. Omit `cost`; the controller replaces any supplied value.

For `commit`, every `evidence_ids` entry must name an existing probe or
evaluation and the list must include the selected candidate's passed promotion
evaluation. The immutable candidate metadata recorded before evaluation must
contain a nonempty `prediction` or `falsifier`. `comparison` has one of two
closed forms: `evaluated_competitor` contains exactly `mode`, a different
`candidate_id`, and `evidence_ids` naming at least one of that candidate's
smoke/promotion evaluations; `documented_non_dominance` contains exactly
`mode`, a nonempty `reason`, and `evidence_ids` including the selected
candidate's promotion. Comparison IDs must also occur in the top-level commit
evidence list.

The controller caps a serialized external action at 1,000,000 bytes, the
ledger at 200 events, live hypotheses at three, and live candidates under one
hypothesis at two. Nonterminal actions must leave one ledger slot for a final
`commit` or `close_negative` event.

## Claim schemas and evidence

The accepted claim kinds are closed:

- `exact_pauli_symmetry` has exactly `kind` and `generator`. It remains
  `PROPOSED` until a `normalized_commutator` probe containing the identical
  generator recipe is recorded. A residual at most `1e-10` supports it; a
  larger residual records refutation and leaves it `PROBED`.
- `ansatz_structure` has exactly `kind` and a nonempty string `family`. The
  controller immediately records a zero-cost admission marker with
  `algebraic_certificate: false` and moves it to `SUPPORTED`.
- `null_control` has exactly `kind`. It receives the same non-algebraic
  admission status. Its candidate is diagnostic and can never pass promotion.

An exact generator recipe uses one of three closed forms:

- `pauli_sum` has exactly `type` and `terms`. It contains 1–256 unique,
  full-register Pauli labels. Each term has exactly `pauli` and optional
  `coeff`; coefficients are finite real JSON numbers with absolute value at
  most `1e6`.
- `global_pauli_sum` has exactly `type`, `pauli`, and optional `selector`.
- `orbit_pauli_sum` has exactly `type`, `seed`, and optional `selector`.

The two generated-sum recipes accept one active single-site Pauli symbol and
only the all-sites selector, generate no more than 256 terms, and use unit
coefficients. Every generator must be finite and Hermitian, have active norm
at least `1e-8`, and be neither zero/identity-only nor a trivial copy of the
Hamiltonian. The commutator residual is scale invariant. Candidate audit also
normalizes reference variance by squared active generator norm.

No external approximate-symmetry, group, point-group, permutation, fermionic,
or reference-moment probe is currently available.

## Typed ansatz IR

`AnsatzSpec` is strict JSON. It contains IR `version` 1, a nonempty `name`,
`num_qubits`, unique declared `parameters`, optional trusted `reference`, and
either ordered `layers` or the top-level `operations` shorthand, but not both.
A layer contains only optional `name` and `operations`. An operation contains
only `macro`, unique in-range `qubits`, `parameters`, and `options`.

Each declared-parameter entry is either its name string or an object containing
exactly the `name` string. A reference object contains only `macro` and
`qubits`. Every layer and the complete candidate must contain an active
operation after parsing; empty padding layers are invalid. An operation's
`parameters` object maps the macro's parameter name to one angle expression.
All three variational macros below require exactly the parameter name `angle`.

An angle is an affine expression over declared scalar parameters. The accepted
representations are a single parameter with optional coefficient, a literal,
or `terms` plus optional `constant`. Terms within one expression cannot repeat
a parameter and cannot use a zero coefficient. Compilation rejects undeclared
or unused parameters, constant-only variational operations, nonzero offsets,
opaque values, unknown fields, unknown macros, invalid support, and duplicate
qubits. Controller audit permits trainable coefficients only from
`{-2, -1, -0.5, 0.5, 1, 2}` and rejects numeric option literals.

Every variational macro follows `U(angle) = exp(-i * angle * G)` and is the
identity at zero:

| Macro | Kind and arity | Exact trusted generator/options |
| --- | --- | --- |
| `PauliRotation` | Variational operation on one or more distinct qubits | Required parameter `angle`; `G` is the active local Pauli word in the sole required string option `pauli`; the word width equals support width and contains no identity factors |
| `XYExchange` | Two-qubit variational operation | `G = XX + YY`; required parameter is `angle`; no options |
| `IsotropicExchange` | Two-qubit variational operation | `G = XX + YY + ZZ`; required parameter is `angle`; no options |
| `X` | Reference-only repeated one-qubit preparation | Applies the trusted operation to every listed reference qubit; it is forbidden in variational layers |

`XYExchange` and `IsotropicExchange` are conditional conservation macros, not
generic structure shortcuts. Audit permits either only when the candidate's
parent is a controller-`SUPPORTED` `exact_pauli_symmetry` hypothesis. The exact
probe is necessary but not sufficient: every instantiated operation must still
pass the existing commutator audit. A registry entry, suggestive macro name,
Hamiltonian pattern, or auto-admitted `ansatz_structure`/`null_control` claim is
not evidence authorizing these macros.

The public backend basis affects trusted lowering and accounting only. It
cannot add logical macros. Candidate-supplied Python, Qiskit instructions,
matrices, callables, optimizer settings, seeds, initial values, or custom gate
registrations are forbidden.

Candidate metadata enforcement is mandatory and claim-dependent:

- exact symmetry requires `preserve`;
- structure requires `unconstrained`;
- control requires `diagnostic`.

Every non-control candidate submission must also preregister a concrete
nonempty `prediction` or `falsifier` in this metadata. A revision must supply a
fresh one for the replacement. Because candidate metadata is immutable, it
cannot be added after seeing that candidate's evaluator result.

Candidate identity is semantic, not cosmetic. The evaluator ignores ansatz and
layer display names, layer-only grouping, parameter spelling, and parameter
declaration order when it computes `candidate_hash`. Parameter incidence and
the ordered physical operation sequence are alpha-normalized. Submission or
revision rejects a family whose semantic hash already exists anywhere in the
campaign, including a retired branch. Rename-only or re-layer-only resubmission
therefore cannot obtain a fresh optimizer/resource-binding trial.

Audit requires at least one logical operation and one trainable parameter. When
the public occupation hint contains occupied positions, the candidate reference
must apply the reference macro on exactly those positions. Otherwise the
candidate reference must be absent, including for an all-zero hint. A rotation
above locality two must equal a declared full-width Hamiltonian term. Under
`preserve`, every operation generator must commute with the claimed charge
within `1e-10`, and the zero-parameter reference must have normalized charge
variance at most `1e-10`.

## Evaluations, costs, and promotion

Stages run in the order `audit`, `smoke`, `promotion`.

| Request or stage | Evaluator-owned behavior | Cost units |
| --- | --- | ---: |
| Hypothesis proposal | Validate and record the claim | 0.10 |
| Exact commutator probe | Compute and record the probe receipt | 0.10 |
| Candidate submission | Validate lifecycle/metadata and record the proposed IR mapping; compilation is deferred to audit | 0.10 |
| Revision | Link and record a complete replacement | 0.10 |
| Retirement, commit, or negative close | Validate and record the transition | 0.00 |
| Audit | Trusted compile, global limits, literal policy, reference and required invariant checks | 0.25 |
| Smoke | COBYLA, at most 32 objective calls, one restart, seed 7 | 2.00 |
| Promotion | COBYLA, at most 96 objective calls, three restarts, seed 997 | 6.00 |

The compiler-derived global caps are 256 logical operations, 128 unique
trainable parameters, 4096 IR nodes, and 64 occurrences of any one parameter.
Parameter sharing does not erase occurrence counts.

Smoke and promotion use a fixed, candidate-independent canonical target with
basis `{rz, sx, x, cx}`, all-to-all connectivity, and transpiler optimization
level 1. The evaluator measures the symbolic template and three
deterministic candidate-specific generic bindings. Eligibility uses the
metric-wise maximum of template and generic-worst results and requires no more
than 512 two-qubit gates, 2048 total gates, and depth 1024. Declared-backend
and final-binding metrics remain evidence but do not replace this canonical
conservative policy.

Smoke and promotion must improve over evaluator-computed zero-parameter
baseline energy by at least `max(1e-6, 1e-6 * abs(baseline_energy))`.
Promotion additionally requires a prior passed smoke and promotion energy no
worse than the best passed smoke energy plus `5e-4`. A passed promotion is not
proof of an exact reference tolerance, a Pareto win, hidden generalization, or
global optimality.

Energy, baseline, parameter names and occurrences, objective calls, optimizer,
seed, candidate hash, compile audit, gate counts, depth, resource eligibility,
probe result, and pass/fail status are evaluator receipts. Agent metadata is
non-authoritative.

## Lifecycle and terminal state

An exact hypothesis starts `PROPOSED`; its single fixed commutator probe moves
it directly to `SUPPORTED` on pass or to `PROBED` on refutation. It does not
advance from `PROBED` to `SUPPORTED`. Non-algebraic claims move directly to
`SUPPORTED` through their admission marker. `REVISED` and `RETIRED` are branch
terminal states. Repeating a fixed probe or candidate evaluation stage under a
new ID is rejected because it is not independent evidence.

Candidates follow `CANDIDATE -> AUDITED -> SMOKE -> PROMOTED`, with failed
evaluation moving the candidate to `RETIRED` and explicit revision creating a
new linked candidate at `CANDIDATE`. A live child candidate must be resolved
before its parent hypothesis can be revised or retired. A `PROMOTED` candidate
cannot be revised or retired; it must remain live or be committed, so a passed
promotion cannot be erased to manufacture a negative close.

`commit` of a `PROMOTED` candidate and `close_negative` are the only terminal
actions; either prevents all future events. Commit also enforces preregistered
prediction/falsifier and comparison evidence. Negative close is accepted only
after every hypothesis and candidate is `REVISED` or `RETIRED`, all cited IDs
resolve to existing evaluator-owned records, and at least one citation is a
substantive probe or evaluation rather than only an automatic non-algebraic
admission. Every investigated hypothesis must be covered by either its cited
refuting probe or a cited evaluation of one of its candidates. Retirement
alone is not a terminal decision.

## Required reporting

Report problem ID and observation hash; claim and probe/admission evidence;
candidate ID and evaluator hash; audit, smoke, and promotion IDs; evaluator
energy and call traces; declared-backend and canonical resource views; unique
parameters and occurrence counts; enforcement and invariant receipts;
security mode; budget spent and remaining; ledger count and tip; terminal
decision; and limitations. Never relabel promotion as hidden evaluation or an
unimplemented scientific guarantee.
