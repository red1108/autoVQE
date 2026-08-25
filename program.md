# AutoVQE research program

Turn `user_problem/hamiltonian.json` into an evidence-backed variational
ansatz through AutoVQE's closed research loop. Read the raw Hamiltonian before
choosing a model. The agent proposes hypotheses, probes, and typed circuits;
the evaluator alone supplies energies, optimized values, and resource counts.

## Ownership and workflow

Treat the input as immutable. Manually write only action JSON under
`.autovqe-runtime/actions/`, apply it with `uv run autovqe research step`, and
read controller state with `uv run autovqe research status`. The controller
snapshots the problem at `research init` and does not reread the original input
during that run. The CLI owns run history and evidence. Request
`uv run autovqe research result` only after a terminal decision; it is the sole
source for the optimized parameter binding.

Action JSON is strict: every unlisted field is rejected. Do not call the
evaluator directly, run another eigensolver or optimizer, edit run history, or
look up a reference solution. Report a harness defect instead of patching the
source during a discovery run.

## Read the Hamiltonian

Qiskit little-endian ordering is used: the rightmost Pauli letter acts on
qubit 0. Separate the identity coefficient from the active Hamiltonian; the
constant shifts every energy but does not select an ansatz. Inspect the raw
terms and use `uv run autovqe inspect --json` for mechanical graph facts.
Summarize:

- active term count, coefficient magnitudes and signs, and locality;
- X-, Y-, and Z-bearing terms and repeated coefficient classes;
- connected components, degrees, boundaries, hubs, and repeated graph motifs;
- higher-locality supports that may represent excitation channels;
- the initial occupation and backend connectivity.

Problem names and declared symmetry metadata are context, not evidence.

## Build falsifiable structure branches

Every circuit belongs to a primary structural hypothesis with a prediction or
falsifier:

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "boundary-hva",
  "family": "boundary-aware HVA",
  "prediction": "beats a disjoint-matching branch at smoke"
}
```

Start with the smallest circuit that tests a physical claim. Bring at least
two genuinely different primary structures through smoke; different depths,
parameter names, symmetry probes, or orderings inside one lineage do not make
a fair comparator.

Useful structures include:

- **Matchings and dimers:** test dominant exchange-like bonds in parallel,
  then add a shifted covering or boundary/bulk split only if evidence supports
  it.
- **Boundary-aware layers:** distinguish endpoints, bulk sites, hubs, or leaves
  before assigning a separate parameter to every operation.
- **Hamiltonian variational layers:** order selected noncommuting cost and
  mixer groups, ranked by coefficient, support, connectivity, and action on
  the initial occupation; one shallow layer should earn more terms or depth.
- **Phase-conditioned excitations:** an odd-Y channel may be objective-flat
  because of phase alignment. Test a preregistered ordering with a trainable
  one-local noncommuting phase rotation. Placement before versus after the
  excitation is part of the hypothesis. Never use a fitted fixed angle.

After all live candidates under a hypothesis are terminal, revise one
structural assumption:

```json
{
  "type": "revise_hypothesis",
  "source_id": "boundary-hva",
  "new_id": "boundary-hva-ordered",
  "family": "boundary-aware ordered HVA",
  "reason": "the first ordering was objective-active but unfavorable",
  "falsifier": "the reordered layer gives no smoke improvement"
}
```

A revision remains in its original lineage. Normalized duplicate family names
and semantically equivalent candidates are rejected.

## Discover symmetry instead of assuming it

Request a normalized-commutator probe for a concrete generator:

```json
{
  "type": "request_symmetry_probe",
  "probe_id": "global-z",
  "generator": {"type": "global_pauli_sum", "pauli": "Z"}
}
```

`global_pauli_sum` permits X, Y, or Z and the optional
`"selector":"all_sites"`. An explicit generator
uses `{"type":"pauli_sum","terms":[...]}`, where each term contains a
full-width `pauli` and optional finite `coeff`. Generators are bounded to 256
distinct terms and coefficient magnitude `1e6`; zero, identity-only, and
near-Hamiltonian copies are rejected.

For U(1)-like particle-number or magnetization conservation, test the relevant
global Z sum or an explicit weighted charge. For SU(2)-like isotropy, probe
global X, Y, and Z separately: one commuting component proves only that
component.

Symmetry is a search constraint, not the ansatz. Continue using coefficients,
graph motifs, locality, occupation, and connectivity to select operations and
ordering. Cite only supported probe IDs. Every candidate operation must
preserve every cited charge, the initial state must occupy a definite sector,
and every conservation-specific gate must touch a relevant, non-spectator
part of a cited charge.

## Express circuits as `AnsatzSpec`

An ansatz contains only `version`, `num_qubits`, and `operations`. Each
operation is driven by one named parameter:

```json
{
  "version": 1,
  "num_qubits": 4,
  "operations": [
    {
      "gate": "PauliRotation",
      "qubits": [0, 1],
      "parameter": "theta",
      "scale": 0.5,
      "pauli": "XY"
    },
    {
      "gate": "XYExchange",
      "qubits": [2, 3],
      "parameter": "phi"
    }
  ]
}
```

`version` defaults to 1, `operations` to an empty list, and `scale` to 1.
Research candidates need at least one operation and parameter. Parameter names
match `[A-Za-z_][A-Za-z0-9_.-]{0,127}`. Reusing a name means intentional
sharing. Allowed scales are `-2, -1, -0.5, 0.5, 1, 2`.

The gate allowlist is exactly:

| Gate | Generator | Constraint |
|---|---|---|
| `PauliRotation` | `exp(-i angle P)` | `pauli` matches the listed qubits |
| `XYExchange` | `exp[-i angle(XX+YY)]` | two qubits, no `pauli` |
| `IsotropicExchange` | `exp[-i angle(XX+YY+ZZ)]` | two qubits, no `pauli` |

Every operation is identity at parameter origin. A Pauli rotation above
locality two must exactly match an input Hamiltonian term. Custom gates, fixed
offsets, extra fields, redundant parameter directions, and semantic duplicates
are rejected. The evaluator canonicalizes parameter directions, so an allowed
global sign or scale does not create another optimizer trajectory.

Within an operation's local Pauli word, letters pair with `qubits` in listed
order; this is separate from the little-endian ordering of full-width labels.

Parameter sharing is a physical claim. Good initial classes include
translation-equivalent bulk edges, boundary versus bulk sites, one matching or
coefficient class, and repeated excitation channels with a justified relative
scale. Split one meaningful class at a time only after evidence shows sharing
is too rigid.

## Submit and evaluate candidates

Submit a typed candidate under its structural hypothesis and cite symmetry
evidence separately:

```json
{
  "type": "submit_candidate",
  "candidate_id": "boundary-hva-p1",
  "hypothesis_id": "boundary-hva",
  "spec": {
    "version": 1,
    "num_qubits": 4,
    "operations": [
      {
        "gate": "XYExchange",
        "qubits": [0, 1],
        "parameter": "theta"
      }
    ]
  },
  "symmetry_evidence_ids": ["global-z"]
}
```

Submission automatically runs the fixed evaluator audit. It validates the
circuit and derives parameter use and conservative resources. Candidate
actions must never provide energy, optimized values, parameter counts, gate
counts, depth, custom operations, or hidden numeric answers.
The evaluator alone writes probe residuals, optimizer traces, optimized
bindings, parameter counts, gate counts, and depth.

Advance one fixed evaluator stage at a time:

```json
{"type": "evaluate_candidate", "candidate_id": "boundary-hva-p1"}
```

The lifecycle is `audit -> AUDITED -> smoke -> SMOKE -> promotion -> PROMOTED`.
A failed stage retires the candidate. Smoke uses 32 evaluations and one
restart; promotion uses 96 evaluations and three restarts. Both must improve
the zero-angle baseline by at least `max(1e-6, 1e-6*abs(baseline))`, and
promotion must reproduce smoke within `5e-4`.

Promotion requires a candidate from a different primary structural root to
have already passed smoke. Before commit, promote that comparator under the
same protocol. Commit compares the target against every passed promotion,
including variants in its own lineage. A lower energy by more than `5e-4`
dominates. At an energy tie, componentwise no-worse parameter count,
conservative two-qubit count, total gate count, and depth, with one strict
improvement, dominates.

## Retire, revise, or terminate

Retire exhausted branches with evidence-based reasons:

```json
{"type": "retire_candidate", "candidate_id": "boundary-hva-p1", "reason": "falsified by smoke"}
```

```json
{"type": "retire_hypothesis", "hypothesis_id": "boundary-hva", "reason": "branch exhausted"}
```

A promoted candidate may be retired only after another passed promotion
dominates it. Otherwise it must be committed or compared with a stronger
branch.

Terminate with exactly one of:

```json
{"type": "commit", "candidate_id": "boundary-hva-p1"}
```

```json
{"type": "close_negative", "reason": "independent objective-active branches failed"}
```

Negative closure requires all branches to be terminal plus either an
objective-active promotion failure or objective-active smoke failures from two
independent roots. Flat phase-only failures do not establish negative closure.
A positive decision proves only the recorded local promotion rule, not exact
ground-state accuracy or generalization.

Keep failed and revised branches: they prevent cycling and preserve evidence.
Commit only a non-dominated result. Report optimized parameters only from
`research result`, and state when no independent reference score was supplied.

## Limits and costs

The maximum budget is 100. Propose and revise cost `0.1`; submit and automatic
audit cost `0.35`; smoke costs `2`; promotion costs `6`; retirement and
terminal decisions cost `0`. Symmetry probe cost follows algebraic work.

A run allows 200 events, three active hypotheses, two active candidates per
hypothesis, 256 operations, 128 parameters, 64 uses per parameter, 512
conservative two-qubit gates, 2048 total gates, depth 1024, and 1 MB per action.
External `record_symmetry_probe` and `record_evaluation` actions are forbidden.
New agent-chosen IDs match `[A-Za-z0-9][A-Za-z0-9_.:-]*` and contain at most 96
characters.

## Failure modes

- Trusting a problem label instead of its Pauli terms.
- Adding every term before a small motif earns more expressivity.
- Calling an objective-flat branch a structural failure.
- Claiming SU(2) from one U(1) commutator.
- Using conservation gates without supported, relevant sector evidence.
- Giving every gate its own parameter without testing sharing.
- Comparing only variants of one primary structure.
- Reporting candidate-authored energies, counts, or optimized values.
- Treating local promotion as an exact ground-state result.
