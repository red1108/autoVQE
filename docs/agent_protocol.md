# Lean agent protocol

This is the complete external protocol for the lean AutoVQE research loop.
JSON objects are strict: omitted required fields and all unlisted fields fail.
The controller snapshots the input at `init`; the original Hamiltonian is not
consulted again during that run.

## Commands and ownership

```bash
uv run autovqe inspect --problem user_problem/hamiltonian.json --json
uv run autovqe research init --problem user_problem/hamiltonian.json \
  --run-dir .autovqe-runtime/research --budget 100
uv run autovqe research step --run-dir .autovqe-runtime/research \
  --action .autovqe-runtime/actions/001.json
uv run autovqe research status --run-dir .autovqe-runtime/research [--full]
uv run autovqe research result --run-dir .autovqe-runtime/research
```

The agent may write action files only. The CLI owns `run.json`, `problem.json`,
and append-only `events.jsonl`. Do not read, edit, or synthesize these files;
read state through `status`. It hides optimized bindings, while `result`
exposes the accepted binding only after a terminal decision.

## Problem document

Required fields are `name` and nonempty `pauli_terms`. Optional fields are
`basis_gates`, `coupling_map`, `initial_state_hint`, `source_note`, and
`symmetry`. Every term is exactly:

```json
{"pauli": "IXYZ", "coeff": 0.25}
```

All labels have equal width; the rightmost letter is qubit 0. Coefficients are
finite real numbers. Duplicate labels are combined and zero sums removed.
`initial_state_hint` is a width-sized integer bit list. `symmetry` accepts only:
`mapping`, `basis`, `orbital_order`, `spin_order`, `spin_orbitals`,
`active_orbitals`, `active_electrons`, `particle_number`, `magnetization`,
`spin_projection`, `total_spin`, and `parity`.

## Hypothesis claims

`propose_hypothesis` accepts exactly one of these claim shapes:

```text
{"kind": "ansatz_structure", "family": "boundary-aware HVA"}
{"kind": "null_control"}
{"kind": "exact_pauli_symmetry", "generator": {"type": "global_pauli_sum", "pauli": "Z"}}
```

An explicit symmetry generator uses:

```json
{
  "type": "pauli_sum",
  "terms": [
    {"pauli": "IZ", "coeff": 1.0},
    {"pauli": "ZI", "coeff": 1.0}
  ]
}
```

`global_pauli_sum` permits `pauli` equal to `X`, `Y`, or `Z` and optional
`"selector": "all_sites"`. `pauli_sum` has at most 256 distinct full-width
labels and optional finite coefficients bounded in magnitude by `1e6`. A zero,
identity-only, or near-copy of the Hamiltonian is rejected.

An exact-symmetry hypothesis produces evidence only. It never owns a
candidate: every non-control candidate belongs to an `ansatz_structure` and
cites supported symmetry through `symmetry_evidence_ids`.

## External action schemas

Optional `metadata` is an object containing only nonempty string `rationale`,
`prediction`, and/or `falsifier`. Every non-control candidate must include a
`prediction` or `falsifier` at submission or revision.

Propose a hypothesis:

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "hva-boundary",
  "claim": {"kind": "ansatz_structure", "family": "boundary-aware HVA"},
  "metadata": {"prediction": "outperforms a product-rotation branch"}
}
```

For an exact symmetry claim only, request its evaluator-owned commutator probe:

```json
{"type": "request_probe", "hypothesis_id": "u1-z"}
```

Submit a candidate (`symmetry_evidence_ids` and `metadata` are optional fields):

```json
{
  "type": "submit_candidate",
  "candidate_id": "hva-p1",
  "hypothesis_id": "hva-boundary",
  "spec": {"version": 1, "name": "hva-p1", "num_qubits": 4, "parameters": ["theta"], "operations": [{"macro": "XYExchange", "qubits": [0, 1], "parameters": {"angle": {"parameter": "theta"}}, "options": {}}]},
  "metadata": {"falsifier": "does not improve its zero-angle baseline at smoke"},
  "symmetry_evidence_ids": ["probe:u1-z"]
}
```

Advance exactly one evaluator-owned stage:

```json
{"type": "evaluate_candidate", "candidate_id": "hva-p1"}
```

Revise a hypothesis after all of its live candidates are terminal:

```json
{
  "type": "revise",
  "entity": "hypothesis",
  "source_id": "old-h",
  "new_id": "new-h",
  "replacement": {"kind": "ansatz_structure", "family": "revised family"},
  "reason": "the observed branch falsified its ordering",
  "metadata": {"prediction": "the revised ordering activates the objective"}
}
```

Revise a candidate (`symmetry_evidence_ids` is optional):

```json
{
  "type": "revise",
  "entity": "candidate",
  "source_id": "hva-p1",
  "new_id": "hva-p2",
  "replacement": {"version": 1, "name": "hva-p2", "num_qubits": 4, "parameters": ["theta"], "operations": [{"macro": "XYExchange", "qubits": [0, 1], "parameters": {"angle": {"parameter": "theta"}}, "options": {}}, {"macro": "XYExchange", "qubits": [2, 3], "parameters": {"angle": {"parameter": "theta"}}, "options": {}}]},
  "reason": "smoke evidence motivates one additional layer",
  "metadata": {"falsifier": "the extra layer gives no material improvement"},
  "symmetry_evidence_ids": ["probe:u1-z"]
}
```

A generic `XX+YY` or `XX+YY+ZZ` construction may be revised to its native
exchange macro without inventing a new structural identity, but only when the
revision adds newly supported symmetry evidence required by that macro.

Retire an entity:

```json
{"type": "retire", "entity": "candidate", "entity_id": "hva-p1", "reason": "falsified by smoke"}
```

Commit a promoted, non-dominated candidate:

```json
{"type": "commit", "candidate_id": "hva-p2"}
```

Or close a fully disposed search:

```json
{"type": "close_negative", "reason": "independent objective-active structures failed"}
```

IDs match `[A-Za-z0-9][A-Za-z0-9_.:-]*`; new IDs are limited to 96 characters.
Agent actions `record_probe` and `record_evaluation` are forbidden.

## AnsatzSpec

The canonical candidate shape is:

```json
{
  "version": 1,
  "name": "shared-edge-layer",
  "num_qubits": 4,
  "parameters": [{"name": "theta"}],
  "operations": [
    {
      "macro": "PauliRotation",
      "qubits": [0, 1],
      "parameters": {
        "angle": {
          "constant": 0.0,
          "terms": [{"parameter": "theta", "coefficient": 1.0}]
        }
      },
      "options": {"pauli": "XY"}
    }
  ]
}
```

Parameter declarations may use the string shorthand `"theta"`. An angle may
use `{"parameter":"theta","coefficient":1.0}`. Compilation requires exactly
one declared parameter term and zero constant; the public research audit also
restricts its coefficient to `-2, -1, -0.5, 0.5, 1, 2`. Every declared
parameter must be used; sharing is allowed, with at most 64 occurrences per
parameter. The evaluator normalizes each independent parameter direction before
optimization and maps the accepted binding back to the submitted coordinates,
so a global sign or scale cannot change the optimizer trajectory.

The operation allowlist is exactly:

| Macro | Generator | Shape |
|---|---|---|
| `PauliRotation` | `exp(-i angle P)` | active `XYZ` word matching `qubits` |
| `XYExchange` | `exp[-i angle(XX+YY)]` | two qubits, no options |
| `IsotropicExchange` | `exp[-i angle(XX+YY+ZZ)]` | two qubits, no options |

Every operation is identity at parameter origin. A `PauliRotation` above
locality two must exactly match a term in the input Hamiltonian. Within its
local Pauli word, letters pair with `qubits` in the listed order. No custom
gates, fixed offsets, multi-parameter angles, unused parameters, or extra
AnsatzSpec fields are accepted.

`XYExchange` and `IsotropicExchange` require cited supported symmetry probes.
At audit, every operation must preserve every cited charge, the initial state
must have a definite sector for each charge, and each conservation gate must
touch a relevant, non-spectator part of at least one cited charge.

## Lifecycle and decisions

An exact-symmetry hypothesis starts `PROPOSED` and must be probed; structure and
null-control hypotheses start `READY`. A candidate advances only by repeated
`evaluate_candidate` actions:

```text
CANDIDATE -> audit -> AUDITED -> smoke -> SMOKE -> promotion -> PROMOTED
```

A failed stage retires the candidate. Audit derives circuit validity and
resources. Smoke uses 32 evaluations and one restart; promotion uses 96 and
three restarts. Both must improve the candidate's zero-angle baseline by at
least `max(1e-6, 1e-6*abs(baseline))`. Promotion must reproduce smoke within
`5e-4`. A typed empty `null_control` may pass smoke but cannot be promoted.

Before promoting a target, a non-control candidate from a different primary
structure root must have passed smoke. Every hypothesis revision remains in
its original lineage even when its claim kind changes; kind-hopping cannot
manufacture a second root. Normalized duplicate family claims are rejected. A
`null_control` never satisfies this comparator requirement. The comparator is
then promoted under the same protocol. If that promotion fails, the loop stays
open so another comparator branch can be built. Commit requires at least one
passed different-root promotion and checks the target against every passed
promotion, including stronger variants from its own root. A target is blocked
when another promotion has energy lower by more than `5e-4`. At energies tied
within `5e-4`, it is also blocked when the target is componentwise no better in
resources and strictly worse in at least one.
Resources include parameters and conservative two-qubit count, total gate
count, and depth. This is the fair comparator rule.

`negative_close` requires all branches retired or revised, plus either one
objective-active promotion failure or objective-active smoke failures from two
independent structure roots. Phase-flat or otherwise inactive failures do not
establish negative closure.

The evaluator alone writes probe residuals, energies, optimizer traces,
optimized bindings, parameter counts, gate counts, and depth. Do not include
these as candidate claims. `research result` deterministically replays the
accepted promotion before exposing its binding and is the sole reporting
source.

The budget maximum is 100. Costs are: propose `0.1`, submit `0.1`, audit `0.25`,
smoke `2`, promotion `6`, revise `0.1`, retire/terminal `0`; symmetry probes are
charged by algebraic size. Runs allow 200 events, three active hypotheses, two
active candidates per hypothesis, 256 operations, 128 parameters, 512
conservative two-qubit gates, 2048 total gates, depth 1024, and 1 MB per action.
