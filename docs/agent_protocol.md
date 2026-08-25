# Agent protocol

This is the complete external protocol for AutoVQE's research loop. Action
JSON is strict: every unlisted field is rejected. The controller snapshots the
problem at `init`; the original file is not consulted again during that run.

## Ownership

The README gives the command sequence. The agent writes action JSON only under
`.autovqe-runtime/actions/`. The CLI owns the run state and evidence; read them
through `research status`, and request `research result` only after a terminal
decision. The result command is the sole source for an optimized binding.

## Hypotheses and symmetry probes

Every circuit belongs to a falsifiable structural hypothesis. Propose one with
a prediction or falsifier:

```json
{
  "type": "propose_hypothesis",
  "hypothesis_id": "boundary-hva",
  "family": "boundary-aware HVA",
  "prediction": "beats a disjoint-matching branch at smoke"
}
```

After its live candidates are terminal, revise a structural assumption with a
new family and a new prediction or falsifier:

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

A revision stays in its original structural lineage. Family names are compared
after normalization, so cosmetic duplicates do not create a comparator.

Symmetry is independent evaluator evidence, not a candidate family. Request a
normalized-commutator measurement directly:

```json
{
  "type": "request_symmetry_probe",
  "probe_id": "global-z",
  "generator": {"type": "global_pauli_sum", "pauli": "Z"}
}
```

An explicit generator instead uses `{"type":"pauli_sum","terms":[...]}`,
where each term has a full-width `pauli` and optional finite `coeff`.
`global_pauli_sum` permits `X`, `Y`, or `Z` and optional
`"selector":"all_sites"`. A generator may contain at most 256 distinct terms;
coefficients have magnitude at most `1e6`; zero, identity-only, or
near-Hamiltonian-copy generators are rejected.

## AnsatzSpec and candidate submission

An ansatz has no separate parameter declarations or nested angle expressions.
Its only top-level fields are `version`, `num_qubits`, and `operations`. Each
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

`version` defaults to 1, `operations` defaults to an empty list, and `scale`
defaults to 1. Research candidates must contain at least one operation and one
parameter. Parameter names match `[A-Za-z_][A-Za-z0-9_.-]{0,127}`; reuse means
intentional sharing. The only scales are `-2, -1, -0.5, 0.5, 1, 2`.

The gate allowlist is exactly:

| Gate | Generator | Additional fields |
|---|---|---|
| `PauliRotation` | `exp(-i angle P)` | `pauli` is required and matches `qubits` |
| `XYExchange` | `exp[-i angle(XX+YY)]` | exactly two qubits; no `pauli` |
| `IsotropicExchange` | `exp[-i angle(XX+YY+ZZ)]` | exactly two qubits; no `pauli` |

Every operation is identity at parameter origin. A `PauliRotation` above
locality two must exactly match a term in the input Hamiltonian. Within a local
Pauli word, letters pair with qubits in listed order. Custom gates, fixed
offsets, extra fields, redundant parameter directions, and semantically
duplicate candidates are rejected. The evaluator normalizes independent
parameter directions before optimization, so a global allowed sign or scale
does not buy another optimizer trajectory.

Submit a candidate under a structural hypothesis. Cite only supported symmetry
probe IDs:

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

Submission automatically performs the evaluator-owned audit and returns its
validity, resources, and violations.

If symmetry evidence is cited, every operation must preserve every cited
charge and the initial state must have a definite sector. `XYExchange` and
`IsotropicExchange` require supported evidence, and each such gate must touch a
relevant, non-spectator part of at least one cited charge.

## Evaluation and decisions

After a passed audit, the same action advances one evaluator stage at a time:

```json
{"type": "evaluate_candidate", "candidate_id": "boundary-hva-p1"}
```

The lifecycle is `CANDIDATE -> audit -> AUDITED -> smoke -> SMOKE -> promotion
-> PROMOTED`. A failed stage retires the candidate. Smoke uses 32 evaluations
and one restart; promotion uses 96 evaluations and three restarts. Both must
improve the candidate's zero-angle baseline by at least
`max(1e-6, 1e-6*abs(baseline))`; promotion must reproduce smoke within `5e-4`.

Before promotion, a candidate from a different primary structural root must
have passed smoke. That comparator is promoted under the same protocol before
commit. Commit compares the target with every passed promotion, including
variants in its own lineage. A lower energy by more than `5e-4` dominates; at
an energy tie, componentwise no-worse resources with one strict improvement
also dominate. Resources are parameter count, conservative two-qubit count,
total gate count, and depth.

Retire exhausted branches with explicit reasons:

```json
{"type": "retire_candidate", "candidate_id": "boundary-hva-p1", "reason": "falsified by smoke"}
```

```json
{"type": "retire_hypothesis", "hypothesis_id": "boundary-hva", "reason": "branch exhausted"}
```

A promoted candidate may be retired only after another passed promotion
dominates it; the controller records that evidence. Otherwise it must be
committed or compared with a stronger branch.

Terminate positively or negatively with:

```json
{"type": "commit", "candidate_id": "boundary-hva-p1"}
```

```json
{"type": "close_negative", "reason": "independent objective-active branches failed"}
```

Negative closure requires every candidate to be retired and every hypothesis
to be retired or revised, plus either an objective-active promotion failure or
objective-active smoke failures from two independent structural roots. Flat
phase-only failures do not establish negative closure.

The evaluator alone writes probe residuals, energies, optimizer traces,
optimized bindings, parameter counts, gate counts, and depth. Candidate actions
must not provide those values. External `record_symmetry_probe` and
`record_evaluation` actions are forbidden. IDs match
`[A-Za-z0-9][A-Za-z0-9_.:-]*` and are limited to 96 characters.

## Limits and costs

The maximum budget is 100. Propose and revise cost `0.1`; submit plus automatic
audit costs `0.35`; smoke costs `2`; promotion costs `6`; retirement and
terminal decisions cost `0`. Symmetry probe cost is derived from algebraic
work. A run allows 200 events, three active hypotheses, two active candidates
per hypothesis, 256 operations, 128 parameters, 64 uses per parameter, 512
conservative two-qubit gates, 2048 total gates, depth 1024, and 1 MB per action.
