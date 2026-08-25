# Ansatz Playbook

Use this playbook to turn Hamiltonian structure into falsifiable experiments.
A pattern suggests a branch; only controller-produced probes and evaluations
support or refute it.

## Read the problem mechanically

Start from:

- Pauli support, locality, signs, and coefficient scales;
- repeated edges, motifs, commuting groups, and graph structure;
- declared encoding and sector metadata;
- the evaluator-owned computational-basis `initial_state_hint`, if present;
- basis gates and coupling constraints.

Full Hamiltonian and generator labels use Qiskit's display order, so the
rightmost character acts on qubit 0. Within a `PauliRotation`, the local Pauli
word is paired left-to-right with its explicit qubit list. Thus
`qubits: [0, 1]` and `pauli: "XY"` means X on q0 and Y on q1; the full-width
Qiskit label is `YX`.

An encoding or sector declaration is context, not a conservation proof. An
initial-state hint is applied by the evaluator and is not candidate circuit
structure. File names, model labels, expected answers, and previously learned
angles are not evidence.

## Keep the search language small

The accepted variational macros are exactly:

| Macro | Generator | Use |
| --- | --- | --- |
| `PauliRotation` | `P` | `exp(-i angle P)` for an active local Pauli word |
| `XYExchange` | `XX + YY` | two-qubit exchange after exact-symmetry evidence |
| `IsotropicExchange` | `XX + YY + ZZ` | two-qubit isotropic exchange after exact-symmetry evidence |

There is no candidate preparation macro, secondary circuit grouping, custom
gate, matrix, or registration hook. Backend gate names are compilation
targets, not additions to this allowlist.

Every variational macro is the identity at zero. `PauliRotation` above
locality two must match a supplied Hamiltonian term. The exchange gates are
admitted only when the candidate cites a supported exact-symmetry probe, every
concrete operation preserves every cited charge, and each special gate touches
at least `1e-3` of the active norm of one charge. A spectator symmetry or an
epsilon-weighted touching term is not a gate certificate; conditioned
Hamiltonian residual and sector variance must remain exact as well.

This boundary lets the agent exploit measured physical structure without
turning the ansatz into an unrestricted universal circuit or hiding a fixed
numeric answer.

## Write a flat typed candidate

An `AnsatzSpec` contains version, name, qubit count, unique scalar parameters,
a flat ordered `operations` array, and affine angle expressions:

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
        "angle": {
          "terms": [
            {"parameter": "theta", "coefficient": 1.0}
          ],
          "constant": 0.0
        }
      },
      "options": {"pauli": "XX"}
    }
  ]
}
```

The shorter `{"parameter": "theta"}` angle form is equivalent. Each angle
must depend on a declared parameter and vanish at the origin. Parameter sharing
is useful when the hypothesis predicts it, but it does not make repeated gates
free: AutoVQE separately counts unique parameters, occurrences, logical
operations, compiled gates, two-qubit gates, and depth. One parameter may
occur at most 64 times.

The controller derives enforcement from the branch and evidence:

- `preserve` when the candidate cites supported symmetry probe IDs;
- `unconstrained` for a structural family;
- `diagnostic` for a null control.

Candidate metadata may contain only text `prediction`, `falsifier`, and
`rationale` fields. Before submitting a promotable candidate, preregister a concrete `prediction`
or `falsifier`. For example: “the selected edge rotations beat their
zero-angle baseline under smoke while staying below the resource cap.” Do not
put a claimed energy, optimized angle, gate count, optimizer, or initial value
in metadata.

## Treat symmetry as research

The productive sequence is:

```text
observe a pattern
  -> propose an exact generator Q
  -> request the controller's [H,Q] probe
  -> choose operations predicted to preserve Q
  -> audit every operation and the prepared initial sector
  -> optimize only after those checks pass
```

For example:

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

The follow-up action is deliberately minimal:

```json
{"type": "request_probe", "hypothesis_id": "total-z"}
```

The controller reuses the claim's generator and creates the evidence ID. It
computes the bounded normalized commutator once. Residual at most `1e-10`
supports this particular exact Pauli charge.

Support for `[H,Q]=0` still does not prove that a circuit preserves `Q`. Audit
prepares the problem's initial state, checks its charge variance, and measures
the normalized commutator of every instantiated operation generator with `Q`.
An exchange-like name never substitutes for this test.

Keep symmetry evidence composable. A candidate can remain under a structural
hypothesis such as an ordering or graph-coloring rule while listing one or more
supported probe IDs in `symmetry_evidence_ids`. Audit enforces every cited
constraint. This also lets a candidate cite several conserved components
without pretending that symmetry is the entire design hypothesis.

### U(1), SU(2), and other structure

Matched `XX + YY` edges motivate testing a total-Z/U(1)-like charge and, if it
passes, an `XYExchange` circuit. Qubit Hamming-weight conservation does not by
itself establish fermionic particle number, spin, or seniority under an
unspecified encoding.

Matched `XX + YY + ZZ` edges motivate isotropic exchange. A single conserved
component is not evidence for full SU(2). With the current Pauli-sum probes,
test global X, Y, and Z as separate exact claims when those generators are part
of the physical argument. Casimir, representation-sector, and general
non-Abelian certification are not yet implemented.

For chemistry, `initial_state_hint` is not a Hartree–Fock certificate, and
`XYExchange` is not UCC. Fermionic single/double, pair-, spin-, and
seniority-preserving blocks require future typed operations and matching
encoding-aware probes.

Translation, permutation, point-group, lattice-gauge, and local-constraint
patterns can still motivate structural hypotheses. Do not claim dedicated
preservation until the corresponding probe and operation audit exist.

## Choose experiments from observed structure

| Observation | Branch worth testing | Current expression | Cheap falsifier or caveat |
| --- | --- | --- | --- |
| Z-only cost | Alternating cost and noncommuting mixer rotations | `PauliRotation` | A generic mixer can leave a required feasible sector |
| `ZZ` plus transverse `X` | Term-factorized TFIM/HVA order | `PauliRotation` per selected term | Try a smaller ordering before adding every term |
| Matched `XX + YY` edges | Total-Z-preserving exchange | `XYExchange` after exact probe | Audit every edge; do not infer fermionic symmetry |
| Matched `XX + YY + ZZ` edges | Isotropic edge evolution | `IsotropicExchange` after exact probe | Test required generators; one charge is not SU(2) |
| General Pauli support | Selected-term HVA/operator pool | `PauliRotation` | Applying all terms may be costly and overly expressive |
| Weak structural evidence | Zero-operation comparison control | empty typed `null_control` candidate | Diagnostic only; it cannot promote |

Do not make every branch a symmetry branch. Ordering, support selection,
parameter sharing, graph coloring, and shallow controls are legitimate
structural experiments when their predictions are explicit.

## Use the closed feedback cycle

The controller chooses the next stage whenever the agent sends:

```json
{"type": "evaluate_candidate", "candidate_id": "candidate-id"}
```

The fixed sequence is:

1. **Audit** — compile, apply the initial preparation, check literals,
   invariants, and conservative resources before optimization.
2. **Smoke** — fixed COBYLA, at most 32 objective calls, one restart, seed 7.
3. **Promotion** — fixed COBYLA, at most 96 calls, three restarts, seed 997.

Audit includes deterministic nonzero audit bindings, so parameter sharing or
a special value cannot hide gate count or depth. Current caps are 512 two-qubit
gates, 2048 total gates, and depth 1024, using the worst of backend-routed and
canonical transpilation results.

Failed candidates are evidence. Use the failure to reduce an operator set,
change an ordering or sharing pattern, revise the physical claim, or retire the
branch. A revision gets a new ID and a new preregistered prediction while the
old result stays visible. Cosmetic renaming of the same physical family does
not earn another optimizer run.

Useful responses include:

| Feedback | Productive next step |
| --- | --- |
| Exact charge is refuted | Retire it or test a physically motivated different charge |
| Prepared state has charge variance | Reconsider the symmetry branch; the candidate cannot change preparation |
| Operation breaks a supported charge | Replace the operation or use an unconstrained structural branch |
| Resource audit fails | Reduce operation count/locality/fan-out before smoke |
| Smoke does not beat zero angle | Retire or make one motivated structural revision |
| Promotion regresses | Treat the family as unstable under the fixed optimizer policy |

## Compare before deciding

A promoted candidate cannot be committed in isolation. Before promotion,
bring a candidate from a different primary hypothesis through smoke. The
controller reserves enough budget for both fixed promotion evaluations and
requires the reserved comparison immediately after the first promotion. It
compares only those equal-fidelity records. Energy-tied comparisons include
unique trainable parameters as well as gates and depth. Then request:

```json
{"type": "commit", "candidate_id": "candidate-id"}
```

For a negative ending, explicitly revise or retire every open hypothesis and
candidate first. Each non-control branch needs a refuted probe, a valid failed
smoke/promotion, or a fair comparison that dominates a promoted candidate.
For a failed numerical run to count, the sampled objective span normalized by
the non-identity Hamiltonian norm must be at least `1e-6`. Establish either
promotion-depth adverse evidence or adverse evidence in two independent
`ansatz_structure` root lineages. A constant Hamiltonian is the only flat
exception. A refuted symmetry, compile audit, null control, or phase-only
candidate cannot satisfy this floor. Then request `close_negative` with a
grounded reason. The controller derives evidence coverage, but it does not
retire branches for the agent.

Use compact `research status` for routine feedback and `research status
--full` for complete history. Use `research result` only after a terminal
decision and report the optimized binding only from that output.

Promotion proves only the local fixed rule. It does not prove exact
ground-state accuracy, Pareto optimality, an anytime advantage, or
generalization. State explicitly when no independent reference score was
provided.

## Planned additions

- approximate, group/permutation, non-Abelian, fermionic, and gauge probes;
- fermionic/UCC, pair/seniority, and constraint-preserving typed operations;
- evaluator-side operator ranking and controlled ansatz growth;
- Pareto, anytime, noise-aware, and held-out evaluation protocols.

Until implemented, these are research directions rather than success claims.

## Background

- [Hamiltonian variational ansatz](https://arxiv.org/abs/1507.08969)
- [QAOA](https://arxiv.org/abs/1411.4028)
- [ADAPT-VQE](https://arxiv.org/abs/1812.11173)
- [Qubit-excitation ADAPT-VQE study](https://www.nature.com/articles/s42005-021-00730-0)
