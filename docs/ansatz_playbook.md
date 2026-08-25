# Ansatz Playbook

Use this playbook to form and test hypotheses, not to assign an ansatz label
from a Hamiltonian pattern. The mechanical observation contains evidence; an
evaluator-produced probe result determines whether one particular claim is supported.

## Start from observations

Before proposing a hypothesis, record:

- Pauli terms and coefficient scales;
- locality distribution and two-body support edges;
- single-letter `X`/`Y`/`Z` counts and complex coefficients;
- declared encoding and sector values, if any;
- the public computational-basis preparation hint, if any;
- backend basis gates and coupling map.

The MVP fixes one indexing dialect. Full Hamiltonian/generator labels use
Qiskit's display order (rightmost character is q0); occupation tuple index
`q` is qubit `q`; and a `PauliRotation` local word is paired left-to-right
with its explicit `qubits` list. For example, `qubits: [0, 1]` with local
`pauli: "XY"` means X on q0 and Y on q1, whose full Qiskit label is `YX`.
Other `EncodingSpec.qubit_order` values are rejected rather than guessed.

Do not use a fixture name, a `model_class` label, a recommendation, an exact
reference energy/state, or previous learned angles as evidence. A declared
sector is a claim supplied with the problem, not proof that every Hamiltonian
term, reference, or proposed circuit preserves it.

## Current typed macro allowlist

The trusted research compiler currently resolves exactly these names:

| Macro | Kind/arity | Parameters/options | Trusted meaning |
| --- | --- | --- | --- |
| `PauliRotation` | variational operation, one or more qubits | `angle`; string option `pauli` with one active `X/Y/Z` per listed qubit | `exp(-i angle P)` via basis changes, a CNOT parity ladder, and `RZ(2 angle)` |
| `XYExchange` | variational operation, exactly two qubits | `angle`; no options | `exp[-i angle(XX + YY)]`, lowered as Qiskit `XXPlusYYGate(4 * angle, 0)` |
| `IsotropicExchange` | variational operation, exactly two qubits | `angle`; no options | `exp[-i angle(XX + YY + ZZ)]` via `RXX/RYY/RZZ` |
| `X` | reference-only, repeated one-qubit targets | no variational parameter | Explicit computational-basis preparation |

There is no public macro registration hook. Unknown names, matrices, callables,
opaque Qiskit instructions, duplicate/out-of-range qubits, and unsupported
options are rejected. Public backend basis names are lowering/accounting
targets only; they never expand this macro allowlist.

The allowlist is closed but not narrow enough to be a complete search policy.
The controller caps a candidate at 256 logical operations, 128 unique
parameters, 4096 IR nodes, and fan-out 64 for any one parameter. A
`PauliRotation` above locality two must exactly match a declared Hamiltonian
term, but arbitrary one/two-body active words remain available. Smoke/promotion
also apply fixed canonical conservative resource caps, taking the metric-wise
maximum of template and generic-worst views; per-problem macro/native-backend
profiles remain future work.

## Typed IR rules

An `AnsatzSpec` contains:

- `version` (currently `1`), `name`, and `num_qubits`;
- a unique list of trainable parameter names;
- an optional `X` reference preparation;
- ordered layers of macro operations;
- affine parameter expressions.

The compiler/evaluator, not the candidate, derives:

- unique trainable parameters and their names;
- every parameter occurrence;
- unused parameters;
- fixed literals and their roles/paths;
- macro, layer, operation, expression, and IR-node counts;
- declared-backend and canonical template/generic-worst/final gate counts and
  depth;
- energy and best-energy traces.

Parameter sharing is legal and scientifically useful, but it is not a way to
hide circuit size. A shared parameter is counted once under
`unique_trainable_params` and once per gate use under
`parameter_occurrences`; operations and physical gates are counted
independently. More than 64 uses of one parameter fails audit, so tying every
gate to a single reported parameter is bounded explicitly.

All declared parameters must occur. A variational angle must depend on a
trainable parameter and be zero when all parameters are zero. Constant-only
rotations and fixed offsets are rejected. At controller audit, coefficient
scales outside `{-2, -1, -0.5, 0.5, 1, 2}` and all numeric options are rejected.
This blocks hidden learned angles but also intentionally limits
Hamiltonian-coefficient-scaled schedules in the MVP.

Candidate metadata must set the enforcement value implied by its claim:
`preserve` for `exact_pauli_symmetry`, `unconstrained` for
`ansatz_structure`, and `diagnostic` for `null_control`. For `preserve`, the
audit computes the normalized commutator of every logical operation generator
with the claim's charge and checks the explicit zero-parameter reference's
normalized charge variance. A residual or variance above `1e-10` fails audit.
The reference must exactly reproduce the public occupation hint and cannot be
introduced when no hint exists.

`XYExchange` and `IsotropicExchange` are conditionally available: the parent
must be a controller-`SUPPORTED` `exact_pauli_symmetry` hypothesis, and every
operation must still pass the charge-commutator audit. Their allowlist entries
and conservation-suggestive names are implementation availability, not
physical evidence. Auto-admitted structure/control claims cannot authorize
them.

## Symmetry is a probeable hypothesis

Use the loop:

```text
observe pattern/declaration
  -> name a candidate generator Q
  -> request [H, Q] probe
  -> design operations predicted to preserve Q
  -> audit operations + public reference against Q
  -> smoke/promote
  -> revise or retire when evidence disagrees
```

The external research controller currently offers one algebraic probe request:
`normalized_commutator` for an `exact_pauli_symmetry` claim. The claim must
contain exactly `kind` and a machine-readable `generator`. Structure and null
control claims use the closed schemas `{"kind":"ansatz_structure",
"family":"..."}` and `{"kind":"null_control"}`; the controller admits them with
an explicit non-algebraic marker rather than a physics certificate.

### Exact commutator

```json
{
  "type": "request_probe",
  "hypothesis_id": "total_z",
  "probe_id": "comm_total_z",
  "probe": {
    "type": "normalized_commutator",
    "generator": {
      "type": "global_pauli_sum",
      "pauli": "Z",
      "selector": "all_sites"
    }
  }
}
```

This computes a scale-normalized Pauli-coefficient norm of `[H,Q]`. “Exact”
means residual at most `1e-10`. The probe rejects a zero/identity generator and
a generator that is merely a trivial copy of `H`. Coefficients must be finite
and real, and active generator norm must be at least `1e-8`.

Generator JSON is also structurally bounded. A `pauli_sum` has exactly `type`
and 1–256 `terms`; labels must be unique, valid, and full width. Each term has
exactly `pauli` and optional `coeff`, where `coeff` is a finite real JSON number
with magnitude at most `1e6`. `global_pauli_sum` uses `pauli` and
`orbit_pauli_sum` uses `seed`; both allow only `X/Y/Z`, optional
`selector: "all_sites"`, strict fields, and at most 256 generated terms.

### Public-reference moments are audit-internal

`reference_moments` is an evaluator primitive, not an accepted external
`request_probe` type. When auditing an exact-symmetry candidate, the controller
computes mean and normalized variance on the required explicit public
reference. Variance at most `1e-10` passes. Division by squared active generator
norm prevents a tiny coefficient from forcing a pass.

The probe alone does not prove that a submitted circuit preserves the charge.
That requires `metadata.enforcement: "preserve"` at candidate audit, which
checks each operation and the explicit reference against the same
machine-readable exact Pauli charge. Approximate-symmetry, group action,
projector/Casimir, and fermionic algebra probes remain future work.

## Candidate hypotheses by observed structure

The table suggests a hypothesis and a cheap attempted falsification. It does
not select a winner.

| Observation | Hypothesis to test | Current encoding | Important caveat |
| --- | --- | --- | --- |
| Z-only terms | Alternating cost rotations and noncommuting single-qubit mixers create useful motion | `PauliRotation` for Z words and one-qubit X/Y words | No named QAOA or constraint-mixer macro; a generic mixer may leave a required feasible sector |
| `ZZ` plus transverse `X` | A term-factorized TFIM/HVA schedule matches the operator split | `PauliRotation` per term | Counterdiabatic/operator-pool selection is not implemented |
| Matched edge `XX+YY` | Exchange moves preserve total Z/Hamming weight | `XYExchange`, after probing global Z sum | Qubit Hamming weight is not automatically fermionic particle/spin/seniority preservation |
| Matched edge `XX+YY+ZZ` | Isotropic edge evolution respects a Heisenberg-like structure | `IsotropicExchange`, only after a matching exact charge probe | Full SU(2) evidence requires more than one charge; the controller has no Casimir/group probe |
| General Pauli support | Selected term rotations form a useful HVA/operator pool | `PauliRotation` | Applying every Hamiltonian term can be large, overly expressive, and symmetry-breaking |
| No robust structural claim | A shallow typed control tests whether the hypothesis adds value | Small allowlisted rotations under `null_control` | It is diagnostic and cannot be promoted; the MVP has no Pareto/holdout comparison |

For a putative U(1) symmetry, probe a candidate total-Z generator and the
reference sector before choosing `XYExchange`. For a putative SU(2)-like model,
separate exact-symmetry hypotheses/commutator probes for global X, Y, and Z are
stronger evidence than a single label, but they still do not constitute a
non-Abelian representation audit.

For chemistry, do not equate an `initial_state_hint` with a Hartree–Fock proof
or `XYExchange` with UCC. The current IR lacks fermionic single/double,
spin-adapted, and pair/seniority-preserving macros and lacks a fermion-to-qubit
algebra probe. A chemistry-specific claim may therefore be scientifically
reasonable but not enforceable by the MVP typed boundary.

For lattice gauge, point-group, translation, permutation, or other constraint
structure, document the hypothesis but do not claim dedicated enforcement. The
generic `preserve` audit can cover an exactly encoded Pauli-sum charge, but
Gauss-law-specific moves, group projection/twirling, orbit parameter tying, and
non-Abelian symmetry-adapted blocks are not in the current allowlist.

## Minimal candidate example

This candidate exposes one trainable Pauli rotation and its explicit reference
preparation:

```json
{
  "version": 1,
  "name": "two_qubit_xx",
  "num_qubits": 2,
  "parameters": ["theta"],
  "reference": {
    "macro": "X",
    "qubits": [0]
  },
  "layers": [
    {
      "name": "move",
      "operations": [
        {
          "macro": "PauliRotation",
          "qubits": [0, 1],
          "parameters": {
            "angle": {
              "terms": [
                { "parameter": "theta", "coefficient": 1.0 }
              ],
              "constant": 0.0
            }
          },
          "options": {
            "pauli": "XX"
          }
        }
      ]
    }
  ]
}
```

The shorter `{"parameter": "theta"}` angle form is equivalent. Use explicit
parameter sharing only when it is part of the hypothesis; do not report only
the unique count while omitting operation/occurrence/gate counts.
Place the claim-required `enforcement` value in the surrounding
`submit_candidate.metadata`, not inside this strict `spec` object. Audit also
requires at least one logical operation and one trainable parameter. Before
evaluation, add a concrete `prediction` or `falsifier` to metadata for any
candidate that may be committed.

## Promotion evidence

The current stage sequence is:

1. `audit`: typed compile plus fixed-literal policy;
2. `smoke`: fixed COBYLA protocol with up to 32 calls, one restart, seed 7;
3. `promotion`: fixed COBYLA protocol with up to 96 calls, three restarts, seed
   997.

Smoke and promotion require improvement over the zero-parameter candidate by
at least `max(1e-6, 1e-6 * abs(baseline_energy))`. Promotion additionally
requires a prior passed smoke and energy no worse than its best energy plus
`5e-4`; null controls are blocked.

Both stages transpile the symbolic template and three deterministic
evaluator-owned generic bindings on the all-to-all canonical `{rz,sx,x,cx}`
target. For each metric, the controller takes
`canonical_conservative_* = max(canonical_template_*,
canonical_generic_worst_*)`; this view must have at most 512 two-qubit gates,
2048 total gates, and depth 1024. Inspect `canonical_template_*`,
`canonical_generic_worst_*`, and `canonical_final_*` alongside declared-backend
metrics and the controller-owned conservative resource policy. Passing these
local gates is not proof of:

- improvement over competing ansatz baselines;
- Pareto efficiency in energy/resources;
- a strong anytime curve;
- generalization to other or hidden Hamiltonians;
- closeness to an exact reference energy.

Record those limitations in the candidate metadata/final report. Do not infer
more than the recorded evaluation establishes.

A passed promotion cannot be revised or retired: it remains live until an
evidence-gated commit, and therefore prevents `close_negative`. Commit cites
the promotion plus an evaluated competitor/control or a documented
non-dominance basis. This preserves a successful result instead of erasing it
to manufacture a negative terminal.

Use `research status` to review branch evidence and `research result` only
after a terminal decision. Do not treat edits to local run files as new
scientific evidence.

## Planned additions

- versioned macro profiles and native-backend/per-profile resource ceilings;
- named HVA/QAOA and constraint-preserving mixer templates;
- fermionic/UCC and pair/seniority-preserving chemistry blocks;
- extensions of exact candidate preservation to approximate/group/non-Abelian
  and fermionic charges;
- approximate, group/permutation, non-Abelian, fermionic, and gauge-law probes;
- ADAPT-style gradient/operator ranking under a fixed evaluator budget;
- public/hidden generalization evaluation, Pareto retention, and anytime
  scoring.

Until implemented, these belong in a hypothesis or roadmap—not in a success
claim.

## Background references

- Hamiltonian variational ansatz: https://arxiv.org/abs/1507.08969
- QAOA: https://arxiv.org/abs/1411.4028
- ADAPT-VQE: https://arxiv.org/abs/1812.11173
- Qubit-excitation ADAPT-VQE study:
  https://www.nature.com/articles/s42005-021-00730-0
