# AutoVQE

[![CI](https://github.com/red1108/autoVQE/actions/workflows/ci.yml/badge.svg)](https://github.com/red1108/autoVQE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD.svg)](pyproject.toml)

AutoVQE is a Hamiltonian-to-ansatz research harness for variational quantum
eigensolvers. It lets an agent inspect a Pauli Hamiltonian, test falsifiable
ideas about its structure, evaluate typed circuits, and retain the evidence
behind both successful and failed branches.

The agent chooses hypotheses and circuit structure. AutoVQE owns initial-state
preparation, optimization, energy, parameter bindings, circuit resources, and
symmetry checks. Candidate-supplied values never count as measurements.

## Quick start with Codex

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and install
the project command:

```bash
git clone https://github.com/red1108/autoVQE.git
cd autoVQE
uv sync
uv run autovqe check
```

Put the problem at `user_problem/hamiltonian.json`. This directory is ignored
by Git. Do not include a reference energy, exact state, target parameters, or
expected ansatz.

Open the clone as a **new Codex workspace** and give Codex this goal:

```text
Read AGENTS.md, README.md, docs/agent_protocol.md, and
docs/ansatz_playbook.md. Treat user_problem/hamiltonian.json as immutable.
Read that Hamiltonian directly and use `uv run autovqe inspect` for its derived
overview. Do not modify AutoVQE source, tests, or docs during this run. Use the
research commands to run a closed cycle: form falsifiable hypotheses, request
physical probes when needed, submit typed AnsatzSpec candidates, and let the
controller advance each candidate through audit, smoke, and promotion. Keep
structural hypotheses separate from any supported symmetry evidence cited by a
candidate. Learn from failures by revising or retiring branches, and bring a
different-hypothesis competitor or control through smoke before promotion.
Continue until the controller accepts positive_commit or negative_close.
Write action JSON only under .autovqe-runtime/actions; do not edit run history
or call evaluator internals directly. Finally run `research result` and
report only evaluator-produced values. If no independent reference score was
provided, say so explicitly.
```

Using a separate worktree for this user-style run keeps development context and
local changes out of the test workspace.

## Problem format

```json
{
  "name": "example",
  "pauli_terms": [
    {"pauli": "ZI", "coeff": -1.0},
    {"pauli": "IZ", "coeff": -1.0},
    {"pauli": "XX", "coeff": 0.2}
  ],
  "basis_gates": ["rx", "ry", "rz", "cx"],
  "coupling_map": [[0, 1], [1, 0]],
  "initial_state_hint": [1, 0]
}
```

Pauli labels use Qiskit's display order: the rightmost character acts on qubit
0. In `initial_state_hint`, item `q` is the bit prepared on qubit `q`. The hint
is input to evaluator-owned circuit preparation; it is not part of an
`AnsatzSpec` and is not proof of a claimed symmetry sector.

The v1 document schema is closed. In addition to the fields above it accepts
an optional text `source_note` and a `symmetry` object containing only
`mapping`, `basis`, `orbital_order`, `spin_order`, `spin_orbitals`,
`active_orbitals`, `active_electrons`, `particle_number`, `magnetization`,
`spin_projection`, `total_spin`, and `parity`. Unknown fields are rejected;
reference answers belong outside the research input. The validated input is
snapshotted at `research init`, and any later content change stops the run.

## Research workflow

Inspect and initialize a run:

```text
uv run autovqe inspect --problem user_problem/hamiltonian.json --json
uv run autovqe research init --problem user_problem/hamiltonian.json --run-dir .autovqe-runtime/research/run-001 --budget 100
```

Write one JSON action to an ignored scratch file and submit it:

```text
uv run autovqe research step --run-dir .autovqe-runtime/research/run-001 --action .autovqe-runtime/actions/next.json
```

Repeat this loop:

```text
observe -> hypothesize -> probe when needed -> submit candidate
        -> audit -> smoke -> fair promotion comparison -> decide
              |       |          |
              +-------+----------+-> revise or retire on failure
```

The external action stays small. A symmetry probe needs only
`type` and `hypothesis_id`; the controller derives its generator and evidence
identifier from the hypothesis. Candidate evaluation needs only `type` and
`candidate_id`; the controller selects the next valid stage. In particular,
the audit compiles and measures resources before any optimization budget is
spent.

Check the compact branch summary at any time:

```text
uv run autovqe research status --run-dir .autovqe-runtime/research/run-001
```

Use `--full` only when complete branch records are needed. Optimized bindings
remain reserved for terminal `research result`; compact status is the normal
agent feedback surface.

A promotion requires another candidate from a different primary hypothesis to
have passed smoke first. Both are then evaluated with the same fixed promotion
protocol. A positive decision compares their evaluator-owned energy, routed
and canonical resources, depth, and unique parameter count. The final action
is just:

```json
{"type": "commit", "candidate_id": "candidate-id"}
```

If no promotable result remains, explicitly revise or retire every open branch.
A negative close requires a refuted physical probe, a valid failed smoke or
promotion experiment, or a promotion retired after a fair dominating
comparison for each non-control hypothesis. A passed compile audit and an
agent-written retirement reason are not scientific evidence. A failed
numerical run counts only when its sampled objective actually varied:
`(max(E)-min(E))/||H_nonidentity||2 >= 1e-6`. Negative closure then requires
either objective-active promotion-depth evidence or objective-active failures
from two independent `ansatz_structure` root lineages. A constant Hamiltonian
is the explicit flat-objective exception. Thus phase-only candidates, null
controls, and refuted symmetry probes cannot manufacture completion. Once all
branches are resolved, the controller can derive the supporting evidence for:

```json
{"type": "close_negative", "reason": "grounded explanation"}
```

After either terminal decision, request the final evaluator-owned result:

```text
uv run autovqe research result --run-dir .autovqe-runtime/research/run-001
```

Only `research result` is the reporting surface for the optimized parameter
binding. A passed promotion establishes the recorded local rule only; it does
not establish exact ground-state accuracy, Pareto optimality, or performance on
new Hamiltonians. Report the absence of an independent reference score when
none was supplied after the run.

## Candidate boundary

Every circuit is a version-1 `AnsatzSpec` with a flat, ordered `operations`
array. The variational operation allowlist is exactly:

- `PauliRotation`
- `XYExchange`
- `IsotropicExchange`

There is no candidate-authored preparation operation, secondary grouping,
custom operation, matrix, or executable circuit payload. Every variational
operation is the identity at zero. Higher-locality `PauliRotation` operations
must match a supplied Hamiltonian term.

`XYExchange` and `IsotropicExchange` are conditional tools, not shortcuts. A
candidate may cite one or more supported exact-symmetry probe IDs independently
of its primary structural hypothesis. Every operation must preserve every
cited charge, the prepared state must have a definite sector, and each special
gate must overlap nontrivially with at least one cited charge. A spectator
or epsilon-weighted touching term cannot justify an exchange gate. The overlap
must carry at least `1e-3` of that charge's active norm, and its conditioned
Hamiltonian residual and sector variance must remain exact. For an SU(2)-like
claim, one conserved
component is not evidence for the whole group; test and cite the generators the
claim actually requires.

Parameter sharing is allowed, but declared parameter directions must be
linearly independent and cannot hide circuit size. AutoVQE separately
counts unique parameters, parameter occurrences, logical operations, gates,
two-qubit gates, and depth, including deterministic nonzero audit bindings.
The resource boundary takes the worst of backend-routed and canonical
transpilation results, so sparse connectivity cannot be hidden by an
all-to-all count. Energy-tied comparisons also treat extra independent
parameters as a real resource cost.
The optimizer policy is fixed by the evaluator: COBYLA with fixed stage
allowances and seeds.

See the [action protocol](docs/agent_protocol.md) for exact JSON schemas and the
[ansatz playbook](docs/ansatz_playbook.md) for structure-driven search advice.

## Repository map

- `autovqe/ansatz_ir.py`, `macros.py`, `compiler.py`: typed candidates and
  circuit compilation.
- `autovqe/problem.py`, `observations.py`: problem loading and mechanical
  observations.
- `autovqe/probes.py`, `evaluator.py`: physical probes, optimization, and
  measurements.
- `autovqe/research.py`, `controller.py`, `history.py`: branch lifecycle,
  budget, evidence, and terminal rules.
- `autovqe/research_cli.py`, `harness.py`: installed command-line workflow.
- `examples/`: optional calibration Hamiltonians, not unseen tests.

## Development

```bash
uv run python -m compileall -q autovqe tests
uv run python -m unittest discover -s tests -v
uv run autovqe check
git diff --check
```

AutoVQE is an alpha research tool. See the [roadmap](docs/roadmap.md),
[release checklist](docs/release_checklist.md), and
[contributing guide](.github/CONTRIBUTING.md).

## Citation and license

If AutoVQE helps your research, cite the repository using
[.github/CITATION.cff](.github/CITATION.cff). Licensed under the
[MIT License](LICENSE).
