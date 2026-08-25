# AutoVQE

[![CI](https://github.com/red1108/autoVQE/actions/workflows/ci.yml/badge.svg)](https://github.com/red1108/autoVQE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD.svg)](pyproject.toml)

AutoVQE is a Hamiltonian-to-ansatz research harness for variational quantum
eigensolvers. It helps an agent inspect a Pauli Hamiltonian, form falsifiable
hypotheses about useful structure, test candidate ansatzes, and retain the
evidence behind the final decision.

The project is designed to avoid two common shortcuts: hard-coding a circuit
for a named fixture and trusting candidate-reported energies or resource
counts. Candidates use a typed ansatz representation; the evaluator derives
optimized energy, parameter counts, gate counts, depth, and symmetry checks.

## Highlights

- Mechanical Hamiltonian observations without hidden reference answers.
- A typed ansatz IR with a small audited macro registry.
- Algebraic probes for commutators, conserved quantities, reference sectors,
  and gradient evidence.
- Symmetry-preserving exchange gates only when the measured Hamiltonian
  structure supports their use.
- A closed hypothesis → probe → candidate → audit → smoke → promotion loop.
- Evaluator-owned metrics and canonical resource accounting to limit reward
  hacking through hard-coded values or misleading parameter reports.
- A legacy solver retained for calibration and compatibility checks.

## Quick start

Install [`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python -m autovqe.harness check
uv run python -m autovqe.harness inspect --problem examples/ising_1d_5q.json
```

The compatibility solver remains available:

```bash
uv run python -m autovqe.harness solve \
  examples/h2_2q.json \
  examples/h2_4q.json \
  examples/ising_1d_5q.json \
  --rel-tol 0.001 \
  --max-stages 2
```

`solve` may use fixture reference data and is therefore a regression path, not
an independent ansatz-discovery result.

## Closed research loop

Initialize a local development run:

```bash
uv run python -m autovqe.harness research init \
  --problem path/to/hamiltonian.json \
  --run-dir .autovqe-runtime/research/demo \
  --budget 100
```

An agent submits one JSON action at a time:

```bash
uv run python -m autovqe.harness research step \
  --problem path/to/hamiltonian.json \
  --run-dir .autovqe-runtime/research/demo \
  --action action.json \
  --allow-unsealed

uv run python -m autovqe.harness research status \
  --run-dir .autovqe-runtime/research/demo \
  --allow-unsealed
```

Actions can register hypotheses, request probes, submit or revise typed
candidates, run fixed evaluation stages, retire disproven branches, and request
a terminal commit or grounded negative close. See
[the protocol](docs/agent_protocol.md) and [the loop guide](docs/agent_loop.md).

`local_unsealed` mode is for development. It tests the workflow but is not a
security boundary because the agent and evaluator share a filesystem identity.
For an isolated Codex campaign, use the operator procedure in
[meta_agent/README.md](meta_agent/README.md).

## Trust boundary

The public Hamiltonian and reference-state preparation may be visible to the
agent. Private references, exact target values, evaluator state, and optimized
parameter bindings are evaluator-side data.

Agent submissions may describe circuit structure and parameter sharing, but
may not provide trusted energy, resource metrics, or hidden numeric literals.
The compiler and evaluator derive those values. A controller-accepted
`positive_commit` proves only that a candidate passed the configured local
promotion rule; an external holdout scorer is still required for a ground-state
or generalization claim.

The supported variational macros are deliberately small:

- `Rx`, `Ry`, `Rz`
- `PauliRotation`
- `XXRotation`, `YYRotation`, `ZZRotation`
- `XYExchange`, `IsotropicExchange`

Every variational macro is identity at zero and decomposes into the canonical
resource basis `{rz, sx, x, cx}`. Conservation-oriented macros require probe
evidence for the corresponding symmetry and still undergo operation-level
commutator checks. See [the ansatz playbook](docs/ansatz_playbook.md).

## Problem format

Problems are JSON files containing Pauli terms and optional public execution
constraints:

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

Bundled examples are calibration fixtures, not release holdouts. A harness
change informed by a holdout result must retire that problem into the
development set.

## Repository map

- `autovqe/ansatz_ir.py`, `macros.py`, and `compiler.py`: typed candidates and
  trusted compilation.
- `autovqe/probes.py` and `evaluator.py`: evaluator-owned evidence and metrics.
- `autovqe/research.py`, `controller.py`, and `ledger.py`: lifecycle, budget,
  terminal rules, and replay.
- `autovqe/harness.py`: public CLI, including the research subcommands.
- `meta_agent/`: Codex campaign templates and bridge tooling.
- `examples/`: public calibration fixtures.
- `docs/`: protocol, ansatz method, release, and roadmap details.

## Development

```bash
uv run python -m unittest discover -s tests
uv run python -m autovqe.harness check
uv run python -m compileall -q autovqe meta_agent tests
git diff --check
```

AutoVQE is an alpha research tool. See the [roadmap](docs/roadmap.md),
[release checklist](docs/release_checklist.md), and
[contributing guide](.github/CONTRIBUTING.md).

## Citation and license

If AutoVQE helps your research, cite the repository using
[.github/CITATION.cff](.github/CITATION.cff). Licensed under the [MIT License](LICENSE).
