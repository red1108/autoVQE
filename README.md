# AutoVQE

[![CI](https://github.com/red1108/autoVQE/actions/workflows/ci.yml/badge.svg)](https://github.com/red1108/autoVQE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD.svg)](pyproject.toml)

AutoVQE is a Hamiltonian-to-ansatz research harness for variational quantum
eigensolvers. It gives an agent a disciplined loop for inspecting a Pauli
Hamiltonian, testing structural hypotheses, proposing typed circuits, and
keeping the evidence behind the final decision.

The central rule is simple: the agent proposes circuit structure, while
AutoVQE computes the energy, optimized parameters, resource counts, and
symmetry checks. Candidate-authored claims about those quantities are never
accepted as measurements.

## What AutoVQE provides

- Mechanical observations derived from the supplied Hamiltonian.
- A strict `AnsatzSpec` representation with a small gate allowlist.
- Algebraic probes for conserved quantities and evaluator-side sector checks.
- Conditional symmetry-preserving gates whose use must be justified by a
  measured symmetry.
- A closed hypothesis → probe → candidate → audit → smoke → promotion loop.
- Evaluator-owned energy, optimization, parameter, gate, and depth results.
- Durable branch history, including rejected and retired ideas.

## Install and inspect

Install [`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python -m autovqe.harness check
uv run python -m autovqe.harness inspect \
  --problem user_problem/hamiltonian.json
```

Use `--json` when a machine-readable, recommendation-free observation is
needed:

```bash
uv run python -m autovqe.harness inspect \
  --problem user_problem/hamiltonian.json \
  --json
```

## Run a research loop

Create a run from an immutable problem file:

```bash
uv run python -m autovqe.harness research init \
  --problem user_problem/hamiltonian.json \
  --run-dir .autovqe-runtime/research/run-001 \
  --budget 100
```

Write one JSON action, then submit it:

```bash
uv run python -m autovqe.harness research step \
  --run-dir .autovqe-runtime/research/run-001 \
  --action action.json
```

Inspect progress at any time:

```bash
uv run python -m autovqe.harness research status \
  --run-dir .autovqe-runtime/research/run-001
```

Continue one action at a time until the controller accepts either
`positive_commit` or `negative_close`. Then request the final evaluator-derived
result:

```bash
uv run python -m autovqe.harness research result \
  --run-dir .autovqe-runtime/research/run-001
```

For a positive decision, the result contains the committed `AnsatzSpec`, its
optimized parameter binding, energy, resource measurements, and cited
evidence. A negative decision contains the investigated branches and the
evidence supporting closure. A local promotion establishes only the recorded
promotion rule; it does not establish exact ground-state accuracy or
cross-problem generalization.

See [the workflow guide](docs/agent_loop.md),
[action protocol](docs/agent_protocol.md), and
[ansatz playbook](docs/ansatz_playbook.md).

## Allowed variational gates

The logical gate surface is intentionally small:

- `PauliRotation`
- `XYExchange`
- `IsotropicExchange`
- `X` for reference preparation only

Every variational gate is the identity at zero and is lowered to the canonical
resource basis `{rz, sx, x, cx}` for comparison. `XYExchange` and
`IsotropicExchange` are available only when an exact conserved quantity has
been established and every instantiated operation preserves it. Higher
locality Pauli rotations must correspond to terms in the supplied
Hamiltonian.

This boundary lets the agent exploit real physical structure without giving it
an unrestricted circuit language that could hide complexity or fixed numeric
answers.

## Evaluator-owned evidence

An action may propose:

- a falsifiable structural or symmetry hypothesis;
- a probe request;
- a typed ansatz and parameter-sharing pattern;
- a request for the next fixed evaluation stage;
- revision, retirement, commit, or negative closure.

An action may not provide authoritative energy, optimized values, parameter
counts, gate counts, depth, optimizer settings, or custom operations. AutoVQE
derives these values from the parsed circuit. Parameter occurrence counts and
generic nonzero bindings prevent a candidate from appearing cheap only because
many gates share one reported parameter or cancel at a special value.

AutoVQE runs in the user's checkout. It is a scientific workflow and validation
layer, not an adversarial execution sandbox. Independent accuracy claims should
be checked later with reference data that was kept outside the research
session.

## Problem format

The input is a JSON file containing Pauli terms and optional public execution
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

Keep evaluation-only reference energies and states out of a user problem.
Files in `examples/` are optional calibration inputs, not unseen evaluation
problems. If an evaluation problem influences implementation choices, move it
into the calibration set and choose a new unseen problem.

## Repository map

- `autovqe/ansatz_ir.py`, `macros.py`, `compiler.py`: typed candidates and
  circuit compilation.
- `autovqe/probes.py`, `evaluator.py`: physical probes, optimization, and
  measurements.
- `autovqe/research.py`, `controller.py`, `history.py`: lifecycle, budget,
  branch history, and terminal rules.
- `autovqe/research_cli.py`, `harness.py`: the public command-line workflow.
- `examples/`: optional calibration Hamiltonians.
- `docs/`: protocol, ansatz method, release, and roadmap notes.

## Development

```bash
uv run python -m compileall -q autovqe tests
uv run python -m unittest discover -s tests -v
uv run python -m autovqe.harness check
git diff --check
```

AutoVQE is an alpha research tool. See the
[roadmap](docs/roadmap.md), [release checklist](docs/release_checklist.md), and
[contributing guide](.github/CONTRIBUTING.md).

## Citation and license

If AutoVQE helps your research, cite the repository using
[.github/CITATION.cff](.github/CITATION.cff). Licensed under the
[MIT License](LICENSE).
