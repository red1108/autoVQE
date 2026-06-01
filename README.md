# AutoVQE

[![CI](https://github.com/red1108/autoVQE/actions/workflows/ci.yml/badge.svg)](https://github.com/red1108/autoVQE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD.svg)](pyproject.toml)

AutoVQE is a compact research harness for Hamiltonian-aware variational quantum
eigensolver experiments. It inspects the operator, proposes structured ansatz
candidates, optimizes them under a fixed evaluator, and reports whether the raw
VQE circuit energy reaches the requested tolerance.

The project is built for agentic VQE research: keep the evaluator fixed, make
one falsifiable ansatz change at a time, and let benchmark evidence decide.

## Highlights

- Hamiltonian inspection for locality, support graph, Pauli structure, and
  candidate-family recommendations.
- Structured ansatz families for U(1) exchange, SU(2)-style Heisenberg HVA,
  TFIM counterdiabatic schedules, Pauli HVA, and shallow HEA baselines.
- Target-driven solve loop with explicit relative/absolute tolerance checks.
- Gate-count and parameter-count reporting after Qiskit transpilation.
- Reproducible JSON fixtures for H2, TFIM, Heisenberg, weighted spin graphs,
  and a 16-qubit N2 stress test.

## Why AutoVQE?

Most VQE demos hard-code a problem and a circuit. AutoVQE keeps the problem file
and evaluator stable, then searches through Hamiltonian-derived circuit
families. That makes it easier to compare ideas such as symmetry-preserving
ansatzes, Hamiltonian variational ansatz layers, or operator-pool candidates
without mixing them with evaluator changes.

The project is intentionally script-first:

- `harness.py` is the public CLI for inspection, benchmark runs, and target
  solving.
- `train.py` proposes and optimizes ansatz candidates.
- `prepare.py` loads problem JSON files, builds Hamiltonians, computes exact
  references for small systems, and reports compiled gate counts.
- `examples/` contains named Hamiltonian fixtures.
- `program.md` is the agent protocol for automated research runs.
- `docs/` contains agent-facing playbooks and benchmark notes.
- `CONTRIBUTING.md` describes the checks expected before a pull request.

## Quick Start

Install [`uv`](https://docs.astral.sh/uv/) first, then run:

```bash
uv sync
uv run harness.py check
uv run harness.py solve --rel-tol 0.001
```

The default solve target is `examples/h2_2q.json`, a fast sanity-check problem.
To run the bundled small suite:

```bash
uv run harness.py solve \
  examples/h2_2q.json \
  examples/h2_4q.json \
  examples/ising_1d_5q.json \
  --rel-tol 0.001
```

`solve` prints a target check using:

```text
abs(best_energy - reference_energy) <= max(abs_tol, rel_tol * abs(reference_energy))
```

Example output:

```text
target_status: passed=True gap=0.000908030202 threshold=0.001857275030 rel_error=0.048890%
solve_rollup:
- examples/h2_2q.json: passed=True ... family=pauli_hva stage=smoke
```

## CLI

```bash
# Inspect Hamiltonian structure and recommended ansatz families.
uv run harness.py inspect --problem examples/ising_1d_5q.json

# Print a Hamiltonian-aware runbook for the default example.
uv run harness.py plan

# Run isolated smoke campaigns over the small examples.
uv run harness.py benchmark

# Include the n=10 hard spin-chain targets.
uv run harness.py benchmark --include-hard

# Run the target-driven solver on a specific problem.
uv run harness.py solve examples/ising_1d_5q.json --rel-tol 0.001
```

Generated experiment files such as `results.tsv`, `run.log`,
`benchmark_runs/`, and `solve_runs/` are ignored by git.

## Problem Format

Problems are JSON files with Pauli terms and optional hardware constraints:

```json
{
  "name": "example",
  "pauli_terms": [
    { "pauli": "ZI", "coeff": -1.0 },
    { "pauli": "IZ", "coeff": -1.0 },
    { "pauli": "XX", "coeff": 0.2 }
  ],
  "basis_gates": ["rx", "ry", "rz", "cx"],
  "coupling_map": [[0, 1], [1, 0]],
  "initial_state_hint": [1, 0]
}
```

If `reference_energy` is omitted and the system is small enough, AutoVQE uses
exact diagonalization to compute it.

## Ansatz Families

The harness classifies each Hamiltonian from its Pauli structure and chooses a
candidate order before running experiments. Built-in families include U(1)
number-preserving exchange layers, Pauli term-evolution HVA,
Heisenberg/exchange HVA, TFIM schedules with parity-preserving
counterdiabatic edge moves, and shallow hardware-efficient baselines. A single
shared-angle `exp(-i theta H)` candidate is not accepted as a VQE ansatz.

Hardware-efficient ansatzes are treated as baselines, not as the default
scientific explanation for every Hamiltonian.

For method-selection context, read `docs/ansatz_playbook.md`. The intended
design is that domain knowledge lives in docs until an experiment justifies
turning it into code.

## Verified Targets

These commands are expected to pass:

```bash
uv run harness.py solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
uv run harness.py solve examples/tfim_n10_g1_open.json examples/heisenberg_n10_open.json --rel-tol 0.001 --max-stages 2
```

Representative current results:

| Problem | Best family | Relative error |
| --- | --- | --- |
| `h2_2q` | `pauli_hva` | 0.0489% |
| `h2_4q` | `pauli_hva` | 0.0961% |
| `ising_1d_5q` | `tfim_counterdiabatic` | 0.0957% |
| `tfim_n10_g1_open` | `tfim_counterdiabatic` | 0.1000% |
| `heisenberg_n10_open` | `heisenberg_hva` | 0.0998% |
| `ising_1d_9q` | `heisenberg_hva` | 0.0998% |
| `n2_16q` | `u1_exchange` | 0.0388% |

The harder spin-chain fixtures are `examples/tfim_n10_g1_open.json` and
`examples/heisenberg_n10_open.json`. They are useful development targets but are
not part of the default CI gate because they are intentionally harder than the
small sanity-check suite. `solve` reports raw circuit VQE energy by default;
classical post-processing must not be counted as a VQE pass.

Large chemistry fixtures are available as explicit targets, including
`examples/h2_4q_pennylane_0p6614.json` and
`examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json`. The N2 fixture is
kept out of default benchmark commands because it is a 16-qubit, 1281-term
stress test.

## Development Checks

```bash
uv run python -m py_compile harness.py train.py prepare.py
uv run harness.py check
uv run harness.py solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
git diff --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for PR expectations and
[docs/release_checklist.md](docs/release_checklist.md) for release checks.

## Project Status

AutoVQE is an alpha research tool. The CLI and problem JSON format are small
and usable, but internals may change as new benchmark evidence lands. See
[ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md) for current direction.

## Citation

If AutoVQE helps your research, cite the repository using
[CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
