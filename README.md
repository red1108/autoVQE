# AutoVQE

[![CI](https://github.com/red1108/autoVQE/actions/workflows/ci.yml/badge.svg)](https://github.com/red1108/autoVQE/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6133BD.svg)](pyproject.toml)

AutoVQE is a Hamiltonian-to-ansatz research tool for variational quantum
eigensolvers. Given a Pauli Hamiltonian, it inspects operator structure,
symmetries, locality, coupling constraints, and reference-state hints, then
proposes ansatz candidates that fit those facts.

It is not built to memorize bundled examples. It is meant to help a researcher
answer the actual VQE design question: for this Hamiltonian, which ansatz family
should I try, and what evidence supports that choice?

## Highlights

- Hamiltonian inspection for locality, support graph, Pauli structure, and
  candidate-family recommendations.
- Structured ansatz families for U(1) exchange, SU(2)-style Heisenberg HVA,
  TFIM counterdiabatic schedules, Pauli HVA, and shallow HEA baselines.
- Target-driven solve loop with explicit relative/absolute tolerance checks when
  a reference energy is available.
- Gate-count and parameter-count reporting after Qiskit transpilation.
- Reproducible JSON fixtures that probe different Hamiltonian regimes: small
  chemistry, TFIM, Heisenberg, weighted spin graphs, and a 16-qubit N2 stress
  test.

## Why AutoVQE?

Most VQE demos hard-code a problem and a circuit. AutoVQE keeps the evaluator
stable and lets the current Hamiltonian drive candidate design. The system should
derive decisions from facts such as conserved sectors, Pauli support graph,
commuting structure, term locality, coefficient scale, reference occupation, and
hardware connectivity. That makes it easier to compare ideas such as
symmetry-preserving ansatzes, Hamiltonian variational ansatz layers, and
operator-pool candidates without writing one-off rules for named fixtures.

The GitHub root is intentionally short. Runtime code lives under `autovqe/`,
while docs, fixtures, and community metadata stay in their own directories:

- `autovqe/harness.py` is the public CLI for inspection, calibration runs, and
  tolerance checks.
- `autovqe/train.py` proposes and optimizes ansatz candidates.
- `autovqe/prepare.py` loads problem JSON files, builds Hamiltonians, computes
  exact references for small systems, and reports compiled gate counts.
- `examples/` contains calibration fixtures for different Hamiltonian regimes.
- `docs/agent_protocol.md` is the agent protocol for automated research runs.
- `docs/` contains agent-facing playbooks, calibration notes, release notes, and
  the roadmap.
- `.github/` contains CI, templates, contributing guidance, security policy,
  citation metadata, and the code of conduct.

## Quick Start

Install [`uv`](https://docs.astral.sh/uv/) first, then run:

```bash
uv sync
uv run python -m autovqe.harness check
uv run python -m autovqe.harness solve --rel-tol 0.001
```

The default solve target is only a fast smoke test. For real use, pass the
Hamiltonian JSON you want to study:

```bash
uv run python -m autovqe.harness inspect --problem path/to/problem.json
uv run python -m autovqe.harness solve path/to/problem.json --rel-tol 0.001
```

To run the bundled small regression suite:

```bash
uv run python -m autovqe.harness solve \
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
uv run python -m autovqe.harness inspect --problem examples/ising_1d_5q.json

# Print a Hamiltonian-aware runbook for the default example.
uv run python -m autovqe.harness plan

# Run isolated smoke campaigns over calibration fixtures.
uv run python -m autovqe.harness benchmark

# Include larger spin-chain regime probes.
uv run python -m autovqe.harness benchmark --include-hard

# Run the target-driven solver on a specific problem.
uv run python -m autovqe.harness solve examples/ising_1d_5q.json --rel-tol 0.001
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

## Calibration Fixtures

Fixtures are not the purpose of AutoVQE. They are probes used to keep the
Hamiltonian-analysis and ansatz-selection logic honest. A change should improve
general behavior for a class of operators, not special-case a file name.

This small regression command is expected to pass:

```bash
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
```

Representative current regression results:

| Problem | Best family | Relative error |
| --- | --- | --- |
| `h2_2q` | `pauli_hva` | 0.0489% |
| `h2_4q` | `pauli_hva` | 0.0961% |
| `ising_1d_5q` | `tfim_counterdiabatic` | 0.0957% |

The larger spin-chain and chemistry fixtures are regime probes. For example,
`examples/tfim_n10_g1_open.json` stresses non-commuting TFIM structure,
`examples/heisenberg_n10_open.json` stresses symmetry-preserving exchange/HVA
logic, `examples/h2_4q_pennylane_0p6614.json` checks a small chemistry mapping,
and `examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json` stresses large
chemistry metadata and U(1)-style sector preservation. If one of these fails,
the correct response is to inspect the Hamiltonian facts and improve the general
candidate policy, not to add "if fixture X, use ansatz Y" logic.

When a reference energy is present, `solve` reports the raw optimized VQE circuit
energy against that reference. Classical post-processing may be studied
separately, but it must be labeled separately and must not be presented as the
VQE circuit result.

## Development Checks

```bash
uv run python -m py_compile autovqe/prepare.py autovqe/train.py autovqe/harness.py
uv run python -m autovqe.harness check
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
git diff --check
```

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for PR expectations and
[docs/release_checklist.md](docs/release_checklist.md) for release checks.

## Project Status

AutoVQE is an alpha research tool. The CLI and problem JSON format are small
and usable, but internals may change as new Hamiltonian-regime evidence lands.
See [docs/roadmap.md](docs/roadmap.md) and
[docs/changelog.md](docs/changelog.md) for current direction.

## Citation

If AutoVQE helps your research, cite the repository using
[.github/CITATION.cff](.github/CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
