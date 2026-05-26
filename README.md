# AutoVQE

AutoVQE is a small, executable research loop for hardware-aware variational
quantum eigensolver experiments. It is inspired by
[autoresearch](https://github.com/karpathy/autoresearch): keep the code simple,
make the objective measurable, and let experiments decide what to try next.

The repository has three moving parts:

- `prepare.py` is the fixed evaluator. It loads a problem, builds the
  Hamiltonian/backend target, transpiles circuits, and computes exact references
  for small systems.
- `train.py` is the research surface. Ansatz families, initial states,
  optimizers, candidate schedules, and compression logic live here.
- `harness.py` is the control loop. It audits Hamiltonians, runs isolated
  campaigns, checks target tolerances, and escalates when a candidate has not
  solved the problem.

`program.md` is the agent protocol. It tells a coding agent how to run VQE
research in this repo without silently changing the benchmark.

## Quick Start

```bash
uv sync
uv run prepare.py
uv run harness.py inspect
uv run harness.py plan
uv run harness.py check
uv run train.py
```

For a target-driven run, use `solve`:

```bash
uv run harness.py solve problem.json --rel-tol 0.001
```

For the bundled example suite:

```bash
uv run harness.py solve problemset/problem1.json problemset/problem2.json problemset/problem3.json --rel-tol 0.001
```

`solve` is the preferred public entrypoint. It runs smoke, standard, and deep
stages as needed, then prints whether the best energy satisfies:

```text
abs(best_energy - reference_energy) <= max(abs_tol, rel_tol * abs(reference_energy))
```

## Common Commands

```bash
# Inspect Hamiltonian structure and recommended ansatz families.
uv run harness.py inspect

# Print the agent runbook for the current problem.
uv run harness.py plan

# Run a bounded smoke campaign and summarize new rows.
uv run harness.py campaign --mode smoke --experiments 8

# Run the target-driven solver.
uv run harness.py solve problem.json --rel-tol 0.001

# Check all bundled problems in isolated result files.
uv run harness.py benchmark --experiments 45 --experiment-seconds 2 --max-evals 300 --timeout 120

# Run fast consistency checks.
uv run harness.py check
```

Generated experiment files such as `results.tsv`, `run.log`, `benchmark_*`, and
`solve_runs/` are ignored by git.

## Problem Format

A problem is a JSON file with Pauli terms and optional backend constraints:

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

For small problems, `prepare.py` computes `reference_energy` by exact
diagonalization if the JSON does not provide one.

## Research Discipline

AutoVQE intentionally keeps the evaluator fixed and the research surface small.

- Do not edit `prepare.py` during a run.
- Do not edit the active `problem.json` during a run.
- Every tunable rotation must be represented as an optimization parameter.
- Problem-aware reference preparation is allowed only through explicit gates that
  are counted in the compiled metrics.
- Hardware-efficient ansatzes are useful baselines, not the first explanation for
  every Hamiltonian.
- A result is not solved until `harness.py solve` proves the requested tolerance.

## Current Built-In Ansatz Families

- Hamiltonian Pauli evolution (`pauli_hva`)
- Heisenberg/exchange HVA (`heisenberg_hva`)
- TFIM grouped and factorized schedules (`tfim_shared`, `tfim_factorized`)
- HF-hint two-state excitation mixers (`two_state_excitation`)
- Hardware-efficient and real-amplitudes style baselines (`hea`, `brick`, `symm`)

The harness chooses an initial family order from the Pauli structure, then
measures actual performance.

## Development Checks

Before pushing changes:

```bash
uv run python -m py_compile harness.py train.py prepare.py
uv run harness.py check
uv run harness.py solve problemset/problem1.json problemset/problem2.json problemset/problem3.json --rel-tol 0.001
git diff --check
```

## License

MIT. See [LICENSE](LICENSE).
