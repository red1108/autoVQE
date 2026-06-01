# Contributing

AutoVQE is a Hamiltonian-to-ansatz research tool. Contributions should keep the
evaluator fixed, derive candidate behavior from the current Hamiltonian, make
that behavior measurable, and report raw VQE circuit energy.

## Setup

```bash
uv sync
uv run python -m autovqe.harness check
```

## Before Changing Code

1. Inspect the target Hamiltonian.

   ```bash
   uv run python -m autovqe.harness inspect --problem <problem.json>
   ```

2. Read the relevant notes in `docs/`.
3. Choose one Hamiltonian-derived candidate policy to test.

Do not add benchmark-name special cases. Candidate selection should follow
operator facts such as locality, support graph, commuting structure, coefficient
scale, reference occupation, hardware connectivity, and conserved sectors.

## Required Checks

Run these before opening a pull request:

```bash
uv run python -m py_compile autovqe/prepare.py autovqe/train.py autovqe/harness.py
uv run python -m autovqe.harness check
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
git diff --check
```

For changes that touch spin-chain ansatz logic, also run:

```bash
uv run python -m autovqe.harness solve examples/tfim_n10_g1_open.json examples/heisenberg_n10_open.json --rel-tol 0.001 --max-stages 2
```

For changes that touch chemistry or reference-state logic, run the affected
chemistry fixture explicitly.

These extra fixtures are probes of general behavior. They are not an invitation
to tune against file names.

## Code Style

- Keep `autovqe/prepare.py` as the fixed evaluator unless the task explicitly changes
  the problem format or measurement logic.
- Put executable candidate circuits and optimizer choices in `autovqe/train.py`.
- Keep `autovqe/harness.py` factual: inspection, isolation, target checks, summaries.
- Put research rationale in `docs/` before turning it into policy.
- Prefer small, measured changes over broad ansatz rewrites.

## Pull Request Notes

Include:

- the problem file,
- best energy and reference energy,
- relative error,
- ansatz family,
- parameter count,
- two-qubit gate count,
- command used to verify the result.

Generated files such as `results.tsv`, `run.log`, `benchmark_runs/`, and
`solve_runs/` should not be committed.

## Community Standards

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). For vulnerability reports, use
[SECURITY.md](SECURITY.md) instead of public issues.
