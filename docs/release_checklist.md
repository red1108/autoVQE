# Release Checklist

Use this checklist before tagging or publishing a release.

## Repository Hygiene

- `git status --short` contains only intentional source, docs, and fixture
  changes.
- No generated run directories are present in the repo root.
- No local agent configuration, virtual environments, caches, or result ledgers
  are tracked.
- `README.md`, `CONTRIBUTING.md`, and `LICENSE` are present.

## Verification

```bash
uv sync
uv run python -m py_compile prepare.py train.py harness.py
uv run harness.py check
uv run harness.py solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
uv run harness.py solve examples/tfim_n10_g1_open.json examples/heisenberg_n10_open.json --rel-tol 0.001 --max-stages 2
git diff --check
```

Optional long checks:

```bash
uv run harness.py solve examples/ising_1d_9q.json --rel-tol 0.001 --max-stages 2
uv run harness.py solve examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json --rel-tol 0.001 --max-stages 1
```

## Release Notes

Record:

- supported Python version,
- dependency versions from `uv.lock`,
- benchmark commands and pass/fail status,
- any benchmarks intentionally excluded from CI because of runtime,
- known limitations.
