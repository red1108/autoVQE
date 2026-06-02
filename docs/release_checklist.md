# Release Checklist

Use this checklist before tagging or publishing a release.

## Repository Hygiene

- `git status --short` contains only intentional source, docs, and fixture
  changes.
- No generated run directories are present in the repo root.
- No local agent configuration, virtual environments, caches, or result ledgers
  are tracked.
- The visible repository root is limited to essentials such as `README.md`,
  `LICENSE`, `pyproject.toml`, `uv.lock`, `autovqe/`, `docs/`, `examples/`,
  and `.github/`.
- Community and release metadata live in `.github/` or `docs/`, not as loose
  root files.
- GitHub issue and pull request templates are present.

## Verification

```bash
uv sync
uv run python -m py_compile autovqe/prepare.py autovqe/train.py autovqe/harness.py
uv run python -m autovqe.harness check
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
git diff --check
```

Optional long checks:

```bash
uv run python -m autovqe.harness solve examples/tfim_n10_g1_open.json examples/heisenberg_n10_open.json --rel-tol 0.001 --max-stages 2
uv run python -m autovqe.harness solve examples/ising_1d_9q.json --rel-tol 0.001 --max-stages 2
uv run python -m autovqe.harness solve examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json --rel-tol 0.001 --max-stages 1
```

## Release Notes

Record:

- supported Python version,
- dependency versions from `uv.lock`,
- calibration commands and observed status,
- any larger regime probes left outside required CI because of runtime or run
  budget,
- known limitations.
