# Release Checklist

Use this checklist before tagging or publishing a release.

## Repository Hygiene

- `git status --short` contains only intentional source, docs, and fixture
  changes.
- The exact release candidate is committed. Record its commit, tree, source
  archive, and committed `uv.lock` SHA-256 before starting an evaluation.
- No generated run directories are present in the repo root.
- No local agent configuration, virtual environments, caches, or result ledgers
  are tracked.
- `.autovqe-runtime/` remains ignored local state; exact-state and holdout truth
  files live outside the checkout entirely.
- The visible repository root is limited to essentials such as `README.md`,
  `AGENTS.md`, `LICENSE`, `pyproject.toml`, `uv.lock`, `autovqe/`, `docs/`,
  `examples/`, and `.github/`.
- Community and release metadata live in `.github/` or `docs/`, not as loose
  root files.
- GitHub issue and pull request templates are present.

## Verification

```bash
uv sync
uv run python -m compileall -q autovqe meta_agent tests
uv run python -m unittest discover -s tests -v
uv run python -m autovqe.harness check
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
git diff --check
```

## Independent user test

- Treat existing examples and every Hamiltonian inspected during development as
  calibration canaries, never as release holdouts.
- Commit the release candidate, clone that exact revision into a separate
  workspace, and start a new Codex session there.
- Give that session only the public repository and a holdout Hamiltonian. Keep
  development conversations, prior runs, and the reference answer outside it.
- Run scientific discovery separately from an agent-only bundle under a
  distinct OS identity. Verify that identity cannot read the source, raw
  problem, private truth, evaluator, key, anchor, or previous trials.
- Keep public-clone self-reports untrusted. Score the committed ansatz and
  evaluator-owned optimized binding outside the participant workspace.
- If a holdout result influences a harness change, retire that problem into the
  development set and replace it before making a release claim.
- Do not call `positive_commit` ground-state success: require the separately
  preregistered private scorer and aggregate trial report.

See [`result_artifact.md`](result_artifact.md) for the terminal artifact
boundary.

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
- known limitations;
- separate user-test and sealed-science results, including all started trials
  and infrastructure failures;
- external scorer and aggregate artifact hashes, plus any preregistration
  deviations.
