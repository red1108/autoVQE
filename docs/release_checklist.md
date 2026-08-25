# Release Checklist

Use this checklist before tagging or publishing AutoVQE.

## Repository hygiene

- `git status --short` contains only intentional source, documentation, test,
  and calibration-input changes.
- The release candidate is committed before evaluation begins.
- Generated run directories, action scratch files, caches, environments, and
  local logs are not tracked.
- `.autovqe-runtime/` remains ignored.
- Unseen Hamiltonians and their reference answers remain outside the committed
  repository.
- The repository root contains only project essentials such as `README.md`,
  `AGENTS.md`, `LICENSE`, `pyproject.toml`, `uv.lock`, `autovqe/`, `tests/`,
  `docs/`, `examples/`, and `.github/`.
- No machine-specific absolute paths, credentials, private answers, or prior
  research outputs appear in tracked files.

## Required checks

```bash
uv sync
uv run python -m compileall -q autovqe tests
uv run python -m unittest discover -s tests -v
uv run python -m autovqe.harness check
git diff --check
```

Confirm that the documented command surface is available:

```bash
uv run python -m autovqe.harness inspect --help
uv run python -m autovqe.harness research init --help
uv run python -m autovqe.harness research step --help
uv run python -m autovqe.harness research status --help
uv run python -m autovqe.harness research result --help
```

## Independent user test

The release test should resemble a future user's first interaction.

1. Commit the release candidate.
2. Create a detached worktree at that exact revision in a separate directory.
3. Open that directory in a new Codex session with no development conversation
   or previous AutoVQE run context.
4. Place one unseen input at `user_problem/hamiltonian.json`. Do not place its
   reference energy or state in the worktree.
5. Ask Codex to use the repository workflow and continue until the controller
   accepts a terminal decision.
6. Do not provide hints about the expected symmetry, circuit family, gate
   sequence, or target energy during the run.
7. Save the final `research result` output and the full branch history.
8. Only after the run ends, compare the returned circuit result with the
   reference answer in a separate evaluation step.

Files in `examples/` are calibration inputs and must not be treated as unseen
tests. If an unseen result causes an implementation or policy change, move that
problem into the calibration set and choose a new one for the next independent
test.

## Review the scientific result

- The input Hamiltonian was not modified.
- Every candidate was valid `AnsatzSpec` data.
- Candidate metadata did not become authoritative energy, optimized angles, or
  resource counts.
- Symmetry-oriented gates were admitted only after the relevant physical test.
- Audit, smoke, and promotion ran in order.
- Failed and retired branches remain visible.
- The terminal decision cites the evidence required by the controller.
- A positive decision is described as a local promotion, not as exact accuracy
  or performance on new problems.
- Any reference comparison happened after the research run and is reported
  separately.

## Release notes

Record:

- supported Python and dependency versions;
- required-check results;
- calibration problems exercised;
- all independent user-test attempts, including negative decisions and
  infrastructure failures;
- measured energy and resource limitations;
- known unsupported physics structures or hardware constraints;
- any change to the action protocol, gate allowlist, optimizer policy, or
  promotion rule.
