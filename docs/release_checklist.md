# Release Checklist

Use this checklist before tagging or publishing AutoVQE.

## Repository hygiene

- `git status --short` contains only intentional source, documentation, test,
  and calibration-input changes.
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
uv sync --frozen
uv run python -m compileall -q autovqe tests
uv run python -m unittest discover -s tests -v
uv run autovqe check
uv build
git diff --check
```

Inspect the source distribution and confirm that `solve_runs/`,
`user_problem/`, `.autovqe-runtime/`, environments, and personal Codex files
are absent.

Confirm that the documented command surface is available:

```bash
uv run autovqe inspect --help
uv run autovqe research init --help
uv run autovqe research step --help
uv run autovqe research status --help
uv run autovqe research result --help
```

## Independent user test

The release test should resemble a future user's first interaction.

1. Create a clean worktree from the version under test in a separate directory.
2. Leave development-only changes and prior run directories in the development
   workspace.
3. Open that directory in a new Codex session with no development conversation
   or previous AutoVQE run context.
4. Place one unseen input at `user_problem/hamiltonian.json`. Do not place its
   reference energy or state in the worktree.
5. Ask Codex to use the repository workflow and continue until the controller
   accepts a terminal decision.
6. Do not provide hints about the expected symmetry, circuit family, gate
   sequence, or target energy during the run.
7. Review the final `research result` and branch history.
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
- The candidate contained only allowed variational operations; initial
  preparation came from evaluator-owned problem input.
- Audit, smoke, and promotion ran in order.
- Resource audit completed before optimizer calls.
- Failed and retired branches remain visible.
- A positive decision used a different-hypothesis competitor or control at the
  same promotion fidelity.
- The controller derived terminal evidence rather than accepting candidate
  measurements or a written non-dominance claim.
- Preterminal history contains no optimized parameter binding; `research
  result` reproduces it from the fixed committed promotion protocol.
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
