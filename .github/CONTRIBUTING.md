# Contributing

AutoVQE is a Hamiltonian-to-ansatz research tool. Contributions should derive
candidate behavior from the supplied Hamiltonian, make each claim measurable,
and keep energy and resource values evaluator-owned.

## Setup

```bash
uv sync
uv run python -m autovqe.harness check
```

## Before changing code

1. Inspect the relevant Hamiltonian.

   ```bash
   uv run python -m autovqe.harness inspect --problem <problem.json>
   ```

2. Read the action protocol and ansatz playbook.
3. State the physical or workflow invariant the change should preserve.
4. Identify tests that distinguish the intended behavior from misuse.

Do not add file-name special cases or fixed answers. Candidate policy should
follow observable facts such as locality, support graph, commuting structure,
coefficient scale, reference occupation, hardware connectivity, and conserved
sectors.

## Required checks

Run these before opening a pull request:

```bash
uv run python -m compileall -q autovqe tests
uv run python -m unittest discover -s tests -v
uv run python -m autovqe.harness check
git diff --check
```

Run focused unit tests for any changed compiler, operation, probe, evaluator,
controller, lifecycle, or CLI behavior. Larger Hamiltonians may be exercised
as optional calibration inputs when the change affects that physical regime.

## Design guidelines

- Keep Hamiltonian parsing and public observations mechanical.
- Represent circuits with `AnsatzSpec`; do not accept opaque executable code.
- Add an operation only with a physical generator, identity-at-zero behavior,
  applicability rules, resource accounting, limits, and misuse tests.
- Keep optimizer policy and reported measurements outside candidate metadata.
- Count parameter occurrences as well as unique parameter names.
- Check resource use at generic nonzero bindings when numeric simplification
  could hide cost.
- Preserve failed and retired research branches.
- Prefer one measured policy change over a broad circuit rewrite.

## Pull request notes

Include:

- the scientific or workflow problem being addressed;
- the observable rule used by the implementation;
- affected action, candidate, or evaluation behavior;
- evaluator-derived energy and resource results when relevant;
- tests run;
- limitations and unsupported regimes.

Do not commit `.autovqe-runtime/`, local action scratch files, virtual
environments, caches, or private reference answers.

## Community standards

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). For vulnerability reports, use
[SECURITY.md](SECURITY.md) instead of public issues.
