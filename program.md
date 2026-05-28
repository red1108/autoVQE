# AutoVQE Agent Protocol

This repo is an autoresearch-style VQE lab. The protocol is deliberately
general: do not map a problem name to a memorized ansatz. Inspect the operator,
extract constraints, propose a measured candidate, and keep only changes that
are proved by the harness.

## File Contract

- `prepare.py`: fixed evaluator. Do not edit it during ordinary research.
- `harness.py`: control plane. It inspects operators, runs isolated campaigns,
  checks targets, and reports evidence.
- `train.py`: experiment surface. Put candidate circuits, schedules, and
  optimizers here.
- `program.md`: operating protocol. Keep it method-agnostic and problem-agnostic.
- `docs/`: background material. It may inform a hypothesis, but it is not a
  substitute for inspecting the current operator.
- `examples/`: fixtures. Do not edit the active fixture during a run unless the
  task is explicitly to add or correct that fixture.

## First Commands

```bash
git status --short --branch
uv run harness.py inspect --problem <problem.json>
uv run harness.py check
```

Use `inspect` output as evidence. Problem names, file names, and prior results
are weak hints. The current Pauli terms, coefficients, locality, support graph,
metadata, reference state, hardware constraints, and target tolerance are the
authoritative inputs.

## Research Loop

1. State the invariant or structural fact the candidate is meant to exploit.
2. Build one candidate family from that fact.
3. Keep every state preparation step and tunable operation explicit in the
   circuit.
4. Run a bounded smoke campaign.
5. Promote only if the ledger shows a real energy or resource improvement.
6. Run target `solve` before reporting success.

Useful commands:

```bash
uv run harness.py campaign --mode smoke --experiments 8 --experiment-seconds 2 --max-evals 60
uv run harness.py solve <problem.json> --rel-tol <target>
```

## General Rules

- Preserve any detected invariant unless deliberately running a baseline.
- If a reference state is used, prepare it with gates inside the circuit.
- A candidate is not valid just because it reaches the target; it must expose a
  real variational search space and respect the stated constraints.
- Do not collapse a Hamiltonian-derived circuit into one global scalar knob and
  call it a solved ansatz.
- Do not hardcode learned angles, exact eigenstates, or reference energies.
- Do not report classical post-processing as raw circuit energy.
- Prefer changes that reduce error without hiding cost. When energies are
  equivalent, prefer fewer two-qubit gates, then fewer total gates, lower depth,
  and fewer parameters.
- Keep generated files such as `results.tsv`, `run.log`, `benchmark_*`, and
  `solve_runs*` out of git.

## Candidate Design

Do not write "if this benchmark, use that ansatz" rules here. Instead, derive
candidate families from general operator facts:

- conserved quantities imply sector-preserving moves,
- commuting structure implies grouped evolution or scheduled phases,
- non-commuting support implies layered or adaptive operator pools,
- hardware constraints imply routing-aware connectivity,
- metadata implies explicit reference preparation and constrained search.

If none of those facts are strong, run a small diagnostic baseline and treat it
as a control, not as the explanation.

## Validation

Before reporting success, run checks that match the scope of the change:

```bash
uv run python -m py_compile harness.py train.py prepare.py
uv run harness.py check
uv run harness.py solve <problem.json> --rel-tol <target>
git diff --check
```

For broad changes, run every affected fixture explicitly. For a single target,
the target solve plus the fixed checks are sufficient. In the final report,
include the command, best energy, reference energy, gap, tolerance, ansatz
family, parameter count, and gate counts.
