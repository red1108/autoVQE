# AutoVQE Agent Protocol

This repo is an experiment in letting an agent do disciplined VQE research. The agent should not guess an ansatz from the problem name. It must inspect the Hamiltonian, choose ansatz families that match the Hamiltonian structure, run controlled experiments, analyze the results, and then improve or compress the circuit.

The style should stay small and direct. Prefer one clear script and measurable experiments over a framework.

The harness owns the research loop. `train.py` proposes and evaluates candidate circuits; `harness.py` decides whether the evidence is good enough, whether to escalate, and whether the target has actually been met. Do not report success from a best-looking row until the harness has compared it against the requested tolerance.

## Files

- `prepare.py` is the fixed evaluator. Do not edit it.
- The active problem file is fixed during a run. Do not edit it while measuring candidates.
- `harness.py` is the research control tool. Use it to inspect the Hamiltonian, run controlled campaigns, summarize results, and recommend the next action.
- `train.py` is the research surface. Modify ansatzes, optimizers, candidate schedules, and logging here.
- `results.tsv` is the experiment ledger. It is ignored by git and should be append-only during a run.
- `run.log` is the latest captured training log.

## Setup

1. Read `README.md`, `program.md`, `prepare.py`, `harness.py`, `train.py`, and the active problem file.
2. Check the worktree with `git status --short --branch`.
3. Verify the environment:

```bash
uv run prepare.py
uv run harness.py inspect
uv run harness.py campaign --mode smoke --dry-run
uv run harness.py check
uv run harness.py results
```

4. If starting a new branch is possible, create `autovqe/<tag>` from the current base branch. If branch operations are blocked, continue and mention the blocker once.
5. Do not ask for confirmation once the user has asked the experiment loop to begin.

## Hamiltonian Audit First

Before changing `train.py`, run:

```bash
uv run harness.py inspect
uv run harness.py plan
```

Use the reported `model_class`, support graph, locality counts, Pauli pattern, and recommendations as the starting point. The Hamiltonian name is weak evidence. The Pauli structure is authoritative.

The audit should answer:

- Is the Hamiltonian Z-diagonal, TFIM-like, XX/XY/XXZ/Heisenberg-like, chemistry-like, or general Pauli?
- Is the support graph sparse, bipartite, or hardware-aligned?
- Are coefficients tied across `XX`, `YY`, and `ZZ` on the same support?
- Are there obvious conserved sectors such as parity, excitation number, magnetization, or spin-like symmetry?
- Which ansatz family should be tried first, and which family should only be used as a baseline?

## Ansatz Selection Policy

Use this decision table unless the audit or experiment evidence contradicts it.

| Hamiltonian class | First candidates | Baseline/fallback |
| --- | --- | --- |
| Z-only / QUBO / classical Ising | QAOA-style cost evolution plus mixer, commuting Z group phase layers | shallow HEA |
| TFIM-like `ZZ + X` | grouped TFIM HVA, QAOA-like cost/mixer | shallow HEA |
| `XX + YY` spin graph | exchange/XY layers, magnetization-preserving pools, qubit-ADAPT edge pool | shallow HEA |
| matched `XX + YY + ZZ` graph | Heisenberg/exchange HVA, edge-color HVA, Neel/reference-prep variants if graph supports it | shallow HEA |
| fermionic chemistry with metadata | HF reference, UCC/UpCCGSD, excitation-preserving or ADAPT-style pools | commuting-group HVA |
| general Pauli | commuting-group HVA, Hamiltonian Pauli pool, qubit-ADAPT-style growth | shallow HEA |

Generic hardware-efficient ansatz is a control, not a scientific explanation. Do not let a deep generic HEA be the only path explored before Hamiltonian-derived candidates have been tested.

Concrete audit rules:

- For TFIM-like `ZZ + X`, inspect the sign of the X-field. A positive `+X` field has `|->` as the field-ground reference; a negative `-X` field has `|+>`. Count that preparation with explicit gates.
- For chemistry-like Hamiltonians with `initial_state_hint`, treat the hint as a Hartree-Fock-style reference only if it is prepared with explicit `x` gates. Then inspect X/Y Pauli supports to propose excitation or UCC-like mixers before a generic HEA.
- For Pauli-evolution ansatzes, remember that Qiskit Pauli labels are big-endian: the rightmost label character acts on qubit 0. Ansatz builders and Hamiltonian audits must use the same convention.
- If a Hamiltonian-derived ansatz stalls above the target tolerance, try a shallow real-amplitudes/HEA baseline as a diagnostic, then return to the structured family or add a targeted operator-pool candidate.

## Initial State Rule

The underlying device state starts from all zeros. Problem-aware reference preparation is allowed only if it is implemented as explicit gates inside `train.py`, counted in the compiled gate metrics, and described in `results.tsv`.

Do not hide a tuned state in constants or uncounted initialization.

## Experiment Loop

For each research cycle:

1. Inspect the Hamiltonian with `harness.py`.
2. Pick one ansatz idea tied to the audit evidence.
3. Modify only `train.py` unless the requested task is to improve the harness or protocol.
4. Run a smoke tournament for candidate families before committing to a full run:

```bash
AUTOVQE_MAX_EXPERIMENTS=6 AUTOVQE_EXPERIMENT_SECONDS=2 AUTOVQE_MAX_EVALS=40 uv run harness.py run --timeout 90
uv run harness.py results
```

The cleaner default is:

```bash
uv run harness.py campaign --mode smoke --experiments 6 --experiment-seconds 2 --max-evals 40 --timeout 90
```

5. Promote promising candidates to normal comparison using the fixed per-experiment budget.
6. Append every run to `results.tsv`.
7. Keep changes that improve energy. If energy is effectively tied, keep only if the circuit improves in this priority order:
   `twoq_count`, `total_gate_count`, `depth`, `num_params`.
8. If energy saturates, stay inside the best Hamiltonian-derived family and compress before returning to broader search.
9. If all structured candidates fail, broaden the operator pool before trying a deeper generic ansatz.

## Time Control

Full comparisons use the fixed per-experiment budget:

```text
2^(n_qubits - 2) seconds
```

The wrapper timeout should cover the whole run:

```bash
uv run harness.py run --timeout <seconds>
```

For campaign-style runs, prefer:

```bash
uv run harness.py campaign --mode smoke --experiments 8
uv run harness.py campaign --mode full --experiments 12
```

To check whether the harness works across the bundled example Hamiltonians, run:

```bash
uv run harness.py benchmark --experiments 45 --experiment-seconds 2 --max-evals 300 --timeout 120
```

The following environment variables are allowed for smoke tests and development only:

- `AUTOVQE_MAX_EXPERIMENTS`: stop `train.py` after this many experiments.
- `AUTOVQE_EXPERIMENT_SECONDS`: override per-experiment optimization seconds.
- `AUTOVQE_MAX_EVALS`: override objective evaluations per experiment.
- `AUTOVQE_MIN_EXPERIMENTS`: override the minimum run count for a controlled campaign.
- `AUTOVQE_EXHAUSTION_PATIENCE`: override stagnation patience.
- `AUTOVQE_TARGET_REL_ERROR`: target relative error versus `reference_energy`.
- `AUTOVQE_TARGET_ABS_ERROR`: target absolute error floor.
- `AUTOVQE_STOP_AT_TARGET`: stop a training stage once the target is met.
- `AUTOVQE_TARGET_EXTRA_COMPRESS`: optional extra experiments after target hit for compression.

Do not report smoke-test results as full-budget results. Mark them clearly in the description if they are logged.

## Solve Mode

When the user gives a concrete accuracy target, use `solve`, not a one-off benchmark:

```bash
uv run harness.py solve examples/h2_2q.json --rel-tol 0.001
uv run harness.py solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001
```

`solve` runs an escalating loop:

1. Audit the Hamiltonian and choose the model class from Pauli structure.
2. Run a smoke stage with isolated results/log files.
3. Compute `abs(best_energy - reference_energy)` and compare it with `max(abs_tol, rel_tol * abs(reference_energy))`.
4. Stop immediately if the target is proved.
5. If not proved, escalate to standard and then deep stages with larger experiment/evaluation budgets.
6. Return nonzero if no stage proves the target or if no reference is available.

This is the Karpathy-style version of the loop: keep the system small, make the objective executable, print the evidence, and let the next action be determined by measured failure instead of by narration.

## Output Format

`train.py` should end with a parseable summary block:

```text
---
energy:           -1.234567
reference_energy: -1.300000
overlap:          0.900000
singleq_count:    40
twoq_count:       24
total_gate_count: 64
depth:            39
num_params:       16
eval_calls:       143
total_seconds:    21.4
```

If a metric is unavailable, omit the line.

## Result Ledger

`results.tsv` has exactly these columns:

```text
commit	energy	singleq_count	twoq_count	total_gate_count	num_params	status	description
```

`description` must include the ansatz family and the Hamiltonian reason, not just hyperparameters. For example:

```text
heisenberg_hva layers=2 shared edge-color | reason=matched XX/YY/ZZ supports
```

Statuses:

- `keep`: improved energy, or near-tied energy with a simpler circuit.
- `discard`: completed but did not beat the incumbent.
- `crash`: implementation or idea failed.

## Research Discipline

- Every tunable rotation must be represented by an explicit optimization parameter and counted in `num_params`.
- Tying parameters is allowed when it reflects a real simplification, symmetry, or shared schedule.
- Hardcoding learned angles is not allowed.
- Do not modify `prepare.energy_from_circuit`.
- Do not change the active problem file during a run.
- Prefer short, named functions over a pile of special cases.
- Delete dead experiment branches in `train.py` once evidence shows they are not useful.
- A pretty circuit that does not beat the ledger is not progress.
