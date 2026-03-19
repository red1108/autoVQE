# AGENTS.md

## Mission
Build **AutoVQE** as the smallest useful repo for hardware-aware VQE research automation.

The repo should feel like `autoresearch`, but for VQE:
- a **fixed harness**,
- a **single editable research script**,
- a **human-written instruction file**,
- a **plain experiment log**,
- and as little extra structure as possible.

## Non-negotiable philosophy
- **Keep the repo tiny.** Prefer editing an existing file to creating a new one.
- **Prefer plain Python over frameworks.** No config zoo, no registries, no plugin systems, no dependency injection.
- **Prefer one working file over many elegant files.** Small, direct, and grep-friendly beats abstract.
- **English only.** All code, comments, docs, commit messages, prompts, log descriptions, and filenames must be in English.
- **Qiskit first.** Use Qiskit as the execution backbone for v0.1.
- **One problem at a time.** AutoVQE v0.1 always works on a single fixed Hamiltonian stored in `problem.json`.
- **Do not add dependencies casually.** Every new dependency must clearly remove more complexity than it adds.
- **When in doubt, delete code.** Simpler with equal behavior is a win.

## The files that matter
The repo should revolve around these files:

- `README.md` — project overview, quick start, repo layout, how to run the agent.
- `prepare.py` — the fixed harness. Loads the problem, builds the Hamiltonian, provides evaluation utilities, computes transpile stats, and prints validation summaries. **Do not modify this during research runs.**
- `train.py` — the single research script. This is the only file experiment agents are allowed to edit during autonomous research.
- `program.md` — human-written instructions for experiment agents.
- `problem.json` — the current Hamiltonian and backend description.
- `results.tsv` — untracked experiment log.
- `pyproject.toml` / `uv.lock` — dependencies and environment lock.

If a new file does not make one of these files simpler or more stable, it probably should not exist.

## AutoVQE v0.1 scope
This is the intentionally narrow scope for the first usable version.

### Inputs
- `problem.json` only.
- Required: `pauli_terms`.
- Optional: `reference_energy`, `symmetry`, `basis_gates`, `coupling_map`, `initial_state_hint`, `name`.

### Objective
Minimize the variational energy

`E(theta) = <psi(theta)| H |psi(theta)>`

where `H` is the fixed Hamiltonian from `problem.json`.

### Primary metric
- `energy` — lower is better.

### Optional benchmark metadata
If available, also report:
- `reference_energy`
- `delta_e = energy - reference_energy`
- `overlap`

These are optional. The system must still work when the true ground-state energy is unknown.

### Soft constraints
Track but do not optimize as the only goal:
- compiled two-qubit gate count
- compiled depth
- number of objective evaluations
- total wall-clock time

### Supported ansatz families in v0.1
Keep it minimal:
- `hea` — hardware-efficient ansatz
- `symm` — symmetry-preserving ansatz
- `qaoa` — for diagonal / Ising-like problems

Do not add adaptive ansatz families in v0.1 unless the simpler version is already solid.

### Execution model
- Local simulation first.
- Hardware awareness comes from **transpilation** and post-transpile resource metrics.
- Real hardware execution is out of scope for the initial version.

## Architecture constraints
- `prepare.py` owns:
  - loading and validating `problem.json`
  - canonicalizing the Hamiltonian
  - optional exact reference computation for small systems
  - backend target construction
  - transpilation helpers
  - energy evaluation helpers
  - summary formatting helpers
- `train.py` owns:
  - `VQEConfig`
  - ansatz builders
  - initialization strategy
  - optimizer loop
  - calling the fixed harness in `prepare.py`
  - the final summary block
- Keep `train.py` self-contained. Tiny helpers can stay inline.
- Avoid creating packages or deep module trees in v0.1.
- Do not introduce YAML/TOML config systems beyond `pyproject.toml` and `problem.json`.
- Avoid CLI flag sprawl. Prefer constants near the bottom of `train.py`.

## Implementation standards
- Prefer dataclasses over class hierarchies.
- Prefer direct functions over base classes.
- Prefer readable loops over generic abstractions.
- Every script must be runnable with `uv run ...`.
- `uv run prepare.py` should validate the problem and print a concise summary.
- `uv run train.py` should run one complete baseline experiment and print a fixed summary block.
- Make final summary lines easy to parse with `grep`.
- A run should fail loudly and simply. Do not hide errors behind layers of wrappers.

## Experiment-mode rules
Once the repo exists and autonomous research begins, the rules get tighter:

- Only edit `train.py`.
- Do not modify `prepare.py`.
- Do not modify `problem.json`.
- Do not add dependencies.
- Baseline first.
- Log every run to `results.tsv`.
- If a change lowers energy, keep it.
- If it is worse or equal, reset it, unless the code becomes clearly simpler at no performance cost.

## Output shape
The final summary from `train.py` should always look like a plain block with one metric per line, e.g.

```text
---
energy:           -1.234567
reference_energy: -1.235100
delta_e:          0.000533
twoq_count:       24
depth:            39
num_params:       16
eval_calls:       143
total_seconds:    21.4
```

If a metric is unavailable, omit the line instead of inventing placeholders.

## Definition of done for v0.1
A good first release has all of the following:
- `problem.json` example included
- `uv sync` works
- `uv run prepare.py` works
- `uv run train.py` works
- baseline uses one of the supported ansatz families and prints the fixed summary block
- `program.md` exists and explains the keep/discard experiment loop
- `results.tsv` format is documented
- the repo still feels small

## What not to build yet
Do **not** start with:
- a benchmark suite framework
- a web UI
- a database-backed service
- a plugin architecture
- multiple backend providers
- heavy adapter layers
- automatic paper reading
- multi-agent orchestration
- real hardware runtime integrations
- distributed execution

If it sounds “platform-like”, it is probably too much for v0.1.
