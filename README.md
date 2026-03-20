# AutoVQE

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), AutoVQE is a automated optimization framework for variational quantum algorithms, focused on hardware-aware VQE. The goal is to provide a simple, reproducible, and extensible setup for systematically exploring ansatz design, optimization strategies, and hardware constraints in VQE research. The emphasis is on a minimal, single-file research script (`train.py`) that can be easily edited and iterated on, while keeping the rest of the repo fixed to ensure consistency and comparability across experiments.

The repo is built around one fixed harness in `prepare.py`, one editable research script in `train.py`, one problem file in `problem.json`, and as little extra structure as possible.

# How it works

Following the philosophy of [autoresearch](https://github.com/karpathy/autoresearch), AutoVQE is built around a very small and iterative workflow.

- `prepare.py` is the fixed harness. It loads and validates the problem from `problem.json`, constructs the Hamiltonian and backend target, computes any reference values if available, and provides shared evaluation and reporting utilities. This file is not part of the research surface.
- `train.py` is the main research script. This is where you implement and modify the VQE ansatz, initialization, optimization loop, and final summary output. You are encouraged to experiment here freely with different ansatz families, optimizers, and search strategies.
- `results.tsv` is a lightweight experiment log. After each run, append one row with the commit hash, final energy, single-qubit gate count, two-qubit gate count, total gate count, parameter count, run status, and a short description of the change.

Each run uses the same fixed 5-minute wall-clock optimization budget, excluding startup and preparation time. This keeps comparisons simple, fair, and iteration-friendly.

## Project Structure

```
prepare.py      — constants, problem prep, transpile + evaluation utilities (do not modify)
train.py        — ansatz, optimizer, and VQE loop (agent modifies this)
program.md      — agent instructions
problem.json    — fixed Hamiltonian
results.tsv     — experiment log
pyproject.toml  — dependencies
```

## Quick Start (Single baseline run)

```bash
uv sync
uv run prepare.py
uv run train.py
```

The expected baseline output from `train.py` is a plain summary block with one metric per line, so simple commands such as `uv run train.py | grep '^energy:'` stay usable.


## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

### For codex
```bash
Hi have a look at program.md and kick off a new experiment.

Do the setup first and then run the experiment loop continuously without stopping early.

- Append every run to results.tsv.
- Never stop even if your best energy is saturated. There are two options:
  1. If you didn't try a different ansatz, try a different ansatz.
  2. If you already tried a different ansatz, try more complex ansatz (increase depth, add more parameters, etc).
  3. If you already tried a more complex ansatz, consider this is a best energy saturation and try to reduce the ansatz complexity (decrease depth, remove parameters, remove gates randomly).
- Do not ask me anything.
- Do not give a final summary until you have completed at least 100 experiments and clearly exhausted both energy improvements and
compression improvements.

If branch operations are blocked by the environment, continue the experiment loop and note the blocker once.
```

## Design Principles

- **Simplicity**: Simple is better. Especially in the quantum domain, complicated design is often not better. Keep the structure as simple as possible.
- **Fixed time budget**: Use a fixed wall-clock time budget for optimization to keep comparisons fair and iteration-friendly. You can try 100 experiments while you sleep.
- **Self-contained**: No complex config.

# License

MIT