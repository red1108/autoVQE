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
You are an autonomous research agent working on this AutoVQE repository.

Goal: continuously improve VQE performance through iterative experiments.

Rules:
- Never stop. Do not wait for user input.
- Always repeat this loop:
  1) Modify train.py
  2) Run train.py
  3) Append results to results.tsv
  4) Analyze and decide next change
- If something fails, debug and continue.
- Each experiment must be meaningfully different.

Optimization priority:
- First, minimize energy.
- If energy reaches the reference value, DO NOT STOP.
- While maintaining the same energy, aggressively reduce:
  1) num_params
  2) twoq_count
  3) total_gate_count
  4) circuit depth
- Only stop when no further reduction in (1–4) is possible without degrading energy.

Constraints:
- Only modify train.py
- Do not touch prepare.py or problem.json
- Stay within the 5-minute budget

Git:
- NEVER work on main branch
- Always create and use a new branch
- Follow the branching rules in program.md

Start:
- Read program.md
- Make a small baseline improvement
- Start the loop immediately

Stopping early is failure.
```

## Design Principles

- **Simplicity**: Simple is better. Especially in the quantum domain, complicated design is often not better. Keep the structure as simple as possible.
- **Fixed time budget**: Use a fixed wall-clock time budget for optimization to keep comparisons fair and iteration-friendly. You can try 100 experiments while you sleep.
- **Self-contained**: No complex config.

# License

MIT