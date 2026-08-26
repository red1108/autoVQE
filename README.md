# AutoVQE

AutoVQE lets a research agent discover and optimize a useful VQE ansatz for a
given Pauli Hamiltonian. The agent reads the problem, edits one transparent
circuit, runs a fixed-time experiment, learns from the result, and repeats.

The repository deliberately has only two Python files:

- `ansatz.py` is the agent's laboratory notebook and the only code it edits.
- `evaluate.py` is the fixed experiment: optimization, energy, and honest
  native-circuit resource accounting.

The eight Hamiltonians in `examples/` are demonstrations, not hidden answers.

## Run

Install Python 3.10+ and [`uv`](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run python evaluate.py examples/h2_4q_bond_70pm.json --hypothesis "baseline"
```

Each evaluation appends one compact comparison row to ignored `results.tsv`.
The evaluator also keeps ignored optimizer state so unchanged parameters can
continue from values that it previously found; all remain variational.

The default optimization budget is `max(15, 60 * 2 ** (n - 16))` seconds, where
`n` is the Hamiltonian width. It is fixed for every candidate in that problem.
Use `--seconds` only to override it. If a problem contains `reference_energy`,
the optional target is a relative energy error (0.01% by default):

```bash
uv run python evaluate.py examples/n2_16q_bond_110pm.json \
  --target-relative-error 0.0001 --hypothesis "number-preserving layer"
```

Without a reference, AutoVQE reports the best energy it actually found; it
does not claim that it found the ground state.

## Run the research loop with Codex

Start a fresh Codex task in this repository and give it a problem path and a
total research budget. The evaluator derives the per-experiment time:

```text
Read program.md and optimize examples/h2_4q_bond_70pm.json. Use the evaluator's
default time for every comparison and spend at most 30 minutes on the whole
search. Leave the best ansatz in ansatz.py and report its final evaluator output.
```

Replace the path and budgets as needed. `program.md` defines the closed
research loop and its anti-cheating boundary.

## Problem and ansatz formats

A problem is JSON with `pauli_terms` and optional `initial_state_hint`,
`basis_gates`, `coupling_map`, and `reference_energy`. Pauli labels use Qiskit
ordering: the rightmost letter acts on qubit 0. See `examples/` for complete
inputs.

An ansatz operation in `ansatz.py` is:

```python
("YX", (0, 1), "theta", 1.0)
("U1", (0, 1), "exchange", 1.0)  # XX + YY
("GIVENS", (0, 1), "mix", 1.0)   # (YX - XY) / 2
("SU2", (2, 3), "spin", 1.0)     # XX + YY + ZZ
```

This applies `Y` to qubit 0 and `X` to qubit 1. The last value may be `-1`,
`-0.5`, `0.5`, or `1`. Reusing a parameter name intentionally shares it.
`ansatz.py` is data-only: imports, functions, and executable expressions are
rejected. Every operation is expanded into one-qubit basis changes, `RZ`, and
`CX` gates before resources are counted. Macros are charged for every Pauli
component, so shared parameters do not hide their cost. There are no opaque
custom unitaries.
