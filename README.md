# AutoVQE

AutoVQE is a small closed-loop harness for turning a Pauli Hamiltonian into a
tested variational ansatz. A research agent reads the Hamiltonian, proposes
physical hypotheses and typed circuits, and learns from fixed evaluator runs.
The harness alone computes energies, optimized parameters, and resources.

## Install

Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
git clone https://github.com/red1108/autoVQE.git
cd autoVQE
uv sync
uv run autovqe check
```

## Provide a Hamiltonian

AutoVQE accepts a Hamiltonian JSON at any path; no special input folder is
required. Save the file wherever convenient and pass that exact path to Codex
or `--problem`. The smallest valid document is:

```json
{
  "name": "my_problem",
  "pauli_terms": [
    {"pauli": "IZ", "coeff": -1.0},
    {"pauli": "ZI", "coeff": -1.0},
    {"pauli": "XX", "coeff": 0.2}
  ]
}
```

Every Pauli label must have the same nonzero length and may contain only
`I`, `X`, `Y`, and `Z`. Coefficients must be finite real numbers. Qiskit
little-endian ordering is used: the rightmost letter acts on qubit 0.

The only optional top-level fields are:

- `basis_gates`: nonempty gate-name strings; defaults to `rx, ry, rz, cx`.
- `coupling_map`: directed pairs of distinct in-range qubit indices.
- `initial_state_hint`: one integer `0` or `1` per qubit.
- `source_note`: nonempty provenance text.
- `symmetry`: optional encoding or sector metadata. Recognized keys are
  `mapping`, `basis`, `orbital_order`, `spin_order`, `spin_orbitals`,
  `active_orbitals`, `active_electrons`, `particle_number`, `magnetization`,
  `spin_projection`, `total_spin`, and `parity`.

Unknown or duplicate fields, complex or non-finite coefficients, inconsistent
Pauli widths, and a Hamiltonian with no non-identity terms after simplification
are rejected. A declared symmetry is context, not proof; the agent must probe
any charge it wants the evaluator to enforce.

When `initial_state_hint` is omitted, the evaluator prepares `|0>` on every
qubit.

Do not include a reference energy, ground state, optimized parameters, or an
answer hidden under another field.

The repository includes eight ready-to-run Hamiltonians under `examples/`.

## Solve with Codex

Start Codex in the repository. This prompt runs the included four-qubit H₂
example:

```text
Read program.md. The Hamiltonian is examples/h2_4q_0p70a.json. Use AutoVQE's
closed research loop to analyze it and discover and optimize an ansatz.
Continue until the controller accepts a terminal decision, then report only
the result returned by the CLI.
```

For another problem, replace the example path with the path to its JSON file.

Codex reads the Hamiltonian, creates the action files, runs the fixed
evaluations, and iterates until the controller accepts a result. The complete
research protocol and physics search guide are in [program.md](program.md).

## What acceptance means

A positive result means the selected candidate passed the fixed audit, smoke,
and promotion protocols and was not dominated by an independently promoted
structural comparator. A negative result closes only the investigated branches
after sufficient objective-active failure evidence.

Neither decision proves the exact ground-state energy or generalization to a
different Hamiltonian. AutoVQE reports that limitation explicitly.

## Manual research commands

The commands below are optional; they are mainly useful for inspecting or
debugging the loop that Codex normally drives:

```bash
uv run autovqe inspect --problem examples/h2_4q_0p70a.json --json
uv run autovqe research init \
  --problem examples/h2_4q_0p70a.json \
  --run-dir .autovqe-runtime/research \
  --budget 100

uv run autovqe research step \
  --run-dir .autovqe-runtime/research \
  --action .autovqe-runtime/actions/001.json

uv run autovqe research status --run-dir .autovqe-runtime/research
uv run autovqe research result --run-dir .autovqe-runtime/research
```
