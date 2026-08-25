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

Create `user_problem/hamiltonian.json`. The smallest valid document is:

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
- `symmetry`: encoding or sector metadata described in
  [the protocol](docs/agent_protocol.md).

Unknown or duplicate fields, complex or non-finite coefficients, inconsistent
Pauli widths, and a Hamiltonian with no non-identity terms after simplification
are rejected. A declared symmetry is context, not proof; the agent must probe
any charge it wants the evaluator to enforce.

When `initial_state_hint` is omitted, the evaluator prepares `|0>` on every
qubit.

Do not include a reference energy, ground state, optimized parameters, or an
answer hidden under another field.

## Run the research loop

Read the raw input and inspect its mechanically derived structure:

```bash
uv run autovqe inspect --problem user_problem/hamiltonian.json --json
uv run autovqe research init \
  --problem user_problem/hamiltonian.json \
  --run-dir .autovqe-runtime/research \
  --budget 100
```

Write one action at a time under `.autovqe-runtime/actions/`, then apply it:

```bash
uv run autovqe research step \
  --run-dir .autovqe-runtime/research \
  --action .autovqe-runtime/actions/001.json

uv run autovqe research status --run-dir .autovqe-runtime/research
```

Continue until the controller accepts `positive_commit` or `negative_close`.
Only then request the result:

```bash
uv run autovqe research result --run-dir .autovqe-runtime/research
```

The exact action schemas and lifecycle are in
[docs/agent_protocol.md](docs/agent_protocol.md). The physics-oriented
search guide is [docs/ansatz_playbook.md](docs/ansatz_playbook.md).

## Clean Codex prompt

Start Codex in a fresh clone or clean worktree containing only the public
repository and your `user_problem/hamiltonian.json`, then give it this goal:

```text
Solve the Hamiltonian in user_problem/hamiltonian.json with AutoVQE's closed
research loop. First read README.md, docs/agent_protocol.md, and
docs/ansatz_playbook.md, then read the raw Hamiltonian and run `uv run
autovqe inspect --json`.

Treat user_problem/hamiltonian.json as immutable. During discovery, do not edit
AutoVQE source, tests, or documentation, and do not read or edit
controller-owned run files or event history. Read state through the CLI and
write only action JSON under .autovqe-runtime/actions/. Route every probe, energy,
optimization, and resource measurement through `uv run autovqe research ...`;
do not import the evaluator or run another solver. Do not inspect bundled
examples, prior run directories, or files outside this worktree for answers.

Investigate falsifiable physical structures, preserve failed branches, and use
only typed AnsatzSpec candidates. Do not supply energies, optimized values, or
resource counts. Continue until the controller accepts positive_commit or a
grounded negative_close. Report the final ansatz and optimized binding only
from `research result`, and state that exact ground-state accuracy is unknown
when no independent reference score was provided.
```

## What acceptance means

A positive result means the selected candidate passed the fixed audit, smoke,
and promotion protocols and was not dominated by an independently promoted
structural comparator. A negative result closes only the investigated branches
after sufficient objective-active failure evidence.

Neither decision proves the exact ground-state energy or generalization to a
different Hamiltonian. AutoVQE reports that limitation explicitly.
