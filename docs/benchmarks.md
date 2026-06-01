# Benchmarks

Benchmarks exist to make agent changes falsifiable. Each problem should have:

- a named Hamiltonian,
- explicit Pauli terms,
- a reference energy when available,
- fixed basis gates and coupling map,
- clear pass criteria.

## Small CI Suite

These should stay fast and reliable:

| Problem | Purpose |
| --- | --- |
| `examples/h2_2q.json` | sanity check for optimization and gate counting |
| `examples/h2_4q.json` | small chemistry-like Pauli Hamiltonian |
| `examples/ising_1d_5q.json` | small non-diagonal spin-chain target |

Expected command:

```bash
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
```

## Hard Spin-Chain Targets

These are development targets, not default CI:

| Problem | Purpose |
| --- | --- |
| `examples/tfim_n10_g1_open.json` | critical TFIM-style non-diagonal spin chain |
| `examples/heisenberg_n10_open.json` | symmetry-aware Heisenberg chain |
| `examples/ising_1d_9q.json` | weighted Heisenberg graph stress target |

These targets must be reported with raw circuit VQE energy. Classical
post-processing may be studied separately, but it is not a benchmark pass.

## Large Chemistry Targets

These are explicit targets, not default benchmark entries:

| Problem | Purpose |
| --- | --- |
| `examples/h2_4q_pennylane_0p6614.json` | PennyLane H2/STO-3G fixture with supplied FCI reference |
| `examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json` | 16-qubit N2/STO-3G active-space stress test |

The N2 fixture has 1281 raw Pauli terms and a sparse exact reference. Use it to
test chemistry metadata, symmetry-preserving references, and scaling behavior.
Do not include it in default CI or broad smoke benchmarks unless the run budget
is explicit.

## Chemistry Targets To Add Later

Do not paste literature energies into problem JSON by hand. Generate the
Hamiltonian and reference from one reproducible pipeline.

Recommended order:

1. linear H6, STO-3G, R=1.5 A
2. linear H6, STO-3G, R=3.0 A
3. linear BeH2, STO-3G

Store enough metadata to make reference energies meaningful:

- geometry,
- basis,
- frozen-core/active-space decision,
- mapping convention,
- reference-energy generator and version.
