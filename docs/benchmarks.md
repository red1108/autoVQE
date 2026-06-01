# Calibration Fixtures

AutoVQE is not a benchmark-fitting repo. These fixtures exist to make ansatz
selection research falsifiable across different Hamiltonian regimes. Each
problem should have:

- a named Hamiltonian,
- explicit Pauli terms,
- a reference energy when available,
- fixed basis gates and coupling map,
- clear scoring criteria when a reference is available.

A fixture failure should lead to a better Hamiltonian-derived rule, operator
pool, symmetry constraint, initialization strategy, or optimizer schedule. It
should not lead to a problem-name special case.

## Small Regression Suite

These should stay fast and reliable so the harness and evaluator do not regress:

| Problem | Purpose |
| --- | --- |
| `examples/h2_2q.json` | sanity check for optimization and gate counting |
| `examples/h2_4q.json` | small chemistry-like Pauli Hamiltonian |
| `examples/ising_1d_5q.json` | small non-diagonal spin-chain target |

Expected command:

```bash
uv run python -m autovqe.harness solve examples/h2_2q.json examples/h2_4q.json examples/ising_1d_5q.json --rel-tol 0.001 --max-stages 2
```

## Spin-Chain Regime Probes

These probe structural cases that matter for a general ansatz-selection tool:

| Problem | Purpose |
| --- | --- |
| `examples/tfim_n10_g1_open.json` | non-commuting TFIM structure; cost/mixer and counterdiabatic policy |
| `examples/heisenberg_n10_open.json` | exchange structure; U(1)/SU(2)-motivated symmetry-preserving candidates |
| `examples/ising_1d_9q.json` | weighted support graph stress case |

They are intentionally useful when developing general Hamiltonian policies. Do
not convert them into "if this file, use that circuit" branches.

## Large Chemistry Regime Probes

These probe chemistry metadata, reference-state preparation, active-space
assumptions, and particle-number-preserving search:

| Problem | Purpose |
| --- | --- |
| `examples/h2_4q_pennylane_0p6614.json` | PennyLane H2/STO-3G fixture with supplied FCI reference |
| `examples/n2_16q_pennylane_sto3g_active14e8o_r2p07416.json` | 16-qubit N2/STO-3G active-space stress test |

The N2 fixture has 1281 raw Pauli terms and a sparse exact reference. Use it to
test chemistry metadata, symmetry-preserving references, U(1)-style sector
preservation, and scaling behavior. It is a stress probe, not the definition of
the project.

## Reporting Rule

When reporting a fixture with a reference energy, report the raw optimized VQE
circuit energy, ansatz family, parameter count, and compiled gate counts.
Classical refinement or exact diagonalization may be useful for analysis, but it
must be labeled separately and must not be reported as the VQE circuit result.

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
