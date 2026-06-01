# Ansatz Playbook

Use this as a decision aid, not as law. The harness provides facts; the agent
chooses one experiment.

## Audit Checklist

Before editing `autovqe/train.py`, answer:

- What is the max Pauli locality?
- Is the Hamiltonian Z-only, TFIM-like, Heisenberg-like, chemistry-like, or
  mixed/general?
- Which Pauli terms share the same support?
- Does the support graph match the hardware coupling map?
- Are there obvious sectors: parity, Hamming weight, magnetization, spin, or
  particle number?
- Is there an `initial_state_hint`, and can it be prepared with explicit gates?

## First Candidate By Hamiltonian

| Hamiltonian evidence | First family | Why |
| --- | --- | --- |
| Z-only / QUBO | QAOA-style cost phase + mixer | The cost terms commute and a mixer supplies non-commuting motion. |
| TFIM-like `ZZ + X` | parity-preserving TFIM HVA / counterdiabatic edge pool | Natural split into ZZ cost and X field; `YZ/ZY` commutator moves can break QAOA plateaus without leaving parity. |
| matched `XX + YY` | exchange / XY blocks | Preserves magnetization when coefficients match. |
| matched `XX + YY + ZZ` | Heisenberg HVA | Edge-local evolution matches the Hamiltonian. |
| chemistry with HF metadata | HF reference + excitation-preserving/UCC-like pool | Particle/spin structure matters more than generic entanglers. |
| general Pauli | Pauli HVA / operator pool | The Hamiltonian itself is the safest source of operators. |
| no clear signal | shallow HEA baseline | Diagnostic only; do not overfit the search around it first. |

## Symmetry Rules

- If a symmetry is detected, first prepare a state inside the intended sector.
- Then use blocks that preserve that sector.
- Count all reference-prep gates.
- If a proposed block breaks a known symmetry, label it as a diagnostic baseline.

Examples:

- U(1) / magnetization: prefer `XX + YY` exchange blocks.
- Chemistry with fixed electron count: treat U(1) particle-number conservation
  as the first symmetry. In qubit terms, preserve Hamming weight after preparing
  the HF/reference determinant.
- SU(2)-like Heisenberg: prefer edge-local `XX + YY + ZZ` blocks and dimer/Neel
  references.
- TFIM global parity: avoid arbitrary single-qubit rotations as the first
  scientific explanation; use `ZZ`, `X`, and parity-preserving `YZ/ZY` edge
  moves first.

## Implementation Rules

- `exp(-i theta P)` must compile to explicit basis changes, CNOT parity ladder,
  and `RZ`.
- Do not use opaque unitaries or hidden exact diagonalization as ansatz energy.
- Do not accept a single shared-angle `exp(-i theta H)` as a VQE ansatz.
- Use independent term, edge, group, or excitation parameters when testing HVA
  style circuits.
- Independent Pauli-string rotations generally do not preserve chemistry U(1);
  use exchange or fermionic-excitation blocks when electron count matters.
- Prefer graph-colored edge layers before all-to-all entanglement.
- For non-chain weighted Heisenberg graphs, term-factorized edge parameters can
  matter more than deeper shared schedules.
- Keep one new idea per run so the ledger is interpretable.

## Research Ideas Worth Adding Later

These belong in `docs/` until they are backed by a passing experiment:

- true ADAPT-VQE operator selection by gradient or short smoke improvement,
- coefficient-aware TFIM schedules for nonuniform couplings,
- anisotropic `XXZ/XYZ` spin blocks,
- chemistry metadata generation for H6/BeH2,
- tapering/inactive-qubit reduction,
- parameter warm starts across related geometries.

## References

- Hamiltonian variational ansatz: https://arxiv.org/abs/1507.08969
- QAOA: https://arxiv.org/abs/1411.4028
- ADAPT-VQE: https://arxiv.org/abs/1812.11173
- Qubit-excitation ADAPT-VQE benchmark context: https://www.nature.com/articles/s42005-021-00730-0
