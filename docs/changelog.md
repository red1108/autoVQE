# Changelog

All notable changes to AutoVQE are recorded here.

The format follows the spirit of Keep a Changelog, and the project uses
semantic versioning while the public API is still small.

## [0.1.0] - 2026-05-31

### Added

- Public CLI for Hamiltonian inspection, benchmark campaigns, and target-driven
  solving.
- JSON problem fixtures for H2, TFIM, Heisenberg, weighted spin graphs, and a
  16-qubit N2 stress test.
- Hamiltonian-aware candidate families:
  - U(1) exchange layers,
  - SU(2)-style Heisenberg HVA,
  - TFIM counterdiabatic schedules,
  - Pauli term-evolution HVA,
  - shallow HEA baselines.
- Documentation for ansatz selection, benchmarks, releases, and contribution
  workflow.
- CI checks for compile, harness self-check, and the small solve suite.

### Changed

- Benchmark pass criteria now use raw circuit VQE energy only.
- Single shared-angle Hamiltonian evolution is treated as ineligible.

### Verified

- Small suite: H2 2q, H2 4q, Ising 5q.
- Hard spin-chain targets: TFIM n10, Heisenberg n10.
- Optional long targets: weighted Heisenberg 9q and N2 16q.
