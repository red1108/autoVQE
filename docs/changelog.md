# Changelog

All notable changes to AutoVQE are recorded here.

The format follows the spirit of Keep a Changelog, and the project uses
semantic versioning while the public API is still small.

## [0.1.0] - 2026-05-31

### Added

- Public CLI for Hamiltonian inspection, calibration campaigns, and target-driven
  solving.
- JSON problem fixtures for H2, TFIM, Heisenberg, weighted spin graphs, and a
  16-qubit N2 stress test.
- Hamiltonian-aware candidate families:
  - U(1) exchange layers,
  - SU(2)-style Heisenberg HVA,
  - TFIM counterdiabatic schedules,
  - Pauli term-evolution HVA,
  - shallow HEA baselines.
- Documentation for ansatz selection, calibration fixtures, releases, and
  contribution workflow.
- CI checks for compile, harness self-check, and the small solve suite.

### Changed

- Reported solve criteria now use raw circuit VQE energy only.
- Single shared-angle Hamiltonian evolution is treated as ineligible.

### Verified

- Small regression suite: H2 2q, H2 4q, Ising 5q.

### Included Regime Probes

- Spin-chain probes: TFIM n10, Heisenberg n10, weighted Heisenberg 9q.
- Chemistry probes: PennyLane H2 4q and N2 16q active-space stress test.
