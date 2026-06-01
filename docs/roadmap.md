# Roadmap

AutoVQE is intentionally small. The roadmap is biased toward changes that make
experiments easier to reproduce and compare.

## Near Term

- Add reproducible chemistry generation scripts for H6 and BeH2 fixtures.
- Add a compact result-summary exporter for benchmark tables.
- Add focused tests for candidate eligibility and symmetry-preserving circuit
  construction.
- Improve optimizer warm starts without hard-coding benchmark-specific angles.

## Research Directions

- ADAPT-VQE style operator ranking from measured gradients or short smoke
  improvements.
- Coefficient-aware schedules for nonuniform TFIM and weighted spin graphs.
- Fermionic excitation blocks when chemistry metadata is available.
- Tapering and inactive-qubit reduction for chemistry fixtures.

## Non-Goals

- Hiding exact diagonalization inside an ansatz.
- Treating classical post-processing as a VQE benchmark pass.
- Growing a large framework before the small harness has earned it.
