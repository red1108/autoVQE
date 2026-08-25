# Changelog

Notable AutoVQE changes are recorded here. The project follows semantic
versioning while its public interface is still evolving.

## Unreleased

### Added

- Typed `AnsatzSpec` candidates with strict affine parameter expressions.
- A small operation allowlist with conditional symmetry-preserving exchanges.
- Exact commutator evidence and reference-sector checks.
- Evaluator-owned energy, optimized parameter, and resource measurements.
- Closed hypothesis, probe, candidate, audit, smoke, promotion, and terminal
  decision lifecycle.
- Local `research init`, `step`, `status`, and `result` commands.
- Semantic duplicate detection and generic-binding resource checks.

### Changed

- Public documentation now focuses on the scientific research workflow.
- Generated run state is kept under `.autovqe-runtime/`.
- Positive decisions are described as local promotions unless an independent
  reference evaluation is reported separately.

## 0.1.0 - 2026-05-31

### Added

- Pauli-Hamiltonian loading and mechanical inspection.
- Initial VQE optimization and hardware-aware circuit measurement.
- Calibration inputs covering molecular and spin-system regimes.
