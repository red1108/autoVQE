# Changelog

Notable AutoVQE changes are recorded here. The project follows semantic
versioning while its public interface is still evolving.

## Unreleased

### Added

- Typed `AnsatzSpec` candidates with strict affine parameter expressions.
- A small operation allowlist with conditional symmetry-preserving exchanges.
- Exact commutator evidence and evaluator-owned initial-sector checks.
- Evaluator-owned energy, optimized parameter, and resource measurements.
- Closed hypothesis, probe, candidate, audit, smoke, promotion, and terminal
  decision lifecycle.
- Installed `uv run autovqe` command with `research init`, `step`, compact
  `status`, optional full status, and `result`.
- Semantic duplicate detection and nonzero audit-binding resource checks.
- Terminal-only optimized-binding materialization without storing the binding
  in preterminal history.
- Conditioned symmetry relevance checks that prevent spectator-weighted terms
  from unlocking conservation-specific gates.

### Changed

- `AnsatzSpec` now contains one flat operation sequence; initial preparation is
  evaluator-owned problem input rather than candidate structure.
- Probe generators and IDs, evaluation stages and IDs, and terminal evidence
  are derived by the controller.
- Positive commit requires a different-hypothesis competitor or control at the
  same promotion fidelity.
- Negative closure requires complete branch grounding plus evaluator-observed
  objective activity, with either promotion depth or two independent structure
  lineages.
- Public documentation now focuses on the scientific research workflow.
- Generated run state is kept under `.autovqe-runtime/`.
- Positive decisions are described as local promotions unless an independent
  reference evaluation is reported separately.

## 0.1.0 - 2026-05-31

### Added

- Pauli-Hamiltonian loading and mechanical inspection.
- Initial VQE optimization and hardware-aware circuit measurement.
- Calibration inputs covering molecular and spin-system regimes.
