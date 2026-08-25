# AutoVQE agent instructions

These instructions apply to Hamiltonian-solving and ansatz-discovery tasks in
this repository. Ordinary maintenance requests may edit the implementation as
the user directs.

- Treat a user-provided `user_problem/hamiltonian.json` as immutable input.
  Do not add a reference energy/state or rename it to match a fixture.
- Read the README's trust-boundary and research-lifecycle sections before
  choosing a workflow. The legacy `solve` command is a compatibility and
  calibration path; do not present its reference-aware success flag as an
  independent research result.
- Express proposed variational circuits as the typed `AnsatzSpec`. Do not add
  candidate-reported energy, parameter counts, gate counts, hidden literals,
  or optimized values. The trusted compiler/evaluator derives those fields.
- Treat hypotheses as falsifiable branches. Use evaluator-owned probes and the
  fixed audit, smoke, and promotion stages; retain failed or retired branches.
- Stop a research loop only after the controller accepts `positive_commit` or
  grounded `negative_close`. A positive commit proves only the recorded local
  promotion rule, not ground-state accuracy or cross-problem generalization.
- In a sealed run, use only the generated bundle client and signed gateway
  publications. Never seek the source checkout, raw private problem,
  evaluator, reference, key, anchor, or previous trials.
- Report a final optimized parameter binding only from the trusted terminal
  result export. If the run is local/unsealed or lacks an external score, label
  it accordingly instead of upgrading the claim.
- Before reporting implementation changes, run the relevant unit tests and
  `uv run python -m autovqe.harness check`.
