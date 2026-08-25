# AutoVQE agent instructions

These instructions apply to Hamiltonian analysis and ansatz discovery in this
repository. Ordinary maintenance requests may edit the implementation as the
user directs.

- Treat `user_problem/hamiltonian.json` as immutable input. Do not add a
  reference energy or state, and do not rename it to match an example.
- During an ansatz-discovery run, do not modify AutoVQE source, tests, or
  documentation. Manually write only action JSON under
  `.autovqe-runtime/actions/`; the CLI alone owns run files and history.
  Report a harness defect instead of patching around it.
- Route every probe, energy, optimization, and resource measurement through
  `uv run autovqe research ...`. Do not import the evaluator directly, run a
  separate eigensolver/optimizer, or edit controller-owned evidence to bypass
  the closed research loop and its budget.
- Read the raw Hamiltonian, README workflow, and `program.md` before choosing a
  strategy. Base hypotheses on visible terms, coefficients, locality, graph
  structure, the initial occupation, and any symmetry that you independently
  probe.
- Express every proposed variational circuit as a typed `AnsatzSpec`. Do not
  submit candidate-authored energy, optimized values, parameter counts, gate
  counts, depth, custom operations, or hidden numeric answers.
- Treat structure hypotheses as falsifiable branches. Candidate submission
  runs the fixed audit automatically; survivors advance through smoke and
  promotion. Preserve failed and retired branches so later decisions retain
  their evidence. Bring a candidate from a different primary structure root
  through smoke before requesting promotion; a symmetry probe or cosmetic
  family duplicate is not a comparator.
- Cite supported symmetry probe IDs separately from the candidate's primary
  structural hypothesis. Use symmetry-preserving gates only after a relevant,
  non-spectator conserved quantity is established and the candidate passes
  operation-level preservation checks for every cited charge.
- Stop only after the controller accepts `positive_commit` or a grounded
  `negative_close`. Negative closure must satisfy the controller's
  objective-activity breadth-or-depth rule; flat phase-only failures do not
  count. A positive decision proves only the recorded local promotion rule,
  not exact ground-state accuracy or generalization.
- Report the optimized parameter binding only from `research result`. If no
  independent reference score was provided, state that limitation explicitly.
- Before reporting implementation changes, run the relevant unit tests and
  `uv run python -m autovqe.harness check`.
