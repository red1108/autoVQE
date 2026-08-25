# AutoVQE agent instructions

These instructions apply to Hamiltonian analysis and ansatz discovery in this
repository. Ordinary maintenance requests may edit the implementation as the
user directs.

- Treat `user_problem/hamiltonian.json` as immutable input. Do not add a
  reference energy or state, and do not rename it to match an example.
- Read the README research workflow, evaluator-owned evidence rules, action
  protocol, and ansatz playbook before choosing a strategy.
- Express every proposed variational circuit as a typed `AnsatzSpec`. Do not
  submit candidate-authored energy, optimized values, parameter counts, gate
  counts, depth, custom operations, or hidden numeric answers.
- Treat hypotheses as falsifiable branches. Use the available physical probes
  and the fixed audit, smoke, and promotion stages. Preserve failed and retired
  branches so later decisions retain their evidence.
- Use symmetry-preserving gates only after the relevant conserved quantity is
  established and the candidate passes operation-level preservation checks.
- Stop only after the controller accepts `positive_commit` or a grounded
  `negative_close`. A positive decision proves only the recorded local
  promotion rule, not exact ground-state accuracy or generalization.
- Report the optimized parameter binding only from `research result`. If no
  independent reference score was provided, state that limitation explicitly.
- Before reporting implementation changes, run the relevant unit tests and
  `uv run python -m autovqe.harness check`.
