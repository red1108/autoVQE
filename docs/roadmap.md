# Roadmap

AutoVQE is an alpha Hamiltonian-to-ansatz research harness. This roadmap keeps
the project focused on better scientific search, honest resource accounting,
and useful feedback to the agent.

## Implemented baseline

- Mechanical Hamiltonian inspection without ansatz recommendations or exact
  reference answers.
- Typed `AnsatzSpec` candidates with affine parameter expressions and a small
  operation allowlist.
- Compiler-derived operation counts, parameter occurrences, fixed literals,
  and canonical circuit resources.
- Evaluator-derived optimization, energy traces, parameter bindings, gate
  counts, and depth.
- Exact normalized Pauli-commutator probes and reference-sector checks.
- Conditional `XYExchange` and `IsotropicExchange` operations with mandatory
  symmetry and per-operation preservation evidence.
- Closed symmetry, structure, and null-control hypothesis types.
- Fixed audit, smoke, and promotion stages with budget accounting.
- Limits on operations, unique parameters, representation size, parameter
  fan-out, two-qubit gates, total gates, and depth.
- Semantic duplicate detection that ignores cosmetic candidate changes.
- Persistent branch lineage with positive and grounded-negative terminal
  decisions.
- Local `research init`, `step`, `status`, and `result` commands.

## Search quality

The first priority is to make each evaluator call more informative.

1. Add evaluator-side gradient ranking for a bounded operator pool.
2. Compare parameter-sharing variants under the same optimization allowance.
3. Add controlled layer growth with explicit marginal energy and resource
   improvement.
4. Preserve a nondominated set over energy, objective calls, unique parameters,
   parameter occurrences, two-qubit gates, total gates, and depth.
5. Add an anytime score based on best energy at fixed objective-call points.
6. Improve branch selection so repeated failures narrow the next hypothesis
   instead of producing unrelated candidates.

The agent must never define its own score or optimization allowance.

## Physics probes

New probes should return a bounded measurement, numerical tolerance, and cost.

- Approximate symmetry with a distinct verdict from exact commutation.
- Translation, point-group, and permutation orbit tests.
- Multiple-generator and Casimir tests for non-Abelian symmetry.
- Particle number, spin projection, total spin, seniority, and excitation-pool
  closure under an explicit fermion-to-qubit encoding.
- Gauge constraints and local Gauss-law checks.
- Reference-sector compatibility for each supported conserved quantity.
- Gradient and operator-pool measurements for ADAPT-style selection.

No gate should be enabled merely because its name suggests a symmetry. Its
applicability must follow from the supplied Hamiltonian and evaluator evidence.

## Ansatz language

Candidate operations should be added only when they have:

- a clear physical generator;
- a precise applicability condition;
- identity at zero parameters;
- a deterministic compiler implementation;
- canonical resource accounting;
- operation, locality, layer, and fan-out limits;
- tests for both intended use and misuse.

Potential additions include:

- grouped Hamiltonian-variational layers;
- constraint-preserving QAOA mixers;
- fermionic single, double, pair, and seniority-preserving excitations;
- translation- or point-group-tied parameters;
- non-Abelian symmetry-adapted blocks;
- lattice-gauge local moves;
- routing-aware graph-colored schedules.

These are research directions, not currently accepted operations.

## Optimizer policy

- Version fixed smoke and promotion settings independently from the candidate
  representation.
- Compare all candidates at one stage with identical call, restart, and seed
  allowances.
- Study optimizer changes separately from ansatz changes.
- Record best-energy traces at common checkpoints.
- Add noise-aware and shot-aware policies without letting candidates select
  favorable settings.
- Detect candidates whose apparent improvement is unstable across fixed
  restarts.

## Hardware-aware evaluation

- Add named execution profiles derived from public hardware constraints.
- Enforce profile-specific connectivity, native operation, locality, and depth
  limits.
- Compare logical structure, canonical compilation, and declared-backend
  compilation without letting one view hide cost in another.
- Add routing-aware penalties and disconnected-support diagnostics.
- Test generic nonzero parameter bindings whenever simplification can depend on
  numeric values.

## Evaluation methodology

- Maintain separate calibration and unseen problem sets.
- Add reproducible chemistry-generation scripts instead of opaque hand-edited
  inputs.
- Test changes across problem families rather than tuning to file names.
- Score final results only after the research run ends.
- Report all started independent trials, including negative decisions and
  failures.
- Evaluate both accuracy and efficiency rather than selecting on one scalar.
- Require a new unseen problem after any result influences the implementation.

## Agent usability

- Provide concise next-action diagnostics after rejected actions.
- Add examples for every action type without embedding problem-specific
  answers.
- Offer machine-readable summaries of live branches and remaining budget.
- Make `research result` easy to consume while keeping every value tied to an
  evaluator computation.
- Add optional assistance for formatting an action, without choosing the
  scientific hypothesis for the agent.

## Non-goals

- Hiding exact diagonalization, an eigenstate, or learned solution angles in a
  candidate.
- Accepting candidate-authored energy, optimized parameters, gate counts, or
  depth.
- Calling classical correction the raw variational-circuit result.
- Treating one shared-angle Hamiltonian evolution as sufficient evidence of a
  useful search space.
- Expanding to an unrestricted universal circuit language before resource and
  applicability policies can evaluate it.
- Describing a local promotion as exact ground-state accuracy or
  cross-problem generalization.
