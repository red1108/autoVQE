# Roadmap

AutoVQE is an alpha research harness. This page separates code that exists now
from proposed evaluation/search controls; an item below is not a capability
claim unless it appears under “Implemented baseline.”

## Implemented baseline

- Agent-safe `inspect --agent-json` with a content-hashed `PublicProblem` and
  structural observation.
- Local `PrivateEvaluationContext` contract for exact reference energy/state
  without serializing them to the agent view.
- Versioned `AnsatzSpec`, affine parameter expressions, trusted macro registry,
  compiler-derived parameter occurrences/fixed literals/macro counts, and
  evaluator-derived circuit metrics.
- Trusted operation macros `PauliRotation`, `XYExchange`, and
  `IsotropicExchange` plus reference-only `X` preparation.
- Exact normalized Pauli-commutator requests and an internal
  public-reference-moment primitive.
- Scale-invariant generator validation plus mandatory
  per-operation/reference preservation audit for an exact-symmetry candidate.
- Strict generator grammar with unique labels, 256-term/global-size and
  coefficient bounds, and exact field allowlists.
- Closed `exact_pauli_symmetry`, `ansatz_structure`, and `null_control` claim
  schemas, with explicit non-algebraic admission markers for structure/control
  experiments and promotion blocked for null controls.
- `research init/step/status`, fixed evaluator-owned audit/smoke/promotion
  protocols, budget accounting, strict lifecycle transitions, and a replayable
  SHA-256 hash-chained JSONL ledger.
- `local_unsealed` development mode plus schema-v3
  `--sealed`/sealed-default operational APIs, HMAC-SHA256 context and ledger-tip
  checkpoints, a signed random 128-bit run ID, path registry, authoritative
  external ledger, monotonic head, workspace mirror, and mutation-wide
  exclusive lock in `AUTOVQE_EVALUATOR_ANCHOR_DIR` outside the run directory.
- Explicit `--allow-unsealed`/`require_sealed=False` opt-in for operating a
  locally initialized development run; `step`, `status`, and their Python APIs
  otherwise require sealed mode.
- Global caps on action/event/search size and candidate operations, unique
  parameters, parameter fan-out, and IR nodes; higher-locality Pauli rotations
  must come from the public Hamiltonian.
- Fixed all-to-all `{rz,sx,x,cx}` canonical resource views and
  smoke/promotion eligibility caps on the metric-wise maximum of symbolic
  template and three-binding generic-worst counts/depth.
- Smoke/promotion gates that require improvement over a zero-parameter baseline
  and promotion consistency with an earlier smoke.
- A terminal operator export that binds a controller-accepted decision to its
  ledger tip and, for a positive result, the committed typed ansatz plus the
  evaluator-owned optimized parameter binding.
- The legacy `campaign`/`benchmark`/`solve` path retained for compatibility and
  regression testing.

## Next: close the search/reward boundary

These controls are the highest-priority additions because the current macro
language and promotion rule are intentionally minimal:

1. Add named macro profiles selected from public execution/physics constraints.
   Refine the current global IR/fan-out and canonical conservative caps with
   per-profile arity/locality/layer and native-backend ceilings. In particular,
   narrow the broad one/two-body `PauliRotation` surface instead of treating a
   closed registry and global caps as sufficient.
2. Make optimizer protocols versioned evaluator policy and compare candidates
   under identical call/restart/seed budgets. Keep optimizer exploration in a
   separate experiment track so it cannot masquerade as ansatz progress.
3. Replace the boolean local promotion check with evaluator-owned
   multi-objective records: energy quality, objective calls, unique parameters,
   parameter occurrences, two-qubit gates, total gates, and depth. Preserve the
   nondominated Pareto set instead of collapsing every tradeoff into one
   agent-visible scalar.
4. Add an anytime metric derived from the best-energy trace at fixed evaluation
   checkpoints or normalized area-under-regret. Define it in evaluator code,
   not in candidate metadata.
5. Split fixtures into public development/smoke cases and a versioned hidden
   generalization set. Promotion within a run and final generalization scoring
   should be distinct events.
6. Return stage-appropriate feedback: detailed schema/audit failures, bounded
   probe facts, aggregate smoke metrics, and a coarser hidden-evaluation
   receipt. Do not reveal hidden fixture identities, per-instance references,
   or enough per-instance data to tune against the holdout.

None of the Pareto, anytime, or hidden-generalization mechanisms in this
section are implemented today.

## Evaluator isolation

Build an external evaluator service that:

- receives only a candidate hash/spec plus protocol version;
- holds `PrivateEvaluationContext` and hidden-set membership outside the agent
  workspace;
- signs receipts and anchors run/ledger tips;
- enforces global candidate, retry, wall-clock, and objective-call quotas;
- deduplicates semantically identical submissions;
- separates public smoke feedback from final hidden feedback.

The present evaluator is local. Sealed mode stores the authoritative ledger,
monotonic head, and signed run-path registry outside the run directory; the
workspace ledger is a verified mirror. Its exclusive external lock spans
verification through append/checkpoint/anchor publication, and ancestry
checking requires the new chain to retain the old anchored tip at the old-count
prefix. That guarantee requires `AUTOVQE_EVALUATOR_KEY` and the whole external
state directory to remain outside agent control. These are evaluator-owned
files, not a remote/durable anchoring service, digital signature, or hidden
evaluation service.
`local_unsealed` is explicitly not a trust boundary. The existence of
`PrivateEvaluationContext` is a data-contract boundary, not proof of service
isolation.

## Automatic agent adapter

Add an optional adapter that can:

- read `observation.json` and the latest controller receipt;
- emit exactly one valid external action;
- stop only on a controller-accepted commit or grounded `close_negative`;
- checkpoint prompts/model/version metadata without placing secrets in the
  agent-visible problem;
- retry formatting errors without repeating paid evaluations.

Today `research step` consumes a user- or script-authored JSON file. No trusted
adapter autonomously drives the sealed action/receipt loop.

## Probe expansion

Keep new probes evaluator-owned and attach an explicit cost and numerical
tolerance to every receipt.

- Approximate symmetry: normalized residuals with declared tolerances,
  scale/error calibration, and a distinct verdict from exact commutation.
- Translation, point-group, and permutation symmetry: orbit construction and
  representation/projector tests rather than the MVP all-site Pauli-sum alias.
- Non-Abelian symmetry: multiple generator/commutator and Casimir/sector tests;
  do not infer it from one commuting charge.
- Fermionic algebra: particle number, spin projection/total spin, seniority, and
  excitation-pool closure using an explicit encoding map.
- Extend the current exact per-operation/reference preservation audit to
  approximate, group, non-Abelian, and fermionic charges and, where needed,
  evaluator-chosen generic parameter bindings.
- Gradient/operator-pool probes for ADAPT-style ranking under fixed call
  budgets.

Approximate, group/permutation, fermionic, generalized candidate-preservation,
and ADAPT probes are future work. The current external controller exposes only
an exact normalized-commutator request. It uses normalized reference moments
internally when auditing an exact-symmetry candidate against the same
machine-readable Pauli charge.

## Ansatz/macro expansion

Candidate macros should be added only with a physical invariant, trusted
implementation, zero-parameter identity test, decomposition/resource audit,
and profile-level limits.

Potential additions:

- grouped Hamiltonian-variational and QAOA cost/mixer templates;
- constraint-preserving QAOA mixers;
- fermionic single/double and pair/seniority-preserving chemistry excitations;
- lattice-gauge Gauss-law-preserving local moves;
- translation/point-group parameter tying and explicit projection/twirling;
- non-Abelian symmetry-adapted blocks;
- routing-aware graph-colored schedules.

These names describe research directions, not current allowlisted macros.

## Reproducibility and calibration

- Version public and hidden evaluation protocols independently from the
  `AnsatzSpec` schema.
- Add reproducible chemistry-generation scripts for H6 and BeH2 rather than
  hand-maintained opaque fixtures.
- Add the private scorer and aggregate Pareto/generalization reports on top of
  the implemented terminal result export.
- Add cross-problem tests that penalize fixture-name rules and candidate
  duplication.
- Add platform/filesystem concurrency, stale-lock recovery, and crash-injection
  tests before promising general multi-writer operation; the current exclusive
  lock serializes sealed mutations and fails closed on cross-file interruption.
- Define migration rules for ledger/schema versions and long-lived runs.

## Legacy migration

- Keep `solve` results reproducible while the typed path matures.
- Label legacy recommendations and reference-aware tolerance checks clearly in
  output and docs.
- Port useful built-in ansatz families into explicit trusted macros/profiles
  only after their audit and symmetry contracts are specified.
- Do not silently equate a legacy solve pass with research promotion or hidden
  generalization.

## Non-goals

- Hiding exact diagonalization, an eigenstate, or learned solution angles inside
  a candidate.
- Accepting candidate-reported energy, parameter, gate, or depth values.
- Calling classical post-processing the raw VQE circuit result.
- Treating a single shared-angle evolution as sufficient evidence of a useful
  variational search space.
- Growing an unrestricted universal gate surface before complexity and
  evaluation policies can audit it.
- Claiming tamper-proof or hidden evaluation from `local_unsealed`, or from a
  sealed HMAC run whose key, authoritative ledger, path registry, or monotonic
  anchor is controlled by the agent.
