# AutoVQE research program

Find the best useful VQE ansatz for the Hamiltonian named by the user. Work as
one closed research loop:

`understand -> propose -> optimize -> compare -> learn -> keep or discard`

Energy is the primary objective. When energies are effectively tied, the
physically simpler circuit wins. Once growth stops helping, remove operations,
reduce depth, and test parameter sharing without losing the energy.

## Boundary

During a solve, edit only `ansatz.py` as code. Treat `evaluate.py`,
`results.tsv`, `.autovqe-state.json`, and the problem file as immutable. Never
run an eigensolver, search for a known answer, insert optimized constants, or
encode a solution in the input, initial state, or fixed gates. The evaluator
is the only source of energies and optimized parameters.

The gate allowlist is a Pauli word, `U1` (`XX+YY`), `GIVENS`
(`(YX-XY)/2`), `PAIR` (`(YX+XY)/2`), or `SU2` (`XX+YY+ZZ`). `GIVENS`
preserves total Z/Hamming weight; `PAIR` mixes sectors whose weights differ by
two while preserving Hamming parity. Neither implies full SU(2). A freely
chosen Pauli word may touch at most two qubits. A higher-weight word is allowed
only when it is an input-Hamiltonian term or a component of a rank-one or
rank-two fermionic excitation under an
explicitly known mapping; expose every component and share its parameter.
Arbitrary Pauli sums, higher-rank excitations, custom unitaries, fixed angles,
and fitted per-operation scales are forbidden.

Before claiming a symmetry, verify that its generator commutes with the
Hamiltonian, the reference occupies a definite target sector, and the complete
logical block preserves it. Full SU(2) requires all three total-spin
components, not only magnetization. Translation, reflection, and point-group
structure should use parameter sharing across symmetry orbits, not new gates.

Every allowed gate is decomposed to the problem's native gate set. `U1`,
`GIVENS`, `PAIR`, and `SU2` count as two, two, two, and three occurrences.
Judge simplicity from occurrences, generator support, native two-qubit gates,
total gates, and depth—not unique parameter count alone.

## Loop

1. Read the raw Hamiltonian. Inspect coefficients, locality, interaction graph,
   initial occupation, repeated structure, and verified conserved quantities.
   A conserving circuit cannot change sector, so preparation must variationally
   reach the intended sector rather than encode an answer.
2. Run the empty or current ansatz once to establish the baseline.
3. Spend the total research budget on bounded structural comparisons, never on
   long or repeatedly continued refinement of a fixed ansatz. State one
   falsifiable change, run it with the evaluator's per-candidate budget, and use
   `--seconds` only when the user overrides that budget.
4. Activate a new preparation seed or identity-initialized layer with `COBYLA`
   or `Powell`; once it yields a useful warm start, switch that same candidate
   to `L-BFGS-B`. Treat this as a bounded handoff within ordinary budgets.
5. Search structure before freedom: select a useful depth with parameters
   shared over graph matchings or verified symmetry orbits. Only afterward
   preserve every gate, order, and scale, split names, warm-start from
   evaluator-owned values, and optimize. Do not interleave growth and splitting.
6. Apply the model's ordered physical ladder before generic Pauli layers. For
   antiferromagnetic Heisenberg systems, prepare low-spin dimers with a shared
   `Y` seed and `GIVENS`; edge-color the interaction graph, then grow alternating
   `SU2` matching cycles with one parameter per matching and cycle before an
   edgewise name split. For TFIM, use one shared `Y` product-state seed, grow
   globally shared alternating interaction-`ZZ` then field-`X` layers to choose
   depth, and only then split each layer's names into reflection orbits.
7. Keep lower energy, or a meaningfully simpler native circuit at an effective
   tie; otherwise revert. Use failures to choose a genuinely different
   structure. Convergence above target requires a structural change, not more
   optimization of the same ansatz.
8. Finish with `evaluate.py ... --restore-best --hypothesis "final best"`; this
   restores the lowest-energy structure without writing optimized numbers into
   `ansatz.py`. If deliberately selecting a measured simpler tie, keep that
   structure and rerun it normally instead.

If `reference_energy` exists, success means meeting the requested relative
error. The reference is for scoring only and must never shape or be copied
into the ansatz. If no reference exists, report `best found`, not `ground
state`. A converged optimizer is not success by itself: ansatz quality is the
energy it can reach within the shared budget. Stop immediately at
`target_reached`; otherwise use the available total research budget.
