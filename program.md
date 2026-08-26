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
(`(YX-XY)/2`), or `SU2` (`XX+YY+ZZ`). `GIVENS` is ordered and preserves total
Z/Hamming weight, not full SU(2). A freely chosen Pauli word may touch at most
two qubits. A higher-weight word is allowed only when it is an input-Hamiltonian
term or a component of a rank-one or rank-two fermionic excitation under an
explicitly known mapping; expose every component and share its parameter.
Arbitrary Pauli sums, higher-rank excitations, custom unitaries, fixed angles,
and fitted per-operation scales are forbidden.

Before claiming a symmetry, verify that its generator commutes with the
Hamiltonian, the reference occupies a definite target sector, and the complete
logical block preserves it. Full SU(2) requires all three total-spin
components, not only magnetization. Translation, reflection, and point-group
structure should use parameter sharing across symmetry orbits, not new gates.

Every allowed gate is decomposed to the problem's native gate set. `U1`,
`GIVENS`, and `SU2` count as two, two, and three occurrences respectively. Judge
simplicity from occurrences, generator support, native two-qubit gates, total
gates, and depth—not unique parameter count alone.

## Loop

1. Read the raw Hamiltonian. Inspect coefficients, locality, interaction
   graph, initial occupation, repeated structure, and plausible conserved
   quantities. Symmetry is one useful clue, not the whole ansatz.
2. Run the empty or current ansatz once to establish the baseline.
3. State one falsifiable structural idea. Change `ansatz.py` in one coherent
   way: add, remove, reorder, split, or share rotations, or change optimizer.
4. Run `evaluate.py --hypothesis "<one-line idea>"` with its problem-defined
   default time budget; use `--seconds` only when the user overrides it. The
   evaluator appends the formatted result to `results.tsv`; never edit it by
   hand.
5. Keep a change that improves energy. For an energy tie, keep it only when it
   makes the native circuit meaningfully simpler. Otherwise revert it.
6. Use failures to choose the next idea; do not blindly enumerate circuits.
   Periodically challenge the current best with a genuinely different
   structure, then simplify the winner.
7. A time-limited promising run is unfinished: repeat it to continue from the
   evaluator-owned values, and optionally refine a nonzero result with L-BFGS-B.
   If it converges above target, return to structure search.
8. Finish with the best ansatz in `ansatz.py` and rerun it for the final report.

If `reference_energy` exists, success means meeting the requested relative
error. The reference is for scoring only and must never shape or be copied
into the ansatz. If no reference exists, report `best found`, not `ground
state`. A converged optimizer is not success by itself: ansatz quality is the
energy it can reach within the shared budget. Stop immediately at
`target_reached`; otherwise use the available total research budget.
