# AutoVQE research program

Find the best useful VQE ansatz for the Hamiltonian named by the user. Work as
one closed research loop:

`understand -> propose -> optimize -> compare -> learn -> keep or discard`

Energy is the primary objective. When energies are effectively tied, the
physically simpler circuit wins. Once growth stops helping or the target is
reached, spend the remaining research effort removing operations, reducing
depth, and testing parameter sharing without losing the energy.

## Boundary

During a solve, edit only `ansatz.py` as code; `results.tsv` may hold local
notes. Treat `evaluate.py` and the problem file as immutable. Never run an
eigensolver, search for a known answer, insert optimized constants, or encode
a solution in the input, initial state, or fixed gates. The evaluator is the
only source of energies and optimized parameters.

The gate allowlist is a Pauli word, `U1` (`XX+YY`), or `SU2`
(`XX+YY+ZZ`). A freely chosen Pauli word may touch at most two qubits. A
higher-weight word is allowed only when it is an input-Hamiltonian term or a
component of a rank-one or rank-two fermionic excitation under an explicitly
known mapping; expose every component and share its parameter. Arbitrary
Pauli sums, higher-rank excitations, custom unitaries, fixed angles, and fitted
per-operation scales are forbidden.

Before claiming a symmetry, verify that its generator commutes with the
Hamiltonian, the reference occupies a definite target sector, and the complete
logical block preserves it. Full SU(2) requires all three total-spin
components, not only magnetization. Translation, reflection, and point-group
structure should use parameter sharing across symmetry orbits, not new gates.

Every allowed gate is decomposed to the problem's native gate set. `U1` and
`SU2` count as two and three parameter occurrences respectively. Judge
simplicity from occurrences, generator support, native two-qubit gates, total
gates, and depth—not unique parameter count alone.

## Loop

1. Read the raw Hamiltonian. Inspect coefficients, locality, interaction
   graph, initial occupation, repeated structure, and plausible conserved
   quantities. Symmetry is one useful clue, not the whole ansatz.
2. Run the empty or current ansatz once to establish the baseline.
3. State one falsifiable structural idea. Change `ansatz.py` in one coherent
   way: add, remove, reorder, split, or share rotations, or change optimizer.
4. Run `evaluate.py` with its problem-defined default time budget; use
   `--seconds` only when the user overrides it. Record the hypothesis, energy,
   and resources in `results.tsv`.
5. Keep a change that improves energy. For an energy tie, keep it only when it
   makes the native circuit meaningfully simpler. Otherwise revert it.
6. Use failures to choose the next idea; do not blindly enumerate circuits.
   Periodically challenge the current best with a genuinely different
   structure, then simplify the winner.
7. Finish with the best ansatz in `ansatz.py` and rerun it once for the final
   report.

If `reference_energy` exists, success means meeting the requested relative
error. The reference is for scoring only and must never shape or be copied
into the ansatz. If no reference exists, report `best found`, not `ground
state`. A converged optimizer is not success by itself: ansatz quality is the
energy it can reach within the shared budget. Likewise, `target_reached` is an
acceptance floor, not permission to end the total research budget early;
confirm competing structures and simplify the best result.
