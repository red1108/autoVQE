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

The ansatz consists only of typed Pauli rotations. They are always decomposed
to the problem's native gate set. A many-qubit rotation is therefore charged
for its full support, depth, and two-qubit gates; it is not a one-gate or
one-parameter shortcut. Do not judge simplicity from unique parameter count
alone. Use all reported costs: parameter occurrences, generator support,
two-qubit gates, total gates, and depth.

## Loop

1. Read the raw Hamiltonian. Inspect coefficients, locality, interaction
   graph, initial occupation, repeated structure, and plausible conserved
   quantities. Symmetry is one useful clue, not the whole ansatz.
2. Run the empty or current ansatz once to establish the baseline.
3. State one falsifiable structural idea. Change `ansatz.py` in one coherent
   way: add, remove, reorder, split, or share rotations, or change optimizer.
4. Run `evaluate.py` with the same per-experiment time budget. Record the
   hypothesis, energy, and resources in `results.tsv`.
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
