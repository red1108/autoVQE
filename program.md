# AutoVQE research program

Find the best useful VQE ansatz for the Hamiltonian named by the user. Work as
one closed research loop:

`understand -> propose -> optimize -> compare -> learn -> keep or discard`

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
Hamiltonian, the state entering its block occupies a definite target sector,
and the complete block preserves it. Full SU(2) requires all three total-spin
components, not only magnetization. Translation, reflection, and point-group
structure should use parameter sharing across symmetry orbits, not new gates.

Every allowed gate is fully expanded and transpiled to supplied native
constraints. `U1`, `GIVENS`, `PAIR`, and `SU2` count as two, two, two, and three
occurrences. Judge simplicity from occurrences, generator support, two-qubit gates,
total gates, and depth—not unique parameter count alone.

## Loop

1. Read the raw Hamiltonian. Inspect coefficients, locality, interaction graph,
   initial occupation, repeated structure, and verified conserved quantities.
   A conserving circuit cannot change sector, so preparation must variationally
   reach the intended sector rather than encode an answer.
2. If this problem has no result, evaluate the empty or current ansatz once;
   otherwise use its existing result as the baseline.
3. Give every candidate the evaluator's fixed default budget
   `max(30, 60*2**(n-16))` seconds; use `--seconds` only when the user overrides
   it. State one falsifiable change per run. Continue until the user interrupts;
   target, convergence, a failed candidate, or a plateau never ends the loop.
4. Start each candidate with `L-BFGS-B`; new parameters receive a small
   deterministic nonzero seed. The evaluator restarts a converged optimizer
   until the candidate budget expires. If energy does not improve, try one
   bounded `COBYLA` or `Powell` activation, then return it to `L-BFGS-B`.
5. Bracket depth with shared parameters in multiplicative jumps instead of
   sweeping adjacent depths. Change depth or sharing in one comparison, never
   both. Split names at fixed structure; if useful, keep that scheme while
   testing further growth.
6. Derive physical ladders from observed terms, coefficients, graph, and
   verified symmetries—not a filename or model label. For antiferromagnetic
   isotropic `XX+YY+ZZ` graphs, test low-spin `Y`/`GIVENS` dimers followed by
   edge-colored `SU2` matching cycles. For `ZZ` graphs with transverse `X`, test
   a shared `Y` seed and alternating `ZZ`/`X` layers, then verified symmetry-orbit
   name splits.
7. The evaluator owns `current_best` and decides every run. Before target, it
   keeps lower energy, or a Pareto-simpler circuit when
   `|delta_E| <= 1e-8*max(1, |E_current_best|)`. The first target-reaching
   candidate is kept. After that, accuracy is a hard constraint: keep only
   target-reaching candidates that reduce at least one of unique parameters,
   occurrences, generator support, two-qubit gates, total gates, and depth
   without increasing another. Otherwise discard and restore `current_best`.
   Use failures to choose a genuinely different structure.
8. Never write optimized numbers into `ansatz.py`. If an edit or run is
   interrupted, use `evaluate.py ... --restore-best --hypothesis "restore"`;
   otherwise keep running until the user stops the task.

If `reference_energy` exists, success means meeting the requested relative
error. The reference is for scoring only and must never shape or be copied
into the ansatz. If no reference exists, report `best found`, not `ground
state`. A converged optimizer is not success by itself: ansatz quality is the
energy it reaches within the shared per-candidate budget. `target_reached`
changes the objective from energy improvement to simplification; it does not
end the research loop.
