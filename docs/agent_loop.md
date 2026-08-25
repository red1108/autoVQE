# Agent Research Loop

AutoVQE is most useful when the agent treats ansatz discovery as a sequence of
falsifiable experiments rather than a one-shot circuit-generation task.

```text
observe Hamiltonian
        |
        v
form a structural claim
        |
        v
request a physical probe when the claim is algebraic
        |
        v
propose typed candidate + preregister prediction
        |
        v
audit -> smoke -> promotion
   |        |         |
   +--------+---------+--> revise or retire on failure
        |
        v
compare credible alternatives
        |
        v
positive_commit or grounded negative_close
```

## Components

| Responsibility | Code | Role |
| --- | --- | --- |
| Problem view | `contracts.py`, `observations.py` | Present Hamiltonian and mechanically derived structure |
| Candidate language | `ansatz_ir.py`, `macros.py` | Define strict typed operations and parameter sharing |
| Compilation | `compiler.py` | Build the circuit and derive audit counts |
| Physical evidence | `probes.py`, `evaluator.py` | Measure symmetries, optimize energy, and count resources |
| Research policy | `controller.py`, `research.py` | Enforce stages, budgets, lineage, and terminal rules |
| Run history | `history.py`, `research_cli.py` | Record actions and reconstruct branch state |
| Public commands | `harness.py` | Expose `inspect`, `check`, and the research workflow |

## Start a run

Use a fresh directory and keep the Hamiltonian unchanged:

```bash
uv run python -m autovqe.harness research init \
  --problem user_problem/hamiltonian.json \
  --run-dir .autovqe-runtime/research/run-001 \
  --budget 100
```

The initialization output is the starting point for the agent. It records the
mechanical problem view and an empty research state.

## One productive iteration

1. Read the Hamiltonian terms, locality distribution, support graph, declared
   reference preparation, and backend constraints.
2. State one falsifiable reason a circuit family might work. Distinguish an
   exact-symmetry claim from a structural heuristic.
3. If the reason depends on a conserved quantity, request the matching
   commutator probe before using conservation-oriented gates.
4. Propose a small `AnsatzSpec` that tests the claim. Include a concrete
   prediction or falsifier before evaluation.
5. Request `audit`. Fix representation, reference, locality, invariant, or
   resource-policy failures before spending optimization budget.
6. Request `smoke`. Compare its energy improvement and resource use with the
   zero-angle baseline.
7. Promote only a candidate with a useful smoke result. Promotion uses a larger
   fixed optimization allowance.
8. If evidence contradicts the hypothesis, revise it with a new identifier or
   retire the branch with a concrete reason.
9. Before committing, test a credible alternative or document why the promoted
   candidate is not dominated by the available evidence.
10. Inspect `research status`, choose the next action, and repeat.

Submit exactly one action per step:

```bash
uv run python -m autovqe.harness research step \
  --run-dir .autovqe-runtime/research/run-001 \
  --action action.json
```

## Choosing hypotheses

Useful observations include:

- Pauli-term locality and coefficient scale;
- repeated support motifs or graph coloring;
- commuting term groups;
- translation or permutation patterns;
- particle-number, parity, spin, or other conserved quantities;
- reference-sector compatibility;
- hardware connectivity and two-qubit cost.

Pattern recognition is a hypothesis generator, not evidence. For example,
seeing repeated `XX + YY` terms may motivate a U(1) claim, but an
`XYExchange` candidate still requires the exact commutator result and the
operation-level preservation audit.

Do not make every branch a symmetry branch. Productive alternatives include:

- ordered rotations from selected Hamiltonian terms;
- different parameter-sharing patterns;
- shallow unconstrained controls;
- graph-colored schedules;

The point is to let the Hamiltonian determine what gets tested.

## Learning from failures

Failures should narrow the next experiment:

| Observation | Productive response |
| --- | --- |
| Claimed charge does not commute | Retire the exact claim; try a smaller charge or a structural branch |
| Reference is outside the claimed sector | Change the hypothesis or use a compatible public reference |
| Audit rejects a gate or literal | Rewrite the candidate within the typed language |
| Generic resource counts are too large | Reduce operations or parameter fan-out; do not rely on cancellation |
| Smoke does not beat zero angle | Retire or make one motivated structural revision |
| Promotion regresses from smoke | Treat the family as unstable under the fixed optimizer policy |
| Two candidates trade energy for cost | Preserve both results and make the comparison explicit |

Do not rename a failed candidate and submit it again. AutoVQE recognizes the
same physical family after cosmetic names and parameter symbols are removed.

## Budget discipline

The budget is part of the scientific method. Cheap structural validation comes
before expensive optimization:

1. inspect;
2. propose;
3. probe if needed;
4. submit;
5. audit;
6. smoke;
7. promote only when justified.

Keep at most a few live branches. A broad list of untested ideas is less useful
than a small set with clear evidence. Reserve enough budget and event capacity
to resolve every branch and request a terminal decision.

## Evaluator feedback

The agent can rely on:

- validated problem structure;
- normalized commutator measurements;
- compiler-derived parameter and operation counts;
- evaluator-derived energy and objective traces;
- canonical and declared-backend resource measurements;
- symmetry-preservation checks;
- lifecycle state and remaining budget.

The agent cannot establish evidence by writing values into candidate metadata.
Optimized angles, energy, and resource counts come from the evaluator.

## Terminal decisions

A positive ending requires a promoted candidate, preregistered expectation,
promotion evidence, and a grounded comparison. A negative ending requires all
branches to be terminal and substantive evidence covering each investigated
hypothesis.

Once the controller accepts a terminal decision, run:

```bash
uv run python -m autovqe.harness research result \
  --run-dir .autovqe-runtime/research/run-001
```

Report only the values returned there. State clearly that local promotion is a
within-run result. Exact accuracy and performance on new Hamiltonians require a
separate evaluation that was not used to shape the search policy.
