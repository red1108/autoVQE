# Agent Loop

For an executable Codex `/goal` workspace, example-free agent contract, and
separated evaluator bridge, see [`../meta_agent/README.md`](../meta_agent/README.md).

The canonical research loop is a typed, one-action-at-a-time protocol:

```text
public observation
  -> falsifiable hypothesis
  -> exact evaluator probe OR explicit non-algebraic admission
  -> typed AnsatzSpec
  -> audit
  -> smoke
  -> promotion
  -> evidence-gated commit

Every failed edge -> revise or retire with evidence
All branches terminal -> grounded close_negative
```

Symmetry is not an answer label. A sector declaration or a visual pattern in
the Pauli terms is an observation that motivates a hypothesis. Exact-symmetry
claims require a commutator receipt; structure/control claims are admitted by a
controller marker that explicitly carries no algebraic certificate.

## Current architecture

| Boundary | Modules | Responsibility |
| --- | --- | --- |
| Problem contracts | `contracts.py`, `observations.py` | Split `PublicProblem`/safe observation from `PrivateEvaluationContext` |
| Candidate language | `ansatz_ir.py`, `macros.py` | Strict JSON IR and closed trusted macro registry |
| Trusted compilation | `compiler.py` | Validate complete IR and derive parameter/literal/macro provenance |
| Trusted measurements | `probes.py`, `evaluator.py` | Compute probe and optimization receipts under fixed protocols |
| State machine | `controller.py`, `research.py` | Sanitize external actions, own results/costs, enforce lifecycle |
| Durable record | `ledger.py`, `research_cli.py` | Keep a sealed authoritative external hash chain, signed path registry and monotonic head; mirror/verify/replay |
| CLI adapter | `research_cli.py`, `harness.py` | `inspect --agent-json` and sealed/unsealed `research init/step/status` |
| Legacy compatibility | `train.py` and non-research harness commands | Existing built-in family search and reference-aware `solve` |

`prepare.py` still loads source problem JSON and supplies the adapter. The safe
observation is produced by allowlisting fields from that object; the source
object itself is not handed to the agent.

## Security modes

`research init` without `--sealed` creates `local_unsealed` for development.
Its content hashes detect inconsistent edits, but an agent with workspace write
access can rewrite `context.json`, `checkpoint.json`, and the whole ledger
together. It is not a trust boundary. `step`/`status` and their Python APIs
nevertheless require sealed mode by default; local use must pass
`--allow-unsealed` or `require_sealed=False` explicitly.

For schema-v3 `sealed` mode, the evaluator/orchestrator keeps
`AUTOVQE_EVALUATOR_KEY` (at least 16 bytes) and
`AUTOVQE_EVALUATOR_ANCHOR_DIR` outside the agent boundary, runs
`research init --sealed`, and invokes sealed-default `step`/`status`
(`--require-sealed` is only an explicit marker). The directory must be outside
the run directory and contains `autovqe-<run_id>.anchor.json`, authoritative
`autovqe-<run_id>.events.jsonl`, and
`autovqe-path-<sha256(abs-run-path)>.registry.json`. The run-directory ledger
is only a verified mirror.

Domain-separated HMAC-SHA256 values authenticate the random 128-bit run ID,
context-bound checkpoint, path registry, and monotonic head. An external lock
serializes the entire verify → dispatch/append → mirror/checkpoint/anchor
update. Head advancement checks that the old tip remains at the old-count
prefix of the authoritative chain. This detects old-pair rollback, longer
forks, and run-path swaps. It requires both the key and all external artifacts
to remain outside agent control; it is not a digital signature, remote
anchoring service, or hidden evaluator.

## Trust flow

| Phase | Untrusted input | Trusted output/check |
| --- | --- | --- |
| Observe | Problem path supplied to CLI | Content-hashed safe `ObservationBundle` |
| Hypothesize | One of three closed claim schemas | Validation plus hypothesis/admission ledger event |
| Probe | Strict bounded exact-symmetry generator recipe | Exact normalized commutator, verdict, fixed cost |
| Design | `AnsatzSpec` and rationale | Closed macro resolution and complete audit |
| Evaluate | Candidate ID and stage request | Fixed optimizer/seed/budget, energy trace, circuit metrics |
| Decide | Revise/retire/commit/close-negative request | Lifecycle and evidence preconditions, event append, replayed state |

An external action cannot append `record_probe` or `record_evaluation`. It
cannot choose the optimizer, seed, objective-call allowance, or evaluation
cost. Energy/count claims placed in metadata remain untrusted; the evaluator
derives the authoritative parameter/gate values.

## One productive iteration

1. Read `observation.json` and cite concrete Pauli/locality/support/metadata
   evidence.
2. Choose exactly one supported schema:
   `exact_pauli_symmetry` with a generator, `ansatz_structure` with a family
   name, or `null_control`.
3. For exact symmetry, request the matching `normalized_commutator` probe.
   Structure/control claims instead receive an explicit non-algebraic admission
   marker.
4. If the commutator refutes the claim, revise or retire it. Do not submit a
   symmetry-preserving candidate merely because the pattern looked plausible.
5. If supported, encode one candidate in `AnsatzSpec`. Keep preparation,
   operations, parameters, sharing, and coefficients explicit. Mark
   the mandatory `metadata.enforcement`: `preserve` for exact symmetry,
   `unconstrained` for structure, or `diagnostic` for a null control. Before
   evaluation, also register a nonempty prediction or falsifier for any
   candidate that may be committed.
6. Request `audit`. Fix or revise any schema, macro, literal, or provenance
   violation before spending optimization budget.
7. Request `smoke` only for an audited candidate.
8. Request `promotion` only when the smoke result is worth the additional fixed
   cost. Each fixed probe and stage runs at most once; a new ID does not turn a
   deterministic repeat into independent evidence.
9. Commit the promoted candidate only with its promotion evidence and an
   evaluated competitor/control or documented non-dominance basis. Otherwise
   preserve negative results through `revise`/`retire`; after every branch is
   terminal, request a grounded `close_negative`.
10. Run `research status` and report the replayed state and ledger tip.

This ordering prevents a large candidate sweep from consuming budget before
its structural premise has been tested. It does not itself prove that the
chosen hypothesis class is complete.

## Lifecycle and budget

An exact hypothesis goes from `PROPOSED` directly to `SUPPORTED` on probe pass
or to `PROBED` on refutation; `PROBED` does not later advance to `SUPPORTED`.
Candidate state is
`CANDIDATE -> AUDITED -> SMOKE -> PROMOTED`, with failed evaluations moving to
`RETIRED` and explicit replacements leaving a `REVISED` link. A promoted
candidate cannot be revised or retired; it must remain live or be committed.
Every non-control candidate submission, including a replacement, preregisters
a prediction or falsifier before its fixed evaluations.

Current controller charges are:

- propose hypothesis: `0.10`;
- normalized commutator probe: `0.10`;
- submit candidate: `0.10`;
- audit: `0.25`;
- smoke: `2.00`;
- promotion: `6.00`;
- revise: `0.10`;
- retire/commit/close-negative: zero.

Agent-supplied costs are ignored and replaced by these values. An action that
would exceed the initialized total budget is rejected before append. Every
nonterminal action also leaves one ledger event slot reserved for the final
commit or negative close.

## Feedback granularity

The current controller returns a `ControllerReceipt` after every action:

- generated ledger event hashes;
- the probe or evaluation result;
- the full replayed `ResearchState`.

Evaluation receipts currently expose optimizer/seed, objective-call count,
energy and best-energy traces, compile audit, and declared-backend plus fixed
canonical resource views. The resource prefixes are `template_*`,
`generic_worst_*`, `final_*`, `canonical_template_*`,
`canonical_generic_worst_*`, and `canonical_final_*`. This is useful for
debugging but is not a deliberately blinded reward channel. Coarser
stage-dependent feedback and a hidden final score are future work.

## What the current boundary prevents

- Unknown or opaque circuit operations cannot enter the compiler.
- Candidate-supplied energy, gate counts, and parameter totals in the IR are
  rejected as unknown fields; metadata claims are ignored and values are
  recomputed.
- Every declared parameter must occur and is counted at each use; fan-out of a
  single parameter is capped at 64.
- Variational macros must be identity at zero; constant-only angles and fixed
  offsets are rejected.
- The controller audit restricts coefficient scales to a small fixed set and
  rejects numeric options.
- Global limits cap candidates at 256 logical operations, 128 unique trainable
  parameters, and 4096 IR nodes; rotations above locality two must name a
  declared Hamiltonian term.
- Smoke/promotion transpile both the symbolic template and three
  evaluator-owned generic bindings on the canonical all-to-all
  `{rz,sx,x,cx}` target. For each metric,
  `canonical_conservative_* = max(canonical_template_*,
  canonical_generic_worst_*)`; the conservative two-qubit count must be
  `<=512`, total gate count `<=2048`, and depth `<=1024`.
- Generator recipes reject extra fields, duplicate labels, more than 256
  terms, non-real/non-finite coefficients, and coefficient magnitude above
  `1e6`.
- A candidate marked `enforcement: preserve` is audited operation-by-operation
  against its parent claim's machine-readable exact Pauli charge, including
  the explicit reference sector.
- Claim schemas and enforcement modes are closed; structure/control admission
  is recorded as non-algebraic, and a diagnostic null control cannot promote.
- `XYExchange` and `IsotropicExchange` require a controller-supported exact
  symmetry parent in addition to per-operation commutator audit; being in the
  macro registry is not applicability evidence.
- Candidate reference preparation must exactly match the public occupation
  hint, and every promotable candidate needs at least one operation and one
  trainable parameter.
- Smoke/promotion require a nontrivial improvement over the zero-parameter
  baseline; promotion must also reproduce a passed smoke within `5e-4`.
- A candidate cannot skip `SUPPORTED`, audit, smoke, or promotion transitions.
- A positive commit or accepted negative close cannot accept further events.
- Search/run limits cap active hypotheses at three, active candidates per
  hypothesis at two, events at 200, and each serialized action at 1,000,000
  bytes.
- In sealed mode, an external lock covers verification through authoritative
  append and checkpoint/anchor publication; ancestry requires the old anchored
  tip to remain at the old-count prefix. The workspace ledger is only a mirror.

## What it does not yet prevent

- `PauliRotation` still covers arbitrary one/two-body active words and
  Hamiltonian-derived higher-locality words. The physical caps use one global
  canonical target; there are no per-problem macro/native-backend profiles yet.
- Promotion is a local baseline-improvement/repeatability check, not Pareto
  selection or hidden generalization.
- `local_unsealed` is not tamper-evident against an agent that can rewrite the
  full run. Sealed mode has evaluator-owned external ledger/anchor/registry
  files and a local exclusive lock, but not a remote/durable anchoring service;
  an interrupted cross-file update may require trusted recovery.
- There is no external hidden-evaluation service, automatic agent/LLM adapter,
  or asynchronous search scheduler.
- Only the exact normalized Pauli commutator is an external probe request.
  Reference moments are an internal exact-symmetry audit primitive;
  approximate/group/fermionic probes are not present.

These are explicit roadmap items rather than implicit guarantees.

## Legacy loop

The older autoresearch-style path still edits/builds candidates in `train.py`
and uses `campaign`/`benchmark`/`solve`. Keep it for compatibility and regression
coverage. It has human-facing classification and reference-aware tolerance
logic, so it is not interchangeable with the safe typed research loop.

The architectural lesson from
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) remains the
stable evaluator boundary. In the new path, the editable experiment surface is
the serialized `AnsatzSpec` action rather than arbitrary evaluator code.
