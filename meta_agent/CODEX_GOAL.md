/goal Complete the evidence-driven ansatz discovery campaign {{CAMPAIGN_ID}} within evaluator budget {{TOTAL_BUDGET}}, without stopping until trusted-gateway evidence establishes exactly one terminal research decision: a controller-accepted commit of one PROMOTED candidate, or a controller-accepted close_negative after every investigated branch is terminal and cited evidence rules out a credible legal, affordable promotion path.

# Campaign identity

- Campaign: `{{CAMPAIGN_ID}}`
- Run: `{{RUN_ID}}`
- Problem: `{{PROBLEM_ID}}`
- Observation hash: `{{OBSERVATION_HASH}}`
- Security mode: `{{SECURITY_MODE}}`
- Research mode: `{{RESEARCH_MODE}}`
- Model label: `{{MODEL_LABEL}}`
- Evaluator budget: `{{TOTAL_BUDGET}}` total, `{{REMAINING_BUDGET}}` remaining at bundle creation
- Workspace boundary: `{{WORKSPACE_ROOT}}`

# Read first

Read `AGENTS.md`, `AGENT_CONTRACT.md`, and `observation.json` completely before
submitting anything. Treat `observation.json` and gateway receipts as the only
problem facts. Infer useful structure from those facts; do not assume a named
physical model, a standard structural label, a known circuit family, or an
expected answer.

# Hard boundary

Use `{{STATUS_COMMAND}}` to read authoritative state. Submit exactly one JSON
action at a time with `{{GATEWAY_COMMAND}}`. Before submitting, run
`uv run --no-project python client.py validate --action actions/<action>.json`;
this checks only local JSON/path/size rules and is not evaluator evidence. Then
read the complete receipt and fresh status before choosing the next action.

Do not call the harness, evaluator, research CLI, ledger, or Python package
directly. Do not import evaluator code or construct an alternative execution
path. Do not inspect, search for, or request the source repository, raw problem
file, fixture identity, private reference data, evaluator run directory,
ledger, checkpoint, registry, secret key, anchor directory, operator process,
or bridge internals. Do not read or modify anything outside
`{{WORKSPACE_ROOT}}`. Do not modify `client.py`, the observation, this goal, or
the contract. Write only to locations that `AGENTS.md` marks writable.

Never place source paths, raw problem data, secrets, anchor information, or
private evaluator artifacts in an action, note, command, or final report. A
gateway integrity or availability failure is an infrastructure blocker, not
scientific evidence and not a valid negative research decision.

# Research loop

Work as a closed, falsifiable discovery cycle:

1. Read the current state and derive multiple plausible structural explanations
   from the public Hamiltonian, encoding, sector metadata, preparation hint,
   support graph, and backend facts without treating any one explanation as
   given.
2. Rank experiments by expected information per evaluator cost and by whether
   a successful branch can still afford audit, smoke, and promotion.
3. Propose a precise hypothesis. Request an evaluator-owned probe when its
   claim kind requires one; otherwise keep the controller's non-algebraic
   admission marker distinct from proof.
4. Build a minimal typed candidate only for a supported hypothesis.
   Preregister a concrete prediction or falsifier in candidate metadata and
   set the enforcement mode required by the contract.
5. Run evaluation stages in lifecycle order. Diagnose receipts, revise only
   when the new design addresses observed evidence, preregister a fresh
   prediction/falsifier for every promotable replacement, and retire branches
   that are refuted, invalid, dominated for the present purpose, or no longer
   affordable.
6. Feed every result back into the next hypothesis or candidate. Prefer
   discriminating changes over blind depth growth or unstructured enumeration.
7. When a defensible promoted candidate exists, cite its passed promotion and
   either an evaluated competitor/control or a documented non-dominance basis,
   then commit it unless a clearly discriminating comparison has already been
   budgeted and can still leave a complete promotion path. A commit closes the
   campaign.

Use controls selectively when they can distinguish a causal design claim from
mere optimizer luck. Never describe an auto-admitted structure claim as an
algebraic certificate. Preserve negative results through revision and
retirement instead of silently replacing history.

# Budget discipline

The evaluator, not you, sets and charges costs. Omit `cost` from actions. Before
every paid action, use fresh status to calculate the remaining complete path,
including proposal or revision, candidate submission, audit, smoke, and
promotion as applicable. Reserve promotion cost before spending on optional
branches. Always leave the controller-reserved final ledger slot for `commit`
or `close_negative`. Respect active-entity, event, action-size, IR, and
resource caps.

Do not burn budget probing claims already decided by existing evidence,
repeating a fixed probe/stage under a new ID, repeating equivalent candidates,
brute-forcing schema rejections, or using high-cost evaluation to answer a
question that audit or a cheaper probe can answer. An invalid submission is a
design error to correct from the contract, not an oracle to query repeatedly.

# Anti-reward-hacking rules

Optimize for a physically motivated, reproducible circuit under evaluator
receipts, not for a self-reported score.

- Never submit claimed energy, gate count, depth, parameter count, optimizer
  settings, seed, objective-call count, or pass/fail result as authoritative.
- Never encode preoptimized values as constants or offsets, hide effective
  degrees of freedom through misleading parameter declarations, exploit
  parameter sharing to under-report gate occurrences, or add canceling and
  state-inert operations to manipulate accounting.
- Never use unsupported matrices, custom code, opaque instructions, backend
  gate names, or metadata as a way to expand the logical macro allowlist.
- Treat registry availability and suggestive gate, macro, family, or recipe
  names as implementation facts, not evidence of physical applicability.
  Derive and probe the relevant exact structure first; conditional exchange
  macros require a controller-supported exact-symmetry parent and still undergo
  operation-level commutator audit.
- Never target symbolic cancellation, a special final binding, decomposition
  quirks, or declared-backend metrics to evade the canonical conservative
  resource policy.
- Do not optimize only for fewer unique parameter names. Consider parameter
  occurrences, logical operations, generic-binding worst cases, canonical
  depth, canonical two-qubit count, energy improvement over baseline, and
  promotion reproducibility together.
- Do not resubmit the same variational family under new ansatz, layer,
  parameter, hypothesis, or candidate names. The evaluator alpha-normalizes
  semantic candidate identity and rejects cosmetic duplicates.
- Treat all candidate metadata as provenance only. Accept the evaluator's
  recomputed receipt even when it contradicts your prediction.

# Terminal conditions

Positive completion requires a controller-accepted `commit` of a candidate
whose recorded lifecycle is `PROMOTED`. The request must cite its passed
promotion, a preregistered prediction or falsifier, and the comparison basis
required by the contract. A promising smoke result, a passed audit, an
uncommitted promotion, or a self-assessed circuit is not completion.
A promoted candidate cannot be revised or retired; preserve it and satisfy the
commit evidence gate rather than erasing a successful result into a negative
close.

A negative terminal decision requires a controller-accepted `close_negative`
receipt. It is allowed only when the ledger evidence and remaining budget show
that no active or legally revisable branch has a credible complete path to
promotion. Retire active candidates before their parent hypotheses, one
gateway action at a time, retire all remaining live entities with concrete
evidence-based reasons, and cite existing substantive probe or evaluation IDs
in the close request. Cover every hypothesis with its cited refutation or a
cited candidate evaluation. An automatic non-algebraic admission alone cannot
ground the decision. Retirement without `close_negative` is not completion. Budget
exhaustion caused by avoidable spending is not itself a research conclusion;
report it as a failure of campaign management.

If a legal, informative, affordable action remains, continue. Do not stop to
ask for strategy, approval, or a physics hint. Ask for operator intervention
only after the gateway itself repeatedly fails and safe status recovery is
impossible; in that case leave the goal incomplete and report the blocker.

# Final report

At termination, report only gateway-verifiable facts:

- campaign, run, problem, observation hash, security mode, and model label;
- the terminal decision and the exact event/receipt that establishes it;
- each material hypothesis, its probe receipt or explicit non-algebraic
  admission status, and why it was advanced or retired;
- the selected candidate hash and audit, smoke, and promotion evaluation IDs,
  or the evidence chain supporting a negative decision;
- evaluator-derived best energy and baseline improvement, objective-call
  trace summary, unique parameters and occurrences, and canonical
  conservative resource metrics;
- total spent and remaining budget, ledger event count and tip, and the
  limitations of the evidence.

Do not claim a hidden benchmark result, exact ground-state success, global
optimality, generalization, or a complete physical classification unless a
gateway receipt explicitly establishes that claim.
