# Trusted terminal-result artifact

`meta_agent.operator export` converts an already terminal research ledger into
one deterministic JSON result. It does not accept an ansatz, energy, resource
count, or parameter binding from the agent at export time.

```powershell
uv run python -m meta_agent.operator export `
  --evaluator-run D:\AutoVQE-Private\runs\campaign-001 `
  --output D:\AutoVQE-Private\runs\campaign-001\published-result.json
```

The command requires a sealed run by default. A local development run must opt
in with `--allow-unsealed`; its artifact is permanently classified
`UNTRUSTED_LOCAL_INTEGRATION`. If `--output` is omitted, the destination is
`<evaluator-run>/final_result.json`. A custom destination must remain inside
the evaluator-owned run tree and its parent must already exist. Export rejects
symbolic links, Windows reparse points, changed parent identities, agent-bundle
overlap, and different existing content. New artifacts use exclusive creation;
repeating an export to an identical regular file is idempotent. Copy a verified
artifact elsewhere only as a separate, explicitly trusted publication step.

## Schema version 1

Every artifact has these top-level fields:

- `schema_version` and `artifact_type` identify the format.
- `decision` is exactly `positive_commit` or `negative_close`.
- `trust` records the security mode, rollback/tamper properties, benchmark
  classification, unmet benchmark prerequisites, and any warning. A sealed
  artifact verifies the AutoVQE protocol records, not the surrounding OS,
  account, network, model configuration, or an external holdout scorer. The
  artifact alone is therefore always `benchmark_grade: false`.
- `provenance` binds the campaign, model label, public problem identity, raw
  input hash, observation hashes, run/session IDs, terminal ledger tip,
  evaluator source-tree hash, and Git identity.
- `budget` records evaluator-owned total, spent, and remaining units.
- `result` is the decision-specific payload described below.
- `limitations` prevents a promotion from being presented as a global-optimum,
  ground-state, cross-problem, or independently computed Pareto claim.
- `artifact_sha256` is a domain-separated SHA-256 over every other field in
  canonical JSON form. It makes repeated exports and accidental corruption
  detectable; the operator's sealed filesystem/key boundary, not this bare
  digest, provides authenticity.

### Positive commit

The positive payload contains the committed `AnsatzSpec`, its semantic
candidate hash, controller-accepted evidence IDs and comparison, and the
trusted evidence records. The controller allows only one fixed promotion per
candidate. Export requires exactly one cited, passed promotion whose evaluator
candidate hash matches the committed spec.

The `promotion` object contains:

- `optimized_parameter_binding`, emitted by the evaluator when it performed
  that promotion and checked against the exact declared parameters;
- evaluator energy and best-energy traces, baseline improvement, optimizer,
  seed, and objective-call count;
- compiler audit plus declared-backend and fixed canonical resource metrics;
- the resource-eligibility decision; and
- a content hash of the replayed evaluation record.

These fields come from the evaluator-owned `record_evaluation` event. Candidate
or commit metadata that claims a score, optimized angle, optimizer setting, or
resource count is not copied into the artifact.

The evaluator persists an optimized binding in its private replay ledger, but
the bridge recursively removes `optimized_parameter_binding` and the legacy
name `best_values` from every nonterminal status and action receipt. The values
become public only through this trusted export after a positive commit. This
removes a direct publication channel; it is not a secrecy guarantee because a
deterministic optimizer can be rerun from public inputs. Anti-hardcoding comes
from the typed IR, compiler literal rules, semantic deduplication, and
evaluator-derived parameter/resource accounting.

### Negative close

The negative payload contains the controller-accepted close reason, evidence
IDs, and the corresponding replayed probe/evaluation records. Private optimizer
bindings are redacted from those evidence records. It has no committed ansatz,
selected binding, or promotion, so a completed negative investigation cannot
be mistaken for a successful ansatz result.

## Reproduction boundary

For a positive result, load the exact input identified by
`provenance.problem.raw_input_sha256`, compile `result.ansatz_spec` with the
source tree identified by `provenance.source.evaluator_source_sha256`, and bind
the values in `result.promotion.optimized_parameter_binding`. This reproduces
the evaluator's selected circuit point for the cited promotion. It does not
rerun the optimizer or establish that another optimizer seed could not find a
different point.
