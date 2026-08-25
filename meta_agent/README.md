# Codex Meta-Agent Workspace

This directory contains tracked templates and operator tooling for launching a
Codex `/goal` ansatz-discovery campaign. Codex should work in a generated
agent-only bundle, not in this source repository and never in the evaluator
run directory.

The bundle gives Codex a public observation, a rendered goal, bundle-local
instructions, the exact agent-side contract, and a narrow `client.py`. It does not contain the raw
problem, AutoVQE source, evaluator state, private references, key, ledger, or
anchor.

## Two security modes

`local_unsealed` is a development mode. It exercises the same one-action
gateway and lifecycle, but processes owned by the same user can rewrite all
local files. Hashes detect inconsistency, not a malicious agent with equivalent
filesystem authority. Results from this mode must be labeled untrusted.

`sealed` is the research mode. A trusted operator process owns the source,
raw problem, evaluator run, authoritative ledger, HMAC key, anchor, registry,
and bridge service. Codex owns only the generated bundle. This separation is a
real trust boundary only when operating-system permissions prevent the Codex
process from reading or writing evaluator-owned paths and environment.

| Location | Owner | Codex access |
| --- | --- | --- |
| Source repository and campaign manifest | Trusted operator | None during the run |
| Raw problem and private evaluation inputs | Trusted operator | None |
| Evaluator run, ledger, checkpoint, anchor, registry | Trusted operator | None |
| Evaluator key and anchor environment variables | Trusted operator process only | None |
| Generated agent bundle | Codex account/process | Read protected inputs; direct edits only in `actions/` and `journal.md`; bundled client creates requests in `outbox/` |
| Bridge endpoint | Trusted operator | Requests only through bundled `client.py` |

The key and anchor directory must never be copied into the bundle, inherited by
the Codex process, printed in logs visible to Codex, or placed under a shared
parent that Codex can traverse.

## Authenticated bridge publications

Every generated session has a campaign-specific Merkle–Lamport SHA-256 public
root. The operator keeps the corresponding random seed and signing cursor only
in the evaluator run. Each `latest_status.json` and request receipt is signed
over its complete canonical content, including message type, protocol, session,
campaign, session-manifest hash, request/action hash when applicable, and the
embedded public evaluator status. The standard-library-only client verifies
the one-time Lamport signature and Merkle proof before it displays a status,
uses its cursor, or accepts a receipt. A trusted receipt replay republishes the
same signed bytes; it never spends the same one-time key on a second payload.

This authentication is defense in depth, not the filesystem boundary. An
agent that can rewrite both `client.py` and `session.json` can replace the
verifier and public root. For a scored run, use a separate Codex identity or
sandbox and OS ACLs: make `client.py`, `session.json`, instructions,
observation, and `inbox/` read-only to Codex; permit the operator to write
`inbox/`; permit Codex to write `actions/`, `journal.md`, and bridge requests
under `outbox/`; and deny Codex all access to the evaluator run and signing
seed. Same-account `local_unsealed` runs do not satisfy this condition.

The hash-based signer has 512 one-time publication slots and fails closed when
they are exhausted. Status republishes, rejected requests, and successful
receipts all consume slots, so restart/status polling is not free. Each signed
status reports authenticated `publication_capacity` telemetry. Never restore,
copy backward, or reconstruct `publication_signer.json` independently of the
campaign: rolling its cursor back can reuse a one-time key and invalidates the
publication-authentication guarantee. If signer state or its lock cannot be
proven current after a crash or backup restore, preserve the run for audit,
abandon that campaign, and prepare a fresh bundle/evaluator run with a new
public root. Do not reset the cursor or delete a signer lock merely to continue.

A signature proves origin and content, not freshness: replaying an old
authentic status is detectable only through the expected timestamp/cursor
context. The trusted compare-and-swap cursor check prevents such a stale status
from authorizing a ledger mutation, but operators should still retain and
monitor the latest published cursor and remaining publication capacity.

## Prepare a development bundle

From the trusted source checkout, create a local-unsealed campaign:

```powershell
uv run python -m meta_agent.operator prepare --campaign meta_agent/campaigns/discovery_001.json --security local_unsealed --model-label "<exact Codex model and reasoning setting>"
```

Use the paths printed by `prepare`. In a trusted operator shell, start the
bridge for the generated evaluator run:

```powershell
uv run python -m meta_agent.operator serve --evaluator-run .autovqe-runtime/evaluator/discovery_001 --allow-unsealed
```

This is convenient for integration testing, but moving Codex into a separate
bundle does not make `local_unsealed` cryptographically trusted.

## Prepare a separated sealed bundle

In the trusted operator shell only, set `AUTOVQE_EVALUATOR_KEY` to an
evaluator-held secret of at least 16 bytes and set
`AUTOVQE_EVALUATOR_ANCHOR_DIR` to a durable operator-owned directory outside
the source checkout, evaluator run, and agent-writable tree. The anchor value,
agent bundle, and evaluator run must use absolute paths. Then run:

```powershell
$agentBundle = 'D:\AutoVQE-Agent\discovery_001'
$evaluatorRun = 'D:\AutoVQE-Private\runs\discovery_001'
$anchorDir = 'D:\AutoVQE-Private\anchors'
$env:AUTOVQE_EVALUATOR_KEY = '<operator-only secret of at least 16 bytes>'
$env:AUTOVQE_EVALUATOR_ANCHOR_DIR = $anchorDir
uv run python -m meta_agent.operator prepare --campaign meta_agent/campaigns/discovery_001.json --security sealed --agent-bundle $agentBundle --evaluator-run $evaluatorRun --model-label "<exact Codex model and reasoning setting>"
```

Start the bridge from that same trusted security context:

```powershell
uv run python -m meta_agent.operator serve --evaluator-run $evaluatorRun
```

Run the bridge under an account that can access the evaluator paths. Run Codex
under a different account or sandbox that can access `<agent-only-path>` but
cannot access `<trusted-path>`, this checkout, the anchor directory, or the
operator environment. Do not open a parent directory that contains both sides
as the Codex workspace.

## Open the generated bundle in Codex

1. Open the exact generated agent-bundle directory as the Codex workspace and
   make it the terminal working directory.
2. Confirm that the bundle contains the rendered `GOAL.md`,
   `AGENT_CONTRACT.md`, `AGENTS.md`, `observation.json`, `session.json`,
   `journal.md`, and `client.py`, but no raw problem, source package, evaluator
   run, key, or anchor artifacts. `AGENTS.md` permits writes only to
   `journal.md` and direct children of `actions/`; `inbox/` and `outbox/` are
   bridge-managed.
3. Read the latest evaluator-published cached state and its publication timestamp
   with:

   ```bash
   uv run --no-project python client.py status
   ```

4. Copy the complete rendered contents of `GOAL.md` into a new Codex
   message. Its first line is already a valid `/goal <objective>` command.
5. During the run, Codex writes one JSON action in the bundle's documented
   writable area. It can catch strict JSON/path/size mistakes locally with:

   ```bash
   uv run --no-project python client.py validate --action actions/<file>.json
   ```

   This preflight does not run evaluator semantics and is not evidence. Submit
   the action with:

   ```bash
   uv run --no-project python client.py submit --action actions/<file>.json
   ```

6. Keep the operator bridge running. A successful `client.py status` read is
   not a bridge-liveness check; confirm the trusted `serve` terminal is still
   running and that `published_at_utc` advances after submissions. Inspect
   operator-side state only from the
   trusted shell; never paste it back into the Codex thread.

The client accepts one action per call. It is not permission to invoke another
AutoVQE entry point or inspect the other side of the bridge.

## Export the terminal result

Once the bridge has accepted a terminal `positive_commit` or `negative_close`,
run this only from the trusted operator environment:

```powershell
uv run python -m meta_agent.operator export `
  --evaluator-run D:\AutoVQE-Private\runs\discovery_001
```

The sealed evaluator is required by default. A development-only local run must
explicitly pass `--allow-unsealed` and remains permanently labeled untrusted.
The export includes evaluator-owned optimized parameter values only after a
positive terminal decision; bridge status and action receipts do not publish
those values during search. The full schema and limitations are in
[`docs/result_artifact.md`](../docs/result_artifact.md).

## Why the goal prompt is shaped this way

[OpenAI's official `/goal` guidance](https://learn.chatgpt.com/use-cases/follow-goals)
recommends one durable objective, one verifiable stopping condition, the files
to read first, the commands or artifacts that prove progress, and checkpointed
work. The rendered prompt follows the recommended opening form:

```text
/goal Complete <objective> without stopping until <verifiable end state>.
```

Use `/goal` with no argument to inspect goal status. Codex also supports
`/goal pause`, `/goal resume`, and `/goal clear`. If the command is unavailable,
enable the goals feature in Codex before starting the research run; do not
replace the goal with an ordinary one-turn request.

## Tracked versus generated data

Tracked files define the reproducible protocol: operator code, campaign
manifests, this template documentation, and tests. Generated bundles,
evaluator runs, ledgers, bridge state, action scratch files, receipts, notes,
and secrets are runtime data and must remain ignored or live outside the
repository.

The default in-repository runtime root is `.autovqe-runtime/`. Do not force-add
anything beneath it. An explicit sealed `--agent-bundle` may live elsewhere;
apply an equivalent ignore and retention policy there. Preserve trusted
evaluator records according to the experiment's audit policy, but never commit
or expose them to the action-producing agent.

Preparation freezes hashes of the evaluator source, problem, prompt, contract,
client, and observation. Do not edit the trusted checkout or raw problem after
`prepare`. There is deliberately no force/clean reuse path: quarantine an old
workspace and prepare fresh empty bundle/evaluator paths for every campaign.

## Operator checklist

- Campaign placeholders render to concrete values; no `{{...}}` token remains
  in the generated bundle.
- The trusted `serve` terminal is running, and
  `uv run --no-project python client.py status` reports the intended campaign,
  problem ID, observation hash, budget, security mode, and a current
  `published_at_utc` before `/goal` starts.
- The source checkout and raw problem remain unchanged after `prepare`.
- The bundle process cannot traverse the source, evaluator, or anchor paths.
- The Codex environment does not contain either evaluator environment
  variable.
- The bridge is the only route from bundle actions to evaluator receipts.
- One campaign uses one fresh bundle and evaluator run; do not reuse a stale
  workspace for a different run.
- Final claims cite gateway receipts and report security mode and limitations.
