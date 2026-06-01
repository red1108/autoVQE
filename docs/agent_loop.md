# Agent Loop

AutoVQE should stay close to the autoresearch shape:

- one fixed evaluator,
- one editable experiment surface,
- one control harness,
- one short agent protocol,
- domain knowledge in docs, not in giant Python policy trees.

The agent's job is not to run every known VQE method. Its job is to make one
small, falsifiable change, run the same measurement loop, and keep the change
only when the evidence improves.

## Recommended Architecture

| Layer | File | Role |
| --- | --- | --- |
| Evaluator | `autovqe/prepare.py` | Load Hamiltonians, compute references, measure energy and compiled gate counts. |
| Experiment surface | `autovqe/train.py` | Build candidate circuits and optimize parameters. This is the main editable file. |
| Harness | `autovqe/harness.py` | Inspect Hamiltonians, isolate runs, compare against targets, summarize evidence. |
| Protocol | `docs/agent_protocol.md` | Short operating contract for agents. |
| Knowledge base | `docs/` | Ansatz decision rules, benchmark notes, source links, and future research ideas. |

Keep the harness factual. It should answer "what does this Hamiltonian look
like?" and "did this run pass?" Avoid turning it into a large expert system.
When a method is domain knowledge rather than executable measurement, put it in
`docs/`.

## Agent Workflow

1. Read `docs/agent_protocol.md`.
2. Run `uv run python -m autovqe.harness inspect --problem <problem>`.
3. Open the relevant doc from `docs/`.
4. Add or adjust one candidate in `autovqe/train.py`.
5. Run a small campaign.
6. If it wins, run `solve`.
7. Keep the edit only if the target check or ledger proves improvement.

## What To Avoid

- Do not add dataclasses that only restate prose.
- Do not encode every possible ansatz as `if/else` policy in `autovqe/harness.py`.
- Do not count classical post-processing as a VQE pass. Raw circuit energy is
  the benchmark result.
- Do not let generated experiment folders accumulate in the repo root.
- Do not broaden the search before the Hamiltonian-derived family has received
  a fair smoke test.

## Source Model

Karpathy's `autoresearch` keeps the repo deliberately small: the evaluator is
not modified, the experiment surface is agent-editable, and the protocol is the
human-written research-org prompt. The key transfer to AutoVQE is the boundary,
not the exact file sizes.

Reference: https://github.com/karpathy/autoresearch
