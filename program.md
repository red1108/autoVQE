# AutoVQE

This is an experiment to have the LLM do its own VQE research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, for example `mar19`. The branch `autovqe/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autovqe/<tag>` from the current main branch.
3. **Read the in-scope files**: The repo is intentionally small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, problem loading, validation, exact reference computation, transpilation, and evaluation helpers. Do not modify.
   - `train.py` — the file you modify. Ansatz, initialization, optimizer, and VQE loop.
   - `problem.json` — the fixed Hamiltonian and backend description for this run.
4. **Verify the environment**: If the environment is not ready, tell the human to run `uv sync`. If the problem needs validation, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row if it does not already exist. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is evaluated on the single fixed Hamiltonian in `problem.json`.

You launch the baseline simply as:

```bash
uv run train.py
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit during autonomous research. Everything inside that file is fair game: ansatz, initialization, optimizer, parameterization, restart strategy, and other VQE choices.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed harness and evaluation helpers.
- Modify `problem.json`. The problem is fixed during a run.
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`.
- Break the plain summary output format in a way that makes simple parsing harder.

**The goal is simple: get the lowest energy.** If a reference energy is available, `delta_e` is useful context but not the primary decision rule. Hardware-aware metrics matter, but they are secondary. Track `singleq_count`, `twoq_count`, `total_gate_count`, `depth`, and `num_params`, but do not keep a change that worsens energy unless it clearly simplifies the code at no real performance cost.

**Simplicity criterion**: All else being equal, simpler is better. A tiny energy improvement that adds ugly complexity is usually not worth it. A tiny energy improvement from deleting code probably is worth it. Equal energy with materially simpler code is a win.

**The first run**: Your very first run should always establish the baseline, so run the training script as is.

## Output Format

Once the script finishes it prints a summary like this:

```text
---
energy:           -1.234567
reference_energy: -1.235100
delta_e:          0.000533
singleq_count:    40
twoq_count:       24
total_gate_count: 64
depth:            39
num_params:       16
eval_calls:       143
total_seconds:    21.4
```

If a metric is unavailable, omit the line instead of inventing placeholders.

## Logging Results

When an experiment is done, log it to `results.tsv` as tab-separated values, not comma-separated values.

The TSV has a header row and 8 columns:

```text
commit	energy	singleq_count	twoq_count	total_gate_count	num_params	status	description
```

1. git commit hash, short form
2. best energy achieved, for example `-1.234567`
3. compiled single-qubit gate count
4. compiled two-qubit gate count
5. compiled total gate count
6. VQE parameter count
7. status: `keep`, `discard`, or `crash`
8. short text description of what the experiment tried

Example:

```text
commit	energy	singleq_count	twoq_count	total_gate_count	num_params	status	description
a1b2c3d	-1.857275	8	2	10	8	keep	baseline hea layers=2
b2c3d4e	-1.857100	12	2	14	12	discard	deeper hea with same optimizer
c3d4e5f	0.000000	0	0	0	0	crash	broken parameter binding
```

## The Experiment Loop

The experiment runs on a dedicated branch such as `autovqe/mar19`.

LOOP FOREVER:

1. Look at the current git branch and commit.
2. Tune `train.py` with one experimental idea by directly editing the file.
3. git commit.
4. Run the experiment:
   `uv run train.py > run.log 2>&1`
5. Read the key results:
   `grep "^energy:\|^singleq_count:\|^twoq_count:\|^total_gate_count:\|^num_params:" run.log`
6. If the grep output is empty, the run crashed. Read the traceback with `tail -n 50 run.log` and decide whether to fix and retry or log a crash and move on.
7. Record the result in `results.tsv`. Leave `results.tsv` untracked by git.
8. If energy improved, keep the commit and advance the branch.
9. If energy is equal or worse, reset back to where you started unless the code became clearly simpler at no performance cost.

**Timeout**: If you are using a fixed time budget for comparison, enforce the same budget for every candidate and kill obviously hung runs.

**Crashes**: If a run crashes because of a small implementation mistake, fix it and retry. If the idea itself is broken, log `crash` in the TSV and move on.

**Never stop**: Once the experiment loop begins, do not pause to ask whether you should continue unless the human explicitly interrupts or changes the plan.
