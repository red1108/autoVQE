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
- Modify `train.py` — this is the only file you edit during autonomous research. Everything inside that file is fair game: ansatz, optimizer, parameterization, restart strategy, training loop, and other VQE choices.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed harness and evaluation helpers.
- Modify `problem.json`. The problem is fixed during a run.
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`.
- Break the plain summary output format in a way that makes simple parsing harder.
- Modify the evaluation harness. the `energy_from_circuit` function is the only way to get energy estimates, and it must not be modified.

**The goal is simple: get the lowest energy.**

**Simplicity criterion**: All else being equal, simpler is better. A tiny energy improvement that adds ugly complexity is usually not worth it. A tiny energy improvement from reducing the number of gates or parameters is a great improvement — that's a simplification win. When evaluating whether to keep a change, weigh the gate count against the improvement magnitude. A 1% energy improvement that doubles the parameter or gate count is probably not worth it. A 1% energy improvement from reduces the parameter/gate count? Definitely keep. An improvement of ~0 but much simpler structure? Keep.

Once a run reaches the reference energy, or is effectively tied with the current best energy, do not stop. Continue searching for a simpler solution that preserves the same or nearly the same energy while reducing, in order of priority: `num_params`, `twoq_count`, `total_gate_count`, and `depth`. Only stop this compression phase when repeated experiments fail to find a candidate that matches or improves the current energy while improving one or more of those four metrics.

**The first run**: Your very first run should always establish the baseline, so run the training script as is.

## Output Format

Once the script finishes it prints a summary like this:

```text
---
energy:           -1.234567
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

1. git commit hash (short, 7 chars)
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

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with one experimental idea by directly hacking the file.
3. git commit.
4. Run the experiment:
   `uv run train.py > run.log 2>&1`
5. Read the key results:
   `grep "^energy:\|^singleq_count:\|^twoq_count:\|^total_gate_count:\|^num_params:" run.log`
6. If the grep output is empty, the run crashed. Read the traceback with `tail -n 50 run.log` and decide whether to fix and retry or log a crash and move on.
7. Record the result in `results.tsv`. Leave `results.tsv` untracked by git.
8. If energy improved, keep the commit and advance the branch.
9. If energy is equal or worse, reset back to where you started unless the code became clearly simpler at no performance cost.
10. If energy is equal or nearly equal to the best known result, switch into compression mode and keep pushing for fewer parameters, fewer two-qubit gates, fewer total gates, and lower depth before considering the search exhausted.

**Timeout**: If you are using a fixed time budget for comparison, enforce the same budget for every candidate and kill obviously hung runs.

**Crashes**: If a run crashes because of a small implementation mistake, fix it and retry. If the idea itself is broken, log `crash` in the TSV and move on.

**Never stop**: Once the experiment loop begins, do not pause to ask whether you should continue unless the human explicitly interrupts or changes the plan.
