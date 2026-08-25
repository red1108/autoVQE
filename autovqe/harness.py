"""Installed command-line interface for AutoVQE."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .ansatz import compile_ansatz
from .evaluator import EvaluationProtocol, evaluate_public_problem
from .problem import DEFAULT_PROBLEM_PATH, canonical_data, load_problem, observe_problem
from .research import (
    execute_action_file,
    initialize_run,
    render_json,
    run_result,
    run_status,
)


DEFAULT_RUN_DIR = Path(".autovqe-runtime/research")


def _json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def _inspect(path: str | Path, *, as_json: bool) -> int:
    problem = load_problem(path)
    observation = observe_problem(problem)
    if as_json:
        print(_json(observation))
        return 0
    structure = observation.structure
    print(f"problem: {problem.problem_id}")
    print(f"qubits: {problem.num_qubits}")
    print(f"pauli_terms: {structure.term_count}")
    print(f"max_locality: {structure.max_locality}")
    print(f"locality_counts: {dict(structure.locality_counts)}")
    print(f"support_edges: {structure.support_graph_edge_count}")
    print(f"declared_symmetries: {list(problem.sector.symmetries)}")
    print(f"basis_gates: {list(problem.backend.basis_gates)}")
    return 0


def _check() -> int:
    problem_document = {
        "name": "self_check",
        "pauli_terms": [
            {"pauli": "IZ", "coeff": -1.0},
            {"pauli": "ZI", "coeff": -1.0},
            {"pauli": "XX", "coeff": 0.2},
        ],
        "basis_gates": ["rz", "sx", "x", "cx"],
        "coupling_map": [[0, 1], [1, 0]],
        "initial_state_hint": [1, 0],
    }
    spec = {
        "version": 1,
        "name": "self_check_ansatz",
        "num_qubits": 2,
        "parameters": [{"name": "theta"}],
        "operations": [
            {
                "macro": "PauliRotation",
                "qubits": [0],
                "parameters": {
                    "angle": {
                        "constant": 0.0,
                        "terms": [{"parameter": "theta", "coefficient": 1.0}],
                    }
                },
                "options": {"pauli": "Y"},
            }
        ],
    }
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "problem.json"
        path.write_text(json.dumps(problem_document), encoding="utf-8")
        problem = load_problem(path)
        observation = observe_problem(problem)
        compiled = compile_ansatz(spec)
        evaluated = evaluate_public_problem(
            problem,
            spec,
            protocol=EvaluationProtocol(max_evals=8, restarts=1, seed=3),
        ).result
        checks.extend(
            (
                ("problem loads", problem.num_qubits == 2),
                ("observation stays compact", "pauli_terms" not in _json(observation)),
                ("structure analysis runs", observation.structure.term_count == 3),
                ("typed ansatz compiles", compiled.audit["unique_trainable_params"] == 1),
                ("compiler derives parameter use", compiled.audit["parameter_occurrences"] == {"theta": 1}),
                ("evaluator optimizes candidate", evaluated.valid and evaluated.best_energy is not None),
                ("evaluator owns resource counts", bool(evaluated.metrics)),
                ("evaluator owns optimized parameters", bool(evaluated.optimized_parameter_binding)),
            )
        )
    for label, passed in checks:
        print(f"{'ok' if passed else 'FAIL'}: {label}")
    return 0 if all(passed for _, passed in checks) else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoVQE Hamiltonian-to-ansatz research harness"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="show Hamiltonian structure")
    inspect.add_argument("--problem", default=str(DEFAULT_PROBLEM_PATH))
    inspect.add_argument("--json", action="store_true")
    commands.add_parser("check", help="run fast scientific self-checks")

    research = commands.add_parser("research", help="run the closed research loop")
    actions = research.add_subparsers(dest="research_command", required=True)
    init = actions.add_parser("init", help="start a research run")
    init.add_argument("--problem", default=str(DEFAULT_PROBLEM_PATH))
    init.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    init.add_argument("--budget", type=float, default=100.0)
    step = actions.add_parser("step", help="apply one JSON action")
    step.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    step.add_argument("--action", required=True)
    status = actions.add_parser("status", help="show branches, evidence, and budget")
    status.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    status.add_argument("--full", action="store_true")
    result = actions.add_parser("result", help="show the accepted terminal result")
    result.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args.problem, as_json=args.json)
        if args.command == "check":
            return _check()
        if args.research_command == "init":
            value = initialize_run(args.problem, args.run_dir, total_budget=args.budget)
        elif args.research_command == "step":
            value = execute_action_file(args.run_dir, args.action)
        elif args.research_command == "status":
            value = run_status(args.run_dir, full=args.full)
        else:
            value = run_result(args.run_dir)
        print(render_json(value))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
