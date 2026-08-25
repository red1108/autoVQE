"""Installed AutoVQE command line."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
from .evaluator import EvaluationProtocol, evaluate_public_problem
from .problem import DEFAULT_PROBLEM_PATH, load_problem, observe_problem
from .research import execute_action_file, initialize_run, render_json, run_result, run_status

DEFAULT_RUN_DIR = Path(".autovqe-runtime/research")

def _inspect(path: str | Path, as_json: bool) -> int:
    problem = load_problem(path)
    observation = observe_problem(problem)
    structure = observation["structure"]
    if as_json:
        print(render_json(observation))
        return 0
    values = {
        "problem": problem.problem_id, "qubits": problem.num_qubits,
        "pauli_terms": structure["term_count"], "max_locality": structure["max_locality"],
        "locality_counts": dict(structure["locality_counts"]),
        "support_edges": structure["support_graph_edge_count"],
        "declared_symmetries": list(dict(problem.symmetry)),
        "basis_gates": list(problem.backend.basis_gates),
    }
    for key, value in values.items():
        print(f"{key}: {value}")
    return 0

def _check() -> int:
    document = {
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
        "num_qubits": 2,
        "operations": [
            {
                "gate": "PauliRotation",
                "qubits": [0],
                "parameter": "theta",
                "pauli": "Y",
            }
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "problem.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        problem = load_problem(path)
        observation = observe_problem(problem)
        result = evaluate_public_problem(problem, spec, protocol=EvaluationProtocol(max_evals=8, restarts=1, seed=3))
    checks = (
        ("problem loads", problem.num_qubits == 2),
        ("observation stays compact", "pauli_terms" not in render_json(observation)),
        ("structure analysis runs", observation["structure"]["term_count"] == 3),
        ("typed ansatz compiles", result.audit.get("unique_trainable_params") == 1),
        ("compiler derives parameter use", result.audit.get("parameter_occurrences") == {"theta": 1}),
        ("evaluator optimizes candidate", result.valid and result.best_energy is not None),
        ("evaluator owns resource counts", bool(result.resources)),
        ("evaluator owns optimized parameters", bool(result.optimized_parameter_binding)),
    )
    for label, passed in checks:
        print(f"{'ok' if passed else 'FAIL'}: {label}")
    return 0 if all(passed for _, passed in checks) else 1

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoVQE Hamiltonian-to-ansatz research harness")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="show Hamiltonian structure")
    inspect.add_argument("--problem", default=str(DEFAULT_PROBLEM_PATH))
    inspect.add_argument("--json", action="store_true")
    commands.add_parser("check", help="run fast scientific self-checks")
    research = commands.add_parser("research", help="run the closed research loop").add_subparsers(dest="research_command", required=True)
    init = research.add_parser("init", help="start a research run")
    init.add_argument("--problem", default=str(DEFAULT_PROBLEM_PATH))
    init.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    init.add_argument("--budget", type=float, default=100.0)
    step = research.add_parser("step", help="apply one JSON action")
    step.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    step.add_argument("--action", required=True)
    for name, help_text in (("status", "show branches, evidence, and budget"), ("result", "show the accepted terminal result")):
        research.add_parser(name, help=help_text).add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args.problem, args.json)
        if args.command == "check":
            return _check()
        if args.research_command == "init":
            value = initialize_run(args.problem, args.run_dir, total_budget=args.budget)
        elif args.research_command == "step":
            value = execute_action_file(args.run_dir, args.action)
        elif args.research_command == "status":
            value = run_status(args.run_dir)
        else:
            value = run_result(args.run_dir)
        print(render_json(value))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
