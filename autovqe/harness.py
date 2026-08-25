"""Command-line entry point for the AutoVQE research harness."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import prepare, research_cli
from .compiler import compile_ansatz
from .contracts import assert_agent_safe, canonical_data
from .evaluator import EvaluationProtocol, evaluate_public_problem
from .observations import adapt_prepare_problem


DEFAULT_PROBLEM = Path("user_problem/hamiltonian.json")
DEFAULT_RUN_DIR = Path(".autovqe-runtime/research")


def _render_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def _inspect(problem_path: str | Path, *, as_json: bool) -> int:
    problem = prepare.load_problem(problem_path)
    views = adapt_prepare_problem(problem)
    if as_json:
        print(_render_json(views.observation_bundle))
        return 0

    structure = views.observation_bundle.structure
    public = views.public_problem
    print(f"problem: {public.problem_id}")
    print(f"qubits: {public.num_qubits}")
    print(f"pauli_terms: {structure.term_count}")
    print(f"max_locality: {structure.max_locality}")
    print(f"locality_counts: {dict(structure.locality_counts)}")
    print(f"support_edges: {len(structure.support_graph_edges)}")
    print(f"declared_symmetries: {list(public.sector.symmetries)}")
    print(f"basis_gates: {list(public.backend.basis_gates)}")
    return 0


def _self_check() -> int:
    checks: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        checks.append((label, bool(condition)))

    problem_payload = {
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
        "parameters": ["theta"],
        "reference": {"macro": "X", "qubits": [0]},
        "operations": [
            {
                "macro": "PauliRotation",
                "qubits": [0],
                "parameters": {"angle": {"parameter": "theta"}},
                "options": {"pauli": "Y"},
            }
        ],
    }

    with tempfile.TemporaryDirectory() as directory:
        problem_path = Path(directory) / "problem.json"
        problem_path.write_text(
            json.dumps(problem_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        problem = prepare.load_problem(problem_path)
        views = adapt_prepare_problem(problem)
        assert_agent_safe(views.observation_bundle)
        check("problem loads", views.public_problem.num_qubits == 2)
        check("public observation hides exact reference", "reference_energy" not in _render_json(views.observation_bundle))
        check("structure analysis runs", views.observation_bundle.structure.term_count == 3)

        compiled = compile_ansatz(spec)
        check("typed ansatz compiles", compiled.audit.unique_trainable_params == 1)
        check("compiler derives parameter use", dict(compiled.audit.parameter_occurrences) == {"theta": 1})

        evaluated = evaluate_public_problem(
            views.public_problem,
            spec,
            protocol=EvaluationProtocol(max_evals=8, restarts=1, seed=3),
        ).receipt
        check("evaluator optimizes candidate", evaluated.valid and evaluated.best_energy is not None)
        check("evaluator owns resource counts", bool(evaluated.metrics))
        check("evaluator owns optimized parameters", bool(evaluated.optimized_parameter_binding))

    failed = [label for label, passed in checks if not passed]
    for label, passed in checks:
        print(f"{'ok' if passed else 'FAIL'}: {label}")
    return 1 if failed else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AutoVQE Hamiltonian-to-ansatz research harness"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect", help="show evaluator-derived Hamiltonian structure"
    )
    inspect_parser.add_argument("--problem", default=str(DEFAULT_PROBLEM))
    inspect_parser.add_argument("--json", action="store_true")

    commands.add_parser("check", help="run fast scientific self-checks")

    research = commands.add_parser(
        "research", help="run the closed ansatz-discovery loop"
    )
    research_commands = research.add_subparsers(
        dest="research_command", required=True
    )

    init_parser = research_commands.add_parser("init", help="start a research run")
    init_parser.add_argument("--problem", default=str(DEFAULT_PROBLEM))
    init_parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    init_parser.add_argument("--budget", type=float, default=100.0)

    step_parser = research_commands.add_parser("step", help="apply one JSON action")
    step_parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    step_parser.add_argument("--action", required=True)

    status_parser = research_commands.add_parser(
        "status", help="show hypotheses, candidates, evidence, and budget"
    )
    status_parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))

    result_parser = research_commands.add_parser(
        "result", help="show the accepted terminal scientific result"
    )
    result_parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args.problem, as_json=args.json)
        if args.command == "check":
            return _self_check()
        if args.command == "research":
            if args.research_command == "init":
                result = research_cli.initialize_run(
                    args.problem,
                    args.run_dir,
                    total_budget=args.budget,
                )
            elif args.research_command == "step":
                result = research_cli.execute_action_file(
                    args.run_dir,
                    args.action,
                )
            elif args.research_command == "status":
                result = research_cli.run_status(args.run_dir)
            elif args.research_command == "result":
                result = research_cli.run_result(args.run_dir)
            else:
                raise AssertionError(
                    f"unhandled research command: {args.research_command}"
                )
            print(research_cli.render_json(result))
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
