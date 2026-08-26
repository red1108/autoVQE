"""Fixed evaluator for one time-bounded AutoVQE experiment."""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

METHODS = {"COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell"}
SCALES = {-1.0, -0.5, 0.5, 1.0}
MACROS = {"U1": ("XX", "YY"), "SU2": ("XX", "YY", "ZZ")}


def load_ansatz() -> tuple[str, list]:
    tree = ast.parse(Path(__file__).with_name("ansatz.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("ansatz.py may contain only literal assignments")
        name = node.targets[0].id
        if name not in {"METHOD", "OPERATIONS"} or name in values:
            raise ValueError("ansatz.py must define METHOD and OPERATIONS once")
        values[name] = ast.literal_eval(node.value)
    if set(values) != {"METHOD", "OPERATIONS"}:
        raise ValueError("ansatz.py must define METHOD and OPERATIONS")
    return values["METHOD"], values["OPERATIONS"]


METHOD, OPERATIONS = load_ansatz()


class TimeLimit(Exception):
    pass


def load_problem(path: str) -> tuple[dict, SparsePauliOp, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("pauli_terms")
    if not isinstance(entries, list) or not entries:
        raise ValueError("pauli_terms must be a non-empty list")
    width = len(entries[0].get("pauli", ""))
    terms = []
    for item in entries:
        label, coeff = item.get("pauli"), item.get("coeff")
        if not isinstance(label, str) or not label or len(label) != width or set(label) - set("IXYZ"):
            raise ValueError("invalid Pauli label")
        if isinstance(coeff, bool) or not isinstance(coeff, (int, float)) or not math.isfinite(coeff):
            raise ValueError("coefficients must be finite real numbers")
        terms.append((label, float(coeff)))
    hint = raw.get("initial_state_hint", [0] * width)
    if not isinstance(hint, list) or len(hint) != width or any(bit not in (0, 1) for bit in hint):
        raise ValueError("invalid initial_state_hint")
    return raw, SparsePauliOp.from_list(terms).simplify(), width


def emit(circuit: QuantumCircuit, word: str, qubits: tuple[int, ...], angle) -> None:
    for qubit, letter in zip(qubits, word, strict=True):
        if letter == "X":
            circuit.h(qubit)
        elif letter == "Y":
            circuit.sdg(qubit); circuit.h(qubit)
    for left, right in zip(qubits[:-1], qubits[1:]):
        circuit.cx(left, right)
    circuit.rz(2 * angle, qubits[-1])
    for left, right in reversed(tuple(zip(qubits[:-1], qubits[1:]))):
        circuit.cx(left, right)
    for qubit, letter in reversed(tuple(zip(qubits, word, strict=True))):
        if letter == "X":
            circuit.h(qubit)
        elif letter == "Y":
            circuit.h(qubit); circuit.s(qubit)


def build(raw: dict, width: int) -> tuple[QuantumCircuit, list[str], Counter, int]:
    if METHOD not in METHODS:
        raise ValueError(f"METHOD must be one of {sorted(METHODS)}")
    circuit = QuantumCircuit(width)
    for qubit, bit in enumerate(raw.get("initial_state_hint", [0] * width)):
        if bit:
            circuit.x(qubit)
    parameters, counts, support = {}, Counter(), 0
    for operation in OPERATIONS:
        if not isinstance(operation, (tuple, list)) or len(operation) != 4:
            raise ValueError("each operation must be (gate, qubits, parameter, scale)")
        gate, qubits, name, scale = operation
        qubits = tuple(qubits) if isinstance(qubits, (tuple, list)) else ()
        words = MACROS.get(gate, (gate,)) if isinstance(gate, str) else ()
        valid = (
            words and all(word and not set(word) - set("XYZ") for word in words)
            and all(len(word) == len(qubits) for word in words)
            and len(set(qubits)) == len(qubits)
            and all(type(q) is int and 0 <= q < width for q in qubits)
            and isinstance(name, str) and name
            and not isinstance(scale, bool) and isinstance(scale, (int, float))
            and float(scale) in SCALES
        )
        if not valid:
            raise ValueError(f"invalid operation: {operation!r}")
        if name not in parameters:
            parameters[name] = Parameter(name)
        for word in words:
            counts[name] += 1
            support += len(word)
            emit(circuit, word, qubits, float(scale) * parameters[name])
    return circuit, list(parameters), counts, support


def resources(circuit: QuantumCircuit, raw: dict, counts: Counter, support: int) -> dict:
    compiled = transpile(
        circuit,
        basis_gates=raw.get("basis_gates") or None,
        coupling_map=raw.get("coupling_map") or None,
        optimization_level=1,
        seed_transpiler=7,
    )
    return {
        "unique_parameters": len(counts),
        "parameter_occurrences": sum(counts.values()),
        "generator_support": support,
        "two_qubit_gates": sum(len(item.qubits) == 2 for item in compiled.data),
        "total_gates": compiled.size(),
        "depth": compiled.depth(),
    }


def run(problem_path: str, seconds: float | None, target_error: float) -> dict:
    raw, hamiltonian, width = load_problem(problem_path)
    seconds = seconds if seconds is not None else max(5.0, 60.0 * 2 ** (width - 16))
    circuit, names, counts, support = build(raw, width)
    measured_resources = resources(circuit, raw, counts, support)
    parameters = [circuit.get_parameter(name) for name in names]
    started, deadline = time.perf_counter(), time.perf_counter() + seconds
    best_energy, best_values, calls = float("inf"), np.zeros(len(names)), 0

    def objective(values) -> float:
        nonlocal best_energy, best_values, calls
        if time.perf_counter() >= deadline:
            raise TimeLimit
        bound = circuit.assign_parameters(dict(zip(parameters, values, strict=True)), inplace=False)
        value = float(np.real(Statevector.from_instruction(bound).expectation_value(hamiltonian)))
        calls += 1
        if value < best_energy:
            best_energy, best_values = value, np.asarray(values).copy()
        return value

    reason = "no_parameters"
    if names:
        try:
            result = minimize(objective, np.zeros(len(names)), method=METHOD, options={"maxiter": 10**9})
            reason = "converged" if result.success else "optimizer_stopped"
        except TimeLimit:
            reason = "time_budget"
    else:
        objective(np.zeros(0))
    reference = raw.get("reference_energy")
    relative_error = None if reference is None else abs(best_energy - float(reference)) / max(abs(float(reference)), 1e-15)
    return {
        "energy": best_energy,
        "optimized_parameters": dict(zip(names, best_values.tolist(), strict=True)),
        "optimizer": METHOD,
        "time_budget_seconds": seconds,
        "evaluations": calls,
        "optimization_seconds": time.perf_counter() - started,
        "termination": reason,
        "resources": measured_resources,
        "reference_energy": reference,
        "relative_error": relative_error,
        "target_reached": None if relative_error is None else relative_error <= target_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--target-relative-error", type=float, default=0.001)
    args = parser.parse_args()
    if (args.seconds is not None and args.seconds <= 0) or args.target_relative_error <= 0:
        parser.error("budgets and tolerances must be positive")
    print(json.dumps(run(args.problem, args.seconds, args.target_relative_error), indent=2))


if __name__ == "__main__":
    main()
