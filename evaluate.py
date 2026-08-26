"""Fixed evaluator for one time-bounded AutoVQE experiment."""
import argparse, ast, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
METHODS = {"COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell"}
SCALES = {-1.0, -0.5, 0.5, 1.0}
MACROS = {"U1": (("XX", 1.0), ("YY", 1.0)), "GIVENS": (("YX", 0.5), ("XY", -0.5)),
          "PAIR": (("YX", 0.5), ("XY", 0.5)), "SU2": (("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0))}
STATE = Path(__file__).with_name(".autovqe-state.json")
def finite(value) -> bool: return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
def load_ansatz() -> tuple[str, list]:
    tree = ast.parse(Path(__file__).with_name("ansatz.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str): continue
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name): raise ValueError("ansatz.py may contain only literal assignments")
        name = node.targets[0].id
        if name not in {"METHOD", "OPERATIONS"} or name in values: raise ValueError("ansatz.py must define METHOD and OPERATIONS once")
        values[name] = ast.literal_eval(node.value)
    if set(values) != {"METHOD", "OPERATIONS"}: raise ValueError("ansatz.py must define METHOD and OPERATIONS")
    return values["METHOD"], values["OPERATIONS"]
METHOD, OPERATIONS = load_ansatz()
class Stop(Exception): pass
def load_problem(path: str) -> tuple[dict, SparsePauliOp, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("pauli_terms")
    if not isinstance(entries, list) or not entries: raise ValueError("pauli_terms must be a non-empty list")
    width, terms = len(entries[0].get("pauli", "")), []
    for item in entries:
        label, coeff = item.get("pauli"), item.get("coeff")
        if not isinstance(label, str) or not label or len(label) != width or set(label) - set("IXYZ"): raise ValueError("invalid Pauli label")
        if not finite(coeff): raise ValueError("coefficients must be finite real numbers")
        terms.append((label, float(coeff)))
    hint = raw.get("initial_state_hint", [0] * width)
    if not isinstance(hint, list) or len(hint) != width or any(bit not in (0, 1) for bit in hint): raise ValueError("invalid initial_state_hint")
    return raw, SparsePauliOp.from_list(terms).simplify(), width
def emit(circuit: QuantumCircuit, word: str, qubits: tuple[int, ...], angle) -> None:
    for qubit, letter in zip(qubits, word, strict=True):
        if letter == "X": circuit.h(qubit)
        elif letter == "Y":
            circuit.sdg(qubit); circuit.h(qubit)
    for left, right in zip(qubits[:-1], qubits[1:]): circuit.cx(left, right)
    circuit.rz(2 * angle, qubits[-1])
    for left, right in reversed(tuple(zip(qubits[:-1], qubits[1:]))): circuit.cx(left, right)
    for qubit, letter in reversed(tuple(zip(qubits, word, strict=True))):
        if letter == "X": circuit.h(qubit)
        elif letter == "Y":
            circuit.h(qubit); circuit.s(qubit)
def build(raw: dict, width: int) -> tuple[QuantumCircuit, list[str], Counter, int, dict, list]:
    if METHOD not in METHODS: raise ValueError(f"METHOD must be one of {sorted(METHODS)}")
    circuit = QuantumCircuit(width)
    for qubit, bit in enumerate(raw.get("initial_state_hint", [0] * width)):
        if bit: circuit.x(qubit)
    parameters, counts, support, roles, ordered = {}, Counter(), 0, {}, []
    for operation in OPERATIONS:
        if not isinstance(operation, (tuple, list)) or len(operation) != 4: raise ValueError("each operation must be (gate, qubits, parameter, scale)")
        gate, qubits, name, scale = operation
        qubits = tuple(qubits) if isinstance(qubits, (tuple, list)) else ()
        components = MACROS.get(gate, ((gate, 1.0),)) if isinstance(gate, str) else ()
        valid = (components and all(word and not set(word) - set("XYZ") and len(word) == len(qubits) for word, _ in components)
                 and len(set(qubits)) == len(qubits) and all(type(q) is int and 0 <= q < width for q in qubits)
                 and isinstance(name, str) and name and finite(scale) and float(scale) in SCALES)
        if not valid: raise ValueError(f"invalid operation: {operation!r}")
        parameters.setdefault(name, Parameter(name))
        roles.setdefault(name, []).append([gate, list(qubits), float(scale)])
        ordered.append([gate, list(qubits), name, float(scale)])
        for word, weight in components:
            counts[name] += 1
            support += len(word)
            emit(circuit, word, qubits, float(scale) * weight * parameters[name])
    return circuit, list(parameters), counts, support, roles, ordered
def load_history() -> list:
    try: history = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): history = []
    if not isinstance(history, list): return []
    return [item for item in history if isinstance(item, dict) and isinstance(item.get("problem"), str)
            and isinstance(item.get("roles"), dict) and isinstance(item.get("values"), dict) and finite(item.get("energy"))
            and all(isinstance(role, list) and finite(item["values"].get(name)) for name, role in item["roles"].items())]
def save_history(history: list) -> None:
    temporary = STATE.with_name(STATE.name + ".tmp")
    temporary.write_text(json.dumps(history), encoding="utf-8")
    temporary.replace(STATE)
def role_counter(role) -> Counter:
    try: return Counter((gate, tuple(qubits), float(scale)) for gate, qubits, scale, *_ in role)
    except (TypeError, ValueError): return Counter()
def structure(operations):
    try: return [(gate, tuple(qubits), float(scale)) for gate, qubits, _, scale in operations]
    except (TypeError, ValueError): return None
def project(item: dict, names: list[str], roles: dict, operations: list) -> tuple[np.ndarray, int, int]:
    initial, warm, covered = np.zeros(len(names)), 0, 0
    previous, values = item.get("operations"), item["values"]
    if structure(previous) == structure(operations):
        mapped = {name: [] for name in names}
        for old, new in zip(previous, operations, strict=True):
            if old[2] in values: mapped[new[2]].append(float(values[old[2]]))
        for index, name in enumerate(names):
            if name in values and role_counter(item["roles"].get(name, [])) == role_counter(roles[name]): mapped[name] = [float(values[name])] * len(roles[name])
            if len(mapped[name]) == len(roles[name]): initial[index], warm, covered = float(np.mean(mapped[name])), warm + 1, covered + len(mapped[name])
        return initial, warm, covered
    old = {name: (role_counter(role), float(values[name])) for name, role in item["roles"].items()}
    legacy = not isinstance(previous, list) and sum(map(len, item["roles"].values())) == sum(map(len, roles.values()))
    for index, name in enumerate(names):
        current, choices = role_counter(roles[name]), []
        if name in old and sum((current & old[name][0]).values()): choices = [old[name]]
        elif legacy:
            containers = [pair for pair in old.values() if current < pair[0]]
            if containers:
                size = min(sum(role.values()) for role, _ in containers)
                choices = [pair for pair in containers if sum(pair[0].values()) == size]
                if len(choices) != 1: choices = []
            else: choices = [pair for pair in old.values() if pair[0] < current]
        weights = [sum((current & role).values()) for role, _ in choices]
        if sum(weights): initial[index], warm, covered = float(np.average([value for _, value in choices], weights=weights)), warm + 1, covered + min(sum(current.values()), sum(weights))
    return initial, warm, covered
def continuation(problem: str, names: list[str], roles: dict, operations: list) -> tuple[np.ndarray, int, list]:
    history = load_history()
    prior = [item for item in reversed(history) if item["problem"] == problem]
    exact = [item for item in prior if item.get("operations") == operations]
    if exact: initial, warm, _ = project(min(exact, key=lambda item: item["energy"]), names, roles, operations)
    elif prior:
        options = [(project(item, names, roles, operations), item) for item in prior]
        initial, warm, _ = max(options, key=lambda pair: (pair[0][2], pair[0][1], -pair[1]["energy"]))[0]
    else: initial, warm = np.zeros(len(names)), 0
    return initial, warm, history
def resources(circuit: QuantumCircuit, raw: dict, counts: Counter, support: int) -> dict:
    compiled = transpile(circuit, basis_gates=raw.get("basis_gates") or None, coupling_map=raw.get("coupling_map") or None,
                         optimization_level=1, seed_transpiler=7)
    return {"unique_parameters": len(counts), "parameter_occurrences": sum(counts.values()), "generator_support": support,
            "two_qubit_gates": sum(len(item.qubits) == 2 for item in compiled.data), "total_gates": compiled.size(), "depth": compiled.depth()}
def adjoint(raw: dict, width: int, names: list[str], operations: list):
    basis, parity = np.arange(1 << width), np.ones(1 << width)
    for qubit in range(width): parity[(basis & (1 << qubit)) != 0] *= -1
    def encode(word, qubits):
        pairs = tuple(zip(qubits, word, strict=True))
        return sum(1 << q for q, letter in pairs if letter in "XY"), sum(1 << q for q, letter in pairs if letter in "YZ"), 1j ** sum(letter == "Y" for _, letter in pairs)
    positions = {name: index for index, name in enumerate(names)}
    rotations = [(positions[name], float(scale) * weight, encode(word, qubits)) for gate, qubits, name, scale in operations
                 for word, weight in MACROS.get(gate, ((gate, 1.0),))]
    terms, initial = [(float(item["coeff"]), encode(item["pauli"], range(width - 1, -1, -1))) for item in raw["pauli_terms"]], np.zeros(len(basis), dtype=complex)
    initial[sum(bit << qubit for qubit, bit in enumerate(raw.get("initial_state_hint", [0] * width)))] = 1
    def apply(state, encoded):
        flip, sign, phase = encoded
        return phase * parity[(source := basis ^ flip) & sign] * state[source]
    def rotate(state, encoded, angle): return math.cos(angle) * state - 1j * math.sin(angle) * apply(state, encoded)
    def value_gradient(values, observe):
        state = initial.copy()
        for index, scale, encoded in rotations: state = rotate(state, encoded, scale * values[index])
        co_state = np.zeros_like(state)
        for coefficient, encoded in terms: co_state += coefficient * apply(state, encoded)
        value, gradient = float(np.vdot(state, co_state).real), np.zeros(len(names))
        observe(value)
        for index, scale, encoded in reversed(rotations):
            gradient[index] -= 2 * scale * np.imag(np.vdot(state, apply(co_state, encoded)))
            state = rotate(state, encoded, -scale * values[index])
            co_state = rotate(co_state, encoded, -scale * values[index])
        return value, gradient
    return value_gradient
def restore_best(problem_path: str) -> None:
    global METHOD, OPERATIONS
    raw, _, width = load_problem(problem_path)
    problem, previous = Path(problem_path).as_posix(), (METHOD, OPERATIONS)
    candidates = sorted((item for item in load_history() if item["problem"] == problem and item.get("method") in METHODS and isinstance(item.get("operations"), list)), key=lambda item: item["energy"])
    for item in candidates:
        METHOD, OPERATIONS = item["method"], item["operations"]
        try:
            built = build(raw, width)
            if built[5] != item["operations"] or any(name not in item["values"] for name in built[1]): raise ValueError("state does not match its circuit")
        except (TypeError, ValueError):
            METHOD, OPERATIONS = previous
            continue
        Path(__file__).with_name("ansatz.py").write_text(f"METHOD = {METHOD!r}\n\nOPERATIONS = {json.dumps(OPERATIONS, indent=4)}\n", encoding="utf-8")
        return
    raise ValueError("no restorable evaluator-owned best ansatz")
def run(problem_path: str, seconds: float | None, target_error: float) -> dict:
    raw, hamiltonian, width = load_problem(problem_path)
    seconds = seconds if seconds is not None else max(30.0, 60.0 * 2 ** (width - 16))
    circuit, names, counts, support, roles, operations = build(raw, width)
    measured_resources, parameters = resources(circuit, raw, counts, support), [circuit.get_parameter(name) for name in names]
    initial, warm, history = continuation(problem := Path(problem_path).as_posix(), names, roles, operations)
    value_gradient = adjoint(raw, width, names, operations) if (analytic := METHOD == "L-BFGS-B" and bool(names)) else None
    reference, started = raw.get("reference_energy"), time.perf_counter()
    deadline, best_energy, best_values, calls = started + seconds, float("inf"), initial.copy(), 0
    def observe(value, values):
        nonlocal best_energy, best_values, calls
        calls += 1
        if value < best_energy:
            best_energy, best_values = value, np.asarray(values).copy()
            if names and reference is not None and abs(value - float(reference)) / max(abs(float(reference)), 1e-15) <= target_error:
                raise Stop("target_reached")
        if time.perf_counter() >= deadline: raise Stop("time_budget")
    def objective(values):
        if time.perf_counter() >= deadline: raise Stop("time_budget")
        if value_gradient is not None:
            result = value_gradient(np.asarray(values), lambda value: observe(value, values))
            if time.perf_counter() >= deadline: raise Stop("time_budget")
            return result
        value = float(np.real(Statevector.from_instruction(circuit.assign_parameters(dict(zip(parameters, values, strict=True)), inplace=False)).expectation_value(hamiltonian)))
        observe(value, values)
        return value
    reason = "no_parameters"
    try:
        if names:
            options = {"maxiter": 10**9} | ({"maxfun": 10**9, "maxls": 50, "ftol": 1e-14, "gtol": 1e-9} if analytic else {})
            result = minimize(objective, initial, method=METHOD, jac=analytic, options=options)
            reason = "converged" if result.success else "optimizer_stopped"
        else: objective(np.zeros(0))
    except Stop as stop: reason = str(stop)
    relative_error = None if reference is None else abs(best_energy - float(reference)) / max(abs(float(reference)), 1e-15)
    values = dict(zip(names, best_values.tolist(), strict=True))
    history.append({"problem": problem, "method": METHOD, "operations": operations, "roles": roles, "values": values, "energy": best_energy})
    best = {}
    for item in history:
        if item["problem"] not in best or item["energy"] < best[item["problem"]]["energy"]: best[item["problem"]] = item
    room = 100 - len(kept := list(best.values())[-100:])
    save_history(kept + ([item for item in history if item not in kept][-room:] if room else []))
    return {"energy": best_energy, "optimized_parameters": values, "warm_started_parameters": warm, "optimizer": METHOD,
            "time_budget_seconds": seconds, "evaluations": calls, "optimization_seconds": time.perf_counter() - started,
            "termination": reason, "resources": measured_resources, "reference_energy": reference, "relative_error": relative_error,
            "target_reached": None if relative_error is None else relative_error <= target_error}
def append_result(problem: str, hypothesis: str, result: dict) -> None:
    path, resource = Path(__file__).with_name("results.tsv"), result["resources"]
    columns = ("problem", "energy", "relative_error", "unique_parameters", "parameter_occurrences", "generator_support", "two_qubit_gates", "total_gates", "depth", "evaluations", "budget_s", "elapsed_s", "optimizer", "termination", "target_reached", "hypothesis")
    relative = result["relative_error"]
    row = (Path(problem).name, f'{result["energy"]:.12g}', "" if relative is None else f"{relative:.3e}", resource["unique_parameters"], resource["parameter_occurrences"], resource["generator_support"], resource["two_qubit_gates"], resource["total_gates"],
           resource["depth"], result["evaluations"], f'{result["time_budget_seconds"]:.3g}', f'{result["optimization_seconds"]:.4g}', result["optimizer"], result["termination"], "" if result["target_reached"] is None else int(result["target_reached"]), hypothesis.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip())
    empty = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as output:
        if empty: output.write("\t".join(columns) + "\n")
        output.write("\t".join(map(str, row)) + "\n")
def main() -> None:
    parser = argparse.ArgumentParser()
    for option, settings in (("problem", {}), ("--seconds", {"type": float}), ("--target-relative-error", {"type": float, "default": 0.0001}), ("--hypothesis", {"required": True}), ("--restore-best", {"action": "store_true"})):
        parser.add_argument(option, **settings)
    args = parser.parse_args()
    if (args.seconds is not None and (not finite(args.seconds) or args.seconds <= 0)) or not finite(args.target_relative_error) or args.target_relative_error <= 0: parser.error("budgets and tolerances must be positive")
    if not args.hypothesis.strip(): parser.error("hypothesis must be non-empty")
    if args.restore_best: restore_best(args.problem)
    result = run(args.problem, args.seconds, args.target_relative_error)
    append_result(args.problem, args.hypothesis, result)
    shown = result if args.restore_best or result["target_reached"] else result | {"optimized_parameters": {}, "optimized_parameter_count": len(result["optimized_parameters"])}
    print(json.dumps(shown, indent=2))
if __name__ == "__main__": main()
