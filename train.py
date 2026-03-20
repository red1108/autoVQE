from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from scipy.optimize import minimize

import prepare


@dataclass(frozen=True)
class VQEConfig:
    ansatz: str = "hea"
    layers: int | None = None
    state_mode: str = "auto"
    param_init: str = "small_random"
    optimizer: str = "cobyla"
    max_evals: int = prepare.MAX_EVALS
    seed: int = prepare.SEED


def build_initial_state(num_qubits: int, mode: str, hint: object | None) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    if mode == "empty":
        return circuit

    if mode == "auto":
        mode = "hf_like" if hint is not None else "empty"

    if mode != "hf_like":
        return circuit

    if isinstance(hint, list) and len(hint) == num_qubits and all(bit in (0, 1) for bit in hint):
        occupied = [index for index, bit in enumerate(hint) if bit]
    elif isinstance(hint, dict) and isinstance(hint.get("occupied_qubits"), list):
        occupied = [int(index) for index in hint["occupied_qubits"]]
    else:
        occupied = list(range(num_qubits // 2))

    for qubit in occupied:
        circuit.x(qubit)
    return circuit


def default_layers(num_qubits: int, configured_layers: int | None) -> int:
    if configured_layers is not None:
        return configured_layers
    if num_qubits <= 2:
        return 1
    if num_qubits <= 6:
        return 2
    return 3


def entangler_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
    if not problem.coupling_map:
        return [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_control, raw_target in problem.coupling_map:
        if raw_control == raw_target:
            continue
        edge = tuple(sorted((raw_control, raw_target)))
        if edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)

    if edges:
        return edges
    return [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]


def build_hea_ansatz(
    problem: prepare.Problem,
    layers: int,
    initial_state: QuantumCircuit,
) -> tuple[QuantumCircuit, ParameterVector]:
    num_params = 2 * layers * problem.num_qubits
    theta = ParameterVector("theta", num_params)
    circuit = QuantumCircuit(problem.num_qubits)
    circuit.compose(initial_state, inplace=True)
    edges = entangler_edges(problem)

    cursor = 0
    for _ in range(layers):
        for qubit in range(problem.num_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rz(theta[cursor], qubit)
            cursor += 1
        for control, target in edges:
            circuit.cx(control, target)

    return circuit, theta


def build_symm_ansatz(num_qubits: int, layers: int, initial_state: QuantumCircuit) -> tuple[QuantumCircuit, ParameterVector]:
    width = (num_qubits + 1) // 2
    num_params = 2 * layers * width
    theta = ParameterVector("theta", num_params)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(initial_state, inplace=True)

    cursor = 0
    for _ in range(layers):
        for left in range(width):
            right = num_qubits - left - 1
            angle_y = theta[cursor]
            cursor += 1
            circuit.ry(angle_y, left)
            if right != left:
                circuit.ry(angle_y, right)
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for left in range(width):
            right = num_qubits - left - 1
            angle_z = theta[cursor]
            cursor += 1
            circuit.rz(angle_z, left)
            if right != left:
                circuit.rz(angle_z, right)

    return circuit, theta


def is_diagonal_problem(problem: prepare.Problem) -> bool:
    for label in problem.hamiltonian.paulis.to_labels():
        if any(symbol not in {"I", "Z"} for symbol in label):
            return False
    return True


def apply_cost_term(circuit: QuantumCircuit, label: str, coeff: float, gamma: Parameter) -> None:
    support = [index for index, symbol in enumerate(label) if symbol == "Z"]
    if not support:
        return
    if len(support) == 1:
        circuit.rz(2.0 * coeff * gamma, support[0])
        return
    if len(support) == 2:
        control, target = support
        circuit.cx(control, target)
        circuit.rz(2.0 * coeff * gamma, target)
        circuit.cx(control, target)
        return
    raise ValueError("qaoa baseline only supports diagonal terms up to two-body Z interactions")


def build_qaoa_ansatz(
    problem: prepare.Problem, layers: int, initial_state: QuantumCircuit
) -> tuple[QuantumCircuit, ParameterVector]:
    if not is_diagonal_problem(problem):
        raise ValueError("qaoa requires a diagonal Hamiltonian for this baseline")

    theta = ParameterVector("theta", 2 * layers)
    circuit = QuantumCircuit(problem.num_qubits)
    circuit.compose(initial_state, inplace=True)
    for qubit in range(problem.num_qubits):
        circuit.h(qubit)

    labels = problem.hamiltonian.paulis.to_labels()
    coeffs = [float(np.real(coeff)) for coeff in problem.hamiltonian.coeffs]
    for layer in range(layers):
        gamma = theta[2 * layer]
        beta = theta[2 * layer + 1]
        for label, coeff in zip(labels, coeffs, strict=True):
            apply_cost_term(circuit, label, coeff, gamma)
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)

    return circuit, theta


def build_ansatz(
    problem: prepare.Problem, config: VQEConfig
) -> tuple[QuantumCircuit, ParameterVector]:
    initial_state = build_initial_state(problem.num_qubits, config.state_mode, problem.initial_state_hint)
    layers = default_layers(problem.num_qubits, config.layers)
    if config.ansatz == "hea":
        return build_hea_ansatz(problem, layers, initial_state)
    if config.ansatz == "symm":
        return build_symm_ansatz(problem.num_qubits, layers, initial_state)
    if config.ansatz == "qaoa":
        return build_qaoa_ansatz(problem, layers, initial_state)
    raise ValueError(f"unsupported ansatz family: {config.ansatz}")


def initial_parameters(config: VQEConfig, num_params: int) -> np.ndarray:
    if config.param_init == "zeros":
        return np.zeros(num_params, dtype=float)
    rng = np.random.default_rng(config.seed)
    if config.param_init == "random":
        return rng.uniform(-0.2, 0.2, size=num_params)
    if config.param_init == "small_random":
        return rng.uniform(-0.05, 0.05, size=num_params)
    raise ValueError(f"unsupported parameter init mode: {config.param_init}")


def bind_parameters(circuit: QuantumCircuit, parameters: ParameterVector, values: np.ndarray) -> QuantumCircuit:
    mapping = dict(zip(parameters, values, strict=True))
    return circuit.assign_parameters(mapping, inplace=False)


def optimize_energy(
    problem: prepare.Problem,
    circuit: QuantumCircuit,
    parameters: ParameterVector,
    config: VQEConfig,
) -> tuple[np.ndarray, float, int]:
    calls = 0

    def objective(values: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        candidate = bind_parameters(circuit, parameters, values)
        return prepare.energy_from_circuit(candidate, problem.hamiltonian)

    initial = initial_parameters(config, len(parameters))
    optimizer = config.optimizer.lower()
    if optimizer == "cobyla":
        method = "COBYLA"
        options = {"maxiter": config.max_evals, "rhobeg": 0.5}
    elif optimizer == "powell":
        method = "Powell"
        options = {"maxfev": config.max_evals, "xtol": 1e-4, "ftol": 1e-8}
    elif optimizer == "nelder-mead":
        method = "Nelder-Mead"
        options = {"maxfev": config.max_evals, "xatol": 1e-4, "fatol": 1e-8}
    else:
        raise ValueError(f"unsupported optimizer: {config.optimizer}")
    result = minimize(
        objective,
        initial,
        method=method,
        options=options,
    )
    best_values = np.asarray(result.x, dtype=float)
    best_energy = float(result.fun)
    return best_values, best_energy, calls


def main() -> None:
    config = VQEConfig()
    problem = prepare.load_problem()
    backend_target = prepare.build_backend_target(problem)
    circuit, parameters = build_ansatz(problem, config)

    started = time.perf_counter()
    best_values, energy, eval_calls = optimize_energy(problem, circuit, parameters, config)
    total_seconds = time.perf_counter() - started

    final_circuit = bind_parameters(circuit, parameters, best_values)
    _, compiled = prepare.transpile_and_report(final_circuit, backend_target)
    overlap = prepare.overlap_with_reference(final_circuit, problem.reference_state)

    summary = [
        ("energy", energy),
        ("reference_energy", problem.reference_energy),
        ("overlap", overlap),
        ("singleq_count", compiled["singleq_count"]),
        ("twoq_count", compiled["twoq_count"]),
        ("total_gate_count", compiled["total_gate_count"]),
        ("depth", compiled["depth"]),
        ("num_params", len(parameters)),
        ("eval_calls", eval_calls),
        ("total_seconds", total_seconds),
    ]
    print(prepare.format_summary(summary))


if __name__ == "__main__":
    main()
