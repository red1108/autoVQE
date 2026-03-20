from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

import prepare


@dataclass(frozen=True)
class VQEConfig:
    layers: int | None = None
    optimizer: str = "adam"
    param_init: str = "small_random"
    epochs: int = 40
    learning_rate: float = 0.2
    gradient_epsilon: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    seed: int = prepare.SEED


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
        edge = tuple(sorted((raw_control, raw_target)))
        if raw_control == raw_target or edge in seen:
            continue
        seen.add(edge)
        edges.append(edge)
    return edges or [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]


def build_hea_ansatz(problem: prepare.Problem, config: VQEConfig) -> tuple[QuantumCircuit, ParameterVector]:
    layers = default_layers(problem.num_qubits, config.layers)
    num_params = 2 * layers * problem.num_qubits
    theta = ParameterVector("theta", num_params)
    circuit = QuantumCircuit(problem.num_qubits)
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
    return circuit.assign_parameters(dict(zip(parameters, values, strict=True)), inplace=False)


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

    values = initial_parameters(config, len(parameters))
    if len(parameters) == 0:
        return values, objective(values), calls

    current_energy = objective(values)
    best_values = values.copy()
    best_energy = current_energy
    epsilon = float(config.gradient_epsilon)
    optimizer = config.optimizer.lower()

    if optimizer == "adam":
        m = np.zeros(len(parameters), dtype=float)
        v = np.zeros(len(parameters), dtype=float)
        beta1 = float(config.adam_beta1)
        beta2 = float(config.adam_beta2)
        adam_epsilon = float(config.adam_epsilon)

        for step in range(1, config.epochs + 1):
            gradient = np.zeros(len(parameters), dtype=float)
            for index in range(len(parameters)):
                shifted = values.copy()
                shifted[index] += epsilon
                shifted_energy = objective(shifted)
                gradient[index] = (shifted_energy - current_energy) / epsilon

            if float(np.linalg.norm(gradient)) < 1e-10:
                break

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            values = values - config.learning_rate * m_hat / (np.sqrt(v_hat) + adam_epsilon)

            current_energy = objective(values)
            if current_energy < best_energy:
                best_energy = current_energy
                best_values = values.copy()

        return best_values, best_energy, calls

    if optimizer in {"gradient_descent", "gradient-descent", "gd"}:
        for _ in range(config.epochs):
            gradient = np.zeros(len(parameters), dtype=float)
            for index in range(len(parameters)):
                shifted = values.copy()
                shifted[index] += epsilon
                shifted_energy = objective(shifted)
                gradient[index] = (shifted_energy - current_energy) / epsilon

            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm < 1e-10:
                break

            values = values - config.learning_rate * gradient / gradient_norm
            current_energy = objective(values)
            if current_energy < best_energy:
                best_energy = current_energy
                best_values = values.copy()

        return best_values, best_energy, calls

    raise ValueError(f"unsupported optimizer: {config.optimizer}")


def main() -> None:
    config = VQEConfig()
    problem = prepare.load_problem()
    backend_target = prepare.build_backend_target(problem)
    circuit, parameters = build_hea_ansatz(problem, config)

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
