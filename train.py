from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit

import prepare

RESULTS_PATH = Path("results.tsv")
TARGET_EXPERIMENTS = 100
ENERGY_IMPROVEMENT_TOL = 1e-6
ENERGY_EQUIVALENCE_TOL = 5e-4


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    layers: int
    optimizer: str
    param_init: str
    learning_rate: float
    seed: int
    edge_mode: str = "full"
    rotation_mode: str = "full"
    time_budget_seconds: float | None = None
    gradient_epsilon: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    description: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    description: str
    status: str
    energy: float
    overlap: float | None
    metrics: dict[str, int]
    num_params: int
    eval_calls: int
    total_seconds: float

    @property
    def compression_key(self) -> tuple[int, int, int, int]:
        return (
            self.metrics["twoq_count"],
            self.metrics["total_gate_count"],
            self.metrics["depth"],
            self.num_params,
        )


def ensure_results_header() -> None:
    header = "commit\tenergy\tsingleq_count\ttwoq_count\ttotal_gate_count\tnum_params\tstatus\tdescription\n"
    if not RESULTS_PATH.exists() or not RESULTS_PATH.read_text():
        RESULTS_PATH.write_text(header)


def head_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "nogit00"


def experiment_id(index: int, description: str) -> str:
    raw = f"{head_snapshot()}|{index}|{description}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:7]


def log_result(result: ExperimentResult) -> None:
    row = "\t".join(
        [
            result.run_id,
            f"{result.energy:.6f}",
            str(result.metrics["singleq_count"]),
            str(result.metrics["twoq_count"]),
            str(result.metrics["total_gate_count"]),
            str(result.num_params),
            result.status,
            result.description,
        ]
    )
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def chain_edges(problem: prepare.Problem) -> list[tuple[int, int]]:
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
    return edges or [(qubit, qubit + 1) for qubit in range(problem.num_qubits - 1)]


def select_edges(
    edges: list[tuple[int, int]],
    mode: str,
    layers: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if mode == "full":
        return [list(edges) for _ in range(layers)]
    if mode == "even":
        subset = [edge for index, edge in enumerate(edges) if index % 2 == 0]
        return [subset for _ in range(layers)]
    if mode == "odd":
        subset = [edge for index, edge in enumerate(edges) if index % 2 == 1]
        return [subset for _ in range(layers)]
    if mode == "alternate":
        return [
            [edge for index, edge in enumerate(edges) if index % 2 == layer % 2]
            for layer in range(layers)
        ]
    if mode == "ends":
        subset = [edges[0]]
        if len(edges) > 1:
            subset.append(edges[-1])
        return [subset for _ in range(layers)]
    if mode == "random_1":
        rng = np.random.default_rng(seed)
        per_layer: list[list[tuple[int, int]]] = []
        for _ in range(layers):
            choice = int(rng.integers(len(edges)))
            per_layer.append([edges[choice]])
        return per_layer
    if mode == "random_2":
        rng = np.random.default_rng(seed)
        per_layer = []
        for _ in range(layers):
            order = rng.permutation(len(edges))
            chosen = sorted(order[: min(2, len(edges))])
            per_layer.append([edges[int(index)] for index in chosen])
        return per_layer
    raise ValueError(f"unsupported edge mode: {mode}")


def build_hea_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    circuit = QuantumCircuit(problem.num_qubits)

    if spec.rotation_mode == "full":
        params_per_layer = 2 * problem.num_qubits
    elif spec.rotation_mode == "ry_only":
        params_per_layer = problem.num_qubits
    elif spec.rotation_mode == "shared":
        params_per_layer = 2
    else:
        raise ValueError(f"unsupported rotation mode: {spec.rotation_mode}")

    theta = ParameterVector("theta", spec.layers * params_per_layer)
    cursor = 0
    for layer in range(spec.layers):
        if spec.rotation_mode == "shared":
            shared_ry = theta[cursor]
            shared_rz = theta[cursor + 1]
            cursor += 2
            for qubit in range(problem.num_qubits):
                circuit.ry(shared_ry, qubit)
            for qubit in range(problem.num_qubits):
                circuit.rz(shared_rz, qubit)
        elif spec.rotation_mode == "ry_only":
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                cursor += 1
        else:
            for qubit in range(problem.num_qubits):
                circuit.ry(theta[cursor], qubit)
                cursor += 1
            for qubit in range(problem.num_qubits):
                circuit.rz(theta[cursor], qubit)
                cursor += 1

        for control, target in edges_by_layer[layer]:
            circuit.cx(control, target)

    return circuit, theta


def build_symm_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    width = (problem.num_qubits + 1) // 2
    theta = ParameterVector("theta", 2 * spec.layers * width)
    circuit = QuantumCircuit(problem.num_qubits)
    cursor = 0

    for _ in range(spec.layers):
        for left in range(width):
            right = problem.num_qubits - left - 1
            angle_y = theta[cursor]
            cursor += 1
            circuit.ry(angle_y, left)
            if right != left:
                circuit.ry(angle_y, right)
        for qubit in range(problem.num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for left in range(width):
            right = problem.num_qubits - left - 1
            angle_z = theta[cursor]
            cursor += 1
            circuit.rz(angle_z, left)
            if right != left:
                circuit.rz(angle_z, right)

    return circuit, theta


def build_brick_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", problem.num_qubits * (spec.layers + 1))
    circuit = QuantumCircuit(problem.num_qubits)
    cursor = 0

    for layer in range(spec.layers):
        for qubit in range(problem.num_qubits):
            circuit.ry(theta[cursor], qubit)
            cursor += 1

        if spec.edge_mode == "full":
            parities = [0, 1]
        elif spec.edge_mode == "even":
            parities = [0]
        elif spec.edge_mode == "odd":
            parities = [1]
        else:
            parities = [layer % 2]

        for parity in parities:
            for qubit in range(parity, problem.num_qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)

    for qubit in range(problem.num_qubits):
        circuit.ry(theta[cursor], qubit)
        cursor += 1

    return circuit, theta


def add_zz_block(circuit: QuantumCircuit, control: int, target: int, angle: Any) -> None:
    circuit.cx(control, target)
    circuit.rz(2.0 * angle, target)
    circuit.cx(control, target)


def build_tfim_shared_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    theta = ParameterVector("theta", 2 * spec.layers)
    circuit = QuantumCircuit(problem.num_qubits)
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    cursor = 0

    for layer in range(spec.layers):
        gamma = theta[cursor]
        beta = theta[cursor + 1]
        cursor += 2
        for control, target in edges_by_layer[layer]:
            add_zz_block(circuit, control, target, gamma)
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * beta, qubit)

    return circuit, theta


def build_tfim_factorized_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    edges_by_layer = select_edges(chain_edges(problem), spec.edge_mode, spec.layers, spec.seed)
    edge_params = sum(len(layer_edges) for layer_edges in edges_by_layer)
    theta = ParameterVector("theta", edge_params + spec.layers * problem.num_qubits)
    circuit = QuantumCircuit(problem.num_qubits)
    cursor = 0

    for layer in range(spec.layers):
        for control, target in edges_by_layer[layer]:
            add_zz_block(circuit, control, target, theta[cursor])
            cursor += 1
        for qubit in range(problem.num_qubits):
            circuit.rx(2.0 * theta[cursor], qubit)
            cursor += 1

    return circuit, theta


def build_ansatz(problem: prepare.Problem, spec: ExperimentSpec) -> tuple[QuantumCircuit, ParameterVector]:
    if spec.family == "hea":
        return build_hea_ansatz(problem, spec)
    if spec.family == "symm":
        return build_symm_ansatz(problem, spec)
    if spec.family == "brick":
        return build_brick_ansatz(problem, spec)
    if spec.family == "tfim_shared":
        return build_tfim_shared_ansatz(problem, spec)
    if spec.family == "tfim_factorized":
        return build_tfim_factorized_ansatz(problem, spec)
    raise ValueError(f"unsupported ansatz family: {spec.family}")


def initial_parameters(spec: ExperimentSpec, num_params: int) -> np.ndarray:
    if spec.param_init == "zeros":
        return np.zeros(num_params, dtype=float)
    rng = np.random.default_rng(spec.seed)
    if spec.param_init == "small_random":
        return rng.uniform(-0.05, 0.05, size=num_params)
    if spec.param_init == "random":
        return rng.uniform(-0.2, 0.2, size=num_params)
    raise ValueError(f"unsupported parameter init mode: {spec.param_init}")


def bind_parameters(circuit: QuantumCircuit, parameters: ParameterVector, values: np.ndarray) -> QuantumCircuit:
    mapping = dict(zip(parameters, values, strict=True))
    return circuit.assign_parameters(mapping, inplace=False)


def training_time_budget(problem: prepare.Problem, spec: ExperimentSpec) -> float:
    if spec.time_budget_seconds is not None:
        return float(spec.time_budget_seconds)
    return float(2**problem.num_qubits)


def optimize_energy(
    problem: prepare.Problem,
    circuit: QuantumCircuit,
    parameters: ParameterVector,
    spec: ExperimentSpec,
) -> tuple[np.ndarray, float, int]:
    calls = 0

    def objective(values: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        candidate = bind_parameters(circuit, parameters, values)
        return prepare.energy_from_circuit(candidate, problem.hamiltonian)

    values = initial_parameters(spec, len(parameters))
    if len(parameters) == 0:
        return values, objective(values), calls

    current_energy = objective(values)
    best_values = values.copy()
    best_energy = current_energy
    epsilon = float(spec.gradient_epsilon)
    deadline = time.perf_counter() + training_time_budget(problem, spec)
    optimizer = spec.optimizer.lower()

    if optimizer == "adam":
        m = np.zeros(len(parameters), dtype=float)
        v = np.zeros(len(parameters), dtype=float)
        step = 0
        while time.perf_counter() < deadline:
            step += 1
            gradient = np.zeros(len(parameters), dtype=float)
            for index in range(len(parameters)):
                if time.perf_counter() >= deadline:
                    break
                shifted = values.copy()
                shifted[index] += epsilon
                shifted_energy = objective(shifted)
                gradient[index] = (shifted_energy - current_energy) / epsilon

            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm < 1e-10:
                break

            m = spec.adam_beta1 * m + (1.0 - spec.adam_beta1) * gradient
            v = spec.adam_beta2 * v + (1.0 - spec.adam_beta2) * (gradient * gradient)
            m_hat = m / (1.0 - spec.adam_beta1**step)
            v_hat = v / (1.0 - spec.adam_beta2**step)
            values = values - spec.learning_rate * m_hat / (np.sqrt(v_hat) + spec.adam_epsilon)

            current_energy = objective(values)
            if current_energy < best_energy:
                best_energy = current_energy
                best_values = values.copy()

        return best_values, best_energy, calls

    if optimizer in {"gradient_descent", "gradient-descent", "gd"}:
        while time.perf_counter() < deadline:
            gradient = np.zeros(len(parameters), dtype=float)
            for index in range(len(parameters)):
                if time.perf_counter() >= deadline:
                    break
                shifted = values.copy()
                shifted[index] += epsilon
                shifted_energy = objective(shifted)
                gradient[index] = (shifted_energy - current_energy) / epsilon

            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm < 1e-10:
                break

            step_direction = gradient / max(1.0, gradient_norm)
            values = values - spec.learning_rate * step_direction
            current_energy = objective(values)
            if current_energy < best_energy:
                best_energy = current_energy
                best_values = values.copy()

        return best_values, best_energy, calls

    raise ValueError(f"unsupported optimizer: {spec.optimizer}")


def should_keep(candidate: ExperimentResult, incumbent: ExperimentResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.energy < incumbent.energy - ENERGY_IMPROVEMENT_TOL:
        return True
    if candidate.energy <= incumbent.energy + ENERGY_EQUIVALENCE_TOL:
        return candidate.compression_key < incumbent.compression_key
    return False


def summarize_run(index: int, result: ExperimentResult) -> None:
    print(
        f"[{index:03d}] {result.status:<7} "
        f"energy={result.energy:.6f} "
        f"twoq={result.metrics['twoq_count']:<3d} "
        f"total={result.metrics['total_gate_count']:<3d} "
        f"depth={result.metrics['depth']:<3d} "
        f"params={result.num_params:<3d} "
        f"{result.description}"
    )


def crash_result(index: int, description: str, total_seconds: float) -> ExperimentResult:
    return ExperimentResult(
        run_id=experiment_id(index, description),
        description=description,
        status="crash",
        energy=0.0,
        overlap=None,
        metrics={
            "singleq_count": 0,
            "twoq_count": 0,
            "total_gate_count": 0,
            "depth": 0,
        },
        num_params=0,
        eval_calls=0,
        total_seconds=total_seconds,
    )


def run_experiment(
    spec: ExperimentSpec,
    problem: prepare.Problem,
    backend_target: prepare.BackendTarget,
    index: int,
) -> ExperimentResult:
    started = time.perf_counter()
    try:
        circuit, parameters = build_ansatz(problem, spec)
        best_values, energy, eval_calls = optimize_energy(problem, circuit, parameters, spec)
        final_circuit = bind_parameters(circuit, parameters, best_values)
        _, compiled = prepare.transpile_and_report(final_circuit, backend_target)
        overlap = prepare.overlap_with_reference(final_circuit, problem.reference_state)
        total_seconds = time.perf_counter() - started
        return ExperimentResult(
            run_id=experiment_id(index, spec.description),
            description=spec.description,
            status="discard",
            energy=energy,
            overlap=overlap,
            metrics=compiled,
            num_params=len(parameters),
            eval_calls=eval_calls,
            total_seconds=total_seconds,
        )
    except Exception as exc:
        total_seconds = time.perf_counter() - started
        description = f"{spec.description} | crash={type(exc).__name__}: {exc}"
        return crash_result(index, description, total_seconds)


def build_baseline_spec() -> ExperimentSpec:
    return ExperimentSpec(
        family="hea",
        layers=2,
        optimizer="adam",
        param_init="small_random",
        learning_rate=0.2,
        seed=prepare.SEED,
        description="baseline hea layers=2 optimizer=adam init=small_random",
    )


def generate_energy_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = [build_baseline_spec()]
    families = ["hea", "symm", "brick", "tfim_shared"]
    for family in families:
        for layers in [1, 2, 3]:
            for optimizer in ["adam", "gradient_descent"]:
                for init_mode in ["zeros", "small_random"]:
                    if family == "hea" and layers == 2 and optimizer == "adam" and init_mode == "small_random":
                        continue
                    specs.append(
                        ExperimentSpec(
                            family=family,
                            layers=layers,
                            optimizer=optimizer,
                            param_init=init_mode,
                            learning_rate=0.2 if optimizer == "adam" else 0.1,
                            seed=prepare.SEED + layers,
                            description=(
                                f"diversify ansatz={family} layers={layers} "
                                f"optimizer={optimizer} init={init_mode}"
                            ),
                        )
                    )

    for family in ["hea", "symm", "brick", "tfim_factorized"]:
        for layers in [4, 5, 6]:
            for init_mode in ["zeros", "small_random"]:
                specs.append(
                    ExperimentSpec(
                        family=family,
                        layers=layers,
                        optimizer="adam",
                        param_init=init_mode,
                        learning_rate=0.15,
                        seed=prepare.SEED + 10 + layers,
                        description=(
                            f"complexify ansatz={family} layers={layers} "
                            f"optimizer=adam init={init_mode}"
                        ),
                    )
                )

    return specs


def generate_compression_specs() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    edge_modes = [
        ("hea", "ry_only", "even"),
        ("hea", "ry_only", "odd"),
        ("hea", "ry_only", "alternate"),
        ("hea", "shared", "even"),
        ("hea", "shared", "random_1"),
        ("brick", "full", "alternate"),
        ("tfim_shared", "full", "ends"),
    ]
    for family, rotation_mode, edge_mode in edge_modes:
        for layers in [1, 2, 3, 4]:
            specs.append(
                ExperimentSpec(
                    family=family,
                    layers=layers,
                    optimizer="adam",
                    param_init="small_random",
                    learning_rate=0.12 if family != "hea" else 0.15,
                    seed=prepare.SEED + 20 + layers,
                    edge_mode=edge_mode,
                    rotation_mode=rotation_mode,
                    description=(
                        f"compress ansatz={family}-{rotation_mode} layers={layers} "
                        f"optimizer=adam edge_mode={edge_mode}"
                    ),
                )
            )

    return specs


def format_best_summary(best_result: ExperimentResult, reference_energy: float | None) -> str:
    metrics = [
        ("energy", best_result.energy),
        ("reference_energy", reference_energy),
        ("overlap", best_result.overlap),
        ("singleq_count", best_result.metrics["singleq_count"]),
        ("twoq_count", best_result.metrics["twoq_count"]),
        ("total_gate_count", best_result.metrics["total_gate_count"]),
        ("depth", best_result.metrics["depth"]),
        ("num_params", best_result.num_params),
        ("eval_calls", best_result.eval_calls),
        ("total_seconds", best_result.total_seconds),
    ]
    return prepare.format_summary(metrics)


def main() -> None:
    ensure_results_header()
    problem = prepare.load_problem()
    backend_target = prepare.build_backend_target(problem)

    all_specs = generate_energy_specs() + generate_compression_specs()
    if len(all_specs) != TARGET_EXPERIMENTS:
        raise RuntimeError(f"expected {TARGET_EXPERIMENTS} experiments, found {len(all_specs)}")

    best_result: ExperimentResult | None = None
    experiment_count = 0

    for spec in all_specs:
        experiment_count += 1
        result = run_experiment(spec, problem, backend_target, experiment_count)
        if result.status != "crash" and should_keep(result, best_result):
            result = ExperimentResult(**{**result.__dict__, "status": "keep"})
            best_result = result
        log_result(result)
        summarize_run(experiment_count, result)

    if best_result is None:
        raise RuntimeError("all experiments crashed")

    print(format_best_summary(best_result, problem.reference_energy))


if __name__ == "__main__":
    main()
