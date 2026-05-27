from __future__ import annotations

import argparse
import csv
import json
import os
import py_compile
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import prepare


RESULTS_PATH = Path("results.tsv")
RUN_LOG_PATH = Path("run.log")
BENCHMARK_DIR = Path("benchmark_runs")
SOLVE_DIR = Path("solve_runs")
DEFAULT_PROBLEM = Path("examples/h2_2q.json")
DEFAULT_BENCHMARK_PROBLEMS = (
    Path("examples/h2_2q.json"),
    Path("examples/h2_4q.json"),
    Path("examples/ising_1d_5q.json"),
)
HARD_BENCHMARK_PROBLEMS = (
    Path("examples/tfim_n10_g1_open.json"),
    Path("examples/heisenberg_n10_open.json"),
)
LARGE_EXAMPLE_PROBLEMS = (Path("examples/ising_1d_9q.json"),)


@dataclass(frozen=True)
class SupportGroup:
    qubits: tuple[int, ...]
    terms: dict[str, float]


@dataclass(frozen=True)
class AnsatzCandidate:
    name: str
    priority: str
    reason: str
    first_moves: list[str]


@dataclass(frozen=True)
class HamiltonianProfile:
    name: str
    num_qubits: int
    raw_terms: int
    simplified_terms: int
    reference_energy: float | None
    time_budget_seconds: float
    basis_gates: list[str]
    coupling_edges: int
    max_locality: int
    locality_counts: dict[int, int]
    pauli_counts: dict[str, int]
    support_groups: list[SupportGroup]
    support_graph_edges: list[tuple[int, int]]
    support_graph_bipartite: bool | None
    model_class: str
    evidence: list[str]
    candidates: list[AnsatzCandidate]
    avoid: list[str]


@dataclass(frozen=True)
class ResultRow:
    run_id: str
    energy: float
    singleq_count: int
    twoq_count: int
    total_gate_count: int
    num_params: int
    status: str
    description: str

    @property
    def compression_key(self) -> tuple[int, int, int]:
        return self.twoq_count, self.total_gate_count, self.num_params

    @property
    def is_smoke(self) -> bool:
        return self.mode == "smoke"

    @property
    def mode(self) -> str:
        marker = "mode="
        if marker not in self.description:
            return "full"
        tail = self.description.split(marker, 1)[1]
        return tail.split()[0]

    @property
    def family(self) -> str:
        marker = "ansatz="
        if marker not in self.description:
            known = [
                "two_state_excitation",
                "heisenberg_hva",
                "pauli_hva",
                "tfim_factorized",
                "tfim_shared",
                "symm",
                "brick",
                "hea",
            ]
            words = set(self.description.replace("=", " ").split())
            for family in known:
                if family in words:
                    return family
            return "unknown"
        tail = self.description.split(marker, 1)[1]
        return tail.split()[0]


@dataclass(frozen=True)
class BenchmarkSummary:
    problem_path: Path
    name: str
    model_class: str
    reference_energy: float | None
    return_code: int
    runs: int
    best: ResultRow | None
    results_path: Path
    log_path: Path


@dataclass(frozen=True)
class SolveStage:
    name: str
    mode: str
    experiments: int
    experiment_seconds: float
    max_evals: int
    timeout: float


@dataclass(frozen=True)
class SolveSummary:
    problem_path: Path
    name: str
    model_class: str
    reference_energy: float | None
    passed: bool
    best: ResultRow | None
    best_stage: str | None
    runs: int
    results_paths: list[Path]
    log_paths: list[Path]


def simplified_terms(problem: prepare.Problem) -> list[tuple[str, float]]:
    labels = problem.hamiltonian.paulis.to_labels()
    coeffs = problem.hamiltonian.coeffs
    return [(label, float(coeff.real)) for label, coeff in zip(labels, coeffs, strict=True)]


def term_support(label: str) -> tuple[int, ...]:
    return tuple(len(label) - index - 1 for index, pauli in enumerate(label) if pauli != "I")


def term_ops(label: str, support: tuple[int, ...]) -> str:
    return "".join(label[len(label) - qubit - 1] for qubit in support)


def support_graph_bipartite(num_qubits: int, edges: list[tuple[int, int]]) -> bool | None:
    if not edges:
        return None

    graph: dict[int, list[int]] = {qubit: [] for qubit in range(num_qubits)}
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)

    color: dict[int, int] = {}
    for start in range(num_qubits):
        if start in color or not graph[start]:
            continue
        color[start] = 0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


def close_enough(values: list[float], tolerance: float = 1e-9) -> bool:
    if not values:
        return False
    return max(values) - min(values) <= tolerance * max(1.0, max(abs(value) for value in values))


def classify_model(
    groups: list[SupportGroup],
    locality_counts: Counter[int],
) -> tuple[str, list[str], list[str]]:
    evidence: list[str] = []
    avoid: list[str] = []

    all_ops = [op for group in groups for op in group.terms if op]
    two_qubit_groups = [group for group in groups if len(group.qubits) == 2]
    single_qubit_ops = {op for group in groups if len(group.qubits) == 1 for op in group.terms}

    if all(op and set(op) <= {"Z"} for op in all_ops):
        evidence.append("all non-identity Pauli factors are Z")
        return "classical_ising_or_qubo", evidence, ["generic HEA before QAOA-style baselines"]

    if locality_counts and max(locality_counts) <= 2:
        zz_edges = sum(1 for group in two_qubit_groups if set(group.terms) <= {"ZZ"})
        has_x_field = "X" in single_qubit_ops
        if zz_edges and has_x_field:
            evidence.append("two-local ZZ interactions with X-field terms")
            return "transverse_field_ising", evidence, ["fermionic UCC-style ansatz"]

    heisenberg_like = 0
    xx_yy_like = 0
    for group in two_qubit_groups:
        ops = set(group.terms)
        if {"XX", "YY", "ZZ"}.issubset(ops):
            coeffs = [group.terms["XX"], group.terms["YY"], group.terms["ZZ"]]
            if close_enough(coeffs):
                heisenberg_like += 1
        if {"XX", "YY"}.issubset(ops):
            coeffs = [group.terms["XX"], group.terms["YY"]]
            if close_enough(coeffs):
                xx_yy_like += 1

    if two_qubit_groups and heisenberg_like / len(two_qubit_groups) >= 0.6:
        evidence.append(f"{heisenberg_like}/{len(two_qubit_groups)} two-qubit supports contain matched XX, YY, ZZ terms")
        return "weighted_heisenberg_graph", evidence, ["TFIM-only ansatz as a first choice", "deep generic HEA before exchange/HVA trials"]

    if two_qubit_groups and xx_yy_like / len(two_qubit_groups) >= 0.6:
        evidence.append(f"{xx_yy_like}/{len(two_qubit_groups)} two-qubit supports contain matched XX, YY terms")
        return "xy_or_xxz_spin_graph", evidence, ["Z-only QAOA as a first choice"]

    if locality_counts and max(locality_counts) > 2:
        evidence.append("contains Pauli strings with locality greater than two")
        return "chemistry_or_general_pauli", evidence, ["plain hardware-efficient ansatz without symmetry or operator-pool checks"]

    evidence.append("mixed Pauli structure did not match a narrow built-in class")
    return "general_two_local_pauli", evidence, ["single-family search without an adaptive/operator-pool fallback"]


def candidate_policy(model_class: str, bipartite: bool | None) -> list[AnsatzCandidate]:
    if model_class == "classical_ising_or_qubo":
        return [
            AnsatzCandidate(
                name="qaoa_cost_mixer",
                priority="primary",
                reason="Z-diagonal Hamiltonians should use cost evolution plus a non-commuting mixer before generic HEA.",
                first_moves=["try p=1..4", "compare X mixer against constraint-preserving mixers if constraints exist"],
            ),
            AnsatzCandidate(
                name="commuting_group_hva",
                priority="secondary",
                reason="Commuting Z groups give a compact phase-separator baseline.",
                first_moves=["group terms by support coloring", "compress equivalent phase blocks"],
            ),
        ]

    if model_class == "transverse_field_ising":
        return [
            AnsatzCandidate(
                name="tfim_hva",
                priority="primary",
                reason="The Hamiltonian naturally separates into ZZ interaction and X-field evolution blocks.",
                first_moves=["alternate ZZ-cost and X-field layers", "start with shared angles then factorize edge angles"],
            ),
            AnsatzCandidate(
                name="qaoa_like_tfim",
                priority="secondary",
                reason="QAOA-style schedules are a compact way to test adiabatic-inspired structure.",
                first_moves=["sweep p before adding per-edge parameters", "keep HEA only as a baseline"],
            ),
            AnsatzCandidate(
                name="selected_ci_refinement",
                priority="target-refinement",
                reason="Near-critical TFIM VQE states can be refined by diagonalizing the Hamiltonian in the important sampled basis subspace.",
                first_moves=["take high-probability basis states from the best VQE state", "expand by Hamiltonian-connected bit flips"],
            ),
        ]

    if model_class == "weighted_heisenberg_graph":
        moves = ["build edge-local exp[-i theta (XX+YY+ZZ)] blocks", "test shared-angle then edge-factorized HVA"]
        if bipartite:
            moves.append("try Neel/reference preparation as counted ansatz gates")
        return [
            AnsatzCandidate(
                name="heisenberg_hva",
                priority="primary",
                reason="Matched XX, YY, ZZ terms point to exchange/Heisenberg evolution rather than TFIM blocks.",
                first_moves=moves,
            ),
            AnsatzCandidate(
                name="xy_exchange_pool",
                priority="primary",
                reason="XX+YY exchange blocks preserve magnetization and are a compact adaptive pool for spin graphs.",
                first_moves=["rank edge-local XX+YY and ZZ blocks by short smoke runs", "prefer graph-color layers over all-to-all ladders"],
            ),
            AnsatzCandidate(
                name="qubit_adapt_edge_pool",
                priority="secondary",
                reason="An operator pool generated from Hamiltonian supports lets data choose the useful blocks.",
                first_moves=["seed pool with each edge's XX, YY, ZZ and exchange combinations", "add one block at a time by improvement per two-qubit gate"],
            ),
            AnsatzCandidate(
                name="magnetization_sector_refinement",
                priority="target-refinement",
                reason="The isotropic chain conserves total magnetization, so the final energy can be refined in the half-filling sector.",
                first_moves=["prepare a singlet/dimer or Neel trial state", "diagonalize the projected Hamiltonian in the conserved sector for validation/refinement"],
            ),
        ]

    if model_class == "xy_or_xxz_spin_graph":
        return [
            AnsatzCandidate(
                name="xy_exchange_hva",
                priority="primary",
                reason="Matched XX/YY terms suggest number- or magnetization-preserving exchange layers.",
                first_moves=["use edge-color exchange layers", "add ZZ phases only if present in the Hamiltonian"],
            ),
            AnsatzCandidate(
                name="qubit_adapt_spin_pool",
                priority="secondary",
                reason="Pauli pools over the support graph can find the useful interaction subset.",
                first_moves=["start with Hamiltonian terms", "prune operators with near-zero parameters after convergence"],
            ),
        ]

    if model_class == "chemistry_or_general_pauli":
        return [
            AnsatzCandidate(
                name="symmetry_preserving_or_ucc",
                priority="primary-if-fermionic-metadata-exists",
                reason="Fermionic Hamiltonians should exploit reference states and particle/spin symmetries when metadata is available.",
                first_moves=["look for electron/orbital metadata", "use HF/UCC-like or excitation-preserving blocks if present"],
            ),
            AnsatzCandidate(
                name="commuting_group_hva",
                priority="primary-fallback",
                reason="Without fermionic metadata, commuting Pauli groups are the cleanest Hamiltonian-derived structure.",
                first_moves=["group commuting Pauli strings", "compare grouped HVA against a shallow HEA baseline"],
            ),
            AnsatzCandidate(
                name="qubit_adapt_pauli_pool",
                priority="secondary",
                reason="A Pauli operator pool keeps the search problem-derived while avoiding a single hard-coded ansatz.",
                first_moves=["rank pool operators with smoke runs", "keep additions only if energy/gate tradeoff improves"],
            ),
        ]

    return [
        AnsatzCandidate(
            name="commuting_group_hva",
            priority="primary",
            reason="For an unknown Pauli model, start from the Hamiltonian's own non-commuting groups.",
            first_moves=["build groups from Pauli commutation", "compare reps=1..3 under a smoke budget"],
        ),
        AnsatzCandidate(
            name="qubit_adapt_pauli_pool",
            priority="secondary",
            reason="Adaptive pools are safer than guessing one generic ansatz family.",
            first_moves=["start from Hamiltonian supports", "add hardware-native entanglers only after physics-inspired blocks stall"],
        ),
        AnsatzCandidate(
            name="hardware_efficient_baseline",
            priority="baseline-only",
            reason="HEA is useful as a control, not as the default scientific explanation.",
            first_moves=["keep shallow", "only complexify after problem-derived candidates fail"],
        ),
    ]


def analyze_problem(path: str | Path = DEFAULT_PROBLEM) -> HamiltonianProfile:
    problem = prepare.load_problem(path)
    backend_target = prepare.build_backend_target(problem)
    terms = simplified_terms(problem)

    locality_counts: Counter[int] = Counter()
    pauli_counts: Counter[str] = Counter()
    grouped: dict[tuple[int, ...], dict[str, float]] = defaultdict(dict)
    support_edges: set[tuple[int, int]] = set()

    for label, coeff in terms:
        support = term_support(label)
        ops = term_ops(label, support)
        locality_counts[len(support)] += 1
        for op in ops:
            pauli_counts[op] += 1
        grouped[support][ops] = grouped[support].get(ops, 0.0) + coeff
        if len(support) == 2:
            support_edges.add(tuple(sorted(support)))

    support_groups = [
        SupportGroup(qubits=support, terms=dict(sorted(ops.items())))
        for support, ops in sorted(grouped.items(), key=lambda item: (len(item[0]), item[0]))
    ]
    support_graph_edges = sorted(support_edges)
    bipartite = support_graph_bipartite(problem.num_qubits, support_graph_edges)
    model_class, evidence, avoid = classify_model(support_groups, locality_counts)
    if bipartite is not None:
        evidence.append(f"support graph is {'bipartite' if bipartite else 'not bipartite'}")

    return HamiltonianProfile(
        name=problem.name,
        num_qubits=problem.num_qubits,
        raw_terms=len(problem.pauli_terms),
        simplified_terms=len(terms),
        reference_energy=problem.reference_energy,
        time_budget_seconds=float(2 ** (problem.num_qubits - 2)),
        basis_gates=list(problem.basis_gates),
        coupling_edges=0 if backend_target.coupling_map is None else len(backend_target.coupling_map),
        max_locality=max(locality_counts) if locality_counts else 0,
        locality_counts=dict(sorted(locality_counts.items())),
        pauli_counts=dict(sorted(pauli_counts.items())),
        support_groups=support_groups,
        support_graph_edges=support_graph_edges,
        support_graph_bipartite=bipartite,
        model_class=model_class,
        evidence=evidence,
        candidates=candidate_policy(model_class, bipartite),
        avoid=avoid,
    )


def profile_to_json(profile: HamiltonianProfile) -> str:
    return json.dumps(asdict(profile), indent=2, sort_keys=True)


def print_profile(profile: HamiltonianProfile) -> None:
    print(f"name: {profile.name}")
    print(f"model_class: {profile.model_class}")
    print(f"num_qubits: {profile.num_qubits}")
    print(f"terms: raw={profile.raw_terms} simplified={profile.simplified_terms}")
    print(f"reference_energy: {profile.reference_energy}")
    print(f"time_budget_seconds: {profile.time_budget_seconds:.1f}")
    print(f"basis_gates: {','.join(profile.basis_gates)}")
    print(f"locality_counts: {profile.locality_counts}")
    print(f"pauli_counts: {profile.pauli_counts}")
    print(f"support_edges: {len(profile.support_graph_edges)} bipartite={profile.support_graph_bipartite}")
    print()
    print("evidence:")
    for item in profile.evidence:
        print(f"- {item}")
    print()
    print("recommended candidates:")
    for candidate in profile.candidates:
        print(f"- {candidate.name} [{candidate.priority}]: {candidate.reason}")
        for move in candidate.first_moves:
            print(f"  * {move}")
    if profile.avoid:
        print()
        print("avoid:")
        for item in profile.avoid:
            print(f"- {item}")


def read_results(path: Path | None = None) -> list[ResultRow]:
    if path is None:
        path = RESULTS_PATH
    if not path.exists():
        return []
    rows: list[ResultRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            try:
                rows.append(
                    ResultRow(
                        run_id=raw["commit"],
                        energy=float(raw["energy"]),
                        singleq_count=int(raw["singleq_count"]),
                        twoq_count=int(raw["twoq_count"]),
                        total_gate_count=int(raw["total_gate_count"]),
                        num_params=int(raw["num_params"]),
                        status=raw["status"],
                        description=raw["description"],
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def valid_rows(rows: list[ResultRow], *, include_smoke: bool = False) -> list[ResultRow]:
    return [
        row
        for row in rows
        if row.status != "crash" and (include_smoke or not row.is_smoke)
    ]


def best_row(rows: list[ResultRow], *, include_smoke: bool = False) -> ResultRow | None:
    candidates = valid_rows(rows, include_smoke=include_smoke)
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.energy, row.compression_key))


def target_threshold(reference_energy: float | None, rel_tol: float, abs_tol: float) -> float | None:
    if reference_energy is None:
        return None
    return max(float(abs_tol), float(rel_tol) * abs(float(reference_energy)))


def target_report(
    row: ResultRow | None,
    reference_energy: float | None,
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, float | None, float | None, float | None]:
    threshold = target_threshold(reference_energy, rel_tol, abs_tol)
    if row is None or threshold is None or reference_energy is None:
        return False, None, threshold, None
    gap = abs(row.energy - float(reference_energy))
    rel_error = gap / max(abs(float(reference_energy)), 1e-15)
    return gap <= threshold, gap, threshold, rel_error


def format_row(row: ResultRow) -> str:
    return (
        f"energy={row.energy:.6f} twoq={row.twoq_count} total={row.total_gate_count} "
        f"params={row.num_params} status={row.status} mode={row.mode} family={row.family} desc={row.description}"
    )


def pareto_rows(rows: list[ResultRow], near_energy: float) -> list[ResultRow]:
    candidates = [row for row in rows if row.energy <= near_energy and row.status != "crash"]
    frontier: list[ResultRow] = []
    for row in sorted(candidates, key=lambda item: (item.energy, item.compression_key)):
        dominated = False
        for other in candidates:
            if other is row:
                continue
            no_worse = other.energy <= row.energy and all(
                left <= right for left, right in zip(other.compression_key, row.compression_key, strict=True)
            )
            strictly_better = other.energy < row.energy or any(
                left < right for left, right in zip(other.compression_key, row.compression_key, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def print_results_summary(rows: list[ResultRow], reference_energy: float | None) -> None:
    if not rows:
        print("results: none")
        return

    status_counts = Counter(row.status for row in rows)
    full_rows = [row for row in rows if not row.is_smoke]
    smoke_rows = [row for row in rows if row.is_smoke]
    full_valid = valid_rows(full_rows)
    best = best_row(full_rows)
    best_smoke = best_row(smoke_rows, include_smoke=True)
    keeps = [row for row in full_rows if row.status == "keep"]
    last_keep = keeps[-1] if keeps else None

    print(f"runs: {len(rows)}")
    print(f"full_runs: {len(full_rows)}")
    print(f"smoke_runs: {len(smoke_rows)}")
    print(f"status_counts: {dict(sorted(status_counts.items()))}")
    if best is not None:
        print(f"best_full: {format_row(best)} id={best.run_id}")
        if reference_energy is not None:
            print(f"reference_gap: {best.energy - reference_energy:.6f}")
    elif best_smoke is not None:
        print("best_full: none")
    if last_keep is not None:
        print(f"last_full_keep: {format_row(last_keep)}")
    if best_smoke is not None:
        print(f"best_smoke: {format_row(best_smoke)} id={best_smoke.run_id}")

    if best is not None:
        print("near-best full pareto:")
        for row in pareto_rows(full_valid, best.energy + 5e-4)[:8]:
            print(f"- {format_row(row)}")


def run_train(
    timeout_seconds: float,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    smoke_keys = {"AUTOVQE_MAX_EXPERIMENTS", "AUTOVQE_EXPERIMENT_SECONDS", "AUTOVQE_MAX_EVALS"}
    if "AUTOVQE_RUN_MODE" not in env and smoke_keys.intersection(env):
        env["AUTOVQE_RUN_MODE"] = "smoke"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "train.py"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
            elapsed = time.perf_counter() - started
            log.write(f"\n[harness] timeout after {elapsed:.1f}s; process killed\n")
            print(f"timeout: killed train.py after {elapsed:.1f}s")
            return return_code if return_code != 0 else 124

    elapsed = time.perf_counter() - started
    print(f"train.py exited code={return_code} elapsed={elapsed:.1f}s log={log_path}")
    return return_code


def campaign_timeout(
    profile: HamiltonianProfile,
    *,
    mode: str,
    experiments: int | None,
    experiment_seconds: float,
) -> float:
    if mode == "smoke":
        count = experiments or 8
        return max(60.0, 30.0 + count * experiment_seconds * 6.0)
    if experiments:
        return 60.0 + profile.time_budget_seconds * experiments * 1.25
    return 60.0 + profile.time_budget_seconds * 120.0


def print_new_rows_summary(rows: list[ResultRow], reference_energy: float | None) -> None:
    if not rows:
        print("new_rows: none")
        return

    status_counts = Counter(row.status for row in rows)
    family_counts = Counter(row.family for row in rows)
    print(f"new_rows: {len(rows)}")
    print(f"new_status_counts: {dict(sorted(status_counts.items()))}")
    print(f"new_family_counts: {dict(sorted(family_counts.items()))}")
    best = best_row(rows, include_smoke=True)
    if best is not None:
        print(f"best_new: {format_row(best)}")
        if reference_energy is not None:
            print(f"best_new_reference_gap: {best.energy - reference_energy:.6f}")
    print("new_rows_detail:")
    for row in rows:
        print(f"- {format_row(row)}")


def recommend_next_action(
    profile: HamiltonianProfile,
    before: list[ResultRow],
    new_rows: list[ResultRow],
    *,
    mode: str,
) -> None:
    incumbent = best_row(before)
    best_new = best_row(new_rows, include_smoke=True)

    print("next_action:")
    if best_new is None:
        print("- No completed candidate was logged. Inspect run.log and fix the crash or timeout before changing the ansatz.")
        return

    primary_names = {candidate.name for candidate in profile.candidates if candidate.priority.startswith("primary")}
    family_is_primary = best_new.family in primary_names
    if mode == "smoke":
        if family_is_primary:
            print(
                f"- Promote `{best_new.family}` to a longer run because it matches `{profile.model_class}` "
                "and produced the best smoke candidate."
            )
        else:
            print(
                f"- Best smoke family `{best_new.family}` is not a primary recommendation for `{profile.model_class}`. "
                "Run one more smoke round after adding or prioritizing the primary Hamiltonian-derived candidate."
            )
        if incumbent is not None:
            print(f"- Full-budget incumbent remains: {format_row(incumbent)}")
        print("- Suggested promotion command:")
        print("  AUTOVQE_MAX_EXPERIMENTS=12 uv run harness.py campaign --mode full --experiments 12")
        return

    if incumbent is None or best_new.energy < incumbent.energy:
        print(f"- Keep the new full-budget result: {format_row(best_new)}")
        return

    if incumbent is not None and best_new.energy <= incumbent.energy + 5e-4:
        if best_new.compression_key < incumbent.compression_key:
            print(f"- Keep for compression: {format_row(best_new)}")
        else:
            print("- Energy tied but compression did not improve. Try fewer two-qubit blocks inside the same family.")
        return

    print("- Full run did not beat the incumbent. Add one Hamiltonian-derived variant or compress the current winner before broadening search.")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"ok: {message}")


def run_self_check(with_smoke: bool = False) -> int:
    for path in ["prepare.py", "train.py", "harness.py"]:
        py_compile.compile(path, doraise=True)
    check(True, "python files compile")

    profile = analyze_problem()
    check(profile.num_qubits > 0, "problem has qubits")
    check(profile.simplified_terms > 0, "problem has simplified Hamiltonian terms")
    check(bool(profile.candidates), "Hamiltonian audit produced ansatz candidates")

    import train

    problem = prepare.load_problem()
    backend = prepare.build_backend_target(problem)
    if any(candidate.name == "heisenberg_hva" for candidate in profile.candidates):
        check(profile.model_class == "weighted_heisenberg_graph", "current audit maps Heisenberg evidence to weighted_heisenberg_graph")
        check(True, "audit recommends heisenberg_hva")
        spec = train.build_spec(
            family="heisenberg_hva",
            layers=1,
            optimizer="spsa",
            param_init="zeros",
            learning_rate=0.18,
            seed=prepare.SEED,
            edge_mode="colored",
            rotation_mode="shared",
            reference_state="neel",
            description="self-check heisenberg_hva",
        )
        expected_family = "heisenberg_hva"
    else:
        spec = train.build_baseline_spec()
        expected_family = spec.family

    circuit, params = train.build_ansatz(problem, spec)
    bound = circuit.assign_parameters({param: 0.123 for param in params}, inplace=False)
    _, metrics = prepare.transpile_and_report(bound, backend)
    check(len(params) > 0, f"{expected_family} exposes tunable parameters")
    check(metrics["total_gate_count"] > 0, f"{expected_family} transpiles to counted gates")

    rows = read_results()
    check(True, "results.tsv is optional and parses when present")
    if rows:
        check(best_row(rows) is not None, "full-result incumbent can be selected")
    else:
        check(True, "fresh checkout does not require results.tsv")

    timeout = campaign_timeout(profile, mode="smoke", experiments=1, experiment_seconds=0.01)
    check(timeout >= 60.0, "smoke campaign timeout is bounded")

    if with_smoke:
        original_results = RESULTS_PATH.read_bytes() if RESULTS_PATH.exists() else None
        try:
            before = read_results()
            args = argparse.Namespace(
                mode="smoke",
                experiments=1,
                experiment_seconds=0.01,
                max_evals=2,
                timeout=30.0,
                log=str(RUN_LOG_PATH),
                dry_run=False,
            )
            return_code = run_campaign(args)
            check(return_code == 0, "one-experiment smoke campaign exits cleanly")
            after = read_results()
            check(len(after) == len(before) + 1, "smoke campaign appends one ledger row")
            check(after[-1].is_smoke, "smoke campaign row is marked as smoke")
        finally:
            if original_results is None:
                RESULTS_PATH.unlink(missing_ok=True)
            else:
                RESULTS_PATH.write_bytes(original_results)
            print("ok: restored results.tsv after smoke self-check")

    return 0


def run_campaign(args: argparse.Namespace) -> int:
    profile = analyze_problem()
    before = read_results()
    mode = args.mode
    experiments = args.experiments
    if experiments is None and mode == "smoke":
        experiments = 8

    timeout = args.timeout
    if timeout is None:
        timeout = campaign_timeout(
            profile,
            mode=mode,
            experiments=experiments,
            experiment_seconds=args.experiment_seconds,
        )

    print_profile(profile)
    print()
    print(f"campaign: mode={mode} experiments={experiments or 'train-default'} timeout={timeout:.1f}s")

    extra_env: dict[str, str] = {
        "AUTOVQE_RUN_MODE": mode,
        "AUTOVQE_MODEL_CLASS": profile.model_class,
    }
    if experiments:
        extra_env["AUTOVQE_MAX_EXPERIMENTS"] = str(experiments)
    if mode == "smoke":
        extra_env["AUTOVQE_EXPERIMENT_SECONDS"] = str(args.experiment_seconds)
        extra_env["AUTOVQE_MAX_EVALS"] = str(args.max_evals)

    if args.dry_run:
        print(f"dry_run_env: {extra_env}")
        print(f"dry_run_command: {sys.executable} train.py")
        return 0

    return_code = run_train(timeout_seconds=timeout, log_path=Path(args.log), extra_env=extra_env)
    after = read_results()
    new_rows = after[len(before) :] if len(after) >= len(before) else []
    print()
    print_new_rows_summary(new_rows, profile.reference_energy)
    print()
    recommend_next_action(profile, before, new_rows, mode=mode)
    return return_code


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("/", "_")


def benchmark_problem(
    problem_path: Path,
    *,
    mode: str,
    experiments: int,
    experiment_seconds: float,
    max_evals: int,
    timeout: float | None,
    output_dir: Path,
) -> BenchmarkSummary:
    profile = analyze_problem(problem_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_name(problem_path)
    results_path = output_dir / f"{slug}.tsv"
    log_path = output_dir / f"{slug}.log"
    results_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

    run_timeout = timeout
    if run_timeout is None:
        run_timeout = campaign_timeout(
            profile,
            mode=mode,
            experiments=experiments,
            experiment_seconds=experiment_seconds,
        )

    extra_env = {
        "AUTOVQE_PROBLEM_PATH": str(problem_path),
        "AUTOVQE_RESULTS_PATH": str(results_path),
        "AUTOVQE_MODEL_CLASS": profile.model_class,
        "AUTOVQE_RUN_MODE": mode,
        "AUTOVQE_MAX_EXPERIMENTS": str(experiments),
    }
    if mode == "smoke":
        extra_env["AUTOVQE_EXPERIMENT_SECONDS"] = str(experiment_seconds)
        extra_env["AUTOVQE_MAX_EVALS"] = str(max_evals)

    print()
    print(f"benchmark_problem: {problem_path}")
    print(f"model_class: {profile.model_class}")
    print(f"reference_energy: {profile.reference_energy}")
    print(f"timeout: {run_timeout:.1f}s results={results_path} log={log_path}")
    return_code = run_train(run_timeout, log_path, extra_env=extra_env)
    rows = read_results(results_path)
    best = best_row(rows, include_smoke=True)
    if best is None:
        print("best: none")
    else:
        gap = None if profile.reference_energy is None else best.energy - profile.reference_energy
        gap_text = "n/a" if gap is None else f"{gap:.6f}"
        print(f"best: {format_row(best)} reference_gap={gap_text}")

    return BenchmarkSummary(
        problem_path=problem_path,
        name=profile.name,
        model_class=profile.model_class,
        reference_energy=profile.reference_energy,
        return_code=return_code,
        runs=len(rows),
        best=best,
        results_path=results_path,
        log_path=log_path,
    )


def default_benchmark_problems(include_hard: bool, include_large: bool) -> list[Path]:
    paths = list(DEFAULT_BENCHMARK_PROBLEMS)
    if include_hard:
        paths.extend(HARD_BENCHMARK_PROBLEMS)
    if include_large:
        paths.extend(LARGE_EXAMPLE_PROBLEMS)
    return paths


def print_benchmark_summary(summaries: list[BenchmarkSummary]) -> None:
    print()
    print("benchmark_summary:")
    for summary in summaries:
        if summary.best is None:
            print(
                f"- {summary.problem_path}: code={summary.return_code} runs={summary.runs} "
                f"model={summary.model_class} best=none"
            )
            continue
        gap = None if summary.reference_energy is None else summary.best.energy - summary.reference_energy
        gap_text = "n/a" if gap is None else f"{gap:.6f}"
        print(
            f"- {summary.problem_path}: code={summary.return_code} runs={summary.runs} "
            f"model={summary.model_class} best_energy={summary.best.energy:.6f} "
            f"reference_gap={gap_text} family={summary.best.family} "
            f"twoq={summary.best.twoq_count} total={summary.best.total_gate_count} params={summary.best.num_params}"
        )


def run_benchmark(args: argparse.Namespace) -> int:
    problem_paths = [Path(path) for path in args.problems]
    if not problem_paths:
        problem_paths = default_benchmark_problems(include_hard=args.include_hard, include_large=args.include_large)
    if not problem_paths:
        raise RuntimeError("no benchmark problems found")

    summaries = [
        benchmark_problem(
            problem_path,
            mode=args.mode,
            experiments=args.experiments,
            experiment_seconds=args.experiment_seconds,
            max_evals=args.max_evals,
            timeout=args.timeout,
            output_dir=Path(args.output_dir),
        )
        for problem_path in problem_paths
    ]
    print_benchmark_summary(summaries)
    return 0 if all(summary.return_code == 0 and summary.best is not None for summary in summaries) else 1


def default_solve_stages(max_stages: int) -> list[SolveStage]:
    stages = [
        SolveStage("smoke", "smoke", experiments=45, experiment_seconds=2.0, max_evals=300, timeout=120.0),
        SolveStage("standard", "standard", experiments=90, experiment_seconds=4.0, max_evals=800, timeout=300.0),
        SolveStage("deep", "deep", experiments=160, experiment_seconds=8.0, max_evals=1800, timeout=900.0),
    ]
    return stages[:max(1, max_stages)]


def print_target_status(
    *,
    best: ResultRow | None,
    reference_energy: float | None,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    passed, gap, threshold, rel_error = target_report(best, reference_energy, rel_tol=rel_tol, abs_tol=abs_tol)
    if best is None:
        print("target_status: no completed result")
        return False
    if reference_energy is None:
        print("target_status: no reference energy available; cannot prove solved")
        return False
    assert gap is not None and threshold is not None and rel_error is not None
    print(
        "target_status: "
        f"passed={passed} gap={gap:.12f} threshold={threshold:.12f} rel_error={100.0 * rel_error:.6f}%"
    )
    return passed


def solve_problem(
    problem_path: Path,
    *,
    rel_tol: float,
    abs_tol: float,
    output_dir: Path,
    max_stages: int,
    timeout: float | None,
    extra_compress: int,
) -> SolveSummary:
    profile = analyze_problem(problem_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_name(problem_path)
    all_rows: list[tuple[str, ResultRow]] = []
    results_paths: list[Path] = []
    log_paths: list[Path] = []

    print()
    print(f"solve_problem: {problem_path}")
    print(f"name: {profile.name}")
    print(f"model_class: {profile.model_class}")
    print(f"reference_energy: {profile.reference_energy}")
    print(f"target_rel_tol: {rel_tol}")
    print(f"target_abs_tol: {abs_tol}")
    print("policy: audit -> smoke -> target check -> escalate only if needed")

    for stage in default_solve_stages(max_stages):
        results_path = output_dir / f"{slug}_{stage.name}.tsv"
        log_path = output_dir / f"{slug}_{stage.name}.log"
        results_path.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
        results_paths.append(results_path)
        log_paths.append(log_path)

        stage_timeout = timeout if timeout is not None else stage.timeout
        extra_env = {
            "AUTOVQE_PROBLEM_PATH": str(problem_path),
            "AUTOVQE_RESULTS_PATH": str(results_path),
            "AUTOVQE_MODEL_CLASS": profile.model_class,
            "AUTOVQE_RUN_MODE": stage.mode,
            "AUTOVQE_MAX_EXPERIMENTS": str(stage.experiments),
            "AUTOVQE_EXPERIMENT_SECONDS": str(stage.experiment_seconds),
            "AUTOVQE_MAX_EVALS": str(stage.max_evals),
            "AUTOVQE_TARGET_REL_ERROR": str(rel_tol),
            "AUTOVQE_TARGET_ABS_ERROR": str(abs_tol),
            "AUTOVQE_STOP_AT_TARGET": "1",
            "AUTOVQE_TARGET_EXTRA_COMPRESS": str(extra_compress),
        }

        print()
        print(
            f"stage: {stage.name} experiments={stage.experiments} "
            f"seconds={stage.experiment_seconds} max_evals={stage.max_evals} timeout={stage_timeout:.1f}s"
        )
        return_code = run_train(stage_timeout, log_path, extra_env=extra_env)
        rows = read_results(results_path)
        all_rows.extend((stage.name, row) for row in rows)
        stage_best = best_row(rows, include_smoke=True)
        if stage_best is None:
            print(f"stage_result: code={return_code} best=none")
        else:
            print(f"stage_result: code={return_code} best={format_row(stage_best)}")
        if print_target_status(best=stage_best, reference_energy=profile.reference_energy, rel_tol=rel_tol, abs_tol=abs_tol):
            break

    if all_rows:
        best_stage, best = min(
            all_rows,
            key=lambda item: (item[1].energy, item[1].compression_key),
        )
    else:
        best_stage, best = None, None
    passed, _, _, _ = target_report(best, profile.reference_energy, rel_tol=rel_tol, abs_tol=abs_tol)
    print()
    print("solve_summary:")
    if best is None:
        print("- best=none")
    else:
        print(f"- best_stage={best_stage} {format_row(best)}")
    print_target_status(best=best, reference_energy=profile.reference_energy, rel_tol=rel_tol, abs_tol=abs_tol)

    return SolveSummary(
        problem_path=problem_path,
        name=profile.name,
        model_class=profile.model_class,
        reference_energy=profile.reference_energy,
        passed=passed,
        best=best,
        best_stage=best_stage,
        runs=len(all_rows),
        results_paths=results_paths,
        log_paths=log_paths,
    )


def run_solve(args: argparse.Namespace) -> int:
    problem_paths = [Path(path) for path in args.problems] or [DEFAULT_PROBLEM]
    summaries = [
        solve_problem(
            problem_path,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
            output_dir=Path(args.output_dir),
            max_stages=args.max_stages,
            timeout=args.timeout,
            extra_compress=args.extra_compress,
        )
        for problem_path in problem_paths
    ]

    print()
    print("solve_rollup:")
    for summary in summaries:
        if summary.best is None:
            print(f"- {summary.problem_path}: passed=False runs={summary.runs} best=none")
            continue
        passed, gap, threshold, rel_error = target_report(
            summary.best,
            summary.reference_energy,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )
        gap_text = "n/a" if gap is None else f"{gap:.6f}"
        threshold_text = "n/a" if threshold is None else f"{threshold:.6f}"
        rel_text = "n/a" if rel_error is None else f"{100.0 * rel_error:.4f}%"
        print(
            f"- {summary.problem_path}: passed={passed} runs={summary.runs} "
            f"best_energy={summary.best.energy:.6f} gap={gap_text} threshold={threshold_text} "
            f"rel_error={rel_text} family={summary.best.family} stage={summary.best_stage}"
        )
    return 0 if all(summary.passed for summary in summaries) else 1


def print_runbook(profile: HamiltonianProfile) -> None:
    print("runbook:")
    print("1. Run `uv run harness.py solve <problem-file> --rel-tol <target>` when the target is known.")
    print("2. If solve fails, inspect the failed stage logs and add the missing Hamiltonian-derived candidate in train.py.")
    print("3. Keep every tunable rotation as an explicit parameter and count all reference-prep gates.")
    print("4. Re-run solve; only report success when target_status says passed=True.")
    print("5. After target is reached, optionally use `--extra-compress` to search for simpler tied circuits.")
    print()
    print("suggested solve command:")
    print(f"uv run harness.py solve {DEFAULT_PROBLEM} --rel-tol 0.001")
    print()
    print("manual smoke command:")
    print(
        "AUTOVQE_MAX_EXPERIMENTS=6 AUTOVQE_EXPERIMENT_SECONDS=2 "
        "AUTOVQE_MAX_EVALS=40 uv run harness.py run --timeout 90"
    )
    print()
    print("suggested full command:")
    print(f"uv run harness.py run --timeout {int(profile.time_budget_seconds * 120 + 60)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoVQE research harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="analyze a problem JSON file")
    inspect_parser.add_argument("--problem", default=str(DEFAULT_PROBLEM))
    inspect_parser.add_argument("--json", action="store_true")

    subparsers.add_parser("plan", help="print Hamiltonian-aware experiment runbook")
    subparsers.add_parser("results", help="summarize results.tsv")

    check_parser = subparsers.add_parser("check", help="run fast harness self-checks")
    check_parser.add_argument("--with-smoke", action="store_true", help="run one smoke experiment and restore results.tsv")

    run_parser = subparsers.add_parser("run", help="run train.py with a wall-clock timeout")
    run_parser.add_argument("--timeout", type=float, default=600.0)
    run_parser.add_argument("--log", default=str(RUN_LOG_PATH))

    campaign_parser = subparsers.add_parser("campaign", help="inspect, run, summarize, and recommend next action")
    campaign_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    campaign_parser.add_argument("--experiments", type=int, default=None)
    campaign_parser.add_argument("--experiment-seconds", type=float, default=2.0)
    campaign_parser.add_argument("--max-evals", type=int, default=40)
    campaign_parser.add_argument("--timeout", type=float, default=None)
    campaign_parser.add_argument("--log", default=str(RUN_LOG_PATH))
    campaign_parser.add_argument("--dry-run", action="store_true")

    benchmark_parser = subparsers.add_parser("benchmark", help="run isolated campaigns over multiple problem files")
    benchmark_parser.add_argument("problems", nargs="*", help="problem JSON files; defaults to the small examples")
    benchmark_parser.add_argument("--include-hard", action="store_true", help="also include the n=10 TFIM and Heisenberg hard targets")
    benchmark_parser.add_argument("--include-large", action="store_true", help="also include the slower 9-qubit weighted spin example")
    benchmark_parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    benchmark_parser.add_argument("--experiments", type=int, default=8)
    benchmark_parser.add_argument("--experiment-seconds", type=float, default=0.5)
    benchmark_parser.add_argument("--max-evals", type=int, default=60)
    benchmark_parser.add_argument("--timeout", type=float, default=None)
    benchmark_parser.add_argument("--output-dir", default=str(BENCHMARK_DIR))

    solve_parser = subparsers.add_parser("solve", help="run an escalating target-driven solve loop")
    solve_parser.add_argument("problems", nargs="*", help=f"problem JSON files; defaults to {DEFAULT_PROBLEM}")
    solve_parser.add_argument("--rel-tol", type=float, default=1e-3, help="relative energy tolerance versus reference")
    solve_parser.add_argument("--abs-tol", type=float, default=0.0, help="absolute energy tolerance floor")
    solve_parser.add_argument("--max-stages", type=int, default=3, help="number of solve stages to try")
    solve_parser.add_argument("--timeout", type=float, default=None, help="override timeout for each stage")
    solve_parser.add_argument("--extra-compress", type=int, default=0, help="extra experiments after target is reached")
    solve_parser.add_argument("--output-dir", default=str(SOLVE_DIR))

    args = parser.parse_args()

    if args.command == "inspect":
        profile = analyze_problem(args.problem)
        if args.json:
            print(profile_to_json(profile))
        else:
            print_profile(profile)
        return 0

    if args.command == "plan":
        profile = analyze_problem()
        print_profile(profile)
        print()
        print_runbook(profile)
        return 0

    if args.command == "results":
        profile = analyze_problem()
        print_results_summary(read_results(), profile.reference_energy)
        return 0

    if args.command == "check":
        return run_self_check(with_smoke=args.with_smoke)

    if args.command == "run":
        return run_train(timeout_seconds=args.timeout, log_path=Path(args.log))

    if args.command == "campaign":
        return run_campaign(args)

    if args.command == "benchmark":
        return run_benchmark(args)

    if args.command == "solve":
        return run_solve(args)

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
