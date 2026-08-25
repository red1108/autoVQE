from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
)
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    InitialStateSpec,
    PauliTerm,
    PublicProblem,
    SectorSpec,
)
from autovqe.controller import ControllerError, ResearchController
from autovqe.evaluator import (
    EvaluationProtocol,
    candidate_identity,
    evaluate_ansatz,
    evaluate_public_problem,
)
from autovqe.probes import (
    MAX_GENERATOR_TERMS,
    ProbeValidationError,
    algebraic_probe_cost_units,
    run_algebraic_probe,
)


def rotation_spec(
    coefficients: tuple[float, ...] = (1.0,),
) -> AnsatzSpec:
    return AnsatzSpec(
        num_qubits=1,
        parameters=("theta",),
        operations=tuple(
            OperationSpec(
                macro="PauliRotation",
                qubits=(0,),
                parameters={
                    "angle": ParameterExpression.parameter(
                        "theta", coefficient
                    )
                },
                options={"pauli": "X"},
            )
            for coefficient in coefficients
        ),
    )


def public_problem(*, occupation: tuple[int, ...] | None = None) -> PublicProblem:
    return PublicProblem.create(
        num_qubits=1,
        pauli_terms=(PauliTerm("Z", 1.0),),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        initial_state=InitialStateSpec(
            kind="computational_basis" if occupation is not None else "unspecified",
            occupation=occupation,
        ),
        backend=BackendSpec(),
    )


def pauli_label(index: int, width: int, alphabet: str = "IXYZ") -> str:
    letters: list[str] = []
    for _ in range(width):
        letters.append(alphabet[index % len(alphabet)])
        index //= len(alphabet)
    return "".join(letters)


class TrustedEvaluatorHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = EvaluationProtocol(max_evals=4, restarts=1, seed=17)

    def test_public_evaluator_rejects_candidate_authored_initialization(self) -> None:
        forged = rotation_spec().to_dict()
        forged["reference"] = {"macro": "X", "qubits": [0]}
        result = evaluate_public_problem(
            public_problem(),
            forged,
            protocol=self.protocol,
        )

        self.assertFalse(result.result.valid)
        self.assertIn("unsupported fields", result.result.violations[0])

    def test_public_evaluator_prepends_declared_initial_state(self) -> None:
        empty = AnsatzSpec(num_qubits=1)
        prepared = evaluate_public_problem(
            public_problem(occupation=(1,)),
            empty,
            protocol=self.protocol,
        )

        self.assertTrue(prepared.result.valid, prepared.result.violations)
        self.assertAlmostEqual(prepared.result.best_energy, -1.0)
        self.assertEqual(prepared.result.optimized_parameter_binding, {})

    def test_low_level_evaluator_rejects_non_hermitian_hamiltonian(self) -> None:
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", 1j)]),
            rotation_spec(),
            protocol=self.protocol,
        )

        self.assertFalse(result.result.valid)
        self.assertIn("Hermitian", result.result.violations[0])

    def test_low_level_evaluator_rejects_non_finite_hamiltonian(self) -> None:
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", float("nan"))]),
            rotation_spec(),
            protocol=self.protocol,
        )

        self.assertFalse(result.result.valid)
        self.assertIn("finite", result.result.violations[0])

    def test_rotation_split_and_cancellation_share_semantic_identity(self) -> None:
        unsplit = rotation_spec((1.0,))
        split = rotation_spec((0.5, 0.5))
        cancel_retry = rotation_spec((1.0, 1.0, -1.0))

        self.assertEqual(candidate_identity(unsplit), candidate_identity(split))
        self.assertEqual(candidate_identity(unsplit), candidate_identity(cancel_retry))

    def test_rotation_split_cannot_buy_a_fresh_controller_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                public_problem(), Path(directory) / "events.jsonl"
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "structure",
                    "claim": {"kind": "ansatz_structure", "family": "rotation"},
                }
            )
            controller.dispatch_external(
                {
                    "type": "submit_candidate",
                    "candidate_id": "unsplit",
                    "hypothesis_id": "structure",
                    "spec": rotation_spec((1.0,)).to_dict(),
                    "metadata": {
                        "falsifier": "the trusted evaluation shows no improvement",
                    },
                }
            )

            with self.assertRaisesRegex(ControllerError, "semantically equivalent"):
                controller.dispatch_external(
                    {
                        "type": "submit_candidate",
                        "candidate_id": "split_retry",
                        "hypothesis_id": "structure",
                        "spec": rotation_spec((0.5, 0.5)).to_dict(),
                        "metadata": {
                            "falsifier": "the trusted evaluation shows no improvement",
                        },
                    }
                )


class ProbeHardeningTests(unittest.TestCase):
    def test_16_qubit_global_symmetry_probe_has_productive_cost(self) -> None:
        hamiltonian = SparsePauliOp.from_list(
            [
                (pauli_label(index + 1, 16), 1.0)
                for index in range(1_245)
            ]
        )
        request = {
            "type": "normalized_commutator",
            "generator": {
                "type": "global_pauli_sum",
                "pauli": "Z",
            },
        }

        self.assertEqual(algebraic_probe_cost_units(hamiltonian, request), 1.25)

    def test_probe_rejects_non_hermitian_hamiltonian(self) -> None:
        request = {
            "type": "normalized_commutator",
            "generator": {
                "type": "pauli_sum",
                "terms": [{"pauli": "X", "coeff": 1.0}],
            },
        }

        with self.assertRaisesRegex(ProbeValidationError, "Hermitian"):
            run_algebraic_probe(
                SparsePauliOp.from_list([("Z", 1j)]), request
            )

    def test_generator_term_cap_is_enforced_before_probe_execution(self) -> None:
        terms = [
            {"pauli": pauli_label(index, 5), "coeff": 1.0}
            for index in range(MAX_GENERATOR_TERMS + 1)
        ]
        request = {
            "type": "normalized_commutator",
            "generator": {"type": "pauli_sum", "terms": terms},
        }

        with self.assertRaisesRegex(ProbeValidationError, "term cap"):
            algebraic_probe_cost_units(
                SparsePauliOp.from_list([("ZIIII", 1.0)]), request
            )

    def test_probe_cost_scales_with_validated_sparse_work(self) -> None:
        hamiltonian = SparsePauliOp.from_list([("Z" + "I" * 7, 1.0)])
        small = {
            "type": "initial_state_moments",
            "generator": {
                "type": "pauli_sum",
                "terms": [{"pauli": "X" + "I" * 7, "coeff": 1.0}],
            },
        }
        large = {
            "type": "initial_state_moments",
            "generator": {
                "type": "pauli_sum",
                "terms": [
                    {"pauli": pauli_label(index, 8), "coeff": 1.0}
                    for index in range(MAX_GENERATOR_TERMS)
                ],
            },
        }

        self.assertGreater(
            algebraic_probe_cost_units(hamiltonian, large),
            algebraic_probe_cost_units(hamiltonian, small),
        )

    def test_controller_debits_the_complexity_derived_probe_cost(self) -> None:
        labels = [
            pauli_label(index, 7, alphabet="IZ")
            for index in range(65)
        ]
        problem = PublicProblem.create(
            num_qubits=7,
            pauli_terms=tuple(PauliTerm(label, 1.0) for label in labels),
            encoding=EncodingSpec(),
            sector=SectorSpec(),
            initial_state=InitialStateSpec(),
            backend=BackendSpec(),
        )
        generator = {
            "type": "pauli_sum",
            "terms": [{"pauli": "Z" + "I" * 6, "coeff": 1.0}],
        }

        with tempfile.TemporaryDirectory() as directory:
            controller = ResearchController(
                problem, Path(directory) / "events.jsonl", total_budget=1.0
            )
            controller.dispatch_external(
                {
                    "type": "propose_hypothesis",
                    "hypothesis_id": "large_h",
                    "claim": {
                        "kind": "exact_pauli_symmetry",
                        "generator": generator,
                    },
                }
            )
            step = controller.dispatch_external(
                {
                    "type": "request_probe",
                    "hypothesis_id": "large_h",
                }
            )

        self.assertEqual(step.result["cost_units"], 0.25)
        self.assertAlmostEqual(step.state_summary["budget"]["spent"], 0.35)


if __name__ == "__main__":
    unittest.main()
