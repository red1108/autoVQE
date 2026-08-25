from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qiskit.quantum_info import SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzSpec,
    LayerSpec,
    OperationSpec,
    ParameterExpression,
    ReferenceSpec as AnsatzReferenceSpec,
)
from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
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
    *,
    reference_qubits: tuple[int, ...] | None = None,
) -> AnsatzSpec:
    reference = (
        None
        if reference_qubits is None
        else AnsatzReferenceSpec(macro="X", qubits=reference_qubits)
    )
    return AnsatzSpec(
        num_qubits=1,
        parameters=("theta",),
        reference=reference,
        layers=(
            LayerSpec(
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
                )
            ),
        ),
    )


def public_problem(*, occupation: tuple[int, ...] | None = None) -> PublicProblem:
    return PublicProblem.create(
        num_qubits=1,
        pauli_terms=(PauliTerm("Z", 1.0),),
        encoding=EncodingSpec(),
        sector=SectorSpec(),
        reference=ReferenceSpec(
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

    def test_public_evaluator_rejects_candidate_introduced_reference(self) -> None:
        result = evaluate_public_problem(
            public_problem(),
            rotation_spec(reference_qubits=(0,)),
            protocol=self.protocol,
        )

        self.assertFalse(result.receipt.valid)
        self.assertIn("cannot introduce a reference", result.receipt.violations[0])

    def test_public_evaluator_requires_the_declared_reference(self) -> None:
        missing = evaluate_public_problem(
            public_problem(occupation=(1,)),
            rotation_spec(),
            protocol=self.protocol,
        )
        exact = evaluate_public_problem(
            public_problem(occupation=(1,)),
            rotation_spec(reference_qubits=(0,)),
            protocol=self.protocol,
        )

        self.assertFalse(missing.receipt.valid)
        self.assertIn("must exactly match", missing.receipt.violations[0])
        self.assertTrue(exact.receipt.valid, exact.receipt.violations)

    def test_low_level_evaluator_rejects_non_hermitian_hamiltonian(self) -> None:
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", 1j)]),
            rotation_spec(),
            protocol=self.protocol,
        )

        self.assertFalse(result.receipt.valid)
        self.assertIn("Hermitian", result.receipt.violations[0])

    def test_low_level_evaluator_rejects_non_finite_hamiltonian(self) -> None:
        result = evaluate_ansatz(
            SparsePauliOp.from_list([("Z", float("nan"))]),
            rotation_spec(),
            protocol=self.protocol,
        )

        self.assertFalse(result.receipt.valid)
        self.assertIn("finite", result.receipt.violations[0])

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
                        "enforcement": "unconstrained",
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
                            "enforcement": "unconstrained",
                            "falsifier": "the trusted evaluation shows no improvement",
                        },
                    }
                )


class ProbeHardeningTests(unittest.TestCase):
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
            "type": "reference_moments",
            "generator": {
                "type": "pauli_sum",
                "terms": [{"pauli": "X" + "I" * 7, "coeff": 1.0}],
            },
        }
        large = {
            "type": "reference_moments",
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
            reference=ReferenceSpec(),
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
            receipt = controller.dispatch_external(
                {
                    "type": "request_probe",
                    "hypothesis_id": "large_h",
                    "probe_id": "large_h_commutator",
                    "probe": {
                        "type": "normalized_commutator",
                        "generator": generator,
                    },
                }
            )

        self.assertEqual(receipt.result["cost_units"], 0.2)
        self.assertAlmostEqual(receipt.state["spent_budget"], 0.3)


if __name__ == "__main__":
    unittest.main()
