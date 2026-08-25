from __future__ import annotations

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    ReferenceSpec,
    SectorSpec,
)
from autovqe.probes import (
    ProbeValidationError,
    reference_moments,
    run_algebraic_probe,
)


class PublicHamiltonianBoundaryTests(unittest.TestCase):
    def test_public_problem_rejects_complex_pauli_coefficients(self) -> None:
        with self.assertRaisesRegex(ValueError, "coefficients must be real"):
            PublicProblem.create(
                num_qubits=1,
                pauli_terms=(PauliTerm("Z", 1.0, 1e-12),),
                encoding=EncodingSpec(),
                sector=SectorSpec(),
                reference=ReferenceSpec(),
                backend=BackendSpec(),
            )


class GeneratorProbeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
        self.reference = QuantumCircuit(1)

    def reference_request(self, coefficient: object, *, pauli: str = "X") -> dict:
        return {
            "type": "reference_moments",
            "generator": {
                "type": "pauli_sum",
                "terms": [{"pauli": pauli, "coeff": coefficient}],
            },
        }

    def test_reference_probe_rejects_identity_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "identity-only"):
            run_algebraic_probe(
                self.hamiltonian,
                self.reference_request(1.0, pauli="I"),
                reference=self.reference,
            )

    def test_reference_probe_rejects_non_hermitian_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "Hermitian"):
            reference_moments(
                self.reference,
                SparsePauliOp.from_list([("X", 1j)]),
            )

    def test_reference_probe_rejects_non_finite_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "finite"):
            run_algebraic_probe(
                self.hamiltonian,
                self.reference_request(float("nan")),
                reference=self.reference,
            )

    def test_generator_recipe_rejects_string_coefficients(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "real JSON number"):
            run_algebraic_probe(
                self.hamiltonian,
                self.reference_request("1.0"),
                reference=self.reference,
            )

    def test_reference_probe_rejects_tiny_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "minimum active norm"):
            run_algebraic_probe(
                self.hamiltonian,
                self.reference_request(1e-12),
                reference=self.reference,
            )

    def test_reference_variance_is_invariant_to_allowed_overall_scale(self) -> None:
        unit_generator = SparsePauliOp.from_list([("X", 1.0)])
        scaled_generator = SparsePauliOp.from_list([("X", 1e-6)])

        _, unit_variance = reference_moments(self.reference, unit_generator)
        _, scaled_variance = reference_moments(self.reference, scaled_generator)

        self.assertAlmostEqual(unit_variance, 1.0)
        self.assertAlmostEqual(scaled_variance, unit_variance)


if __name__ == "__main__":
    unittest.main()
