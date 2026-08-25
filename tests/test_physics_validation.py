from __future__ import annotations

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from autovqe.contracts import (
    BackendSpec,
    EncodingSpec,
    PauliTerm,
    PublicProblem,
    InitialStateSpec,
    SectorSpec,
)
from autovqe.probes import (
    ProbeValidationError,
    initial_state_moments,
    run_algebraic_probe,
)


class PublicHamiltonianValidationTests(unittest.TestCase):
    def test_public_problem_rejects_complex_pauli_coefficients(self) -> None:
        with self.assertRaisesRegex(ValueError, "coefficients must be real"):
            PublicProblem.create(
                num_qubits=1,
                pauli_terms=(PauliTerm("Z", 1.0, 1e-12),),
                encoding=EncodingSpec(),
                sector=SectorSpec(),
                initial_state=InitialStateSpec(),
                backend=BackendSpec(),
            )


class GeneratorProbeValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
        self.initial_state = QuantumCircuit(1)

    def initial_state_request(self, coefficient: object, *, pauli: str = "X") -> dict:
        return {
            "type": "initial_state_moments",
            "generator": {
                "type": "pauli_sum",
                "terms": [{"pauli": pauli, "coeff": coefficient}],
            },
        }

    def test_initial_state_probe_rejects_identity_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "identity-only"):
            run_algebraic_probe(
                self.hamiltonian,
                self.initial_state_request(1.0, pauli="I"),
                initial_state=self.initial_state,
            )

    def test_initial_state_probe_rejects_non_hermitian_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "Hermitian"):
            initial_state_moments(
                self.initial_state,
                SparsePauliOp.from_list([("X", 1j)]),
            )

    def test_initial_state_probe_rejects_non_finite_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "finite"):
            run_algebraic_probe(
                self.hamiltonian,
                self.initial_state_request(float("nan")),
                initial_state=self.initial_state,
            )

    def test_generator_recipe_rejects_string_coefficients(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "real JSON number"):
            run_algebraic_probe(
                self.hamiltonian,
                self.initial_state_request("1.0"),
                initial_state=self.initial_state,
            )

    def test_initial_state_probe_rejects_tiny_generator(self) -> None:
        with self.assertRaisesRegex(ProbeValidationError, "minimum active norm"):
            run_algebraic_probe(
                self.hamiltonian,
                self.initial_state_request(1e-12),
                initial_state=self.initial_state,
            )

    def test_initial_state_variance_is_scale_invariant(self) -> None:
        unit_generator = SparsePauliOp.from_list([("X", 1.0)])
        scaled_generator = SparsePauliOp.from_list([("X", 1e-6)])

        _, unit_variance = initial_state_moments(self.initial_state, unit_generator)
        _, scaled_variance = initial_state_moments(self.initial_state, scaled_generator)

        self.assertAlmostEqual(unit_variance, 1.0)
        self.assertAlmostEqual(scaled_variance, unit_variance)


if __name__ == "__main__":
    unittest.main()
