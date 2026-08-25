from __future__ import annotations

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from autovqe import probes
from autovqe.ansatz import OperationSpec, ParameterExpression
from autovqe.problem import InitialStateSpec, PauliTerm, PublicProblem


def _problem() -> PublicProblem:
    return PublicProblem.create(
        num_qubits=2,
        pauli_terms=(
            PauliTerm("ZZ", 1.0),
            PauliTerm("XX", 0.25),
            PauliTerm("YY", 0.25),
        ),
        initial_state=InitialStateSpec(
            kind="computational_basis", occupation=(1, 0)
        ),
    )


def _exchange() -> OperationSpec:
    return OperationSpec(
        macro="XYExchange",
        qubits=(0, 1),
        parameters={"angle": ParameterExpression.parameter("theta")},
    )


class LeanPublicProbeTests(unittest.TestCase):
    def test_normalized_commutator_is_the_only_public_probe(self) -> None:
        request = {
            "type": "normalized_commutator",
            "generator": {"type": "global_pauli_sum", "pauli": "Z"},
        }
        result = probes.run_public_probe(_problem(), request)

        self.assertEqual(result.probe_type, "normalized_commutator")
        self.assertTrue(result.metrics["exact"])
        self.assertEqual(result.cost_units, 0.25)

        with self.assertRaisesRegex(probes.ProbeValidationError, "unsupported probe"):
            probes.run_public_probe(
                _problem(),
                {"type": "initial_state_moments", "generator": request["generator"]},
            )

    def test_orbit_recipe_alias_is_not_supported(self) -> None:
        with self.assertRaisesRegex(probes.ProbeValidationError, "unsupported generator"):
            probes.generator_from_recipe(
                2, {"type": "orbit_pauli_sum", "seed": "Z"}
            )

    def test_observables_must_be_finite_and_hermitian(self) -> None:
        with self.assertRaisesRegex(probes.ProbeValidationError, "finite"):
            probes.validate_hamiltonian_observable(
                SparsePauliOp.from_list([("Z", float("nan"))])
            )
        with self.assertRaisesRegex(probes.ProbeValidationError, "Hermitian"):
            probes.validate_generator_observable(
                SparsePauliOp.from_list([("X", 1j)])
            )

    def test_probe_cost_tracks_sparse_term_products(self) -> None:
        def label(index: int) -> str:
            alphabet = "IXYZ"
            letters = []
            for _ in range(16):
                letters.append(alphabet[index % 4])
                index //= 4
            return "".join(letters)

        hamiltonian = SparsePauliOp.from_list(
            [(label(index + 1), 1.0) for index in range(1_245)]
        )
        request = {
            "type": "normalized_commutator",
            "generator": {"type": "global_pauli_sum", "pauli": "Z"},
        }
        self.assertEqual(probes.algebraic_probe_cost_units(hamiltonian, request), 1.25)


class LeanInternalEvidenceTests(unittest.TestCase):
    def test_initial_state_moments_are_scale_invariant(self) -> None:
        state = QuantumCircuit(1)
        unit = SparsePauliOp.from_list([("X", 1.0)])
        scaled = SparsePauliOp.from_list([("X", 1e-6)])

        self.assertAlmostEqual(probes.initial_state_moments(state, unit)[1], 1.0)
        self.assertAlmostEqual(
            probes.initial_state_moments(state, scaled)[1], 1.0
        )

    def test_exchange_requires_a_touching_nondiluted_charge(self) -> None:
        global_z = probes.generator_from_recipe(
            3, {"type": "global_pauli_sum", "pauli": "Z"}
        )
        values = probes.validate_special_operation_relevance(
            3,
            _exchange(),
            global_z,
            symmetry_residual=0.0,
            sector_variance=0.0,
        )
        self.assertLessEqual(values[2], probes.EXACT_SYMMETRY_TOLERANCE)

        with self.assertRaisesRegex(probes.ProbeValidationError, "no nontrivial charge"):
            probes.validate_special_operation_relevance(
                3,
                _exchange(),
                SparsePauliOp.from_list([("ZII", 1.0)]),
                symmetry_residual=0.0,
                sector_variance=0.0,
            )

    def test_energy_rejects_nonhermitian_hamiltonian(self) -> None:
        with self.assertRaisesRegex(probes.ProbeValidationError, "Hermitian"):
            probes.energy_from_circuit(
                QuantumCircuit(1), SparsePauliOp.from_list([("Z", 1j)])
            )

    def test_removed_dense_and_gradient_interfaces_stay_absent(self) -> None:
        self.assertFalse(hasattr(probes, "unitary_commutation_residual"))
        self.assertFalse(hasattr(probes, "gradient_snapshot"))
        self.assertFalse(hasattr(probes, "run_algebraic_probe"))


if __name__ == "__main__":
    unittest.main()
