import unittest
from unittest.mock import patch

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from autovqe import probes
from autovqe.ansatz_ir import OperationSpec, ParameterExpression


class AlgebraicProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.heisenberg = SparsePauliOp.from_list(
            [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]
        )

    def test_heisenberg_commutes_with_global_spin_generators(self) -> None:
        for pauli in ("X", "Y", "Z"):
            generator = probes.generator_from_recipe(
                2, {"type": "global_pauli_sum", "pauli": pauli}
            )
            self.assertLess(probes.normalized_commutator(self.heisenberg, generator), 1e-12)

    def test_xxz_near_miss_keeps_u1_but_breaks_su2(self) -> None:
        xxz = SparsePauliOp.from_list(
            [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.2)]
        )
        global_z = probes.generator_from_recipe(
            2, {"type": "global_pauli_sum", "pauli": "Z"}
        )
        global_x = probes.generator_from_recipe(
            2, {"type": "global_pauli_sum", "pauli": "X"}
        )
        self.assertLess(probes.normalized_commutator(xxz, global_z), 1e-12)
        self.assertGreater(probes.normalized_commutator(xxz, global_x), 1e-3)

    def test_probe_computes_the_commutator_once_after_preflight(self) -> None:
        request = {
            "type": "normalized_commutator",
            "generator": {
                "type": "global_pauli_sum",
                "pauli": "Z",
            },
        }
        original = probes.normalized_commutator
        with patch(
            "autovqe.probes.normalized_commutator", wraps=original
        ) as measured:
            result = probes.run_algebraic_probe(self.heisenberg, request)

        self.assertTrue(result.valid)
        self.assertEqual(measured.call_count, 1)

    def test_hamiltonian_itself_is_rejected_as_vacuous_symmetry(self) -> None:
        with self.assertRaisesRegex(probes.ProbeValidationError, "trivial copy"):
            probes.validate_symmetry_generator(self.heisenberg, self.heisenberg)

    def test_initial_state_moments_identify_definite_charge(self) -> None:
        initial_state = QuantumCircuit(2)
        initial_state.x(0)
        global_z = probes.generator_from_recipe(
            2, {"type": "global_pauli_sum", "pauli": "Z"}
        )
        mean, variance = probes.initial_state_moments(initial_state, global_z)
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(variance, 0.0)

    def test_macro_commutation_is_checked_at_nonzero_binding(self) -> None:
        theta = Parameter("theta")
        conserving = QuantumCircuit(2)
        conserving.append(XXPlusYYGate(theta), [0, 1])
        global_z = probes.generator_from_recipe(
            2, {"type": "global_pauli_sum", "pauli": "Z"}
        )
        residual = probes.unitary_commutation_residual(
            conserving, global_z, parameter_values={theta: 0.37}
        )
        self.assertLess(residual, 1e-12)

        breaking = QuantumCircuit(2)
        breaking.ry(0.37, 0)
        self.assertGreater(
            probes.unitary_commutation_residual(breaking, global_z), 1e-3
        )

    def test_global_charge_is_relevant_to_local_exchange(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        global_z = probes.generator_from_recipe(
            3, {"type": "global_pauli_sum", "pauli": "Z"}
        )

        touching_norm, fraction, residual, conditioned, conditioned_variance = (
            probes.validate_special_operation_relevance(
                3,
                operation,
                global_z,
                symmetry_residual=0.0,
                sector_variance=0.0,
            )
        )

        self.assertGreater(touching_norm, probes.EXACT_SYMMETRY_TOLERANCE)
        self.assertGreater(fraction, probes.MIN_SPECIAL_CHARGE_FRACTION)
        self.assertLessEqual(residual, probes.EXACT_SYMMETRY_TOLERANCE)
        self.assertEqual(conditioned, 0.0)
        self.assertEqual(conditioned_variance, 0.0)

    def test_disjoint_spectator_charge_is_not_relevant_to_exchange(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        spectator_z = SparsePauliOp.from_list([("ZII", 1.0)])

        with self.assertRaisesRegex(
            probes.ProbeValidationError, "no nontrivial charge"
        ):
            probes.validate_special_operation_relevance(
                3,
                operation,
                spectator_z,
                symmetry_residual=0.0,
                sector_variance=0.0,
            )

    def test_tiny_touching_term_cannot_unlock_a_special_gate(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        hamiltonian = SparsePauliOp.from_list(
            [("IZZ", 1.0), ("IZI", 1e-3)]
        )
        touching_only = SparsePauliOp.from_list([("IXX", 1.0)])
        diluted = SparsePauliOp.from_list(
            [("ZII", 1.0), ("IXX", 5e-8)]
        )

        self.assertGreater(
            probes.normalized_commutator(hamiltonian, touching_only),
            probes.EXACT_SYMMETRY_TOLERANCE,
        )
        self.assertLessEqual(
            probes.normalized_commutator(hamiltonian, diluted),
            probes.EXACT_SYMMETRY_TOLERANCE,
        )
        with self.assertRaisesRegex(
            probes.ProbeValidationError, "relative to the full charge"
        ):
            probes.validate_special_operation_relevance(
                3,
                operation,
                diluted,
                symmetry_residual=probes.normalized_commutator(
                    hamiltonian, diluted
                ),
                sector_variance=0.0,
            )

    def test_conditioned_residual_blocks_spectator_dilution_above_fraction_floor(self) -> None:
        operation = OperationSpec(
            macro="XYExchange",
            qubits=(0, 1),
            parameters={"angle": ParameterExpression.parameter("theta")},
        )
        hamiltonian = SparsePauliOp.from_list(
            [("IZZ", 1.0), ("IIX", 1e-8)]
        )
        relevant = SparsePauliOp.from_list(
            [("IIZ", 1.0), ("IZI", 1.0)]
        )
        diluted = SparsePauliOp.from_list(
            [("ZII", 1.0), ("IIZ", 1e-3), ("IZI", 1e-3)]
        )
        relevant_residual = probes.normalized_commutator(hamiltonian, relevant)
        diluted_residual = probes.normalized_commutator(hamiltonian, diluted)
        self.assertGreater(relevant_residual, probes.EXACT_SYMMETRY_TOLERANCE)
        self.assertLessEqual(diluted_residual, probes.EXACT_SYMMETRY_TOLERANCE)

        with self.assertRaisesRegex(
            probes.ProbeValidationError, "too weak on the special operation support"
        ):
            probes.validate_special_operation_relevance(
                3,
                operation,
                diluted,
                symmetry_residual=diluted_residual,
                sector_variance=0.0,
            )

    def test_gradient_snapshot_uses_evaluator_side_parameters(self) -> None:
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)
        gradients, calls = probes.gradient_snapshot(
            circuit,
            SparsePauliOp.from_list([("Z", 1.0)]),
            values=[np.pi / 2],
        )
        self.assertEqual(calls, 2)
        self.assertAlmostEqual(gradients[0], -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
