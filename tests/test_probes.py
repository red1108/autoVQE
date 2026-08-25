import unittest

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from autovqe import probes


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

    def test_hamiltonian_itself_is_rejected_as_vacuous_symmetry(self) -> None:
        with self.assertRaisesRegex(probes.ProbeValidationError, "trivial copy"):
            probes.validate_symmetry_generator(self.heisenberg, self.heisenberg)

    def test_reference_moments_identify_definite_charge(self) -> None:
        reference = QuantumCircuit(2)
        reference.x(0)
        global_z = probes.generator_from_recipe(
            2, {"type": "global_pauli_sum", "pauli": "Z"}
        )
        mean, variance = probes.reference_moments(reference, global_z)
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
