from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from autovqe.ansatz_ir import (
    AnsatzSpec,
    OperationSpec,
    ParameterExpression,
)
from autovqe.compiler import compile_ansatz


def _exp_hermitian(angle: float, generator: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    phases = np.exp(-1.0j * angle * eigenvalues)
    return (eigenvectors * phases) @ eigenvectors.conj().T


def _compiled_unitary(operation: OperationSpec, angle: float) -> np.ndarray:
    compiled = compile_ansatz(
        AnsatzSpec(
            num_qubits=2,
            parameters=("theta",),
            operations=(operation,),
        )
    )
    bound = compiled.circuit.assign_parameters(
        {compiled.parameters["theta"]: angle}, inplace=False
    )
    return Operator(bound).data


class MacroAngleConventionTests(unittest.TestCase):
    def assert_macro_generator(
        self,
        operation: OperationSpec,
        generator: SparsePauliOp,
        *,
        angle: float = 0.371,
    ) -> None:
        actual = _compiled_unitary(operation, angle)
        expected = _exp_hermitian(angle, generator.to_matrix())
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_pauli_rotation_means_exp_minus_i_angle_p(self) -> None:
        self.assert_macro_generator(
            OperationSpec(
                macro="PauliRotation",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
                options={"pauli": "XY"},
            ),
            # The local word is support-ordered: X on q0 and Y on q1.  In a
            # full Qiskit label the most-significant q1 character comes first.
            SparsePauliOp.from_list([("YX", 1.0)]),
        )

    def test_xy_exchange_means_exp_minus_i_angle_xx_plus_yy(self) -> None:
        self.assert_macro_generator(
            OperationSpec(
                macro="XYExchange",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
            ),
            SparsePauliOp.from_list([("XX", 1.0), ("YY", 1.0)]),
        )

    def test_isotropic_exchange_means_exp_minus_i_angle_heisenberg(self) -> None:
        self.assert_macro_generator(
            OperationSpec(
                macro="IsotropicExchange",
                qubits=(0, 1),
                parameters={"angle": ParameterExpression.parameter("theta")},
            ),
            SparsePauliOp.from_list(
                [("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)]
            ),
        )


if __name__ == "__main__":
    unittest.main()
