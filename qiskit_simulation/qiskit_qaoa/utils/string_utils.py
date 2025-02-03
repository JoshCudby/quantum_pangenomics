import numpy as np
from qiskit.quantum_info import SparsePauliOp


def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def bitstring_to_energy(bitstring: list, op: SparsePauliOp):
    pauli_string = ''.join(['X' if x == 1 else 'I' for x in bitstring])
    x_op = SparsePauliOp(pauli_string, np.array([1]))
    opt_energy_operator = x_op.adjoint() @ op @ x_op
    return np.sum(opt_energy_operator.coeffs)