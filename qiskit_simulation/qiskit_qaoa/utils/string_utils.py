import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .logging import get_logger

logger = get_logger(__name__)

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def bitstring_to_energy(bitstring: list, op: SparsePauliOp):
    assert all(x in [0, 1] for x in bitstring), "Bitstring should be binary integers"
    pauli_string = ''.join(['X' if x == 1 else 'I' for x in bitstring])
    x_op = SparsePauliOp(pauli_string, np.array([1]))
    opt_energy_operator = x_op.adjoint() @ op @ x_op
    return np.sum(opt_energy_operator.coeffs)


def print_optimal_solution_properties(
        optimal: list[int],
        op: SparsePauliOp,
        sample: dict,
        offset: float
):
    optimal.reverse()
    logger.info(f'Optimal bitstring: {optimal}')
    logger.info(f'Optimal cost: {bitstring_to_energy(optimal, op) + offset}')
    try:
        logger.info(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')
    except KeyError:
        logger.info('Did not sample optimal.')