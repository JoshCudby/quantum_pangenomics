import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .logging import get_logger

logger = get_logger(__name__)

_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.int8)


def evaluate_sparse_pauli_samples(samples: list[str], observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")

    bytes = np.array([packed_uint8 & np.frombuffer(int(x, 2).to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8) for x in samples])
    reduced = np.bitwise_xor.reduce(bytes, axis=-1)
    return np.real_if_close(np.sum(observable.coeffs * _PARITY[reduced], axis=-1))


def evaluate_sparse_pauli(sample: str, observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    sample_int = int(sample, 2)
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(sample_int.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.real_if_close(np.sum(observable.coeffs * _PARITY[reduced]))


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