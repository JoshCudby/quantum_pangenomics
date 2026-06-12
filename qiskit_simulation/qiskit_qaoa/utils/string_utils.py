"""Bitstring evaluation and solution scoring utilities.

Provides fast vectorised functions for computing the expectation value of a
``SparsePauliOp`` observable over a set of measured bitstrings, converting a
binary solution vector to its QUBO/Ising energy, and logging properties of the
classically-optimal solution.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .logging import get_logger

logger = get_logger(__name__)

_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.int8)


def evaluate_sparse_pauli_samples_all(observable: SparsePauliOp) -> np.typing.NDArray:
    """Utility for the evaluation of the expectation value of a measured state."""
    coeffs = np.array(observable.coeffs, dtype=np.float16)
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    
    packed = np.arange(2**observable.num_qubits, dtype=np.uint64)[:, None].view(np.uint8)[:, :packed_uint8.shape[1]]
    bytes = np.bitwise_and(packed[:, None, :], packed_uint8[None, :, :])
    reduced = np.bitwise_xor.reduce(bytes, axis=-1)
    return np.sum(coeffs * _PARITY[reduced], axis=-1)


def evaluate_sparse_pauli_samples(samples: list[str], observable: SparsePauliOp) -> np.typing.NDArray:
    """Compute the energy of each bitstring sample under a SparsePauliOp observable.

    Uses a packed-uint8 / XOR-parity trick for efficiency: each bitstring is
    converted to a byte array, ANDed with the packed Pauli Z mask, and reduced
    by XOR parity to give a ±1 eigenvalue per term.

    Args:
        samples: A list of binary strings (e.g. ``['0101', '1100']``).  Each
            string represents a computational-basis measurement outcome, with
            the leftmost character corresponding to the highest-index qubit
            (Qiskit convention).
        observable: The ``SparsePauliOp`` whose expectation value is
            evaluated for each sample.

    Returns:
        A 1-D real numpy array of length ``len(samples)`` containing the
        energy ``<s|H|s>`` for each sample ``s``.
    """
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
    """Convert an integer to a big-endian binary list of length ``num_bits``.

    Args:
        integer: A non-negative integer to convert.
        num_bits: Total number of bits; the result is zero-padded to this length.

    Returns:
        A list of integers in ``{0, 1}`` with the most-significant bit first.
    """
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def bin_rep(k, n):
    """Convert an integer to a little-endian binary list of length ``n``.

    Args:
        k: A non-negative integer to convert.
        n: Total number of bits.

    Returns:
        A list of integers in ``{0, 1}`` with the least-significant bit first.
    """
    return [int(x) for x in np.binary_repr(k, n)[::-1]]


def bitstring_to_energy(bitstring: list, op: SparsePauliOp):
    """Compute the Ising energy of a binary solution vector.

    Constructs the corresponding X-Pauli operator and uses the sandwich
    formula ``<x|H|x>`` to extract the energy without explicit matrix
    exponentiation.

    Args:
        bitstring: A list of integers in ``{0, 1}``, ordered from qubit 0
            (index 0) upward.
        op: The ``SparsePauliOp`` Ising Hamiltonian.

    Returns:
        The real energy (float) of the configuration ``bitstring`` under
        ``op``.

    Raises:
        AssertionError: If any element of ``bitstring`` is not in ``{0, 1}``.
    """
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
    """Log the bitstring, cost, and sampling probability of the optimal solution.

    Reverses the bitstring in-place (converting from QUBO variable ordering to
    the Qiskit qubit ordering used by ``op``), evaluates its energy, and looks
    up its empirical probability in the provided sample distribution.

    Args:
        optimal: The optimal binary solution as a list of integers in
            ``{0, 1}``, ordered from the lowest QUBO variable index.  This
            list is modified in-place by reversal.
        op: The normalised ``SparsePauliOp`` Ising Hamiltonian (without the
            constant offset).
        sample: A dict mapping bitstring keys to empirical probabilities,
            as returned by ``sample_optimized_circuit``.
        offset: The constant QUBO energy offset to add back when reporting
            the true objective value.
    """
    optimal.reverse()
    logger.info(f'Optimal bitstring: {optimal}')
    logger.info(f'Optimal cost: {bitstring_to_energy(optimal, op) + offset}')
    try:
        logger.info(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')
    except KeyError:
        logger.info('Did not sample optimal.')