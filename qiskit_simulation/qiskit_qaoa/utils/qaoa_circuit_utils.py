"""Mixer operators and initial state preparation for constrained QAOA.

Implements the Shukla–Vedula state preparation method for a uniform superposition
over ``{0, …, M-1}`` on ``ceil(log2(M+1))`` qubits, the walk-state
initialiser ``state_prep``, and the corresponding constrained mixer operator
``get_mixer_operator`` that preserves the Hamming-weight subspace required by
the pangenome path QUBO.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def uniform_over_range(num_qubits: int, M: int):
    """Prepare a uniform superposition over ``|0>, |1>, ..., |M-1>`` on ``num_qubits`` qubits.

    If ``M`` is a power of 2, a layer of Hadamard gates on the lowest
    ``log2(M)`` qubits suffices.  Otherwise the Shukla–Vedula decomposition
    is used: the binary representation of ``M`` drives a sequence of
    controlled-Ry and controlled-Hadamard gates that partition the
    probability amplitude correctly.

    Args:
        num_qubits: Number of qubits in the register.  Must satisfy
            ``M <= 2**num_qubits``.
        M: The number of basis states over which to prepare a uniform
            distribution.  Must be a positive integer.

    Returns:
        A ``QuantumCircuit`` on ``num_qubits`` qubits that prepares the
        state ``(1/sqrt(M)) * sum_{k=0}^{M-1} |k>``.

    Raises:
        Exception: If ``M`` is out of the range ``[0, 2**num_qubits]``.
    """
    if M not in range(2 ** num_qubits +1):
        print(M)
        print(num_qubits)
        raise Exception('Bad M: out of range')
    for i in range(num_qubits+1):
        if M == 2 ** i:
            print(f'M={M} a power of 2. Use Hadamard circuit.')
            circuit = QuantumCircuit(num_qubits)
            for j in range(i):
                circuit.h(j)
                
            return circuit
    
    circuit = QuantumCircuit(num_qubits)

    try:
        M_binary = np.binary_repr(M, num_qubits)
    except Exception as e:
        print(M)
        print(num_qubits)
        raise e
    M_binary = M_binary[::-1]
    ran = np.arange(len(M_binary))
    mask = [M_binary[x] == '1' for x in range(len(M_binary))]
    l = ran[mask]
    
    for i in range(1, len(l)):
        circuit.x(l[i])
    if l[0] > 0:
        for i in range(l[0]):
            circuit.h(i)

    MM = 2 ** l[0]

    circuit.ry(-2 * np.arccos(np.sqrt(MM/M)), l[1])

    for i in range(l[0], l[1]):
        circuit.ch(l[1], i, ctrl_state=0)

    for m in range(1, len(l)-1):
        circuit.cry(
            -2 * np.arccos(np.sqrt(2 ** l[m] / (M - MM) )), 
            l[m], l[m+1], ctrl_state=0
        )
        for i in range(l[m], l[m+1]):
            circuit.ch(l[m+1], i, ctrl_state=0)
        MM += 2 ** l[m]

    return circuit


def state_prep(N: int, T: int) -> QuantumCircuit:
    """Prepare the initial state for constrained QAOA over ``T`` time steps with ``N`` nodes.

    Builds a circuit of ``T`` registers, each of ``ceil(log2(2*N+1))`` qubits,
    and applies ``uniform_over_range(n, 2*N+1)`` to each register independently.
    This encodes the fact that at each time step a node index in ``{0, …, 2N}``
    can be visited (using both orientations of ``N`` nodes plus a sentinel).

    Args:
        N: Number of graph nodes (segment orientations per strand).
        T: Walk length (number of QAOA time steps / registers).

    Returns:
        A ``QuantumCircuit`` on ``ceil(log2(2*N+1)) * T`` qubits.
    """
    n = int(np.ceil(np.log2(2*N+1)))
    uni = uniform_over_range(n, 2*N+1)
    circuit = QuantumCircuit(n * T)
    for t in range(T):
        circuit.append(
            uni,
            list(range(t * n, (t+1) * n))   
        )
    return circuit


def get_mixer_operator(N: int, T: int, parameter=Parameter('beta')) -> QuantumCircuit:
    """Construct the constrained QAOA mixer that preserves the walk-state subspace.

    Implements the Grover-style mixer: uncompute the initial state preparation,
    apply a multi-controlled phase to the all-zero ancilla state, then reapply
    the state preparation.  This ensures that the mixer only mixes within the
    feasible subspace of valid walk states (no amplitude leaks into invalid
    configurations).

    Args:
        N: Number of graph nodes.
        T: Walk length (number of registers).
        parameter: The ``Parameter`` object representing the mixer angle
            ``beta``.  Defaults to a parameter named ``'beta'``.

    Returns:
        A parameterised ``QuantumCircuit`` on ``ceil(log2(2*N+1)) * T`` qubits
        implementing the constrained mixer layer ``U_B(beta)``.
    """
    # TODO: use ancillas to reduce depth of mcp?
    num_qubits = int(np.ceil(np.log2(2*N+1))) * T
    state_prep_circuit = state_prep(N, T)
    mixer = QuantumCircuit(num_qubits)
    mixer.append(
        state_prep_circuit.inverse(),
        range(num_qubits)
    )
    # mixer.save_statevector('after_prep')
    mixer.x(-1)
    mixer.mcp(-parameter, list(range(num_qubits - 1)), -1, ctrl_state=0)
    mixer.x(-1)
    # mixer.save_statevector('after_phase')
    mixer.append(
        state_prep_circuit,
        range(num_qubits)
    )
    # mixer.save_statevector('after_unprep')
    return mixer