import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def uniform_over_range(num_qubits: int, M: int):
    """
    Returns a circuit that prepares a uniform superposition over |0>,|1>,...,|M-1> on num_qubits qubits.
    Uses a Hadamard layer if M is a power of 2, else uses the method of Shukla and Vedula.
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