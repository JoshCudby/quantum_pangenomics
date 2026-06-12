from pytket import Circuit
from sympy import Symbol
from .logging import get_logger

logger = get_logger(__name__)


def Q_to_Ising(Q, offset):
    n_qubits = Q.shape[0]
    J = {}
    h = {i : 0 for i in range(n_qubits)}
    # xi binary, zi spin
    # recall zi^2 = 1 whereas xi^2 = xi
    for i in range(n_qubits):
        # Q[i, i]xi^2 = Q[i, i](1 - 2zi + zi^2)/4 -> h[i] = - Q[i, i]/ 2, O += Q[i, i] / 2
        h[n_qubits - i - 1] -= Q[i, i] / 2
        offset += Q[i, i] / 2
        # Calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Q[i, j]xi xj = Q[i, j] (1 - zi - zj + zi zj)/4 -> J[i, j] = Q[i, j] / 4, h[i], h[j] -= Q[i, j]/4, O+= Q[i, j] / 4
            J[(n_qubits - i - 1, n_qubits - j - 1)] = Q[i, j] / 4
            h[n_qubits - i - 1] -= Q[i, j] / 4
            h[n_qubits - j - 1] -= Q[i, j] / 4
            offset += Q[i, j] / 4
    return h, J, offset


def qaoa_circuit(
        n_qubits: int, 
        reps: int,
        terms: list
):
    # test_circuit = Circuit(1)
    # test_circuit.Rz(0.5, 0)
    # logger.info(test_circuit.get_unitary())
    # 2**-0.5 [[1-i, 0],[0,1+i]] = [[e^(-i pi/4), 0], [0, e^(i pi/4)]]
    # test_circuit = Circuit(2)
    # test_circuit.CX(0, 1)
    # test_circuit.Rz(0.5, 1)
    # test_circuit.CX(0, 1)
    # diag(e^(-i pi/4), e^(i pi/4), e^(i pi/4), e^(-i pi/4))
    # logger.info(test_circuit.get_unitary())

    circuit = Circuit(n_qubits)
    p_keys = []

    # Initial State
    for i in range(n_qubits):
        circuit.H(i)
    for d in range(reps):
        # Hamiltonian unitary
        gamma_d = Symbol(f"γ_{d}")
        for index in range(len(terms)):
            qubits = terms[index][0]
            coef = terms[index][1]
            if len(qubits) == 2:
                circuit.CX(qubits[0], qubits[1])
                circuit.Rz(gamma_d * coef, qubits[1])
                circuit.CX(qubits[0], qubits[1])
            elif len(qubits) == 1:
                circuit.Rz(-gamma_d * coef, qubits[0])
            else:
                raise Exception('Terms should act on 1 or 2 qubits')
        p_keys.append(gamma_d)

        # Mixing unitary
        beta_d = Symbol(f"β_{d}")
        for i in range(n_qubits):
            circuit.Rx(beta_d, i)
        p_keys.append(beta_d)
    
    return circuit, p_keys
