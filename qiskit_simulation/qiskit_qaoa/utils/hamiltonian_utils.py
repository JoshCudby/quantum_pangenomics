import numpy as np
import pickle
from qiskit_optimization import QuadraticProgram


def get_qp(data_file) -> QuadraticProgram:
    data = np.load(data_file, allow_pickle=True)
    Q, offset, _, _  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    return mod


def get_objective_and_hamiltonian(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    hamiltonian, ising_offset = mod.to_ising()
    hamiltonian = hamiltonian.sort(weight=True)
    return mod.objective, hamiltonian, ising_offset


def get_offset(data_file):
    data = np.load(data_file, allow_pickle=True)
    Q, offset, _, _  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation
    return offset


def get_ising_offset(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    _, ising_offset = mod.to_ising()
    return ising_offset