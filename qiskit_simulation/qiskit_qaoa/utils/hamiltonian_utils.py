import numpy as np
from qiskit_optimization import QuadraticProgram


def get_objective_and_hamiltonian(data_file):
    data = np.load(data_file, allow_pickle=True)
    Q, offset, T, N  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation


    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    hamiltonian, offset = mod.to_ising()
    hamiltonian = hamiltonian.sort(weight=True)
    return mod.objective, hamiltonian