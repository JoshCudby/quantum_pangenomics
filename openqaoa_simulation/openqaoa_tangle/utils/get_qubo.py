import numpy as np
from openqaoa import QUBO


def get_qubo(filename) -> QUBO:
    data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.npy'

    data = np.load(data_file, allow_pickle=True)
    Q, offset, _, _  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation


    terms = []
    weights = []

    for i in range(Q.shape[0]):
        for j in range(i, Q.shape[0]):
            if not Q[i, j] == 0:
                terms.append([i, j])
                weights.append(Q[i, j])
                
                
    terms.append([])
    weights.append(offset)
    ising_terms, ising_weights = QUBO.convert_qubo_to_ising(Q.shape[0], terms, weights)
    ising_qubo = QUBO(Q.shape[0], ising_terms, ising_weights)
    return ising_qubo