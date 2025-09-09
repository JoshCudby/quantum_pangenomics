import numpy as np
import networkx as nx
import re
import pickle
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram


rng = np.random.default_rng(seed=1)


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


def monomial_to_pauli(monomial, size):
    indices = [int(re.search(r'[0-9]+', atom.name).group(0)) for atom in monomial.atoms()]
    pauli_str = ['I'] * size
    for i in indices:
        pauli_str[i] = 'Z'
    return ''.join(pauli_str)


def indices_to_pauli(t: int, k: int, n: int, T: int):
    p = ['I'] * n * T
    p[t*n + k] = 'Z'
    return SparsePauliOp(''.join(p), np.array([1]))


def hamiltonian_to_doubles_graph(hamiltonian: SparsePauliOp) -> nx.Graph:
    edges = []
    weights = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) == 2:
            edge = np.nonzero(t.paulis[0].z)[0]
            edges.append(edge)
            weights.append(t.coeffs[0])
            
    program_graph = nx.Graph()
    for i in range(hamiltonian.num_qubits):
        program_graph.add_node(i)
    for idx in range(len(weights)):
        program_graph.add_edge(edges[idx][0],edges[idx][1],weight=weights[idx])
    return program_graph


def hamiltonian_to_interactions(hamiltonian: SparsePauliOp, fraction_4=0.0, fraction_6=0.8) -> list[tuple]:
    interactions: list[tuple] = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) < 2 or np.sum(t.paulis[0].z) > 6:
            pass
        elif np.sum(t.paulis[0].z) <= 4 and rng.random() > fraction_4:
            edge = tuple(np.nonzero(t.paulis[0].z)[0])
            interactions.append(edge)
        elif rng.random() > fraction_6:
            edge = tuple(np.nonzero(t.paulis[0].z)[0])
            interactions.append(edge)
    return interactions