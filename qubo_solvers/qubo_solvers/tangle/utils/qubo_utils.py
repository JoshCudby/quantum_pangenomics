import networkx as nx
import numpy as np
from itertools import product
from math import floor
from dimod.reference.samplers import SimulatedAnnealingSampler
from dimod import BQM
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_to_path, dwave_sample_bqm


def get_tangle_qubo_matrix(graph: nx.DiGraph) -> np.ndarray:
    """Generates a matrix describing the max path problem qubo cost function.
    The cost function is C(x) = x^T Q x, where Q is the matrix returned by this function.

    Args:
        graph (nx.DiGraph): the directed graph describing the max path problem.
        penalty (int): the penalty for breaking constraints.

    Returns:
        np.ndarray: a 2D array Q representing the cost function.
    """
    nodes = list(graph.nodes)
    W = len(nodes)
    alpha = 1.2
    total_weight = int(sum(dict(graph.nodes.data('weight')).values()))
    T = floor(total_weight * alpha)
    
    offset = 0
    
    lambda_t = 10
    lambda_g = 10
    lambda_start = 5
    lambda_w = 1
    
    qubo_matrix = np.zeros(shape=(T, W+1, T, W+1), dtype=int)
    
    # Walk constraint
    T_qubo_matrix = lambda_t * (np.ones((W+1, W+1), dtype=np.int16) - 2 * np.diagflat(np.ones((W+1), dtype=np.int16)))
    for t in range(T):
        qubo_matrix[t, :, t, :] += T_qubo_matrix
    offset += T * lambda_t
         
    # Graph step constraint
    G_qubo_matrix = lambda_g * np.ones((W+1, W+1), dtype=np.int16)
    G_qubo_matrix[:, W] = 0
    for i, j in graph.edges:
        G_qubo_matrix[nodes.index(i), nodes.index(j)] = 0
        G_qubo_matrix[nodes.index(j), nodes.index(i)] = 0
    for t in range(T - 1):
        qubo_matrix[t, :, t+1, :] = G_qubo_matrix
    
    # Weight constraint
    def W_qubo_matrix(weight):
        return lambda_w * (
                np.ones((T, T), dtype=np.int16) - (2 * weight) * np.diagflat(np.ones((T), dtype=np.int16))
            )
    for i in range(W):
        qubo_matrix[:, i, :, i] += W_qubo_matrix(graph.nodes[nodes[i]]["weight"])
    offset += lambda_w * sum(graph.nodes[nodes[i]]["weight"] ** 2 for i in range(W))
        
    # Set start/end nodes
    start_nodes=set()
    end_nodes= set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'start':
            print(f'Found start node:{node}')
            start_nodes.add(nodes.index(node))
        if val == 'end':
            print(f'Found end node:{node}')
            end_nodes.add(nodes.index(node))
    
    start_nodes = list(start_nodes)        
    end_nodes = list(end_nodes)
    print(f'Start nodes: {start_nodes}, End nodes: {end_nodes}')
    exist_start_nodes = len(start_nodes) > 0
    exist_end_nodes = len(end_nodes) > 0
    
    if exist_start_nodes:
        S_qubo_matrix = lambda_start * (
            np.ones((len(start_nodes), len(start_nodes)), dtype=np.int16) 
            - 2 * np.diagflat(np.ones((len(start_nodes)), dtype=np.int16))
            )
        for i, j in product(start_nodes, start_nodes):
            qubo_matrix[0, i, 0, j] += S_qubo_matrix[i, j]
        offset += lambda_start
    
    if exist_end_nodes:
        E_qubo_matrix = np.zeros((W+1, W+1), dtype=np.int16)
        E_qubo_matrix[:, W] = lambda_start * np.array([i not in end_nodes for i in range(W)] + [0], dtype=np.int16)
        for t in range(T-1):
            qubo_matrix[t, :, t+1, :] += E_qubo_matrix
    
    qubo_matrix = qubo_matrix.reshape((T * (W+1), T * (W+1)))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)
    
    return qubo_matrix, offset, T, W


def dwave_sample_max_path_problem(graph: nx.Graph, sampler=None, time_limit=None, penalty=None):
    """Solves the max path problem on a node-weighted graph.

    Args:
        graph (nx.Graph): The underlying graph.
        sampler (Sampler, optional): The sampler to use. Defaults to SimulatedAnnealingSampler.
        time_limit (int, optional): The time limit passed to the sampler.
        penalty (int, optional): The penalty for breaking constraints. Defaults to total weight of graph.
    """
    if sampler is None:
        sampler = SimulatedAnnealingSampler()
    
    dg = graph_to_max_path_digraph(graph)
    W = len(dg.nodes) - 1
    
    if penalty is None:
        penalty = W
    
    qubo_matrix = get_tangle_qubo_matrix(dg, penalty)
    bqm = BQM(qubo_matrix, 'BINARY')
    bqm.offset = penalty * (W + 3)
    print(f'Number of nodes: {W + 1}')
    print(f'Number of edges: {len(dg.edges)}')
    print(f'Number of QUBO vars: {len(bqm.variables)}')
    
    best_sample, best_energy = dwave_sample_bqm(sampler, bqm, time_limit=time_limit, label=f"Max Path QUBO W = {W}")
    
    best_path = dwave_sample_to_path(best_sample, dg)
    return best_sample, best_energy, best_path
