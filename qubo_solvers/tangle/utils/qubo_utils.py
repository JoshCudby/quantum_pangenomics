import networkx as nx
import numpy as np
from itertools import product
from dimod.reference.samplers import SimulatedAnnealingSampler
from dimod import BQM
from utils.graph_utils import graph_to_max_path_digraph
from utils.sampling_utils import dwave_sample_to_path, dwave_sample_bqm


def get_max_path_problem_qubo_matrix(graph: nx.DiGraph, penalty: int) -> np.ndarray:
    """Generates a matrix describing the max path problem qubo cost function.
    The cost function is C(x) = x^T Q x, where Q is the matrix returned by this function.

    Args:
        graph (nx.DiGraph): the directed graph describing the max path problem.
        penalty (int): the penalty for breaking constraints.

    Returns:
        np.ndarray: a 2D array Q representing the cost function.
    """
    nodes = list(graph.nodes)
    end_node = nodes[-1]
    W = len(nodes) - 1
    
    qubo_matrix = np.zeros(shape=(W+1, W+1, W+1, W+1), dtype=int)
    # Reward travelling along real edges
    for t in range(W):
        for i, j in product(range(W+1), range(W+1)):
            if (nodes[i], nodes[j]) not in graph.edges:
                qubo_matrix[t, i, t+1, j] += penalty
            else:
                qubo_matrix[t, i, t+1, j] += -1 if (nodes[i] != end_node and nodes[j] != end_node) else 0
    
    # Penalise not being in exactly 1 location at each time
    for t in range(W+1):
        for i in range(W+1):
            qubo_matrix[t, i, t, i] -= penalty
            for j in range(i+1, W+1):
                qubo_matrix[t, i, t, j] += 2 * penalty
           
    # Penalise multiple visits to a real node     
    for i in range(W):
        for t1 in range(W+1):
            for t2 in range(t1+1, W+1):
                qubo_matrix[t1, i, t2, i] += penalty
    
    # Reward starting at the start
    for i in range(W):
        try:
            if graph.nodes[nodes[i]]["start"] == "start":
                qubo_matrix[0, i, 0, i] -= penalty
        except:
            pass
        
    # Reward ending at the end
    qubo_matrix[W, W, W, W] -= penalty
    
    qubo_matrix = qubo_matrix.reshape(((W+1)**2, (W+1)**2))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)
    
    return qubo_matrix


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
    
    qubo_matrix = get_max_path_problem_qubo_matrix(dg, penalty)
    bqm = BQM(qubo_matrix, 'BINARY')
    bqm.offset = penalty * (W + 3)
    print(f'Number of nodes: {W + 1}')
    print(f'Number of edges: {len(dg.edges)}')
    print(f'Number of QUBO vars: {len(bqm.variables)}')
    
    best_sample, best_energy = dwave_sample_bqm(sampler, bqm, time_limit=time_limit, label=f"Max Path QUBO W = {W}")
    
    best_path = dwave_sample_to_path(best_sample, dg)
    return best_sample, best_energy, best_path
