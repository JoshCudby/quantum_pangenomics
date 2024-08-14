import numpy as np
import networkx as nx
from itertools import product
from math import floor


def qubo_matrix_from_graph(graph: nx.DiGraph, alpha: float | None=None) -> tuple[np.ndarray, float, int, int]:
    """Constructs the QUBO matrix corresponding to a graph. Also returns the offset of the model, the max time and the number of nodes.

    Args:
        graph (nx.DiGraph): the node-weighted graph describing the problem.
        alpha (float, optional): the proportion of extra time allowed to paths over the maximum weight.

    Returns:
        tuple[np.ndarray, float, int, int]: qubo_matrix, offset, T_max, V
    """
    nodes = list(graph.nodes)
    V = int(len(nodes))
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    
    # T_max = total weight + "a bit"
    if alpha is None:
        alpha = 1.2
    T_max = floor(total_weight * alpha)

    # Penalty Values
    lambda_t = 4 * T_max
    lambda_g = T_max
    lambda_end = floor(1 * T_max)
    lambda_w = floor(1 * T_max)

    qubo_matrix = np.zeros((T_max, V + 1, T_max, V + 1))
    
    # Path
    for t in range(T_max):
        for i in range(V):
            qubo_matrix[t, i, t, i] -= lambda_t
            qubo_matrix[t, V, t, i] += 2 * lambda_t
        qubo_matrix[t, V, t, V] -= lambda_t
        
        for i, j in product(range(V), range(V)):
            if not (i == j):
                qubo_matrix[t, i, t, j] += lambda_t
        
    # Graph
    for t in range(T_max - 1):
        for i, j in product(range(V), range(V)):
            if (nodes[i], nodes[j]) in graph.edges:
                qubo_matrix[t, i, t+1, j] -= lambda_g

    # Staying in end
    for t in range(T_max - 1):
        for i in range(V):
            qubo_matrix[t, i, t + 1, V] -= (lambda_g - 1)
        qubo_matrix[t, V, t + 1, V] -= (lambda_g - 1)
        
    # Leaving end
    for t in range(T_max - 1):
        for i in range(V):
            qubo_matrix[t, V, t + 1, i] += lambda_end
                
    # Weights
    for i in range(0, V, 2):
        for t in range(T_max):
            qubo_matrix[t, i, t, i] -= (2 * graph.nodes[nodes[i]]["weight"] - 1) * lambda_w
            qubo_matrix[t, i + 1, t, i + 1] -= (2 * graph.nodes[nodes[i]]["weight"] - 1) * lambda_w
        
        for t1, t2 in product(range(T_max), range(T_max)):
            qubo_matrix[t1, i, t2, i + 1] += 2 * lambda_w
            if not (t1 == t2):
                qubo_matrix[t1, i, t2, i] += lambda_w
                qubo_matrix[t1, i + 1, t2, i + 1] += lambda_w

    qubo_matrix = qubo_matrix.reshape((T_max * (V+1)), (T_max * (V+1)))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)
    
    offset = -1 * (lambda_g * (1 - T_max) + (T_max - total_weight)) + (T_max * lambda_t + lambda_w * sum(graph.nodes[nodes[i]]["weight"] ** 2 for i in range(0, V, 2)))
    return qubo_matrix, offset, T_max, V