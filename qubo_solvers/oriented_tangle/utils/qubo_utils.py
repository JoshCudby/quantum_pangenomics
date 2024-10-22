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
    V = int(len(nodes) / 2)
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    
    # T_max = total weight + "a bit"
    if alpha is None:
        alpha = 1.2
    T_max = floor(total_weight * alpha)

    # Penalty Values
    lambda_t = 10
    lambda_g = 10
    lambda_w = 1

    # Note: we add an end node with parity 0 and 1, we only want 1 of them. We will delete the other at the end.
    qubo_matrix = np.zeros((T_max, V + 1, 2, T_max, V + 1, 2), dtype=np.int8)
    
    # Path constraint
    for t in range(T_max):
        for i, b in product(range(V), range(2)):
            qubo_matrix[t, i, b, t, i, b] -= lambda_t
            qubo_matrix[t, V, 0, t, i, b] += 2 * lambda_t
                
        qubo_matrix[t, V, 0, t, V, 0] -= lambda_t
        
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if not (i == j and bi == bj):
                qubo_matrix[t, i, bi, t, j, bj] += lambda_t
    
    # Graph step constraints
    for t in range(T_max - 1):
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if not (nodes[2 * i + bi], nodes[2 * j + bj]) in graph.edges:
                qubo_matrix[t, i, bi, t+1, j, bj] += lambda_g
        for i, bi in product(range(V), range(2)):
            qubo_matrix[t, V, 0, t+1, i, bi] += lambda_g
                
    # Weights constraints
    for i in range(V):
        for t in range(T_max):
            for b in range(2):
                qubo_matrix[t, i, b, t, i, b] -= (2 * graph.nodes[nodes[2 * i]]["weight"] - 1) * lambda_w
        
        for t1, t2 in product(range(T_max), range(T_max)):
            for b1, b2 in product(range(2), range(2)):
                if not (t1 == t2 and b1 == b2):
                    qubo_matrix[t1, i, b1, t2, i, b2] += lambda_w

    qubo_matrix = qubo_matrix.reshape((T_max * (V+1) * 2), (T_max * (V+1) * 2))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)

    # Delete rows and columns corresponding to the extra end node we do not need
    qubo_matrix = np.delete(qubo_matrix, [np.ravel_multi_index((t, V, 1), dims=(T_max, V+1, 2)) for t in range(T_max)], 0)
    qubo_matrix = np.delete(qubo_matrix, [np.ravel_multi_index((t, V, 1), dims=(T_max, V+1, 2)) for t in range(T_max)], 1)
    
    offset = lambda_t * T_max  + lambda_w * int(sum(graph.nodes[nodes[2 * i]]["weight"] ** 2 for i in range(V)))
    
    return qubo_matrix, offset, T_max, V