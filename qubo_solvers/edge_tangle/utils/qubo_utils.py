import numpy as np
import networkx as nx
from itertools import product
from math import floor
from dimod import CQM, Binary


def cqm_from_graph(graph: nx.DiGraph, alpha: float | None=None) -> tuple[CQM, int, int]:
    nodes = list(graph.nodes)
    V = int(len(nodes))
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    alpha = 1.2
    T_max = int(total_weight * alpha)

    variables_array_shape = (T_max, V + 1)
    
    def var_index(multi_index):
        return np.ravel_multi_index(multi_index, variables_array_shape)
    
    
    cqm = CQM()
    variables = [Binary(i) for i in range(T_max * (V+1))]
    
    # Include a 0 * variable term so that the variables are added to the model
    cqm.set_objective(
        sum(
            (graph.nodes[nodes[i]]["weight"] - sum(variables[var_index((t, i))] + variables[var_index((t, i + 1))] for t in range(T_max))) ** 2
            for i in range(0, V, 2)
        ) 
        + 0 * sum(variables)
    )
    
    # Graph step constraints
    for i,j in product(range(V), range(V)):
        if not (nodes[i], nodes[j]) in graph.edges:
            cqm.add_constraint_from_iterable(
                [(var_index((t, i)), (var_index((t + 1, j))), 1) for t in range(T_max - 1)],
                sense='==',
                rhs=0
            )
          
    # Path constraints  
    for t in range(T_max):
        cqm.add_constraint_from_iterable(
            [(var_index((t, i)), 1) for i in range(V + 1)],
            sense='==',
            rhs=1
        )
        
    # Don't leave end constraints
    for t in range(T_max - 1):
        cqm.add_constraint_from_iterable(
            [(var_index((t, V)), (var_index((t + 1, i))), 1) for i in range(V)],
            sense='==',
            rhs=0
        )
    
    return cqm, T_max, V


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
    lambda_t = 4 
    lambda_g = 2
    lambda_w = 1

    qubo_matrix = np.zeros((T_max, V + 1, T_max, V + 1), dtype=np.int8)
    
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
        qubo_matrix[t, :, t + 1, :] += lambda_g
        for edge in graph.edges:
            qubo_matrix[t, nodes.index(edge[0]), t + 1, nodes.index(edge[1])] -= lambda_g
        for i in range(V + 1):
            qubo_matrix[t, i, t + 1, V] -= lambda_g
                
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
    
    offset = (T_max * lambda_t + lambda_w * sum(graph.nodes[nodes[i]]["weight"] ** 2 for i in range(0, V, 2)))
    return qubo_matrix, offset, T_max, V