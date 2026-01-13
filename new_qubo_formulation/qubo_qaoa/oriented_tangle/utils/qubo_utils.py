import numpy as np
import numpy.typing as npt
import networkx as nx
from itertools import product
from typing import Optional
from math import floor


def qubo_matrix_from_graph(
    graph: nx.DiGraph, 
    penalties: Optional[list[int]]=None, 
    T: Optional[int] = None
) -> tuple[npt.NDArray, float, int, int]:
    """Constructs the QUBO matrix corresponding to a graph. Also returns the offset of the model, the max time and the number of nodes.

    Args:
        graph (nx.DiGraph): the node-weighted graph describing the problem.
        alpha (float, optional): the proportion of extra time allowed to paths over the maximum weight.

    Returns:
        tuple[np.ndarray, float, int, int]: qubo_matrix, offset, T, V
    """
    nodes = list(graph.nodes)
    V = int(len(nodes) / 2)
    
    # T = total weight + "a bit"
    if T is None:
        T = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    print(f'V: {V}, T: {T}')
    
    if penalties is None:
        # Penalty Values
        lambda_t = 100
        lambda_g = 50
        lambda_w = 1    
    else:
        lambda_t = penalties[0]
        lambda_g = penalties[1]
        lambda_w = penalties[2]
    print(f'Penalties. t: {lambda_t}, g: {lambda_g}, w: {lambda_w}')
    lambda_start = lambda_g
    
    qubo_matrix = np.zeros((T, V, 2, T, V, 2), dtype=np.int16)
    offset = 0
    
    # One node at a time constraints
    Q_walk_prime = np.ones((V, 2, V, 2), dtype=np.int8)
    for i, sigma in product(range(V), range(2)):
        Q_walk_prime[i, sigma, i, sigma] = -1
    Q_walk_prime *= lambda_t
    for t in range(T):
        qubo_matrix[t, :, :, t, :, :] += Q_walk_prime
    offset += T * lambda_t
    
    # Traverse graph edges constraints
    Q_graph_prime = np.zeros((V, 2, V, 2), dtype=np.int8)
    
    for i, sigma, j, tau in product(range(V), range(2), range(V), range(2)):
        if (nodes[2 * i + sigma], nodes[2 * j + tau]) in graph.edges:
            Q_graph_prime[i, sigma, j, tau] = -1 
    Q_graph_prime *= lambda_g
    offset += (T-1) * lambda_g

    for t in range(T - 1):
        qubo_matrix[t,     :, :, t + 1, :, :] += Q_graph_prime
    
    # Set start/end nodes
    start_nodes=set()
    end_nodes= set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'start':
            print(f'Found start node:{node}')
            node_index_in_qubo = floor(nodes.index(node)/ 2)
            start_nodes.add(node_index_in_qubo)
        if val == 'end':
            print(f'Found end node:{node}')
            node_index_in_qubo = floor(nodes.index(node)/ 2)
            end_nodes.add(node_index_in_qubo)
    
    start_nodes = list(start_nodes)        
    end_nodes = list(end_nodes)
    print(f'Start nodes: {start_nodes}, End nodes: {end_nodes}')
    
    if len(start_nodes) > 0:
        for i, j, sigma, tau in product(start_nodes, start_nodes, range(2), range(2)):
            qubo_matrix[0, i, sigma, 0, j, tau] += lambda_start * (-1) ** ((i==j) * (sigma == tau))
    
    if len(end_nodes) > 0:
        for i, j, sigma, tau in product(end_nodes, end_nodes, range(2), range(2)):
            qubo_matrix[T-1, i, sigma, T-1, j, tau] += lambda_start * (-1) ** ((i==j) * (sigma == tau))
    offset += (1 if len(start_nodes) else 0)  * lambda_start \
                + (1 if len(end_nodes) else 0)  * lambda_start 
                
    # Number of visits constraints
    for i in range(V):
        Q_weight_prime = np.ones((T, 2, T, 2), dtype=np.int8)
        for t, sigma in product(range(T), range(2)):
            Q_weight_prime[t, sigma, t, sigma] -= int(2 * graph.nodes[nodes[2 * i]]["weight"])

        qubo_matrix[:, i, :, :, i, :] += Q_weight_prime * lambda_w

    qubo_matrix = qubo_matrix.reshape((T * V * 2), (T * V * 2))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)
    
    offset += lambda_w * int(sum(graph.nodes[nodes[2 * i]]["weight"] ** 2 for i in range(V))) \

    
    # normalisation = np.max(np.abs(qubo_matrix))
    # qubo_matrix = qubo_matrix / normalisation
    # offset = offset / normalisation
    
    return qubo_matrix, offset, T, V