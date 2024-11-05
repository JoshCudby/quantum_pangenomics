import numpy as np
import networkx as nx
from itertools import product
from math import floor
import re


def get_original_vertex_name(vertex_name):
    pattern = r'(.+)_([\+\-])+'
    match = re.search(pattern, vertex_name)
    if match is None:
        raise Exception('Could not retrieve vertex name')
    else:
        return match.group(1)
    

def qubo_matrix_from_graph(graph: nx.DiGraph, alpha: float | None=None) -> tuple[np.ndarray, float, int, int]:
    """Constructs the QUBO matrix corresponding to a graph. Also returns the offset of the model, the max time and the number of nodes.

    Args:
        graph (nx.DiGraph): the node-weighted graph describing the problem.
        alpha (float, optional): the proportion of extra time allowed to paths over the maximum weight.

    Returns:
        tuple[np.ndarray, float, int, int]: qubo_matrix, offset, T_max, V
    """    
    nodes = list(graph.nodes)
    N = int(len(nodes) / 2)
    total_weight = int(sum(dict(graph.nodes.data('weight')).values()) / 2)
    
    # T_max = total weight + "a bit"
    if alpha is None:
        alpha = 1.2
    T_max = floor(total_weight / 2 * alpha)

    # Penalty Values
    lambda_t = 10
    lambda_g = 10
    lambda_w = 1

    # Note: we add an end node with parity 0 and 1, we only want 1 of them. We will delete the other at the end.
    Q = np.zeros((2, T_max, N + 1, 2, 2, T_max, N + 1, 2), dtype=np.int8)
    
    # One node at a time constraints
    Q_walk_prime = np.ones((N+1, 2, N+1, 2), dtype=np.int8)
    for i, sigma in product(range(N), range(2)):
        Q_walk_prime[i, sigma, i, sigma] = -1
        Q_walk_prime[N, 1, :, :] = 0
        Q_walk_prime[:, :, N, 1] = 0
        Q_walk_prime[N, 0, N, 0] = -1
    Q_walk_prime *= lambda_t

    for X, t in product(range(2), range(T_max)):
        Q[X, t, :, :, X, t, :, :] += Q_walk_prime
      
        
    # Traverse graph edges constraints
    Q_graph_prime = np.zeros((N+1, 2, N+1, 2), dtype=np.int8)

    # Set end nodes
    Q_graph_prime[N, 0, 0:N, :] = 1
    end_nodes= set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            node_index_in_qubo = floor(nodes.index(node)/ 2)
            end_nodes.add(node_index_in_qubo)
    if len(end_nodes) > 0:
        print(f'Setting end nodes: {end_nodes}')
        Q_graph_prime[0:N, :, N, 0] = 1
        Q_graph_prime[list(end_nodes), :, N, 0] = 0

    for i, sigma, j, tau in product(range(N), range(2), range(N), range(2)):
        if not (nodes[2 * i + sigma], nodes[2 * j + tau]) in graph.edges:
            Q_graph_prime[i, sigma, j, tau] = 1 
    Q_graph_prime *= int(0.5 * lambda_g)

    for X, t in product(range(2), range(T_max - 1)):
        Q[X, t, :, :, X, t + 1, :, :] += Q_graph_prime
        Q[X, t + 1, :, :, X, t, :, :] += Q_graph_prime.reshape(2*(N+1),2*(N+1)).T.reshape(N+1,2,N+1,2)
        
    # Set start nodes
    start_nodes= set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'start':
            node_index_in_qubo = floor(nodes.index(node)/ 2)
            start_nodes.add(node_index_in_qubo)
    
    start_nodes = list(start_nodes)
    if len(start_nodes) > 0:
        print(f'Setting start node: {nodes[2 * start_nodes[0]]}')
        Q_start_prime = np.ones((2, 2), dtype=np.int8)
        for sigma in range(2):
            Q_start_prime[sigma, sigma] = -1
        Q_start_prime *= lambda_g    
        
        for X in range(2):
            Q[X, 0, start_nodes[0], :, X, 0, start_nodes[0], :] += Q_start_prime
        

    # Number of visits constraints
    for i in range(N):
        Q_weight_prime = np.ones((2, T_max, 2, 2, T_max, 2), dtype=np.int8)
        for X, t, sigma in product(range(2), range(T_max), range(2)):
            Q_weight_prime[X, t, sigma, X, t, sigma] -= int(2 * graph.nodes[nodes[2 * i]]["weight"])

        Q[:, :, i, :, :, :, i, :] += Q_weight_prime * lambda_w
        
        
    Q = Q.reshape((2 * T_max * (N+1) * 2), (2 * T_max * (N+1) * 2))

    # Delete rows and columns corresponding to the extra end node we do not need
    indices = np.array([(2 * (N + 1)) * (x+1) - 1 for x in range(2 * T_max)])
    Q = np.delete(Q, indices, 0)
    Q = np.delete(Q, indices, 1)
    
    offset = lambda_t * T_max * 2  \
        + lambda_w * int(sum(graph.nodes[nodes[2 * i]]["weight"] ** 2 for i in range(N))) \
            + (2 if len(start_nodes) > 0 else 0)  * lambda_g   
    return Q, offset, T_max, N