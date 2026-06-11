"""Utilities for building the QUBO matrix for the standard (unoriented) tangle formulation."""

import networkx as nx
import numpy as np
from itertools import product
from math import floor


def get_tangle_qubo_matrix(graph: nx.Graph) -> np.ndarray:
    """Build the QUBO matrix for the maximum-coverage path problem on an unoriented graph.

    Binary variables are indexed as ``(t, i)`` where ``t = 0 .. T-1`` is the
    timestep and ``i = 0 .. W`` is the node index.  Index ``i = W`` is a
    sentinel "skip/end" node that absorbs the walk once it leaves the tangle.
    The number of timesteps is ``T = floor(total_weight * 1.2)`` — a slack
    factor of 1.2 above the total copy-number sum.

    Four penalty terms are assembled into a 4-D tensor of shape
    ``(T, W+1, T, W+1)`` before being flattened to a 2-D matrix:

    * ``lambda_t = 10`` — one-hot constraint: exactly one node is active per
      timestep.
    * ``lambda_g = 10`` — edge constraint: consecutive active nodes must share a
      graph edge (or transition to/from the sentinel).
    * ``lambda_start = 5`` — boundary constraint: the first timestep must begin
      at a designated start node (if present in the graph), and transitions to
      the sentinel are forbidden unless the current node is an end node.
    * ``lambda_w = 1`` — coverage objective: penalises deviation from target
      copy numbers via
      ``lambda_w * (sum_t x[t,i] - weight[i])^2``, which expands to
      ``lambda_w * (1 - 2*weight[i]) * x[t,i]
      + lambda_w * sum_{t1 != t2} x[t1,i] * x[t2,i] + const``.

    The 4-D tensor is reshaped to ``(T*(W+1), T*(W+1))``, symmetrised as
    ``0.5 * (Q + Q^T)``, then normalised by ``max(|Q|)`` so that all entries
    lie in ``[-1, 1]``.  The offset is normalised by the same factor.

    Args:
        graph (nx.Graph): Node-weighted undirected graph.  Each node must have
            a ``weight`` attribute (int copy number) and optionally a ``start``
            attribute (``'start'``, ``'end'``, or ``None``).

    Returns:
        tuple[np.ndarray, float, int, int]: ``(Q, offset, T, W)`` where
            ``Q`` is the normalised 2-D QUBO matrix, ``offset`` is the
            normalised constant energy term, ``T`` is the number of timesteps,
            and ``W`` is the number of graph nodes.
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
        for i, j in product(range(len(start_nodes)), range(len(start_nodes))):
            qubo_matrix[0, start_nodes[i], 0, start_nodes[j]] += S_qubo_matrix[i, j]
        offset += lambda_start
    
    if exist_end_nodes:
        E_qubo_matrix = np.zeros((W+1, W+1), dtype=np.int16)
        E_qubo_matrix[:, W] = lambda_start * np.array([i not in end_nodes for i in range(W)] + [0], dtype=np.int16)
        for t in range(T-1):
            qubo_matrix[t, :, t+1, :] += E_qubo_matrix
    
    qubo_matrix = qubo_matrix.reshape((T * (W+1), T * (W+1)))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)
    
    normalisation = np.max(np.abs(qubo_matrix))
    qubo_matrix = qubo_matrix / normalisation
    offset = offset / normalisation

    return qubo_matrix, offset, T, W
