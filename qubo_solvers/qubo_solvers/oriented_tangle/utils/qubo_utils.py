"""Utilities for building QUBO matrices for the strand-aware (oriented) tangle formulation."""

import numpy as np
import networkx as nx
from itertools import product
from math import floor
from qubo_solvers.logging import get_logger

logger = get_logger(__name__)


def edge2node_qubo_matrix_from_graph(graph: nx.DiGraph, alpha: float | None=None, penalties: list | None=None) -> tuple[np.ndarray, float, int, int]:
    """Build the QUBO matrix for the oriented tangle with per-strand copy-number weights.

    This is the edge-to-node converted variant of the oriented formulation.
    Unlike :func:`qubo_matrix_from_graph`, the coverage target can differ
    between the ``+`` and ``-`` strands of the same segment: ``i_+`` uses
    ``copy_numbers[2*index]`` and ``i_-`` uses ``copy_numbers[2*index+1]``.

    The 6-D tensor layout and penalty structure are otherwise identical to
    :func:`qubo_matrix_from_graph`.  See that function for full details of the
    variable indexing, sentinel node, symmetrisation, and deletion of the
    extra parity-1 end-node rows/columns.

    Note:
        Normalisation is disabled in this variant (the commented-out block at
        the end of the function); the raw integer matrix is returned.

    Args:
        graph (nx.DiGraph): Strand-aware directed graph produced by
            :func:`~qubo_solvers.oriented_tangle.utils.graph_utils.edge2node_oriented_graph`.
            Each node must have a ``weight`` attribute giving its individual
            (strand-specific) copy number.
        alpha (float | None): Slack factor for computing ``T_max``
            (``T_max = floor(total_weight * alpha)``).  Defaults to ``1.1``.
        penalties (list | None): Override penalty weights as
            ``[lambda_t, lambda_g, lambda_w]``.  Defaults to
            ``[100, 50, 1]``.

    Returns:
        tuple[np.ndarray, float, int, int]: ``(qubo_matrix, offset, T_max, V)``
            where ``qubo_matrix`` is a 2-D integer array, ``offset`` is the
            constant energy term, ``T_max`` is the number of timesteps, and
            ``V`` is the number of unoriented nodes (len(nodes) / 2).
    """
    nodes = list(graph.nodes)
    V = int(len(nodes) / 2)
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes))
    
    # T_max = total weight + "a bit"
    if alpha is None:
        alpha = 1.1
    T_max = floor(total_weight * alpha)
    logger.info(f'V: {V}, T: {T_max}')
    
    if penalties is None:
        # Penalty Values
        lambda_t = 100
        lambda_g = 50
        lambda_w = 1    
    else:
        lambda_t = penalties[0]
        lambda_g = penalties[1]
        lambda_w = penalties[2]
    logger.info(f'Penalties. t: {lambda_t}, g: {lambda_g}, w: {lambda_w}')
    
    # Note: we add an end node with parity 0 and 1, we only want 1 of them. We will delete the other at the end.
    qubo_matrix = np.zeros((T_max, V + 1, 2, T_max, V + 1, 2), dtype=np.int16)
    
    # Path constraint
    for t in range(T_max):
        for i, b in product(range(V), range(2)):
            qubo_matrix[t, i, b, t, i, b] -= lambda_t
            qubo_matrix[t, V, 0, t, i, b] += 2 * lambda_t
                
        qubo_matrix[t, V, 0, t, V, 0] -= lambda_t
        
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if not (i == j and bi == bj):
                qubo_matrix[t, i, bi, t, j, bj] += lambda_t
    
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
    exist_start_nodes = len(start_nodes) > 0
    exist_end_nodes = len(end_nodes) > 0
    
    # Graph step constraints
    for t in range(T_max - 1):
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if (nodes[2 * i + bi], nodes[2 * j + bj]) not in graph.edges:
                qubo_matrix[t, i, bi, t+1, j, bj] += lambda_g
        for i, bi in product(range(V), range(2)):
            qubo_matrix[t, V, 0, t+1, i, bi] += 5 * lambda_g
            if exist_end_nodes:
                if i not in end_nodes:
                    qubo_matrix[t, i, bi, t+1, V, 0] += lambda_g
    if exist_start_nodes:
        start_node = start_nodes[0]
        for b in range(2):
            qubo_matrix[0, start_node, b, 0, start_node, b] -= lambda_g
            qubo_matrix[0, start_node, b, 0, start_nodes, 1 - b] += lambda_g
        
                
    # Weights constraints
    # ( sum_t ( x_{t,i,b} ) - w[i, b] ) ^2 = sum_t ( (1 - 2 * w[i, b]) x_{t,i,b} ) + sum_{t1 /= t2} x_{t1,i,b} x_{t2,i,b} + w[i,b]^2
    for i in range(V):
        for b in range(2):  
            for t in range(T_max):
                qubo_matrix[t, i, b, t, i, b] -= (2 * graph.nodes[nodes[2 * i + b]]["weight"] - 1) * lambda_w
        
            for t1, t2 in product(range(T_max), range(T_max)):
                if not (t1 == t2):
                    qubo_matrix[t1, i, b, t2, i, b] += lambda_w

    qubo_matrix = qubo_matrix.reshape((T_max * (V+1) * 2), (T_max * (V+1) * 2))
    qubo_matrix = 0.5 * (qubo_matrix + qubo_matrix.T)

    # Delete rows and columns corresponding to the extra end node we do not need
    qubo_matrix = np.delete(qubo_matrix, [np.ravel_multi_index((t, V, 1), dims=(T_max, V+1, 2)) for t in range(T_max)], 0)
    qubo_matrix = np.delete(qubo_matrix, [np.ravel_multi_index((t, V, 1), dims=(T_max, V+1, 2)) for t in range(T_max)], 1)
    
    offset = lambda_t * T_max  + lambda_w * int(sum(graph.nodes[nodes[2 * i + b]]["weight"] ** 2 for i in range(V) for b in range(2))) + (1 if exist_start_nodes else 0)  * lambda_g
    
    # normalisation = np.max(np.abs(qubo_matrix))
    # qubo_matrix = qubo_matrix / normalisation
    # offset = offset / normalisation
    
    return qubo_matrix, offset, T_max, V


def qubo_matrix_from_graph(graph: nx.DiGraph, alpha: float | None=None, penalties: list | None=None) -> tuple[np.ndarray, float, int, int]:
    """Build the QUBO matrix for the oriented tangle with shared per-segment copy-number weights.

    Binary variables are indexed as ``(t, i, b)`` where ``t = 0 .. T_max-1``
    is the timestep, ``i = 0 .. V`` is the node index (``V`` is the sentinel
    end-node), and ``b in {0, 1}`` is the strand (0 = ``+``, 1 = ``-``).
    The internal tensor has shape ``(T_max, V+1, 2, T_max, V+1, 2)`` — the
    ``"+1"`` accommodates the sentinel end-node, and the factor of 2 encodes
    strand orientation.

    The sentinel end-node is added with both parities; parity-1 copies
    (``b=1``) are deleted at the end by removing the corresponding
    rows/columns from the flattened matrix (indexed via
    ``np.ravel_multi_index``), leaving only the parity-0 sentinel.

    Coverage targets apply per segment (not per strand): for node ``i`` the
    weight is taken from ``graph.nodes[nodes[2*i]]["weight"]``, so both
    ``i_+`` and ``i_-`` contribute to the same coverage penalty.

    Three penalty terms are used (defaults: ``lambda_t=100``,
    ``lambda_g=50``, ``lambda_w=1``):

    * ``lambda_t``: one-hot constraint — exactly one ``(i, b)`` pair is active
      per timestep.
    * ``lambda_g``: edge constraint — consecutive ``(i, b)`` pairs must be
      connected by a directed edge in the graph.
    * ``lambda_w``: coverage objective — penalises deviation from the segment's
      copy-number target.

    The matrix is symmetrised as ``0.5 * (Q + Q^T)`` after reshaping.
    Normalisation is currently disabled.

    Args:
        graph (nx.DiGraph): Strand-aware directed graph produced by
            :func:`~qubo_solvers.oriented_tangle.utils.graph_utils.oriented_graph_with_copy_numbers`.
            Each node must have a ``weight`` attribute (copy number shared
            between strands).
        alpha (float | None): Slack factor for computing ``T_max``
            (``T_max = floor(total_weight * alpha)``).  Defaults to ``1.1``.
        penalties (list | None): Override penalty weights as
            ``[lambda_t, lambda_g, lambda_w]``.  Defaults to
            ``[100, 50, 1]``.

    Returns:
        tuple[np.ndarray, float, int, int]: ``(qubo_matrix, offset, T_max, V)``
            where ``qubo_matrix`` is a 2-D integer array, ``offset`` is the
            constant energy term, ``T_max`` is the number of timesteps, and
            ``V`` is the number of unoriented nodes (len(nodes) / 2).
    """
    nodes = list(graph.nodes)
    V = int(len(nodes) / 2)
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    
    # T_max = total weight + "a bit"
    if alpha is None:
        alpha = 1.1
    T_max = floor(total_weight * alpha)
    logger.info(f'V: {V}, T: {T_max}')
    
    if penalties is None:
        # Penalty Values
        lambda_t = 100
        lambda_g = 50
        lambda_w = 1    
    else:
        lambda_t = penalties[0]
        lambda_g = penalties[1]
        lambda_w = penalties[2]
    logger.info(f'Penalties. t: {lambda_t}, g: {lambda_g}, w: {lambda_w}')
    
    # Note: we add an end node with parity 0 and 1, we only want 1 of them. We will delete the other at the end.
    qubo_matrix = np.zeros((T_max, V + 1, 2, T_max, V + 1, 2), dtype=np.int16)
    
    # Path constraint
    for t in range(T_max):
        for i, b in product(range(V), range(2)):
            qubo_matrix[t, i, b, t, i, b] -= lambda_t
            qubo_matrix[t, V, 0, t, i, b] += 2 * lambda_t
                
        qubo_matrix[t, V, 0, t, V, 0] -= lambda_t
        
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if not (i == j and bi == bj):
                qubo_matrix[t, i, bi, t, j, bj] += lambda_t
    
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
    exist_start_nodes = len(start_nodes) > 0
    exist_end_nodes = len(end_nodes) > 0
    
    # Graph step constraints
    for t in range(T_max - 1):
        for i, j, bi, bj in product(range(V), range(V), range(2), range(2)):
            if (nodes[2 * i + bi], nodes[2 * j + bj]) not in graph.edges:
                qubo_matrix[t, i, bi, t+1, j, bj] += lambda_g
        for i, bi in product(range(V), range(2)):
            qubo_matrix[t, V, 0, t+1, i, bi] += lambda_g
            if exist_end_nodes:
                if i not in end_nodes:
                    qubo_matrix[t, i, bi, t+1, V, 0] += lambda_g
    if exist_start_nodes:
        start_node = start_nodes[0]
        for b in range(2):
            qubo_matrix[0, start_node, b, 0, start_node, b] -= lambda_g
            qubo_matrix[0, start_node, b, 0, start_nodes, 1 - b] += lambda_g
        
                
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
    
    offset = lambda_t * T_max  + lambda_w * int(sum(graph.nodes[nodes[2 * i]]["weight"] ** 2 for i in range(V))) + (1 if exist_start_nodes else 0)  * lambda_g
    
    # normalisation = np.max(np.abs(qubo_matrix))
    # qubo_matrix = qubo_matrix / normalisation
    # offset = offset / normalisation
    
    return qubo_matrix, offset, T_max, V