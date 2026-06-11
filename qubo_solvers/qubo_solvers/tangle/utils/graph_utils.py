"""Utilities for constructing NetworkX graphs from GFA files for the tangle QUBO formulation."""

import networkx as nx
import gfapy


def graph_with_copy_numbers(filename: str, copy_numbers: list, nodes: list | None=None) -> nx.Graph:
    """Read a GFA file into an undirected graph with copy-number node weights.

    Each GFA segment becomes a node with two attributes: ``weight`` (integer
    copy number) and ``start`` (the segment's ``st`` tag from the GFA, one of
    ``'start'``, ``'end'``, or ``None``).  Only segments whose names appear in
    ``nodes`` are added; segments absent from ``nodes`` are skipped and the
    copy-number index is adjusted accordingly.

    Note:
        ``nodes`` typically comes from Pathfinder output and may be a strict
        subset of ``gfa.names`` if Pathfinder filtered low-coverage segments.
        When ``nodes`` is ``None`` all GFA segments are included and
        ``copy_numbers`` must have one entry per segment in GFA order.

    Args:
        filename (str): Path to the ``.gfa`` file to read.
        copy_numbers (list): Integer copy number for each included node, in the
            same order as the GFA segment list (skipped nodes are not counted).
        nodes (list | None): Ordered list of node names to include.  If
            ``None``, all segments in the GFA are included.

    Returns:
        nx.Graph: Undirected graph with node attributes ``weight`` and
            ``start``, and edges derived from GFA edge records.
    """
    gfa = gfapy.Gfa.from_file(filename)

    if nodes is None:
        nodes = gfa.names

    graph = nx.Graph()
    index_offset = 0
    for index, segment_line in enumerate(gfa.segments):
        if segment_line.name in nodes:
            graph.add_node(segment_line.name, weight=copy_numbers[index + index_offset], start=segment_line.st)
        else:
            index_offset -= 1
    for edge_line in gfa.edges:
        if edge_line.sid1.name in nodes and edge_line.sid2.name in nodes:
            graph.add_edges_from([
                (edge_line.sid1.name, edge_line.sid2.name)
            ])
    return graph


def toy_graph(exact_solution=True) -> nx.Graph:
    """Return a small, hand-crafted graph for testing the QUBO formulation.

    The graph has five nodes (``'0'``–``'4'``) arranged so that node ``'1'``
    lies at a branch point.  Node weights are chosen such that the coverage
    objective is exactly satisfiable when ``exact_solution=True`` (node ``'1'``
    has weight 3, matching the number of distinct paths through it) and
    infeasible when ``exact_solution=False`` (weight 4, which cannot be
    achieved by any walk).

    Args:
        exact_solution (bool): If ``True`` (default), use weights that admit a
            perfect QUBO optimum.  If ``False``, use weights that make the
            coverage constraint unsatisfiable.

    Returns:
        nx.Graph: Test graph with node attributes ``weight`` and ``start``.
    """
    weight_1 = 3 if exact_solution else 4
        
    g = nx.Graph()
    g.add_nodes_from([
        ('0', {"weight": 1, "start": "start"}),
        ('1', {"weight": weight_1}),
        ('2', {"weight": 1}),
        ('3', {"weight": 1}),
        ('4', {"weight": 1, "start": "end"}),
    ])
    g.add_edges_from([
        ('0', '1'), ('1', '2'), ('1', '3'), ('1', '4'), ('2', '3'), ('2', '4'), ('3', '4'),
    ])
    return g

