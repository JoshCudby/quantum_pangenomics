"""Utilities for constructing strand-aware directed graphs from GFA files for the oriented tangle QUBO formulation."""

import gfapy
import networkx as nx
from qubo_solvers.logging import get_logger

logger = get_logger(__name__)


def edge2node_oriented_graph(filename: str, copy_numbers: list[str]):
    """Read a GFA file into a strand-aware directed graph with per-strand copy-number weights.

    Each GFA segment ``i`` becomes two nodes — ``i_+`` and ``i_-`` — whose
    weights are taken from separate entries in ``copy_numbers``:
    ``i_+`` receives ``copy_numbers[2*index]`` and ``i_-`` receives
    ``copy_numbers[2*index+1]``.  This differs from
    :func:`oriented_graph_with_copy_numbers`, where both strands share the
    same value.

    Directed edges are added in both orientations: for each GFA edge
    ``(v1, v2)``, the forward edge ``(v1.orient, v2.orient)`` is added first,
    then ``gfapy``'s ``invert()`` method is used to derive and add the
    reverse-complement edge ``(v2_inv, v1_inv)``.

    Args:
        filename (str): Path to the ``.gfa`` file to read.
        copy_numbers (list[str]): Copy numbers for oriented nodes, with length
            equal to twice the number of GFA segments.  Entry ``2*index`` is
            for the ``+`` strand and ``2*index+1`` is for the ``-`` strand of
            segment ``index``.

    Returns:
        nx.DiGraph: Directed graph with node attributes ``weight`` and
            ``start``.
    """
    gfa = gfapy.Gfa.from_file(filename, vlevel=0)
    graph = nx.DiGraph()
    for index, segment_line in enumerate(gfa.segments):
        graph.add_node(f'{segment_line.name}_+', weight=copy_numbers[2*index], start=segment_line.st)
        graph.add_node(f'{segment_line.name}_-', weight=copy_numbers[2*index+1], start=segment_line.st)
    for edge_line in gfa.edges:
        v1 = edge_line.sid1
        v2 = edge_line.sid2
        graph.add_edges_from([
            (f'{v1.name}_{v1.orient}', f'{v2.name}_{v2.orient}'),
        ])
        v1.invert()
        v2.invert()
        graph.add_edges_from([
            (f'{v2.name}_{v2.orient}', f'{v1.name}_{v1.orient}'),
        ])
    return graph


def oriented_graph_with_copy_numbers(filename, copy_numbers, nodes: list | None=None):
    """Read a GFA file into a strand-aware directed graph with shared copy-number weights.

    Each GFA segment ``i`` becomes two nodes — ``i_+`` and ``i_-`` — that both
    receive the same copy number: ``copy_numbers[index]``, where ``index`` is
    the segment's position in the GFA segment list.  This mirrors the
    unoriented formulation where a single coverage target applies to a node
    regardless of traversal direction.

    Directed edges are added in both orientations using the same inversion
    approach as :func:`edge2node_oriented_graph`: the forward edge is added
    from the GFA record directly, and the reverse-complement edge is derived
    via ``gfapy``'s ``invert()`` method.

    Note:
        The ``nodes`` parameter is not yet implemented (subgraph filtering is
        a TODO).  All segments in the GFA are always included.

    Args:
        filename (str): Path to the ``.gfa`` file to read.
        copy_numbers (list): Integer copy number for each GFA segment, one
            entry per segment (not per oriented node).
        nodes (list | None): Reserved for future subgraph filtering; currently
            unused.

    Returns:
        nx.DiGraph: Directed graph with node attributes ``weight`` and
            ``start``.
    """
    # TODO: deal with subgraphs via the nodes parameter
    gfa = gfapy.Gfa.from_file(filename, vlevel=0)
    graph = nx.DiGraph()
    for index, segment_line in enumerate(gfa.segments):
        graph.add_node(f'{segment_line.name}_+', weight=copy_numbers[index], start=segment_line.st)
        graph.add_node(f'{segment_line.name}_-', weight=copy_numbers[index], start=segment_line.st)
    for edge_line in gfa.edges:
        v1 = edge_line.sid1
        v2 = edge_line.sid2
        graph.add_edges_from([
            (f'{v1.name}_{v1.orient}', f'{v2.name}_{v2.orient}'),
        ])
        v1.invert()
        v2.invert()
        graph.add_edges_from([
            (f'{v2.name}_{v2.orient}', f'{v1.name}_{v1.orient}'),
        ])
    return graph
