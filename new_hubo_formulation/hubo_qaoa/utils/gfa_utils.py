"""GFA parsing and conversion to an orientation-aware directed graph with HUBO encoding parameters.

Reads a GFA (Graphical Fragment Assembly) file and constructs a NetworkX DiGraph in which
each segment is represented by two nodes (one per orientation: ``+`` and ``-``).
Alongside the graph the module computes the binary-encoding width ``n`` and the total
path weight ``total_weight`` that are consumed by the downstream Hamiltonian builder.
"""

import gfapy
import networkx as nx
import numpy as np
from typing import Sequence


def gfa_file_to_graph(filepath: str, copy_numbers: Sequence[float | int]):
    """Parse a GFA file into an orientation-aware directed graph with HUBO encoding parameters.

    Each GFA segment ``i`` is expanded into two nodes ``i+`` and ``i-`` representing
    the forward and reverse-complement orientations respectively.  For every GFA edge
    ``(u, v)`` a directed arc ``u → v`` is added; the reverse-complement arc
    ``v̄ → ū`` is added automatically so the graph is orientation-consistent.
    Each node carries a ``weight`` attribute equal to the segment's copy number and a
    ``start`` attribute taken from the ``st`` tag of the GFA segment line.

    The binary encoding width ``n = ⌈log₂(V)⌉`` is the number of qubits required to
    index any one of the ``V`` nodes at a single QAOA timestep.

    ``total_weight`` is the sum of all node weights divided by two (since each segment
    contributes two orientation nodes).  It is used as the number of timesteps ``T``
    when building the Hamiltonian, so that the total circuit width is ``T × n`` qubits.

    Args:
        filepath: Path to a GFA-format pangenome graph file.
        copy_numbers: Copy number for each segment, in the same order as the segments
            appear in the GFA file.  The length must equal the number of segments.

    Returns:
        A four-tuple ``(graph, n, V, total_weight)`` where:

        * ``graph`` (``nx.DiGraph``) – orientation-aware directed graph with ``V`` nodes.
          Each node is labelled ``<name>_+`` or ``<name>_-`` and carries ``weight``
          and ``start`` attributes.
        * ``n`` (``int``) – number of binary-encoding qubits per timestep,
          ``⌈log₂(V)⌉``.
        * ``V`` (``int``) – total number of nodes (twice the number of segments).
        * ``total_weight`` (``int``) – sum of copy numbers, used as the number of
          QAOA timesteps ``T`` and for Hamiltonian normalisation.

    Raises:
        Exception: If the length of ``copy_numbers`` does not match the number of
            segments in the GFA file.
    """
    gfa = gfapy.Gfa.from_file(filepath, vlevel=0)

    graph = nx.DiGraph()
    if not len(gfa.segments) == len(copy_numbers):
        raise Exception(f'Got {len(copy_numbers)} copy numbers but .gfa has {len(gfa.segments)} segments.')
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

    nodes = list(graph.nodes)
    V = len(nodes)
    n = int(np.ceil(np.log2(V)))
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    return graph, n, V, total_weight