"""GFA file parsing for qiskit_simulation.

Converts a GFA (Graphical Fragment Assembly) pangenome graph into a
``networkx.DiGraph`` suitable for QUBO problem construction, computing
the number of binary variables (``n``) and the QAOA walk length (``T``)
from the segment copy numbers.
"""

import gfapy
import networkx as nx
import numpy as np
from typing import Sequence


def gfa_file_to_graph(filepath: str, copy_numbers: Sequence[float | int]):
    """Parse a GFA file and build a directed segment graph with copy-number weights.

    Each GFA segment becomes two nodes (``<name>_+`` and ``<name>_-``)
    representing the forward and reverse orientations.  Each GFA edge creates
    directed arcs between the appropriate orientation nodes and their reverse-
    complement counterparts.

    Args:
        filepath: Path to the ``.gfa`` file.
        copy_numbers: A sequence of numeric copy numbers, one per segment in
            the GFA file (in the order they appear).  Must have the same
            length as the number of segments.

    Returns:
        A tuple ``(graph, n, V, T)`` where:

        - ``graph``: A ``networkx.DiGraph`` with nodes labelled
          ``'<seg>_+'`` / ``'<seg>_-'`` carrying ``weight`` (copy number)
          and ``start`` (segment start position) attributes.
        - ``n``: Number of bits required to represent any node index
          (``ceil(log2(V+1))``).
        - ``V``: Total number of nodes in the graph.
        - ``T``: Upper-bound walk length, set to ``ceil(1.1 * total_weight)``
          where ``total_weight`` is half the sum of all copy numbers.

    Raises:
        Exception: If ``len(copy_numbers)`` does not match the number of
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
    n = int(np.ceil(np.log2(V+1)))
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    T = int(1.1 * total_weight)
    return graph, n, V, T