import gfapy
import networkx as nx
import numpy as np
from typing import Sequence


def gfa_file_to_graph(filepath: str, copy_numbers: Sequence[float | int]):
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