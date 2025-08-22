import gfapy
import networkx as nx
import numpy as np


def gfa_file_to_graph(filepath: str, copy_numbers: list[float]):
    gfa = gfapy.Gfa.from_file(filepath, vlevel=0)

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

    nodes = list(graph.nodes)
    V = len(nodes)
    n = int(np.ceil(np.log2(V+1)))
    total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
    T = int(1.1 * total_weight)
    return graph, n, V, T