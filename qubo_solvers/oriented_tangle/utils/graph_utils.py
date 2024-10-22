import gfapy
import networkx as nx

def oriented_graph_from_file(filename):
    """Reads a .gfa file into an oriented graph, where each node has a positive and negative version.

    Args:
        filename (str): filepath to read.

    Returns:
        nx.Graph: corresponding oriented graph.
    """
    gfa = gfapy.Gfa.from_file(filename, vlevel=0)
    graph = nx.DiGraph()
    for segment_line in gfa.segments:
        graph.add_node(f'{segment_line.name}_+', weight=segment_line.SC)
        graph.add_node(f'{segment_line.name}_-', weight=segment_line.SC)
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


def normalise_node_weights(graph: nx.Graph, normalisation: float) -> nx.Graph:
    """Normalises weights of nodes in a graph by a constant factor.

    Args:
        graph (nx.Graph): a node-weighted graph, with node attribute "weight".
        normalisation (float): the constant factor to normalise weights by.

    Returns:
        nx.Graph: a graph with node attributes "weight".
    """
    for node in graph.nodes:
        graph.nodes[node]["weight"] = round(graph.nodes[node]["weight"] / normalisation)
    return graph