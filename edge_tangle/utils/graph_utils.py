import gfapy
import networkx as nx
import re


def invert_orient(orient: str):
    if orient == '+':
        return '-'
    elif orient == '-':
        return '+'
    else:
        raise Exception('Bad orient')
    

def oriented_graph_from_file(filename):
    """Reads a .gfa file into the dual graph, where edges become nodes.

    Args:
        filename (str): filepath to read.

    Returns:
        nx.Graph: corresponding oriented graph.
    """
    gfa = gfapy.Gfa.from_file(filename, vlevel=0)
    graph = nx.DiGraph()
    for edge_line in gfa.edges:
        v1 = edge_line.sid1
        v2 = edge_line.sid2
        forward_node = f'{v1.name}{v1.orient}_{v2.name}{v2.orient}'
        graph.add_node(forward_node, weight=edge_line.EC)
        backward_node = f'{v2.name}{invert_orient(v2.orient)}_{v1.name}{invert_orient(v1.orient)}'
        graph.add_node(backward_node, weight=edge_line.EC)
        for node in graph.nodes:
            node_matches = re.search(
                r'(.+[\+\-])_(.+[\+\-])',
                node
            )
            node_start = node_matches.group(1)
            node_end = node_matches.group(2)
            if node_start == f'{v2.name}{v2.orient}':
                graph.add_edges_from([(forward_node, node)])
            if node_start == f'{v1.name}{invert_orient(v1.orient)}':
                graph.add_edges_from([(backward_node, node)])
            if node_end == f'{v1.name}{v1.orient}':
                graph.add_edges_from([(node, forward_node)])
            if node_end == f'{v2.name}{invert_orient(v2.orient)}':
                graph.add_edges_from([(node, backward_node)])
                
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