import networkx as nx
import gfapy
from math import remainder
from scipy.optimize import minimize


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


def graph_from_gfa_file(filename: str) -> nx.Graph:
    """Reads a .gfa file into a graph.

    Args:
        filename (str): filepath to read.

    Returns:
        nx.Graph: corresponding graph.
    """
    gfa = gfapy.Gfa.from_file(filename)
    graph = nx.Graph()
    for segment_line in gfa.segments:
        if segment_line.SC is not None:
            weight = segment_line.SC
        elif segment_line.LN is not None and segment_line.KC is not None:
            weight = segment_line.KC / segment_line.LN
        else:
            raise Exception('Could not compute graph weights from .gfa file')
        graph.add_node(segment_line.name, weight=weight, start=segment_line.st)
    for edge_line in gfa.edges:
        graph.add_edges_from([
            (edge_line.sid1.name, edge_line.sid2.name)
        ])
    return graph


def toy_graph(exact_solution=True) -> nx.Graph:
    """Returns a small fixed graph instance.
    Weights are chosen so that all weights can be satisfied if and only if exact_solution=True.

    Args:
        exact_solution (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
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

