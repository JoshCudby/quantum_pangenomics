import networkx as nx
import gfapy


def graph_with_copy_numbers(filename: str, copy_numbers: list, nodes: list) -> nx.Graph:
    """Reads a .gfa file into a graph.

    Args:
        filename (str): filepath to read.

    Returns:
        nx.Graph: corresponding graph.
    """
    gfa = gfapy.Gfa.from_file(filename)
    gfa.connected_components()
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

