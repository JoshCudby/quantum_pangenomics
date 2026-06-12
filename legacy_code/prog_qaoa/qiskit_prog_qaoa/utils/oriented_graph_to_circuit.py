import numpy as np
import networkx as nx
from gfapy import Gfa

from qiskit_prog_qaoa.utils.oriented_circuit_utils import get_prog_qaoa_circuit
from qiskit_prog_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def gfa_file_to_oriented_prog_qaoa_circuit(data_file: str, p: int, lamda: float):
    gfa = Gfa.from_file(data_file)


    graph = nx.DiGraph()
    for segment_line in gfa.segments:
        graph.add_node(f'{segment_line.name}_+',  weight=segment_line.SC)
        graph.add_node(f'{segment_line.name}_-',  weight=segment_line.SC)
        
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

    n = len(gfa.segments)
    nodes_weights = list(graph.nodes(data="weight", default=0)) # type: ignore

    K = max(x[1] for x in nodes_weights)  # K should be more than max weight to allow for over-visiting a high weight node.
    K = int(min(K, 5))
    total_weight = int(sum(x[1] for x in nodes_weights) / 2)
    T = int(np.floor(total_weight * 1.1)) 

    return get_prog_qaoa_circuit(p=p, n=n, K=K, T=T, graph=graph, lamda=lamda), n, K, T, graph