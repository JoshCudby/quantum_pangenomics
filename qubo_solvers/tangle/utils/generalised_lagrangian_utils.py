import networkx as nx
import numpy as np
from itertools import product
from dimod.reference.samplers import SimulatedAnnealingSampler
from dimod import Sampler, BQM
from utils.graph_utils import setup_graph_for_tangle_qubo
from utils.sampling_utils import get_constraint_values, get_path, dwave_sample_bqm


def _tangle_problem_iteration(sampler: Sampler, graph: nx.DiGraph, lamda: list, mu: float, P: int):
    """
    Deprecated.
    Perform one iteration of the tangle problem using the generalised Lagrangian method.

    Args:
        sampler (Sampler): The classical or quantum sampler to perform the annealing.
        graph (nx.DiGraph): The node-weighted graph which underlies the problem.
        lamda (list): The linear generalised Lagrangian variables.
        mu (float): The quadratic generalised Lagrangian variable.
        P (int): The penalty for traversing non-edges.

    Returns:
        (dict, float, list): Returns the best solution, best energy and list of node-weight constraint values.
    """
    bqm = _tangle_problem_bqm(graph, lamda, mu, P)
    best_sample, best_energy = dwave_sample_bqm(sampler, bqm)
    constraint_values = get_constraint_values(best_sample, graph)
    return best_sample, best_energy, constraint_values


def tangle_problem(graph: nx.DiGraph, sampler=None, lamda=None, mu=0.5, growth_factor=1.1, P=10):
    """
    Deprecated.
    Solve the tangle problem using the generalised Lagrangian method.

    Args:
        graph (nx.DiGraph): node-weighted graph underlying the tangle problem.
        sampler (Sampler, optional): The sampler to use. Defaults to None.
        lamda (np.ndarray, optional): Initial values for the linear Lagrangian variables. Defaults to None.
        mu (float, optional): Initial value for the quadratic Lagrangian variable. Defaults to 0.5.
        growth factor (float, optional): Growth rate for quadratic Lagrangian variable after each iteration. Defaults to 1.1.
        P (int, optional): The penalty for traversing non-edges. Defaults to 10.

    Returns:
        (dict, float64, list, float): Returns the best variable assignment and the corresponding energy as well as the lagrangian variable values.
    """
    if sampler is None:
        sampler = SimulatedAnnealingSampler()
    if lamda is None:
        lamda = np.array([0] * len(list(graph.nodes)), dtype=float)
        
    if not all("weight" in x[1].keys() for x in graph.nodes.data()):
        raise Exception("Graph is not node-weighted")
    if not len(lamda) == len(list(graph.nodes)):
        raise Exception("Require one linear Lagrangian variable per graph node")
    if mu <= 0:
        raise Exception("Quadratic Lagrangian variable should be strictly positive")
    if growth_factor <= 1:
        raise Exception("Growth factor should be strictly greater than 1")
        
    best_sample, best_energy, constraint_values = _tangle_problem_iteration(sampler, graph, lamda, mu, P)
    print(f'Best path={get_path(best_sample)}\nBest energy={best_energy}\nConstraint values={list(zip(graph.nodes, constraint_values))}\n')
    while not all(constraint_values <= 0):
        lamda += (constraint_values > 0) * mu * constraint_values
        mu *= growth_factor
        
        best_sample, best_energy, constraint_values = _tangle_problem_iteration(sampler, graph, lamda, mu, P)
        print(f'Best path={get_path(best_sample)}\nBest energy={best_energy}\nConstraint values={list(zip(graph.nodes, constraint_values))}\n')

    return best_sample, best_energy, lamda, mu


def _tangle_problem_bqm(graph: nx.Graph, lamda: list, mu: float, P: int) -> BQM:
    """
    Deprecated.
    Returns a Binary Quadratic Model for the tangle problem.
    
    The tangle problem is to find the longest path through a node-weighted graph, where any node can be visited at most its weight many times.

    Args:
        graph (nx.Graph): the node-weighted graph which underlies the tangle problem
    """
    bqm = BQM({}, {}, 0, "BINARY")
    t_max = sum(graph.nodes.data()[node]["weight"] for node in graph.nodes) + 1
    
    qubo_graph = setup_graph_for_tangle_qubo(graph, t_max)
    nodes = list(qubo_graph.nodes)
    edges = list(qubo_graph.edges)
    
    # Reward travelling along true edges; penalise travelling not along edges
    for t in range(t_max - 1):
        for i, j in product(range(len(nodes)), range(len(nodes))):
            bqm.add_interaction(
                (nodes[i], t), 
                (nodes[j], t + 1), 
                -1 if ((nodes[i], nodes[j]) in edges) else P
            )
                
        # Travelling the virtual edges should not be rewarded
        bqm.add_interaction((nodes[-2], t), (nodes[-1], t + 1), 1)
        bqm.add_interaction((nodes[-1], t), (nodes[-1], t + 1), 1)
    
    # Penalise not starting at start or ending at end
    bqm.add_linear((nodes[0], 0), -P)
    bqm.add_linear((nodes[-1], t_max - 1), -P)
    bqm.offset += 2 * P
    
    # Penalise multiple locations at one time
    for t in range(t_max):
        bqm.offset += P
        for i in range(len(nodes)): 
            bqm.add_linear((nodes[i], t), -P)
            for j in range(i):
                bqm.add_interaction((nodes[i], t), (nodes[j], t), 2 * P) 
                
    # Generalised Lagrangian Penalties
    for i in range(len(nodes) - 1):
        weight = qubo_graph.nodes.data()[nodes[i]]["weight"]
        bqm.offset += mu / 2 * weight ** 2 - lamda[i] * weight
        for t1 in range(t_max):
            bqm.add_linear((nodes[i], t1), mu / 2 * (1 - 2 * weight) + lamda[i])
            for t2 in range(t1):
                bqm.add_interaction((nodes[i], t2), (nodes[i], t1), mu)
    
    return bqm