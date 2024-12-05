import numpy as np
import networkx as nx
import re
from math import floor
from greedy import SteepestDescentSolver
from dimod import Sampler, BQM
from dwave.system import LeapHybridSampler


def dwave_sample_bqm(sampler: Sampler, bqm: BQM, time_limit=None, label="QUBO", num_reads=30):
    """Perform a batch of annealing with greedy post-processing on a given Binary Quadratic Model.

    Args:
        sampler (Sampler): The sampler to anneal with.
        bqm (BQM): The model to anneal.
        time_limit (int, optional): The time limit. Quantum solver only.
        label (str, optional): The label for sample submission on DWave platform. Quantum solver only.
        num_reads (int, optional): Number of runs in batch. Defaults to 30. Classical Solver only.
        
    Returns:
        (dict, float): Returns the best sample and best energy of the batch.
    """
    
    if isinstance(sampler, LeapHybridSampler):
        if time_limit == -1:
            time_limit = sampler.min_time_limit(bqm)
            print(f"Using default min time limit: {time_limit}")
        sampleset = sampler.sample(bqm, time_limit, label=label)
    else:
        sampleset = sampler.sample(bqm, label=label, num_reads=num_reads)
    
    try:
        print(f"D-Wave access time: {round(sampleset.info['run_time'] / 10 ** 6)}")
    except:
        pass
    
    greedy_solver = SteepestDescentSolver()
    post_processed = greedy_solver.sample(bqm, initial_states=sampleset)
    
    best_sample = post_processed.first.sample
    best_energy = post_processed.first.energy
    return best_sample, best_energy


def get_path(sample: dict):
    """Deprecated"""
    return sorted([i for i in list(sample.keys()) if sample[i]], key=lambda e: e[1])


def get_node_visits(sample: dict):
    """Deprecated"""
    path = get_path(sample)
    nodes = set([key[0] for key in list(sample.keys())])
    node_visits = {node : 0 for node in nodes}
    for path_node in path:
        node_visits[path_node[0]] += 1
        
    return node_visits


def get_constraint_values(sample: dict, graph: nx.DiGraph):
    """Deprecated"""
    node_visits = get_node_visits(sample)
    constraint_values = np.array([node_visits[x] - graph.nodes.data()[x]["weight"] for x in list(graph.nodes)])
    return constraint_values


def _index_to_node_time(idx, num_nodes):
    """Converts a qubo index to a (time-step, node_index) pair

    Args:
        idx (int): index of qubo variable
        num_nodes (int): number of nodes in graph

    Returns:
        (int, int): A pair describing the corresponding path time-step and the node index
    """
    rem = idx % num_nodes
    div = floor(idx / num_nodes)
    return (div, rem)


def dwave_sample_to_path(sample: dict, dg: nx.DiGraph) -> list:
    """Gets the actual path as a list of (time, node) pairs from an output of a DWave Sampler.

    Args:
        sample (dict): the qubo variables as a dict.
        dg (nx.DiGraph): the directed graph underlying the problem.

    Returns:
        list: a list of (time_step, node) pairs.
    """
    on_vars = []
    for i in range(len(sample.keys())):
        if sample[i] == 1:
            on_vars.append(i)
    path = [_index_to_node_time(x, len(dg.nodes)) for x in on_vars]
    path = [(e[0], list(dg.nodes)[e[1]]) for e in path]
    return path


def qubo_vars_to_path(qubo_vars: list[int], g: nx.Graph) -> list:
    """Gets the actual path as a list of (time, node) pairs from an array of qubo variable values.

    Args:
        qubo_vars (list[int]): the qubo variables as a list.
        dg (nx.DiGraph): the directed graph underlying the problem.

    Returns:
        list: a list of (time_step, node) pairs.
    """
    on_vars = []
    for i, var in enumerate(qubo_vars):
        if var == 1:
            on_vars.append(i)
    path = [_index_to_node_time(i, len(g.nodes) + 1) for i in on_vars]
    path = [(e[0], list(g.nodes)[e[1]] if e[1] < len(g.nodes) else 'end') for e in path]
    return path


def print_path(path: list):
    """Pretty print a path"""
    num_per_line = 8
    if len(path) < num_per_line:
        print(path)
        return
    for i in range(floor(len(path) / num_per_line)):
        print(path[i * num_per_line: (i + 1) * num_per_line])
    if not (i + 1) * num_per_line == len(path) - 1:
        print(path[(i + 1)*num_per_line:])

        
        
def validate_path(path: list, graph: nx.Graph):
    """Checks the constraints for a path on a graph.
    
    In particular:
     - does the path go along graph edges at each time step
     - is each node visited the correct number of times
     - is exactly one node visited per time step

    Args:
        path (list): _description_
        graph (nx.Graph): _description_
    """
    print(f"Best path:")
    print_path(path)
    if len(path) == 0:
        print("No path")
        return
    
    end_nodes = set()
    start_nodes = set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            end_nodes.add(node)
        elif val == 'start':
            start_nodes.add(node)
    end_nodes.add('end')
    
    if len(start_nodes) > 0 and not path[0][1] in start_nodes:
        print(f'Did not start at start')
    
    time_offset = 0
    i = 0
    while i < len(path):
        if i + time_offset == path[i][0]:
            i += 1
            continue
        if path[i][0] < i + time_offset:
            print(f'Extra node at time {path[i][0]}')
            time_offset -= 1
            i += 1
            continue
        if path[i][0] > i + time_offset:
            print(f'Skipped time {path[i][0] - 1}')
            time_offset += 1
            i += 1
            continue
    
    node_dict = {node: 0 for node in graph.nodes}
    node_dict['end'] = 0
    
    for x in range(len(path) - 1):
        v1 = path[x][1]
        node_dict[v1] += 1
        v2 = path[x + 1][1]            
        if v1 == 'end' and not v2 == 'end':
            print(f'Left the end node at path entry {x}')
        elif (not v1 == 'end') and (not v2 == 'end') and (not (v1, v2) in graph.edges):
            print(f'Broke graph edge at path entry {x}')
        elif len(end_nodes) > 0 and (v2 == 'end') and (not v1 in end_nodes):
            print(f'Went to end node illegally at path entry {x}')
    node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        visits = node_dict[nodes[i]]
        missing_visits = graph.nodes[nodes[i]]["weight"] - visits
        if  missing_visits != 0:
            print(f'Did not meet node weight for node: {nodes[i]}. Missing visits: {missing_visits}')
    