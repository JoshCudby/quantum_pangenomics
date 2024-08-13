import numpy as np
import networkx as nx
import re
import gurobipy as gp
from gurobipy import GRB
from math import floor


def dwave_sample_qubo(qubo_matrix: np.ndarray, offset: float, time_limit=None, label="Oriented QUBO") -> tuple[dict, float]:
    """Perform a batch of annealing with greedy post-processing on a given Binary Quadratic Model.

    Args:
        sampler (Sampler): The sampler to anneal with.
        bqm (BQM): The model to anneal.
        time_limit (int, optional): The time limit.
        label (str, optional): The label for sample submission on DWave platform.
        
    Returns:
        (dict, float): Returns the best sample and best energy of the batch.
    """
    
    from dimod import BQM
    from dwave.system import LeapHybridSampler
    bqm = BQM(qubo_matrix, 'BINARY')
    bqm.offset = offset
    sampler = LeapHybridSampler()
    if time_limit == -1:
        time_limit = sampler.min_time_limit(bqm)
        print(f"Using default min time limit: {time_limit}")
    sampleset = sampler.sample(bqm, time_limit, label=label)
    
    try:
        print(f"D-Wave access time: {round(sampleset.info['run_time'] / 10 ** 6)}")
    except:
        pass
    
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    return best_sample, best_energy


def gurobi_sample_qubo(qubo_matrix: np.ndarray, graph: nx.Graph, offset: int, T_max: int, time_limit: int):
    total_weight = int(sum(graph.nodes[node]["weight"] for node in list(graph.nodes)) / 2)
    
    print(f'Offset: {offset}')
    print(f'Total weight: {total_weight}')
    print(f'T_max: {T_max}')   

    with gp.Env() as env, gp.Model(env=env) as model:
        model_vars = model.addMVar(shape=qubo_matrix.shape[0], vtype=GRB.BINARY, name="x")
        model.setObjective(model_vars @ qubo_matrix @ model_vars, GRB.MINIMIZE)
        model.Params.BestObjStop = - offset
        model.Params.TimeLimit = time_limit
        model.Params.Seed = np.random.default_rng().integers(0, 1000)
        model.optimize()
        energy = model.ObjVal + offset
        return model_vars.X, energy


def sample_array_to_path(sample_array: np.ndarray, nodes: list, V: int):
    nz = np.nonzero(sample_array == 1)
    return [(nz[0][i], nodes[nz[1][i]] if nz[1][i] in range(V) else 'end') for i in range(nz[0].shape[0])]


def sample_list_to_path(sample_list: np.ndarray, graph: nx.Graph, T_max: int, V: int):
    sample_array = sample_list.reshape((T_max, V + 1))
    return sample_array_to_path(sample_array, list(graph.nodes), V)
    

def print_path(path: list):
    """Pretty print a path"""
    num_per_line = 6
    for i in range(floor(len(path) / num_per_line)):
        print(path[i * num_per_line: (i + 1) * num_per_line])
    if not (i + 1) * num_per_line == len(path):
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
    
    time_offset = 0
    for i in range(len(path)):
        if i < path[i][0] + time_offset:
            print(f'Skipped time {i}')
            time_offset -= 1
        elif i > path[i][0] + time_offset:
            print(f'Visited 2 nodes at time {i}')
            time_offset += 1
    
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
    node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(0, len(nodes), 2):
        visits = node_dict[nodes[i]] + node_dict[nodes[i + 1]]
        missing_visits = graph.nodes[nodes[i]]["weight"] - visits
        if  missing_visits != 0:
            print(f'Did not meet node weight for node: {nodes[i]}. Missing visits: {missing_visits}')

