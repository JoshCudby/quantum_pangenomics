import numpy as np
import re
import networkx as nx
from math import floor


def dwave_sample_qubo(qubo_matrix: np.ndarray, offset: float, time_limit=None, label="Oriented QUBO") -> tuple[dict, float]:
    """Perform a batch of annealing on a given Binary Quadratic Model.

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


def sample_array_to_paths(sample_array: np.ndarray, nodes: list, N: int):
    nz = np.nonzero(sample_array == 1)
    path_list_concat = [
        (
            nz[0][i],
            nz[1][i],
            'end' if nz[2][i] == N else nodes[nz[2][i] * 2 + nz[3][i]]
        ) for i in range(nz[0].shape[0])
    ]
    path_one = [step for step in path_list_concat if step[0] == 0]
    path_two = [step for step in path_list_concat if step[0] == 1]
    return [path_one, path_two]


def sample_list_to_paths(solution, nodes, T, N):
    indices = np.array([(2 * (N + 1)) * (x+1) - 1 for x in range(2 * T)])
    for index in indices:
        solution = np.insert(solution, [index], 0)
    solution = solution.reshape((2, T, N + 1, 2))
    return sample_array_to_paths(solution, nodes, N)


def get_original_vertex_name(vertex_name):
    pattern = r'(.+)_([\+\-])+'
    match = re.search(pattern, vertex_name)
    if match is None:
        raise Exception('Could not retrieve vertex name')
    else:
        return match.group(1)
    
    
def print_path(path: list):
    """Pretty print a path"""
    num_per_line = 6
    if len(path) < num_per_line:
        print(path)
        return
    
    for i in range(floor(len(path) / num_per_line)):
        print(path[i * num_per_line: (i + 1) * num_per_line])
    if not (i + 1) * num_per_line == len(path):
        print(path[(i + 1)*num_per_line:])
        
        
def oriented_node_to_perl_format(node):
    pattern = r'(.+)_([\+\-])+'
    match = re.search(pattern, node)
    return ('>' if match.group(2) == '+' else '<') + match.group(1)
        

def print_paths_to_perl_format(paths:list):
    for path in paths:
        path_str=""
        for step in path:
            if not step[2] == 'end':
                path_str += oriented_node_to_perl_format(step[2])
            else:
                if len(path_str) > 0:
                    print(path_str)
                    path_str = ""
        if len(path_str) > 0:
            print(path_str)
            

def validate_path(paths: list, graph: nx.Graph):
    """Checks the constraints for a path on a graph.
    
    In particular:
     - does the path go along graph edges at each time step
     - is each node visited the correct number of times
     - is exactly one node visited per time step

    Args:
        path (list): _description_
        graph (nx.Graph): _description_
    """
    for path in paths:
        print_path(path)
    
    for idx, path in enumerate(paths):
        time_offset = 0
        i = 0
        while i < len(path):
            if path[i][1] == i + time_offset:
                i += 1
                continue
            if path[i][1] < i + time_offset:
                print(f'Extra {"x" if idx == 0 else "y"} node at time {path[i][1]}')
                time_offset -= 1
                i += 1
                continue
            if path[i][1] > i + time_offset:
                print(f'Skipped {"x" if idx == 0 else "y"} at time {path[i][1]}')
                time_offset += 1
                i += 1
                continue
    
    node_dict = {node: 0 for node in graph.nodes}
    node_dict['end'] = 0
    
    for idx, path in enumerate(paths):
        for x in range(len(path) - 1):
            v1 = path[x][2]
            node_dict[v1] += 1
            v2 = path[x + 1][2]
            if v1 == 'end' and not v2 == 'end':
                print(f'Left the end node at {"x" if idx == 0 else "y"} path entry {x}')
            elif (not v1 == 'end') and (not v2 == 'end') and (not (v1, v2) in graph.edges):
                print(f'Broke graph edge at {"x" if idx == 0 else "y"} path entry {x}')
        node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(int(len(nodes) / 2)):
        visits = node_dict[nodes[2 * i]] + node_dict[nodes[2 * i + 1]]
        missing_visits = graph.nodes[nodes[2 * i]]["weight"] - visits
        if  missing_visits != 0:
            print(f'Did not meet node weight for node: {get_original_vertex_name(nodes[2 * i])}. Missing visits: {missing_visits}')
