import numpy as np
import networkx as nx
import re
import gurobipy as gp
import subprocess
from qubo_solvers.definitions import MQLIB_DIR
from gurobipy import GRB
from math import floor


def mqlib_sample_qubo(oriented_out_dir: str, filename: str, offset: int, graph: nx.Graph, T: int, V: int, time_limit: int):
    input_filepath = f"{oriented_out_dir}/mqlib_input_{filename}.txt"

    # Run the MQLib solver and capture output
    process = subprocess.run([f"{MQLIB_DIR}/bin/MQLib", "-fQ", input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
    out = process.stdout.decode("utf-8")

    # First line of output includes run data. 3rd line contains the solution.
    out_data = [x for x in out.split('\n') if len(x) > 0]
    solution = out_data[2].split()
    solution = [int(x) for x in solution]
    solution_energy = int(out_data[0].split(',')[3])
    energy = offset - solution_energy
    path = sample_list_to_path(solution, graph, T, V)
    return solution, energy, path


def dwave_sample_qubo(
    qubo_matrix: np.ndarray, offset: float, graph: nx.Graph, T: int, V: int, time_limit=None, label="Oriented QUBO"
    ) -> tuple[dict, float]:
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
    path = sample_list_to_path(np.array(list(best_sample.values())), graph, T, V)
    return best_sample, best_energy, path


def gurobi_sample_qubo(qubo_matrix: np.ndarray, offset: int, graph: nx.Graph, T: int, V: int, time_limit: int):
    total_weight = int(sum(graph.nodes[node]["weight"] for node in list(graph.nodes)) / 2)
    
    print(f'Offset: {offset}')
    print(f'Total weight: {total_weight}')
    print(f'T_max: {T}')

    with gp.Env() as env, gp.Model(env=env) as model:
        model_vars = model.addMVar(shape=qubo_matrix.shape[0], vtype=GRB.BINARY, name="x")
        model.setObjective(model_vars @ qubo_matrix @ model_vars, GRB.MINIMIZE)
        model.Params.BestObjStop = - offset
        model.Params.TimeLimit = time_limit
        model.Params.Seed = np.random.default_rng().integers(0, 1000)
        model.optimize()
        energy = model.ObjVal + offset
        path = sample_list_to_path(model_vars.X, graph, T, V)
        return model_vars.X, energy, path


def sample_array_to_path(sample_array: np.ndarray, nodes: list, V: int):
    nz = np.nonzero(sample_array == 1)
    return [
        (
            int(nz[0][i]), 
            nodes[nz[1][i] * 2 + nz[2][i]] if nz[1][i] in range(V) else 'end'
        ) for i in range(nz[0].shape[0])
    ]


def sample_list_to_path(sample_list: np.ndarray, graph: nx.Graph, T_max: int, V: int):
    for idx in [t * (V + 1) * 2 + V * 2 + 1 for t in range(T_max)]:
        sample_list = np.insert(sample_list, idx, 0)
    sample_array = sample_list.reshape((T_max, V + 1, 2))
    return sample_array_to_path(sample_array, list(graph.nodes), V)
    

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
        
        
def get_original_vertex_name(vertex_name):
    pattern = r'(.+)_([\+\-])+'
    match = re.search(pattern, vertex_name)
    if match is None:
        raise Exception('Could not retrieve vertex name')
    else:
        return match.group(1)
        

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
    print("Best path:")
    print_path(path)
    
    end_nodes = set()
    start_nodes = set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            end_nodes.add(node)
        elif val == 'start':
            start_nodes.add(node)
    if len(end_nodes) > 0:
        end_nodes.add('end')
    
    if len(start_nodes) > 0 and path[0][1] not in start_nodes:
        print('Did not start at start')
    
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
        elif (not v1 == 'end') and (not v2 == 'end') and ((v1, v2) not in graph.edges):
            print(f'Broke graph edge at path entry {x}')
        elif len(end_nodes) > 0 and (v2 == 'end') and (v1 not in end_nodes):
            print(f'Went to end node illegally at path entry {x}')
    node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(int(len(nodes) / 2)):
        visits = node_dict[nodes[2 * i]] + node_dict[nodes[2 * i + 1]]
        missing_visits = graph.nodes[nodes[2 * i]]["weight"] - visits
        if  missing_visits != 0:
            print(f'Did not meet node weight for node: {get_original_vertex_name(nodes[2 * i])}. Missing visits: {missing_visits}')

