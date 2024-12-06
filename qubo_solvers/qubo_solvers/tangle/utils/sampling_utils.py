import numpy as np
import networkx as nx
import gurobipy as gp
import subprocess
from gurobipy import GRB
from math import floor
from greedy import SteepestDescentSolver
from dimod import BQM
from dwave.system import LeapHybridSampler
from qubo_solvers.definitions import MQLIB_DIR


def mqlib_sample_qubo(tangle_out_dir: str, filename: str, offset: int, graph: nx.Graph, time_limit: int):
    input_filepath = f"{tangle_out_dir}/mqlib_input_{filename}.txt"

    # Run the MQLib solver and capture output
    process = subprocess.run([f"{MQLIB_DIR}/bin/MQLib", "-fQ", input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
    out = process.stdout.decode("utf-8")

    # First line of output includes run data. 3rd line contains the solution.
    out_data = [x for x in out.split('\n') if len(x) > 0]
    solution = out_data[2].split()
    solution = [int(x) for x in solution]
    solution_energy = int(out_data[0].split(',')[3])
    energy = offset - solution_energy
    path = qubo_vars_to_path(solution, graph)
    return solution, energy, path


def gurobi_sample_qubo(qubo_matrix: np.ndarray, graph: nx.Graph, offset: int, time_limit: int):
    with gp.Env() as env, gp.Model(env=env) as model:
        model_vars = model.addMVar(shape=qubo_matrix.shape[0], vtype=GRB.BINARY, name="x")
        model.setObjective(model_vars @ qubo_matrix @ model_vars, GRB.MINIMIZE)
        model.Params.BestObjStop = - offset
        model.Params.TimeLimit = time_limit
        model.Params.Seed = np.random.default_rng().integers(0, 1000)
        model.optimize()
        
        energy = model.ObjVal + offset
        path = qubo_vars_to_path(model_vars.X, graph)
        return energy, path, model_vars.X


def dwave_sample_qubo(qubo_matrix: np.ndarray, offset: int, graph: nx.Graph, time_limit=None, label=None):
    """Solves the max path problem on a node-weighted graph.

    Args:
        qubo_matrix (np.ndarray): The QUBO matrix to sample from.
        time_limit (int, optional): The time limit passed to the sampler.
    """
    if label is None:
        label = "Tangle"
    sampler = LeapHybridSampler()
    
    bqm = BQM(qubo_matrix, 'BINARY')
    bqm.offset = offset
    print(f'Number of QUBO vars: {len(bqm.variables)}')
        
    if time_limit == -1:
        time_limit = sampler.min_time_limit(bqm)
        print(f"Using default min time limit: {time_limit}")
    sampleset = sampler.sample(bqm, time_limit, label=label)
 
    try:
        print(f"D-Wave access time: {round(sampleset.info['run_time'] / 10 ** 6)}")
    except:
        pass
    
    greedy_solver = SteepestDescentSolver()
    post_processed = greedy_solver.sample(bqm, initial_states=sampleset)
    
    best_sample = post_processed.first.sample
    best_energy = post_processed.first.energy
    path = dwave_sample_to_path(best_sample, graph)

    return list(best_sample.values()), best_energy, path


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


def on_vars_to_path(on_vars, nodes):
    path = [_index_to_node_time(x, len(nodes) + 1) for x in on_vars]
    path = [(e[0], nodes[e[1]] if e[1] < len(nodes) else 'end') for e in path]
    return path


def dwave_sample_to_path(sample: dict, g: nx.Graph) -> list:
    """Gets the actual path as a list of (time, node) pairs from an output of a DWave Sampler.

    Args:
        sample (dict): the qubo variables as a dict.
        g (nx.Graph): the graph underlying the problem.

    Returns:
        list: a list of (time_step, node) pairs.
    """
    on_vars = []
    for i in range(len(sample.keys())):
        if sample[i] == 1:
            on_vars.append(i)
    return on_vars_to_path(on_vars, list(g.nodes))
    


def qubo_vars_to_path(qubo_vars: list[int], g: nx.Graph) -> list:
    """Gets the actual path as a list of (time, node) pairs from an array of qubo variable values.

    Args:
        qubo_vars (list[int]): the qubo variables as a list.
        g (nx.Graph): the graph underlying the problem.

    Returns:
        list: a list of (time_step, node) pairs.
    """
    on_vars = []
    for i, var in enumerate(qubo_vars):
        if var == 1:
            on_vars.append(i)
    return on_vars_to_path(on_vars, list(g.nodes))


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
    