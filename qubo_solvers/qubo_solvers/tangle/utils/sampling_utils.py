import numpy as np
import networkx as nx
import gurobipy as gp
import subprocess
from gurobipy import GRB
from math import floor
from greedy import SteepestDescentSolver
from dimod import BQM
from dwave.system import LeapHybridSampler
from qubo_solvers.definitions import MQLIB_DIR, QuboDescription
from qubo_solvers.logging import get_logger

rng = np.random.default_rng()

logger = get_logger(__name__)

def mqlib_sample_qubo(qubo_description: QuboDescription):
    input_filepath = f"{qubo_description.data_dir}/mqlib_input_{qubo_description.filename}.txt"

    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        
        for _ in range(qubo_description.jobs):
            # Run the MQLib solver and capture output
            process = subprocess.run([f"{MQLIB_DIR}/bin/MQLib", "-fQ", input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-s", str(rng.integers(0, 65535)), "-ps"], capture_output=True)
            out = process.stdout.decode("utf-8")

            try:
                # First line of output includes run data. 3rd line contains the solution.
                out_data = [x for x in out.split('\n') if len(x) > 0]
                solution = out_data[2].split()
                solution = [int(x) for x in solution]
                solution_energy = float(out_data[0].split(',')[3])
                energy = qubo_description.offset - solution_energy
                path = qubo_vars_to_path(solution, qubo_description.graph)
                paths[time_limit].append((solution, energy, path))
            except ValueError:
                logger.error('Could not parse mqlib data')
                logger.error(out)
                paths[time_limit].append(([], np.inf, []))
            
    return paths


def gurobi_sample_qubo(qubo_description: QuboDescription):
    total_weight = int(sum(qubo_description.graph.nodes[node]["weight"] for node in list(qubo_description.graph.nodes)) / 2)
    
    logger.info(f'Offset: {qubo_description.offset}')
    logger.info(f'Total weight: {total_weight}')
    logger.info(f'T_max: {qubo_description.T}')
    
    paths = {}
    with gp.Env() as env, gp.Model(env=env) as model:
        model_vars = model.addMVar(shape=qubo_description.Q.shape[0], vtype=GRB.BINARY, name="x")
        model.setObjective(model_vars @ qubo_description.Q @ model_vars, GRB.MINIMIZE)
        model.Params.BestObjStop = - qubo_description.offset
        
        for time_limit in qubo_description.time_limits:
            paths[time_limit] = []
            model.Params.TimeLimit = time_limit
            for _ in range(qubo_description.jobs):
                model.Params.Seed = rng.integers(0, 100000)
                model.reset()
                model.optimize()
                energy = model.ObjVal + qubo_description.offset
                path = qubo_vars_to_path(model_vars.X, qubo_description.graph)
                paths[time_limit].append((model_vars.X, energy, path))
    
    return paths    


def dwave_sample_qubo(qubo_description: QuboDescription):
    """Solves the max path problem on a node-weighted graph.

    Args:
        qubo_matrix (np.ndarray): The QUBO matrix to sample from.
        time_limit (int, optional): The time limit passed to the sampler.
    """
    bqm = BQM(qubo_description.Q, 'BINARY')
    bqm.offset = qubo_description.offset
    sampler = LeapHybridSampler()
    
    paths = {}
    for time_limit in qubo_description.time_limits:
        paths[time_limit] = []
        for _ in range(qubo_description.jobs):
            sampleset = sampler.sample(bqm, time_limit, label=f'tangle_{qubo_description.filename}')
            try:
                logger.info(f"D-Wave access time: {round(sampleset.info['run_time'] / 10 ** 6)}")
            except KeyError:
                pass
            greedy_solver = SteepestDescentSolver()
            post_processed = greedy_solver.sample(bqm, initial_states=sampleset)

            best_sample = post_processed.first.sample
            best_energy = post_processed.first.energy
            path = dwave_sample_to_path(best_sample, qubo_description.graph)
            paths[time_limit].append((best_sample, best_energy, path))
            
    return paths


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
    path = [(int(e[0]), nodes[e[1]] if e[1] < len(nodes) else 'end') for e in path]
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
        logger.info(path)
        return
    for i in range(floor(len(path) / num_per_line)):
        logger.info(path[i * num_per_line: (i + 1) * num_per_line])
    if not (i + 1) * num_per_line == len(path) - 1:
        logger.info(path[(i + 1)*num_per_line:])

        
        
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
    logger.info("Best path:")
    print_path(path)
    if len(path) == 0:
        logger.info("No path")
        return
    
    end_nodes = set()
    start_nodes = set()
    for node, val in dict(graph.nodes.data('start')).items():
        if val == 'end':
            end_nodes.add(node)
        elif val == 'start':
            start_nodes.add(node)
    if len(end_nodes):
        end_nodes.add('end')
    
    if len(start_nodes) > 0 and path[0][1] not in start_nodes:
        logger.info('Did not start at start')
    
    time_offset = 0
    i = 0
    while i < len(path):
        if i + time_offset == path[i][0]:
            i += 1
            continue
        if path[i][0] < i + time_offset:
            logger.info(f'Extra node at time {path[i][0]}')
            time_offset -= 1
            i += 1
            continue
        if path[i][0] > i + time_offset:
            logger.info(f'Skipped time {path[i][0] - 1}')
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
            logger.info(f'Left the end node at path entry {x}')
        elif (not v1 == 'end') and (not v2 == 'end') and ((v1, v2) not in graph.edges):
            logger.info(f'Broke graph edge at path entry {x}')
        elif len(end_nodes) > 0 and (v2 == 'end') and (v1 not in end_nodes):
            logger.info(f'Went to end node illegally at path entry {x}')
    node_dict[v2] += 1
    
    nodes = list(graph.nodes)
    for i in range(len(nodes)):
        visits = node_dict[nodes[i]]
        missing_visits = graph.nodes[nodes[i]]["weight"] - visits
        if  missing_visits != 0:
            logger.info(f'Did not meet node weight for node: {nodes[i]}. Missing visits: {missing_visits}')
    