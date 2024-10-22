import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import os
from datetime import datetime
from utils.qubo_utils import graph_to_max_path_digraph, get_max_path_problem_qubo_matrix
from utils.graph_utils import graph_from_gfa_file, toy_graph, normalise_node_weights
from utils.sampling_utils import qubo_vars_to_path
from utils.sampling_utils import print_path, validate_path


if len(sys.argv) > 1:
    filename = sys.argv[1]
    graph = graph_from_gfa_file(f"data/{filename}")
    
else:
    filename = 'toy'
    graph = toy_graph(exact_solution=True)

if len(sys.argv) > 2:
    try:
        normalisation = int(sys.argv[2])
    except ValueError:
        normalisation = 1
else:
    normalisation = 1
    
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)
print(list(zip(list(graph.nodes), [graph.nodes[node]["normalised_weight"] for node in graph.nodes])))
    
if len(sys.argv) > 3:
    try:
        time_limit = int(sys.argv[3])
    except ValueError:
        time_limit = 60
else:
    time_limit = 60

dg = graph_to_max_path_digraph(graph)
W = len(dg.nodes) - 1
penalty = W

qubo_matrix = get_max_path_problem_qubo_matrix(dg, penalty)
    
offset = penalty * (W + 3)

with gp.Env() as env, gp.Model(env=env) as model:
    model_vars = model.addMVar(shape=qubo_matrix.shape[0], vtype=GRB.BINARY, name="x")
    model.setObjective(model_vars @ qubo_matrix @ model_vars, GRB.MINIMIZE)
    model.Params.BestObjStop = -W - offset + 1
    model.Params.TimeLimit = time_limit
    model.Params.Seed = np.random.default_rng().integers(0, 1000)
    model.optimize()
    
    energy = model.ObjVal + offset
    path = qubo_vars_to_path(model_vars.X, dg)
    print(f"Best path:")
    print_path(path)
    validate_path(path, graph)
    print(f"Energy of path: {energy}")
    
    print('Objective value: %g' % model.ObjVal)
    print(f'Offset: {offset}')
    print(f'Best possible score: {-W - offset + 1}')
    
    save_dir = "out"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    now = datetime.now().strftime("%d%m%Y_%H%M")
    save_file = save_dir + f"/gurobi_{filename}_{now}"   
        
    to_save = np.array([model_vars.X, energy, path], dtype=object)
    np.save(save_file, to_save)
    print('Compilation Data')
    print(f'[{time_limit}, {energy}],')