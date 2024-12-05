import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
from datetime import datetime
from qubo_solvers.definitions import DATA_DIR, OUT_DIR
from qubo_solvers.tangle.utils.graph_utils import graph_from_gfa_file, normalise_node_weights
from qubo_solvers.tangle.utils.sampling_utils import qubo_vars_to_path
from qubo_solvers.tangle.utils.sampling_utils import validate_path


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "test.gfa"

if len(sys.argv) > 2:
    try:
        normalisation = int(sys.argv[2])
    except ValueError:
        normalisation = 1
else:
    normalisation = 1
    
if len(sys.argv) > 3:
    try:
        time_limit = int(sys.argv[3])
    except ValueError:
        time_limit = 5
else:
    time_limit = 5

tangle_out_dir = f"{OUT_DIR}/tangle"
qubo_data_filepath = f"{tangle_out_dir}/qubo_data_{filename}.npy"

Q, offset, T_max, V = np.load(qubo_data_filepath, allow_pickle=True)
print(Q, offset, T_max, V)

graph = graph_from_gfa_file(f"{DATA_DIR}/{filename}")
print(f"Normalising by: {normalisation}")
graph = normalise_node_weights(graph, normalisation)


with gp.Env() as env, gp.Model(env=env) as model:
    model_vars = model.addMVar(shape=Q.shape[0], vtype=GRB.BINARY, name="x")
    model.setObjective(model_vars @ Q @ model_vars, GRB.MINIMIZE)
    model.Params.BestObjStop = - offset
    model.Params.TimeLimit = time_limit
    model.Params.Seed = np.random.default_rng().integers(0, 1000)
    model.optimize()
    
    energy = model.ObjVal + offset
    path = qubo_vars_to_path(model_vars.X, graph)
    validate_path(path, graph)
    print(f"Energy of path: {energy}")
    
    print('Objective value: %g' % model.ObjVal)
    print(f'Offset: {offset}')
    
        
    now = datetime.now().strftime("%d%m%Y_%H%M")
    save_file = tangle_out_dir + f"/gurobi_{filename}_{now}"   
        
    to_save = np.array([model_vars.X, energy, path], dtype=object)
    np.save(save_file, to_save)
    print('Compilation Data')
    print(f'[{time_limit}, {energy}],')