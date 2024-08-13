import gurobipy as gp
from gurobipy import GRB
import numpy as np
import sys
import os
from datetime import datetime
from utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from utils.qubo_utils import qubo_matrix_from_graph
from utils.sampling_utils import gurobi_sample_qubo, sample_list_to_path, validate_path

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
        time_limit = 60
else:
    time_limit = 60


graph = oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)

qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

sample, energy = gurobi_sample_qubo(qubo_matrix, graph, offset, T_max, time_limit)
path = sample_list_to_path(sample, graph, T_max, V)


validate_path(path, graph)
print(f"Energy of path: {energy}")


save_dir = "out/oriented"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/gurobi_{filename}_{now}"   
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)


print('Compilation Data')
print(f'[{time_limit}, {energy}],')