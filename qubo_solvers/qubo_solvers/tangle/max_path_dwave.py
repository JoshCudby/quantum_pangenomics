import sys
import os
import numpy as np
from datetime import datetime
from qubo_solvers.definitions import DATA_DIR, OUT_DIR
from qubo_solvers.tangle.utils.graph_utils import graph_from_gfa_file, toy_graph, normalise_node_weights
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, dwave_sample_to_path, print_path, validate_path

if len(sys.argv) > 1:
    filename = sys.argv[1]
    graph = graph_from_gfa_file(f"{DATA_DIR}/{filename}")

else:
    graph = toy_graph(exact_solution=False)

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
        print('Could not parse quantum time limit')
        time_limit = None
else:
    time_limit = None


tangle_out_dir = f"{OUT_DIR}/tangle"
qubo_data_filepath = f"{tangle_out_dir}/qubo_data_{filename}.npy"

Q, offset, T_max, V = np.load(qubo_data_filepath, allow_pickle=True)
graph = graph_from_gfa_file(f"{DATA_DIR}/{filename}")

print(f"Normalising by: {normalisation}")
graph = normalise_node_weights(graph, normalisation)

sample, energy = dwave_sample_qubo(Q, offset, time_limit, label=f"tangle_{filename}")
path = dwave_sample_to_path(sample, graph)

validate_path(path, graph)
print(f"Energy of path: {energy}")
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = tangle_out_dir + f"/dwave_{filename}_{now}"   
    
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)

print('Compilation Data')
print(f'[{time_limit}, {energy}],')