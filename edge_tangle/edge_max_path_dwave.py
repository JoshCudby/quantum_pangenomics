import sys
import os
import numpy as np
from datetime import datetime
from utils.graph_utils import dual_oriented_graph_from_file, normalise_node_weights
from utils.qubo_utils import qubo_matrix_from_graph
from utils.sampling_utils import dwave_sample_qubo, sample_list_to_path, validate_path

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
        print('Could not parse quantum time limit')
        time_limit = None
else:
    time_limit = None
   

graph = dual_oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)
qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)


sample, energy = dwave_sample_qubo(qubo_matrix, offset, time_limit, label=f'edge_{filename}')
path = sample_list_to_path(np.array(list(sample.values())), graph, T_max, V)


validate_path(path, graph)
print(f"Energy of path: {energy}")


save_dir = "out/edge"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/dwave_{filename}_{now}"   
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)


print('Compilation Data')
print(f'[{time_limit}, {energy}],')