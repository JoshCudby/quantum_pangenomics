import sys
from pathlib import Path
import numpy as np
from datetime import datetime
from utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from utils.sampling_utils import dwave_sample_qubo, sample_list_to_paths, validate_path

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
    
if len(sys.argv) > 4:
    try:
        jobs = int(sys.argv[4])
    except ValueError:
        print('Could not parse number of jobs')
        jobs = 1
else:
    jobs = 1 
   

graph = oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)

save_dir = 'out/diploid'
to_load = f'{save_dir}/qubo_data_{filename}_normalisation_{normalisation}.npy'
qubo_matrix, offset, T_max, N = np.load(to_load, allow_pickle=True)

for _ in range(jobs):
    sample, energy = dwave_sample_qubo(qubo_matrix, offset, time_limit, label=f'diploid_{filename}')
    path = sample_list_to_paths(np.array(list(sample.values())), list(graph.nodes), T_max, N)


    validate_path(path, graph)
    print(f"Energy of path: {energy}")

    path='out/diploid'
    Path(save_dir).mkdir(exist_ok=True)
    now = datetime.now().strftime("%d%m%Y_%H%M")
    save_file = save_dir + f"/dwave_{filename}_{now}"   
    to_save = np.array([sample, energy, path], dtype=object)
    np.save(save_file, to_save)


    print('Compilation Data')
    print(f'[{time_limit}, {energy}],')