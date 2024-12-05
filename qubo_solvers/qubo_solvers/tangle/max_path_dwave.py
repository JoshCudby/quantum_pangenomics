import sys
import os
import numpy as np
from datetime import datetime
from dwave.system import LeapHybridSampler
from dimod.reference import SimulatedAnnealingSampler
from utils.qubo_utils import dwave_sample_max_path_problem
from utils.graph_utils import graph_from_gfa_file, toy_graph, normalise_node_weights
from utils.sampling_utils import print_path, validate_path

if len(sys.argv) > 1:
    filename = sys.argv[1]
    graph = graph_from_gfa_file(f"data/{filename}")

else:
    graph = toy_graph(exact_solution=False)

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
        print('Could not parse quantum time limit')
        time_limit = None
else:
    time_limit = None


if len(sys.argv) > 4 and sys.argv[4] == 'q':
    solver = "quantum"
    sampler = LeapHybridSampler()
    print("Using Leap Hybrid Solver")
else:
    solver = "classical"
    sampler = SimulatedAnnealingSampler()
    print("Using Classical Solver")    

sample, energy, path = dwave_sample_max_path_problem(graph, sampler, time_limit=time_limit)

print(f"Best path:")
print_path(path)
validate_path(path, graph)
print(f"Energy of path: {energy}")

save_dir = "out"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/dwave_{solver}_{filename}_{now}"   
    
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)

print('Compilation Data')
print(f'[{time_limit}, {energy}],')