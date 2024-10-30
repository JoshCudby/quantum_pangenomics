import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from utils.sampling_utils import validate_paths, sample_list_to_paths, print_paths_to_perl_format


if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = "test.gfa"

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
        time_limit = 10
else:
    time_limit = 10

filename = os.path.basename(filepath)

graph = oriented_graph_from_file(filepath)
print(f'Normalising by: {normalisation}')
graph = normalise_node_weights(graph, normalisation)

save_dir = "out/diploid"
to_load = f'{save_dir}/qubo_data_{filename}_normalisation_{normalisation}.npy'
_, offset, T_max, N = np.load(to_load, allow_pickle=True)
           
mqlib_input_filepath = f'{save_dir}/mqlib_input_{filename}_normalisation_{normalisation}.txt'

seed =  np.random.default_rng().integers(0, 1000)

# TODO: call this from shell script?
# Run the MQLib solver and capture output
process = subprocess.run(["MQLib/bin/MQLib", "-fQ", mqlib_input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps", "-s", str(seed)], capture_output=True)
out = process.stdout.decode("utf-8")

# First line of output includes run data. 3rd line contains the solution.
out_data = [x for x in out.split('\n') if len(x) > 0]
solution = out_data[2].split()
solution = np.array([int(x) for x in solution])
solution_energy = int(out_data[0].split(',')[3])
energy = offset - solution_energy
paths = sample_list_to_paths(solution, list(graph.nodes), T_max, N)

validate_paths(paths, graph)
print(f"Energy of paths: {energy}")

print_paths_to_perl_format(paths)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/mqlib_output_{filename}_normalisation_{normalisation}_{now}"   
    
to_save = np.array([solution, energy, paths], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')