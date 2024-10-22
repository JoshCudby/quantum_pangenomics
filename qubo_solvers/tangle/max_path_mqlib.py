import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from utils.sampling_utils import qubo_vars_to_path
from utils.graph_utils import graph_from_gfa_file, normalise_node_weights
from utils.qubo_utils import get_max_path_problem_qubo_matrix, graph_to_max_path_digraph
from utils.sampling_utils import print_path, validate_path


if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise Exception('No MQLib file to read')

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
    
filepath = f'out/mqlib_input_{filename}.txt'

graph = graph_from_gfa_file(f"data/{filename}")

print(f'Normalising by: {normalisation}')
graph = normalise_node_weights(graph, normalisation)

dg = graph_to_max_path_digraph(graph)
W = len(dg.nodes) - 1
penalty = W
offset = W * (W + 3)

qubo_matrix = get_max_path_problem_qubo_matrix(dg, penalty)

# Write to MQLib Format
non_zero = np.nonzero(qubo_matrix)
non_zero_count = int(non_zero[0].shape[0] / 2 + qubo_matrix.shape[0] / 2)
f = open(filepath, 'w')
f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
for i in range(qubo_matrix.shape[0]):
    for j in range(i, qubo_matrix.shape[0]):
        if not qubo_matrix[i, j] == 0: 
            f.write(f'{i + 1} {j + 1} {-qubo_matrix[i, j]}\n')
f.close()
            

# Run the MQLib solver and capture output
process = subprocess.run(["MQLib/bin/MQLib", "-fQ", filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
out = process.stdout.decode("utf-8")

# First line of output includes run data. 3rd line contains the solution.
out_data = [x for x in out.split('\n') if len(x) > 0]
solution = out_data[2].split()
solution = [int(x) for x in solution]
solution_energy = int(out_data[0].split(',')[3])
energy = offset - solution_energy

path = qubo_vars_to_path(solution, dg)
print(f"Best path:")
print_path(path)
validate_path(path, graph)
print(f"Energy of path: {energy}")

save_dir = "out"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([solution, energy, path], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')