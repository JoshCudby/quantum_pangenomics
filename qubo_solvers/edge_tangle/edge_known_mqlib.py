import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from utils.graph_utils import dual_oriented_graph_from_file, normalise_node_weights
from utils.sampling_utils import validate_path, sample_list_to_path
from math import floor


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
        time_limit = 10
else:
    time_limit = 10
    

graph = dual_oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by: {normalisation}')
graph = normalise_node_weights(graph, normalisation)
nodes = list(graph.nodes)
V = int(len(nodes))
total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)

# T_max = total weight + "a bit"
alpha = 1.2
T_max = floor(total_weight * alpha)

lambda_t = 4
lambda_w = 1

offset = (T_max * lambda_t + lambda_w * sum(graph.nodes[nodes[i]]["weight"] ** 2 for i in range(0, V, 2)))

# Input in MQLib Format
filepath = f'out/edge/mqlib_input_{filename}.txt'

# Run the MQLib solver and capture output
process = subprocess.run(["../modules/MQLib/bin/MQLib", "-fQ", filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
out = process.stdout.decode("utf-8")

# First line of output includes run data. 3rd line contains the solution.
out_data = [x for x in out.split('\n') if len(x) > 0]
if not len(out_data) > 2:
    print(out_data)
    exit(1)
solution = out_data[2].split()
solution = np.array([int(x) for x in solution])
solution_energy = int(out_data[0].split(',')[3])
energy = offset - solution_energy
path = sample_list_to_path(solution, graph, T_max, V)

validate_path(path, graph)
print(f"Energy of path: {energy}")

save_dir = "out/oriented"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = save_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([solution, energy, path], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')