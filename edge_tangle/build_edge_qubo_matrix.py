import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from utils.graph_utils import dual_oriented_graph_from_file, normalise_node_weights
from utils.qubo_utils import qubo_matrix_from_graph
from utils.sampling_utils import validate_path, sample_list_to_path


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
    

graph = dual_oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by: {normalisation}')
graph = normalise_node_weights(graph, normalisation)

qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

# Write to MQLib Format
filepath = f'out/edge/mqlib_input_{filename}.txt'
non_zero = np.nonzero(qubo_matrix)
non_zero_count = int(non_zero[0].shape[0] / 2 + qubo_matrix.shape[0] / 2)
f = open(filepath, 'w')
f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
to_write = ''
for i in range(qubo_matrix.shape[0]):
    for j in range(i, qubo_matrix.shape[0]):
        if not qubo_matrix[i, j] == 0: 
            to_write += f'{i + 1} {j + 1} {-qubo_matrix[i, j]}\n'

f.write(to_write)
f.close()
            