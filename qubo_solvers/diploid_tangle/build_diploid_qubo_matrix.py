import sys
import numpy as np
import os
from pathlib import Path
from utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from utils.qubo_utils import qubo_matrix_from_graph

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
   
filename = os.path.basename(filepath)

graph = oriented_graph_from_file(filepath)
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)
qubo_matrix, offset, T_max, N = qubo_matrix_from_graph(graph)

out_path='out/diploid'
Path(out_path).mkdir(exist_ok=True)
to_save = np.array([qubo_matrix, offset, T_max, N], dtype=object)
np_data_filepath = f'{out_path}/qubo_data_{filename}_normalisation_{normalisation}'
np.save(np_data_filepath, to_save)

# Write to MQLib Format
mqlib_data_filepath = f'{out_path}/mqlib_input_{filename}_normalisation_{normalisation}.txt'
ut_qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(ut_qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])
f = open(mqlib_data_filepath, 'w')
f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
to_write = ''
for i in range(len(non_zero[0])):
    to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-qubo_matrix[non_zero[0][i], non_zero[1][i]]} \n'
    if i % 500 == 0:
        f.write(to_write)
        to_write = ''

f.write(to_write)
f.close()