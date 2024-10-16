import sys
import numpy as np
from pathlib import Path
from utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from utils.qubo_utils import qubo_matrix_from_graph

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
   

graph = oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by {normalisation}')
graph = normalise_node_weights(graph, normalisation)
qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

path='out/diploid'
Path(path).mkdir(exist_ok=True)
to_save = np.array([qubo_matrix, offset, T_max, V], dtype=object)
filepath = f'{path}/qubo_data_{filename}_normalisation_{normalisation}'
np.save(filepath, to_save)

# Write to MQLib Format
filepath = f'out/diploid/mqlib_input_{filename}_normalisation_{normalisation}.txt'
ut_qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(ut_qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])
f = open(filepath, 'w')
f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
to_write = ''
for i in range(len(non_zero[0])):
    to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-qubo_matrix[non_zero[0][i], non_zero[1][i]]} \n'
    if i % 500 == 0:
        f.write(to_write)
        to_write = ''

f.write(to_write)
f.close()