import numpy as np
import sys
from utils.graph_utils import dual_oriented_graph_from_file, normalise_node_weights
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
    
print(f'Making graph from: {filename}')
graph = dual_oriented_graph_from_file(f"data/{filename}")
print(f'Normalising by: {normalisation}')
graph = normalise_node_weights(graph, normalisation)

print(f'Building qubo matrix')
qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

print(f'Writing to MQLib format')
filepath = f'out/edge/mqlib_input_{filename}.txt'
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
            