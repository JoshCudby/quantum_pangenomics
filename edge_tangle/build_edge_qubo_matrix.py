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
qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])
np_to_write = np.zeros((non_zero_count, 3))
np_to_write[:, 0] = non_zero[0] + 1
np_to_write[:, 1] = non_zero[1] + 1
np_to_write[:, 2] = -qubo_matrix[non_zero[0][:], non_zero[1][:]]
np.savetxt(filepath, np_to_write, fmt='%1d %1d %+1d' , header=f'{qubo_matrix.shape[0]} {non_zero_count}', comments='')