import sys
import numpy as np
import os
from qubo_solvers.definitions import DATA_DIR, OUT_DIR
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph

if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = f"{DATA_DIR}/test.gfa"

if len(sys.argv) > 2:
    try:
        normalisation = int(sys.argv[2])
    except ValueError:
        normalisation = 1
else:
    normalisation = 1
   
filename = os.path.basename(filepath)
out_dir = f"{OUT_DIR}/oriented"

graph = oriented_graph_from_file(filepath)
print(f"Normalising by {normalisation}")
graph = normalise_node_weights(graph, normalisation)
qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

to_save = np.array([qubo_matrix, offset, T_max, V], dtype=object)
filepath = f"{out_dir}/qubo_data_{filename}"
np.save(filepath, to_save, allow_pickle=True)

# Write to MQLib Format
filepath = f"{out_dir}/mqlib_input_{filename}.txt"
ut_qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(ut_qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])
f = open(filepath, "w")
f.write(f"{qubo_matrix.shape[0]} {non_zero_count}\n")
to_write = ""
for i in range(len(non_zero[0])):
    to_write += f"{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-qubo_matrix[non_zero[0][i], non_zero[1][i]]} \n"
    if i % 500 == 0:
        f.write(to_write)
        to_write = ""

f.write(to_write)
f.close()