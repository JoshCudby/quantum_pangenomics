import sys
import numpy as np
import os
from qubo_solvers.definitions import DATA_DIR, OUT_DIR
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers, normalise_node_weights
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph
from qubo_solvers.pathfinder_coverage import run_pathfinder_coverage

if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = f"{DATA_DIR}/test.gfa"
    
    
coverage_suffix = "coverage"
run_pathfinder_coverage(filepath, coverage_suffix)

with open(f"{filepath}_{coverage_suffix}", "r") as f:
    lines = f.readlines()
if len(lines) < 3:
    raise Exception(f"Could not read copy numbers from {filepath}_{coverage_suffix}")
copy_numbers = [int(x) for x in lines[2].split()]
   
filename = os.path.basename(filepath)
out_dir = f"{OUT_DIR}/oriented"

graph = oriented_graph_with_copy_numbers(filepath, copy_numbers)
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