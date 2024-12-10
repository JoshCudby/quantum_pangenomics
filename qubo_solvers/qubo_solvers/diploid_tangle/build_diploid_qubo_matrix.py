import sys
import numpy as np
import os
from pathlib import Path
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, COVERAGE_SUFFIX
from qubo_solvers.pathfinder_coverage import run_pathfinder_coverage
from qubo_solvers.diploid_tangle.utils.graph_utils import oriented_graph_with_copy_numbers
from qubo_solvers.diploid_tangle.utils.qubo_utils import qubo_matrix_from_graph

print("Started Building Matrix")
if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = f"{DATA_DIR}/test.gfa"


run_pathfinder_coverage(filepath, COVERAGE_SUFFIX)

with open(f"{filepath}_{COVERAGE_SUFFIX}", "r") as f:
    lines = f.readlines()
if len(lines) < 3:
    raise Exception(f"Could not read copy numbers from {filepath}_{COVERAGE_SUFFIX}")
copy_numbers = [int(x) for x in lines[2].split()]

filename = os.path.basename(filepath)

graph = oriented_graph_with_copy_numbers(filepath, copy_numbers)

qubo_matrix, offset, T_max, N = qubo_matrix_from_graph(graph)

out_path=f'{OUT_DIR}/diploid'
Path(out_path).mkdir(exist_ok=True)
to_save = np.array([qubo_matrix, offset, T_max, N], dtype=object)
np_data_filepath = f'{out_path}/qubo_data_{filename}'
np.save(np_data_filepath, to_save)

# Write to MQLib Format
mqlib_data_filepath = f'{out_path}/mqlib_input_{filename}.txt'
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