import sys
import numpy as np
import os
from pathlib import Path
from qubo_solvers.tangle.utils.graph_utils import graph_with_copy_numbers
from qubo_solvers.tangle.utils.qubo_utils import get_tangle_qubo_matrix
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, COVERAGE_SUFFIX
from qubo_solvers.pathfinder_coverage import run_pathfinder_coverage

if len(sys.argv) > 1:
    filepath = sys.argv[1]
    filename = os.path.basename(filepath)
else:
    filename = "test.gfa"
    filepath = f"{DATA_DIR}/{filename}"
    
run_pathfinder_coverage(filepath, COVERAGE_SUFFIX)

with open(f"{filepath}_{COVERAGE_SUFFIX}", "r") as f:
    lines = f.readlines()
if len(lines) < 3:
    raise Exception(f"Could not read copy numbers from {filepath}_{COVERAGE_SUFFIX}")
copy_numbers = [int(x) for x in lines[2].split()]

graph = graph_with_copy_numbers(filepath, copy_numbers)

qubo_matrix, offset, T_max, V = get_tangle_qubo_matrix(graph)

tangle_out_dir = f"{OUT_DIR}/tangle"
Path(tangle_out_dir).mkdir(exist_ok=True, parents=True)

savepath = f'{tangle_out_dir}/qubo_data_{filename}'
to_save = np.array([qubo_matrix, offset, T_max, V], dtype=object)
np.save(savepath, to_save, allow_pickle=True)

# Write to MQLib Format
mqlib_savepath = f'{tangle_out_dir}/mqlib_input_{filename}.txt'

ut_qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(ut_qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])
f = open(mqlib_savepath, 'w')
f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
to_write = ''
for i in range(len(non_zero[0])):
    to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-qubo_matrix[non_zero[0][i], non_zero[1][i]]} \n'
    if i % 500 == 0:
        f.write(to_write)
        to_write = ''

f.write(to_write)
f.close()