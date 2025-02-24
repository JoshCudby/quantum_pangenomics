import sys
import numpy as np
import os
from pathlib import Path
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, COVERAGE_SUFFIX
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers
from qubo_solvers.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph
from qubo_solvers.pathfinder_coverage import run_pathfinder_coverage
from qubo_solvers.logging import get_logger

logger = get_logger(__name__)


if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = f'{DATA_DIR}/test.gfa'

if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = f'{OUT_DIR}/oriented'
Path(out_dir).mkdir(exist_ok=True, parents=True)

    
filename = os.path.basename(filepath)

logger.info(f'Getting coverage from {filepath}')
copy_numbers = run_pathfinder_coverage(out_dir, filepath, COVERAGE_SUFFIX)
logger.info(f'Copy numbers: {copy_numbers}')

filename = os.path.basename(filepath)

logger.info(f'Getting graph from {filepath}')
graph = oriented_graph_with_copy_numbers(filepath, copy_numbers)
qubo_matrix, offset, T_max, V = qubo_matrix_from_graph(graph)

logger.info('Saving np data')
to_save = np.array([qubo_matrix, offset, T_max, V], dtype=object)
filepath = f'{out_dir}/qubo_data_{filename}'
np.save(filepath, to_save, allow_pickle=True)

logger.info('Writing to MQLib format')
# Write to MQLib Format
mqlib_filepath = f'{out_dir}/mqlib_input_{filename}.txt'

ut_qubo_matrix = np.triu(qubo_matrix)
non_zero = np.nonzero(ut_qubo_matrix)
non_zero_count = int(non_zero[0].shape[0])

with open(mqlib_filepath, 'w') as f:
    f.write(f'{qubo_matrix.shape[0]} {non_zero_count}\n')
    to_write = ''
    for i in range(len(non_zero[0])):
        to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-qubo_matrix[non_zero[0][i], non_zero[1][i]]} \n'
        if i % 500 == 0:
            f.write(to_write)
            to_write = ''

    f.write(to_write)