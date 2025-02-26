import numpy as np
import os
import pickle
import argparse
from pathlib import Path
from qubo_solvers.tangle.utils.graph_utils import graph_with_copy_numbers
from qubo_solvers.tangle.utils.qubo_utils import get_tangle_qubo_matrix
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, COVERAGE_SUFFIX
from qubo_solvers.pathfinder_coverage import run_pathfinder_coverage
from qubo_solvers.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', default=f'{DATA_DIR}/test.gfa')
    parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-d', '--data-dir', default=f"{OUT_DIR}/tangle")

    args = parser.parse_args()

    logger.info(f'Building qubo matrix for {args.filepath}')

    filename = os.path.basename(args.filepath)
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    
    if args.copy_numbers is None:
        logger.info('Running pathfinder to get coverage')
        copy_numbers, nodes = run_pathfinder_coverage(args.data_dir, args.filepath, COVERAGE_SUFFIX)
    else:
        logger.info(f'Copy numbers provided: {args.copy_numbers}')
        copy_numbers = args.copy_numbers
        nodes = None

    graph = graph_with_copy_numbers(args.filepath, copy_numbers, nodes)

    Q, offset, T_max, V = get_tangle_qubo_matrix(graph)


    savepath = f'{args.data_dir}/qubo_data_{filename}.pkl'

    to_save = {
        'Q': Q.tolist(),
        'offset': offset,
        'T_max': T_max,
        'V': V,
        'graph': graph
    }
    with open(savepath, 'wb') as file:
        pickle.dump(to_save, file)

    logger.info('Writing to MQLib format')
    mqlib_savepath = f'{args.data_dir}/mqlib_input_{filename}.txt'

    ut_qubo_matrix = np.triu(Q)
    non_zero = np.nonzero(ut_qubo_matrix)
    non_zero_count = int(non_zero[0].shape[0])
    f = open(mqlib_savepath, 'w')
    f.write(f'{Q.shape[0]} {non_zero_count}\n')
    to_write = ''
    for i in range(len(non_zero[0])):
        to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-Q[non_zero[0][i], non_zero[1][i]]} \n'
        if i % 500 == 0:
            f.write(to_write)
            to_write = ''

    f.write(to_write)
    f.close()

    return 0


if __name__ == "__main__":
    exit(main())