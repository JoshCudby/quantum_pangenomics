import numpy as np
import os
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, Solver, COVERAGE_SUFFIX
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers


def setup(*args):
    if len(args) > 1:
        if args[1] in set(item.value for item in Solver):
            solver = Solver(args[1])
        else:
            raise Exception(f'Solver {args[1]} not implemented yet.')
    else:
        raise Exception('No solver specified.')
    
    if len(args) > 2:
        filepath = args[2]
    else:
        filepath = f'{DATA_DIR}/test.gfa'
    filename = os.path.basename(filepath)


    if len(args) > 3:
        try:
            time_limit = int(args[3])
        except ValueError:
            time_limit = 10
    else:
        time_limit = 10

    if len(args) > 4:
        qubo_data_dir = args[4]
    else:
        qubo_data_dir = f'{OUT_DIR}/oriented'
    qubo_data_path = f'{qubo_data_dir}/qubo_data_{filename}.npy'
        
    with open(f'{qubo_data_dir}/{filename}_{COVERAGE_SUFFIX}', 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise Exception(f'Could not read copy numbers from {filepath}_{COVERAGE_SUFFIX}')
    copy_numbers = [int(x) for x in lines[2].split()]
        
    graph = oriented_graph_with_copy_numbers(filepath, copy_numbers)
    
    Q, offset, T_max, V = np.load(qubo_data_path, allow_pickle=True)

    return filepath, filename, qubo_data_dir, graph, time_limit, Q, offset, T_max, V, solver