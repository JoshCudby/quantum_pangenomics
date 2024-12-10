import numpy as np
import os
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, Solver
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers, normalise_node_weights


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
        filepath = f"{DATA_DIR}/test.gfa"

    if len(args) > 3:
        try:
            normalisation = int(args[3])
        except ValueError:
            normalisation = 1
    else:
        normalisation = 1

    if len(args) > 4:
        try:
            time_limit = int(args[4])
        except ValueError:
            time_limit = 10
    else:
        time_limit = 10
        
    filename = os.path.basename(filepath)
    graph = oriented_graph_with_copy_numbers(filepath)
    print(f'Normalising by: {normalisation}')
    graph = normalise_node_weights(graph, normalisation)
    
    oriented_out_dir = f'{OUT_DIR}/oriented'
    to_load = f'{oriented_out_dir}/qubo_data_{filename}.npy'
    Q, offset, T_max, V = np.load(to_load, allow_pickle=True)
    return filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V, solver