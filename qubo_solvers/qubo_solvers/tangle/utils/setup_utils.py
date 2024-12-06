import os
import numpy as np
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, Solver
from qubo_solvers.tangle.utils.graph_utils import graph_from_gfa_file, normalise_node_weights


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
            print('Could not parse time limit')
            time_limit = 5
    else:
        time_limit = 5
        
    filename = os.path.basename(filepath)

    tangle_out_dir = f"{OUT_DIR}/tangle"
    qubo_data_filepath = f"{tangle_out_dir}/qubo_data_{filename}.npy"

    Q, offset, T_max, V = np.load(qubo_data_filepath, allow_pickle=True)
    graph = graph_from_gfa_file(f"{DATA_DIR}/{filename}")

    print(f"Normalising by: {normalisation}")
    graph = normalise_node_weights(graph, normalisation)
    
    return filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V, solver