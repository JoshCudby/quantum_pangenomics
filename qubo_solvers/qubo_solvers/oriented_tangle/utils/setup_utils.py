import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, MQLIB_DIR
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from qubo_solvers.oriented_tangle.utils.sampling_utils import validate_path, sample_list_to_path

def setup(*args):
    if len(args) > 1:
        filepath = args[1]
    else:
        filepath = f"{DATA_DIR}/test.gfa"

    if len(args) > 2:
        try:
            normalisation = int(args[2])
        except ValueError:
            normalisation = 1
    else:
        normalisation = 1

    if len(args) > 3:
        try:
            time_limit = int(args[3])
        except ValueError:
            time_limit = 10
    else:
        time_limit = 10
        
    filename = os.path.basename(filepath)
    graph = oriented_graph_from_file(filepath)
    print(f'Normalising by: {normalisation}')
    graph = normalise_node_weights(graph, normalisation)
    
    oriented_out_dir = f'{OUT_DIR}/oriented'
    to_load = f'{oriented_out_dir}/qubo_data_{filename}.npy'
    Q, offset, T_max, V = np.load(to_load, allow_pickle=True)
    return filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V