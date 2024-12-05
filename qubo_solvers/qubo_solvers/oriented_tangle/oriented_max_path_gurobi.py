import numpy as np
import sys
import os
from datetime import datetime
from qubo_solvers.oriented_tangle.utils.setup_utils import setup
from qubo_solvers.oriented_tangle.utils.sampling_utils import gurobi_sample_qubo, sample_list_to_path, validate_path

filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V = setup(sys.argv)

sample, energy = gurobi_sample_qubo(Q, graph, offset, T_max, time_limit)
path = sample_list_to_path(sample, graph, T_max, V)

validate_path(path, graph)
print(f"Energy of path: {energy}")

now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = oriented_out_dir + f"/gurobi_{filename}_{now}"   
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)

print('Compilation Data')
print(f'[{time_limit}, {energy}],')