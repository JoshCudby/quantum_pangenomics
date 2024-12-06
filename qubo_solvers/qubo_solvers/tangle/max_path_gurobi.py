import numpy as np
import sys
from datetime import datetime
from qubo_solvers.tangle.utils.setup_utils import setup
from qubo_solvers.tangle.utils.sampling_utils import gurobi_sample_qubo
from qubo_solvers.tangle.utils.sampling_utils import validate_path

filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V = setup(sys.argv)
energy, path, solution = gurobi_sample_qubo(Q, graph, offset, time_limit)

validate_path(path, graph)
print(f"Energy of path: {energy}")
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = tangle_out_dir + f"/gurobi_{filename}_{now}"   
    
to_save = np.array([solution, energy, path], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')