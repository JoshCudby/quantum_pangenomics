import sys
import os
import numpy as np
from datetime import datetime
from qubo_solvers.tangle.utils.setup_utils import setup
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, validate_path

filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V = setup(sys.argv)

sample, energy, path = dwave_sample_qubo(Q, offset, time_limit, label=f"tangle_{filename}")

validate_path(path, graph)
print(f"Energy of path: {energy}")
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = tangle_out_dir + f"/dwave_{filename}_{now}"   
    
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)

print('Compilation Data')
print(f'[{time_limit}, {energy}],')