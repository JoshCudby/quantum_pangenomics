import sys
import os
import numpy as np
from datetime import datetime
from qubo_solvers.definitions import DATA_DIR
from qubo_solvers.oriented_tangle.utils.setup_utils import setup
from qubo_solvers.oriented_tangle.utils.sampling_utils import dwave_sample_qubo, sample_list_to_path, validate_path
    
if len(sys.argv) > 4:
    try:
        jobs = int(sys.argv[4])
    except ValueError:
        print('Could not parse number of jobs')
        jobs = 1
else:
    jobs = 1 
   
filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V = setup(sys.argv)

for _ in range(jobs):
    sample, energy = dwave_sample_qubo(Q, offset, time_limit, label=f'oriented_{filename}')
    path = sample_list_to_path(np.array(list(sample.values())), graph, T_max, V)


    validate_path(path, graph)
    print(f"Energy of path: {energy}")
  
    now = datetime.now().strftime("%d%m%Y_%H%M")
    save_file = oriented_out_dir + f"/dwave_{filename}_{now}"   
    to_save = np.array([sample, energy, path], dtype=object)
    np.save(save_file, to_save)


    print('Compilation Data')
    print(f'[{time_limit}, {energy}],')