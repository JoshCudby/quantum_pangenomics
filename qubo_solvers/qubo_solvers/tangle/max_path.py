import sys
import numpy as np
from datetime import datetime
from qubo_solvers.definitions import Solver
from qubo_solvers.tangle.utils.setup_utils import setup
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_path

filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V, solver = setup(*sys.argv)

if solver == Solver.DWAVE:
    sample, energy, path = dwave_sample_qubo(Q, offset, graph, time_limit, label=f"tangle_{filename}")
elif solver == Solver.MQLIB:
    sample, energy, path = mqlib_sample_qubo(tangle_out_dir, filename, offset, graph, time_limit)
elif solver == Solver.GUROBI:
    sample, energy, path = gurobi_sample_qubo(Q, graph, offset, time_limit)

validate_path(path, graph)
print(f"Energy of path: {energy}")
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = tangle_out_dir + f"/{solver.value}_{filename}_{now}"   
    
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)

print('Compilation Data')
print(f'[{time_limit}, {energy}],')