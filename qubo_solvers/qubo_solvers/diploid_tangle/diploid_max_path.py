import numpy as np
import sys
from datetime import datetime
from qubo_solvers.definitions import Solver
from qubo_solvers.diploid_tangle.utils.setup_utils import setup
from qubo_solvers.diploid_tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_paths

filepath, filename, diploid_out_dir, graph, time_limit, Q, offset, T_max, N, solver = setup(*sys.argv)

if solver == Solver.DWAVE:
    sample, energy, paths = dwave_sample_qubo(Q, offset, graph, T_max, N, time_limit, label=f"oriented_tangle_{filename}")
elif solver == Solver.MQLIB:
    sample, energy, paths = mqlib_sample_qubo(diploid_out_dir, filename, graph, offset, T_max, N, time_limit)
elif solver == Solver.GUROBI:
    sample, energy, paths = gurobi_sample_qubo(Q, offset, graph, T_max, N, time_limit)


validate_paths(paths, graph)
print(f"Energy of path: {energy}")

now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = diploid_out_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([sample, energy, paths], dtype=object)
np.save(save_file, to_save)
compile_path = diploid_out_dir + f"/{solver.value}.{filename}.compiled.txt"
with open(compile_path, "a") as f:
    f.write(f'[{time_limit}, {energy}],')