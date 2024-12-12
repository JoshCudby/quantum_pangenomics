import numpy as np
import sys
from datetime import datetime
from qubo_solvers.definitions import Solver
from qubo_solvers.oriented_tangle.utils.setup_utils import setup
from qubo_solvers.oriented_tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_path

filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V, solver = setup(*sys.argv)

if solver == Solver.DWAVE:
    sample, energy, path = dwave_sample_qubo(Q, offset, graph, T_max, V, time_limit, label=f"oriented_tangle_{filename}")
elif solver == Solver.MQLIB:
    sample, energy, path = mqlib_sample_qubo(oriented_out_dir, filename, offset, graph, T_max, V, time_limit)
elif solver == Solver.GUROBI:
    sample, energy, path = gurobi_sample_qubo(Q, offset, graph, T_max, V, time_limit)


validate_path(path, graph)
print(f"Energy of path: {energy}")

now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = oriented_out_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([sample, energy, path], dtype=object)
np.save(save_file, to_save)
compile_path = oriented_out_dir + f"/{solver.value}.{filename}.compiled.txt"
with open(compile_path, "w") as f:
    f.write(f'[{time_limit}, {energy}],')