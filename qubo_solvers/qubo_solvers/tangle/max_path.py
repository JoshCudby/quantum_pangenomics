import pickle 
from collections import Counter
from datetime import datetime
from qubo_solvers.definitions import Solver
from qubo_solvers.tangle.utils.setup_utils import setup
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_path

# TODO: move setup here?
qubo_description = setup()

if qubo_description.solver == Solver.DWAVE:
    paths = dwave_sample_qubo(qubo_description)
elif qubo_description.solver == Solver.MQLIB:
    paths = mqlib_sample_qubo(qubo_description)
elif qubo_description.solver == Solver.GUROBI:
    paths = gurobi_sample_qubo(qubo_description)

for time_limit in qubo_description.time_limits:
    for i in range(qubo_description.jobs):
        validate_path(paths[time_limit][i][2], qubo_description.graph)
        print(f'Energy of path: {paths[time_limit][i][1]}')

    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = qubo_description.data_dir + f'/{qubo_description.solver.value}_{qubo_description.filename}_{now}'   


with open(save_file, 'wb') as f:
    pickle.dump(paths, f)
    
compile_path = qubo_description.data_dir + f'/{qubo_description.solver.value}.{qubo_description.filename}.compiled.txt'
counts = {
    time_limit: Counter([float(paths[time_limit][i][1]) for i in range(len(paths[time_limit]))]) for time_limit in qubo_description.time_limits
}
with open(compile_path, 'a') as f:
    for time_limit in qubo_description.time_limits:
        f.write(f'{time_limit}: {counts[time_limit]},')
        