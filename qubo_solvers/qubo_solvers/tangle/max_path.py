import pickle 
import os
import argparse
from collections import Counter
from datetime import datetime
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_path
from qubo_solvers.definitions import DATA_DIR, OUT_DIR, Solver, QuboDescription
from qubo_solvers.logging import get_logger


logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filepath', default=f'{DATA_DIR}/test.gfa')
parser.add_argument('-t', '--times', help='delimited list input', 
    type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('-j', '--jobs', type=int)
parser.add_argument('-s', '--solver', required=True)
parser.add_argument('-d', '--data-dir', default=f"{OUT_DIR}/tangle")


def setup() -> QuboDescription:
    args = parser.parse_args()
    
    if args.solver in set(item.value for item in Solver):
        solver = Solver(args.solver)
    else:
        raise Exception(f'Solver {args.solver} not implemented yet.')
    
    filename = os.path.basename(args.filepath)

    try:
        with open(f'{args.data_dir}/qubo_data_{filename}.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        raise Exception('Run build_tangle_qubo_matrix first!')
    
    return QuboDescription(
        filename=filename, data_dir=args.data_dir, graph=data['graph'], time_limits=args.times, 
        Q=data['Q'], offset=data['offset'], T=data['T_max'], V=data['V'], solver=solver, jobs=args.jobs
    )


def main():
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
            logger.info(f'Energy of path: {paths[time_limit][i][1]}')

        
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


if __name__ == "__main__":
    exit(main())    