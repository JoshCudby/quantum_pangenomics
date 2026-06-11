"""CLI tool that loads a pre-built QUBO pickle and solves it with the chosen backend."""

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
    """Parse CLI arguments, load the QUBO pickle, and return a QuboDescription.

    CLI arguments:
        -f / --filepath: Path to the original ``.gfa`` file whose base name is
            used to locate the pre-built pickle (default: ``<DATA_DIR>/test.gfa``).
        -t / --times: Comma-separated list of integer solver time limits in
            seconds.
        -j / --jobs: Number of independent solver runs per time limit.
        -s / --solver: Solver name (required); must be a member of
            ``Solver`` (``dwave``, ``mqlib``, or ``gurobi``).
        -d / --data-dir: Directory containing the QUBO pickle (default:
            ``<OUT_DIR>/tangle``).

    Returns:
        QuboDescription: Fully populated descriptor ready to pass to a solver.

    Raises:
        Exception: If the solver name is not recognised.
        Exception: If the pickle file does not exist (``build_tangle_qubo_matrix``
            has not been run).
    """
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
    """Dispatch to the chosen solver, validate results, and write output files.

    Calls ``setup()`` to obtain a ``QuboDescription``, then routes to the
    appropriate sampling function (``dwave_sample_qubo``, ``mqlib_sample_qubo``,
    or ``gurobi_sample_qubo``).  Each returned path is validated against the
    graph topology via ``validate_path``, and the per-run energies are logged.

    Two files are written to ``qubo_description.data_dir``:

    * A pickle named ``<solver>_<filename>_<timestamp>`` containing the full
      ``paths`` dict mapping time limit → list of (sample, energy, path) tuples.
    * A compiled summary text file ``<solver>.<filename>.compiled.txt`` with
      lines of the form ``<time_limit>: Counter({energy: count, ...}),`` appended
      on each run.
    """
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