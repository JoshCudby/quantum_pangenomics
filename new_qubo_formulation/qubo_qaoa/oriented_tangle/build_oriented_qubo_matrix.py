"""CLI tool to build an orientation-aware QUBO matrix from a GFA pangenome graph.

This script reads a sequence graph in GFA format, constructs the orientation-
aware QUBO for the tangle resolution problem, and writes two output artefacts
to ``--data-dir``:

* ``qubo_data_<filename>.pkl`` — a pickle file containing the QUBO matrix
  ``Q`` (as a nested list), the Ising constant ``offset``, the maximum
  timestep count ``T_max``, the node count ``V``, and the NetworkX ``graph``.
* ``mqlib_input_<filename>.txt`` — the upper-triangular QUBO in MQLib sparse
  text format (header line ``n_vars n_nonzeros``, then rows
  ``i j -Q[i,j]`` 1-indexed with negated weights).

Usage::

    python build_oriented_qubo_matrix.py \\
        -f path/to/graph.gfa \\
        -c 1.0,2.0,1.0 \\
        -p 10,5,3 \\
        -d /output/dir
"""
import numpy as np
import pickle
import os
import argparse
from pathlib import Path
from qubo_qaoa.oriented_tangle.utils.graph_utils import oriented_graph_with_copy_numbers
from qubo_qaoa.oriented_tangle.utils.qubo_utils import qubo_matrix_from_graph
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Build and save the QUBO matrix for an oriented pangenome tangle.

    Parses CLI arguments, constructs the orientation-aware directed graph from
    the input GFA file with the given copy numbers, computes the QUBO matrix
    via ``qubo_matrix_from_graph`` using the supplied penalty weights, and
    serialises the results to ``--data-dir``.

    CLI Arguments:
        -f / --filepath: Path to the input ``.gfa`` graph file.
        -c / --copy-numbers: Comma-separated list of copy numbers (floats),
            one per segment in the GFA file, in GFA segment order.
        -p / --penalties: Comma-separated list of three integer penalty
            weights ``[lambda_t, lambda_g, lambda_w]`` used to enforce the
            timestep, graph-consistency, and walk constraints in the QUBO.
        -d / --data-dir: Output directory.  Created if it does not exist.

    Output files written to ``--data-dir``:
        qubo_data_<basename>.pkl: Pickle dict with keys ``Q`` (list of lists),
            ``offset`` (float), ``T_max`` (int), ``V`` (int), ``graph``
            (``nx.DiGraph``).
        mqlib_input_<basename>.txt: MQLib sparse QUBO format with negated
            upper-triangular entries.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath')
    parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
        type=lambda s: [float(item) for item in s.split(',') if len(item)])
    parser.add_argument('-p', '--penalties', help='delimited list input', 
        type=lambda s: [int(item) for item in s.split(',') if len(item)])
    parser.add_argument('-d', '--data-dir')

    args = parser.parse_args()

    logger.info(f'Building qubo matrix for {args.filepath}')
    filename = os.path.basename(args.filepath)
    Path(args.data_dir).mkdir(exist_ok=True, parents=True)


    # logger.info(f'Copy numbers provided: {args.copy_numbers}')
    copy_numbers = args.copy_numbers
    nodes = None

    # copy_numbers = [max(int(x), 1) for x in copy_numbers]
    logger.info(f'Copy numbers: {copy_numbers}')


    logger.info(f'Getting graph from {args.filepath}')
    graph = oriented_graph_with_copy_numbers(args.filepath, copy_numbers, nodes)
    Q, offset, T_max, V = qubo_matrix_from_graph(graph, penalties=args.penalties)
    
    logger.info('Saving data')
    savepath = f'{args.data_dir}/qubo_data_{filename}.pkl'
    to_save = {
        'Q': Q.tolist(),
        'offset': offset,
        'T_max': T_max,
        'V': V,
        'graph': graph
    }
    with open(savepath, 'wb') as file:
        pickle.dump(to_save, file)


    logger.info('Writing to MQLib format')
    # Write to MQLib Format
    mqlib_filepath = f'{args.data_dir}/mqlib_input_{filename}.txt'

    ut_qubo_matrix = np.triu(Q)
    non_zero = np.nonzero(ut_qubo_matrix)
    non_zero_count = int(non_zero[0].shape[0])

    with open(mqlib_filepath, 'w') as f:
        f.write(f'{Q.shape[0]} {non_zero_count}\n')
        to_write = ''
        for i in range(len(non_zero[0])):
            to_write += f'{non_zero[0][i] + 1} {non_zero[1][i] + 1} {-Q[non_zero[0][i], non_zero[1][i]]} \n'
            if i % 500 == 0:
                f.write(to_write)
                to_write = ''

        f.write(to_write)
    logger.info('Finished building oriented qubo matrix')


if __name__ == "__main__":
    exit(main())