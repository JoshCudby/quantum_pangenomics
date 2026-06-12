"""Command-line argument parser for QAOA simulation entry points.

Defines the standard set of CLI flags shared across QAOA simulation scripts,
including the input file path, circuit depth (reps), memory limit, solver
method, sampling shots, regularisation alpha, and flags for hardware/noisy
simulation and parameter initialisation strategy.
"""

import argparse


def get_parser():
    """Build and return the argument parser for QAOA simulation scripts.

    Returns:
        An ``argparse.ArgumentParser`` configured with the following arguments:

        - ``-f`` / ``--filename``: Path to the input data file (pickle).
        - ``-p`` / ``--reps``: Number of QAOA layers (default 4).
        - ``-m`` / ``--memory``: Memory limit in MB (default 4000).
        - ``-M`` / ``--method``: Solver/optimiser method string (default '').
        - ``-n`` / ``--shots``: Number of measurement shots (default 2000).
        - ``-a`` / ``--alpha``: Regularisation or penalty strength (default 0.25).
        - ``--hardware``: Flag to enable real-hardware execution.
        - ``--noisy``: Flag to enable noisy simulation.
        - ``--init``: Parameter initialisation strategy; one of ``'ramp'``,
          ``'random'``, or ``'fixed'`` (default ``'random'``).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename')
    parser.add_argument('-p', '--reps', type=int, default=4)
    parser.add_argument('-m', '--memory', type=int, default=4000)
    parser.add_argument('-M', '--method', type=str, default='')
    parser.add_argument('-n', '--shots', type=int, default=2000)
    parser.add_argument('-a', '--alpha', type=float, default=0.25)
    parser.add_argument('--hardware', action='store_true', default=False)
    parser.add_argument('--noisy', action='store_true', default=False)
    parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
    return parser