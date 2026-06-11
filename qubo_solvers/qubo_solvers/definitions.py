"""Core constants, enumerations, and dataclasses shared across all qubo_solvers submodules."""

import os
from dataclasses import dataclass
from networkx import Graph
from numpy import ndarray
from pathlib import Path
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = Path(ROOT_DIR).parent.absolute()
MQLIB_DIR = os.path.join(PARENT_DIR, 'MQLib')
DATA_DIR = '/lustre/scratch127/qpg/jc59/data'
OUT_DIR = '/lustre/scratch127/qpg/jc59/out'
SCRATCH_DIR = '/lustre/scratch127/qpg/jc59'
QUANTUM_DIR = '/nfs/users/nfs_j/jc59/quantumwork'

class Solver(Enum):
    """Enumeration of supported QUBO solvers.

    Members:
        DWAVE: D-Wave quantum annealer, accessed via the D-Wave Ocean SDK.
        MQLIB: MQLib heuristic Max-Cut / QUBO solver (classical, CPU-based).
        GUROBI: Gurobi mixed-integer programming solver (classical, commercial).
    """

    DWAVE = 'dwave'
    MQLIB = 'mqlib'
    GUROBI = 'gurobi'

COVERAGE_SUFFIX = "coverage"


@dataclass
class QuboDescription:
    """Container for everything needed to run a QUBO solve job.

    Attributes:
        filename: Base name of the source GFA file (no directory component).
        data_dir: Directory where pickles, MQLib inputs, and solution files are
            read from and written to.
        graph: NetworkX graph built from the GFA file, with node attributes
            ``weight`` (copy number) and ``start`` (segment tag).
        time_limits: Solver time budgets in seconds; one solve is run per value.
        jobs: Number of independent solver runs per time limit.
        Q: The normalised QUBO matrix as a 2-D NumPy array.
        offset: Constant term of the QUBO energy (added back when reporting
            energies).
        T: Number of timesteps (path length) encoded in the QUBO.
        V: Number of graph nodes (not counting the sentinel end-node).
        solver: Which backend to use for sampling.
    """

    filename: str
    data_dir: str
    graph: Graph
    time_limits: list[int]
    jobs: int
    Q: ndarray
    offset: int
    T: int
    V: int
    solver: Solver