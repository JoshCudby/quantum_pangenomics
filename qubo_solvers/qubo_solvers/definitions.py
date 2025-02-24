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
    DWAVE = 'dwave'
    MQLIB = 'mqlib'
    GUROBI = 'gurobi'

COVERAGE_SUFFIX = "coverage"


@dataclass
class QuboDescription:
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