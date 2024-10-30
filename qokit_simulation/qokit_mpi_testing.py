import numpy as np
import scipy
import matplotlib.pyplot as plt
from qokit.fur import choose_simulator, get_available_simulator_names
from qokit import parameter_utils
from qokit.qaoa_objective import get_qaoa_objective
from itertools import combinations_with_replacement

rng = np.random.default_rng(10)


print(get_available_simulator_names("x"))

import mpi4py.MPI
size = mpi4py.MPI.COMM_WORLD.Get_size()
print(size)