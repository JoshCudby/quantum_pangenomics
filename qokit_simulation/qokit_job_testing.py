import numpy as np
from itertools import combinations, combinations_with_replacement
import scipy
import matplotlib.pyplot as plt

from qokit.utils import brute_force
from qokit import get_qaoa_objective
from qokit.fur import choose_simulator, get_available_simulator_names

data = np.load('../qubo_solvers/out/tangle/qubo_data_test.npy', allow_pickle=True)
Q, offset, W = data
# Move terms to upper triangular part
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2
N_vars = Q.shape[0]

# Get Hamiltonian terms
terms = [(Q[i, j], [i, j]) for i in range(N_vars) for j in range(i, N_vars)]


print(get_available_simulator_names("x"))
simclass = choose_simulator(name='auto')
sim = simclass(N_vars, terms=terms)
print(type(sim))
sim.get_cost_diagonal()

p = 3
gamma, beta = np.random.rand(2, 3)
_result = sim.simulate_qaoa(gamma, beta) # Result depends on the type of simulator. 
sv = sim.get_statevector(_result)
print(sv)
probs = sim.get_probabilities(_result)
probs.sum()

probs = sim.get_probabilities(_result, preserve_state=False)
sv2 = sim.get_statevector(_result)
print("Using numpy") if np.allclose(sv, sv2) else print("Yohoo, I'm using a memory-economic simulator!")
print(sv2)
