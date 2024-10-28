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
print(N_vars)

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

overlap = sim.get_overlap(_result)
print("Ground state overlap:", overlap)
# Below we test that for positive-valued cost function, the maximum can be achieved 
# by either inverting the values, or negating the values.
costs_abs = np.abs(sim.get_cost_diagonal())
print("Overlap with ground state for absolute cost:", sim.get_overlap(_result, costs=costs_abs))
overlap_inv = sim.get_overlap(_result, costs=1/costs_abs)
print("Overlap with highest state (inverted costs):", overlap_inv)
overlap_neg = sim.get_overlap(_result, costs=-costs_abs)
print("Overlap with highest state (negative):", overlap_neg)
assert overlap_inv == overlap_neg, "You may have values of mixed sign in your cost."