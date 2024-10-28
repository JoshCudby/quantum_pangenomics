import numpy as np
from itertools import combinations, combinations_with_replacement
import scipy
import matplotlib.pyplot as plt

from qokit.utils import brute_force
from qokit import get_qaoa_objective
from qokit.fur import choose_simulator, get_available_simulator_names

# Small test
N_vars = 4
np.random.seed(100)
terms = [(np.random.normal(), spin_pair) for spin_pair in combinations_with_replacement(range(N_vars), r=2)]

# print(get_available_simulator_names("x"))
# simclass = choose_simulator(name='auto')
# sim = simclass(N_vars, terms=terms)
# sim.get_cost_diagonal()

# p = 3
# gamma, beta = np.random.rand(2, 3)
# _result = sim.simulate_qaoa(gamma, beta) # Result depends on the type of simulator. 
# sv = sim.get_statevector(_result)
# print(sv)
# probs = sim.get_probabilities(_result)
# probs.sum()

# probs = sim.get_probabilities(_result, preserve_state=False)
# sv2 = sim.get_statevector(_result)
# print("Using numpy") if np.allclose(sv, sv2) else print("Yohoo, I'm using a memory-economic simulator!")
# print(sv2)

# Get objective
p = 5
f = get_qaoa_objective(N_vars, p, terms=terms, parameterization='theta')
initial_gamma = -1*np.linspace(0, 1, p)
initial_beta = np.linspace(1, 0, p)

res = scipy.optimize.minimize(f, np.hstack([initial_gamma, initial_beta]), method='COBYLA', options={'rhobeg': 0.01})
print(f"Expected QAOA solution quality: {res.fun:.5f}")

def f_from_terms_ground_truth(s):
    """ground truth for debugging
    """
    out = 0
    for coeff, (i, j) in terms:
        out += coeff * s[i] * s[j]
    return out

print(f"True minimum: {brute_force(f_from_terms_ground_truth, N_vars, minimize=True)[0]:.5f}")
