import numpy as np
import scipy
import matplotlib.pyplot as plt
from qokit.fur import choose_simulator, get_available_simulator_names
from qokit import parameter_utils
from qokit.qaoa_objective import get_qaoa_objective


rng = np.random.default_rng(10)

data = np.load('../qubo_solvers/out/tangle/qubo_data_test.npy', allow_pickle=True)
Q, offset, W = data
# Move terms to upper triangular part
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2
N = Q.shape[0]
print(N)

# Get Hamiltonian terms
terms = [(Q[i, j], [i, j]) for i in range(N) for j in range(i, N)]


# print(get_available_simulator_names("x"))
# simclass = choose_simulator(name='auto')
# sim = simclass(N, terms=terms)
# print(type(sim))
# sim.get_cost_diagonal()

# Initial
p = 1
gamma, beta = rng.random((2, 3))
u, v = parameter_utils.to_fourier_basis(gamma, beta)
init_freq = np.hstack([u, v])

f = get_qaoa_objective(N, p, terms=terms, parameterization='freq', objective='overlap')
print(f"Success probability at p={p} before optimization is {1-f(init_freq)}")

res = scipy.optimize.minimize(f, init_freq, method='COBYLA', options={'rhobeg': 0.01/N})
u_opt, v_opt = res.x[:p], res.x[p:]
print(f"Success probability at p={p} after optimization is {1-f(np.hstack([u_opt, v_opt]))}")