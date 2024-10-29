import numpy as np
import scipy
import matplotlib.pyplot as plt
from qokit.fur import choose_simulator, get_available_simulator_names
from qokit import parameter_utils
from qokit.qaoa_objective import get_qaoa_objective
from itertools import combinations_with_replacement


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

# Small test
# N = 4
# np.random.seed(10)
# terms = [(np.random.normal(), spin_pair) for spin_pair in combinations_with_replacement(range(N), r=2)]


print(get_available_simulator_names("x"))
# simclass = choose_simulator(name='auto')
# sim = simclass(N, terms=terms)
# print(type(sim))
# sim.get_cost_diagonal()

# Initial
p = 1
gamma, beta = rng.random((2, p))
# u, v = parameter_utils.to_fourier_basis(gamma, beta)
theta = np.hstack([gamma, beta])
print(f'Original theta: {theta}')

f = get_qaoa_objective(N, p, terms=terms, parameterization='theta', objective='overlap')
print(f"Success probability at p = {p} before optimization is {1-f(theta)}")

res = scipy.optimize.minimize(f, theta, method='COBYLA', options={'rhobeg': 0.01/N})
gamma_opt, beta_opt = res.x[:p], res.x[p:]
theta = np.hstack([gamma_opt, beta_opt])
print(f"Success probability at p = {p} after optimization is {1-f(theta)}")
print(f'Optimised theta at p = {p}: {theta}')

while(1-f(theta) < 10 ** -4 and p < 10):
    init_gamma = np.interp(np.linspace(0, 1, p + 1), np.linspace(0, 1, p), gamma_opt)
    init_beta = np.interp(np.linspace(0, 1, p + 1), np.linspace(0, 1, p), beta_opt)
    init_theta = np.hstack([init_gamma, init_beta])
    
    p = p + 1
    f = get_qaoa_objective(N, p, terms=terms, parameterization='theta', objective='overlap')
    print(f'Init theta at p = {p}: {init_theta}')
    print(f"Success probability at p = {p} before optimization is {1-f(init_theta)}")

    res = scipy.optimize.minimize(f, init_theta, method='COBYLA', options={'rhobeg': 0.01/N})
    gamma_opt, beta_opt = res.x[:p], res.x[p:]
    theta = np.hstack([gamma_opt, beta_opt])
    print(f"Success probability at p = {p} after optimization is {1-f(theta)}")
    print(f'Optimised theta at p = {p}: {theta}')