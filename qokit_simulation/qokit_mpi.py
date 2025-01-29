import pickle
import numpy as np
import scipy
from qokit.fur import choose_simulator, get_available_simulator_names
from qokit.qaoa_objective import get_qaoa_objective
from itertools import combinations_with_replacement
from mpi4py import MPI


rng = np.random.default_rng(100)

with open('/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa', 'rb') as f:
    data = pickle.load(f)

Q = data['qubo_matrix']
offset = data['offset']
Q = np.array(Q)
# Move terms to upper triangular part
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2

def Q_to_Ising(Q, offset):
    n_qubits = Q.shape[0]
    J = {}
    h = {i : 0 for i in range(n_qubits)}
    # xi binary, zi spin
    # recall zi^2 = 1 whereas xi^2 = xi
    for i in range(n_qubits):
        # Q[i, i]xi^2 = Q[i, i](1 - 2zi + zi^2)/4 -> h[i] = - Q[i, i]/ 2, O += Q[i, i] / 2
        h[n_qubits - i - 1] -= Q[i, i] / 2
        offset += Q[i, i] / 2
        # Calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Q[i, j]xi xj = Q[i, j] (1 - zi - zj + zi zj)/4 -> J[i, j] = Q[i, j] / 4, h[i], h[j] -= Q[i, j]/4, O+= Q[i, j] / 4
            J[(n_qubits - i - 1, n_qubits - j - 1)] = Q[i, j] / 4
            h[n_qubits - i - 1] -= Q[i, j] / 4
            h[n_qubits - j - 1] -= Q[i, j] / 4
            offset += Q[i, j] / 4
    return h, J, offset


h, J, offset = Q_to_Ising(Q, offset)
N = Q.shape[0]
# print(N)
# Get Hamiltonian terms
terms = [(val, key) for key, val in J.items()] + [(val, (key,)) for key, val in h.items()]


# print(get_available_simulator_names("x"))

# Initial
p = 4
gamma, beta = rng.random((2, p))
theta = np.hstack([gamma, beta])
# print(f'Original theta: {theta}')

f = get_qaoa_objective(N, p, terms=terms, parameterization='theta', objective='overlap')
# print(f"Success probability at p = {p} before optimization is {1-f(theta)}")

res = scipy.optimize.minimize(f, theta, method='COBYLA', options={'rhobeg': 0.01/N, 'tol': 10**-7})
gamma_opt, beta_opt = res.x[:p], res.x[p:]
theta = np.hstack([gamma_opt, beta_opt])
# print(f"Success probability at p = {p} after optimization is {1-f(theta)}")
# print(f'Optimised theta at p = {p}: {theta}')

while(1-f(theta) < 10 ** -2 and p < 6):
    init_gamma = np.interp(np.linspace(0, 1, p + 1), np.linspace(0, 1, p), gamma_opt)
    init_beta = np.interp(np.linspace(0, 1, p + 1), np.linspace(0, 1, p), beta_opt)
    init_theta = np.hstack([init_gamma, init_beta])
    
    p = p + 1
    f = get_qaoa_objective(N, p, terms=terms, parameterization='theta', objective='overlap')
    # print(f'Init theta at p = {p}: {init_theta}')
    # print(f"Success probability at p = {p} before optimization is {1-f(init_theta)}")

    res = scipy.optimize.minimize(f, init_theta, method='COBYLA', options={'rhobeg': 0.01/N, 'tol': 10**-7})
    gamma_opt, beta_opt = res.x[:p], res.x[p:]
    theta = np.hstack([gamma_opt, beta_opt])
    # print(f"Success probability at p = {p} after optimization is {1-f(theta)}")
    # print(f'Optimised theta at p = {p}: {theta}')


if MPI.COMM_WORLD.Get_rank() == 0:
    simulator_cls = choose_simulator(name="auto")
    sim = simulator_cls(N, terms=terms)
    result = sim.simulate_qaoa(gamma_opt, beta_opt, None)
    probs = sim.get_probabilities(result)
    print("Max probs, argmax probs")
    print(max(probs), np.argmax(probs))
    found_optimum = list(np.binary_repr(np.argmax(probs), N))
    found_optimum = [int(x) for x in found_optimum]
    print("Found optimum")
    print(found_optimum)

    costs = sim.get_cost_diagonal()
    overlap = sim.get_overlap(result, costs)
    print("Overlap")
    print(overlap)


    optimum_trivial = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    # 0, 1, 2, 3, 1, 2, 4
    optimum_trivial = [
        1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1,
        ]
    optimum_index = sum([optimum_trivial[i] * 2 ** (i) for i in range(N)])
    print("Index of true optimum")
    print(optimum_index)
    print("Probs of true optimum")
    print(probs[optimum_index])

    print("True optimum reversed @ Q")
    print(optimum_trivial[-1::-1] @ Q @ optimum_trivial[-1::-1] + data['offset'])
    print("True optimum J h")
    print(
        sum(J[(i, j)] * (-1) ** (optimum_trivial[i] + optimum_trivial[j]) for (i, j) in J.keys())
        + sum(h[i] * (-1) ** optimum_trivial[i] for i in range(N))
        + offset
        )

    print("Found optimum Q")
    print(found_optimum[-1::-1] @ Q @ found_optimum[-1::-1] + data['offset'])
    print("Found optimum J h")
    print(
        sum(J[(i, j)] * (-1) ** (found_optimum[i] + found_optimum[j]) for (i, j) in J.keys())
        + sum(h[i] * (-1) ** found_optimum[i] for i in range(N))
        + offset   
    )

    # 2114 = [1000,0100,0010]
    # OOM on test data - MPI?