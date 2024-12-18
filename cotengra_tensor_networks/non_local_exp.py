import jax
import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
import quimb as qu
from concurrent.futures import ThreadPoolExecutor
import sys
import tqdm
from itertools import product
from skopt import Optimizer

seed = 666
p = 4

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
    
def Q_to_Ising(Q, offset):
    n_qubits = Q.shape[0]
    J = {}
    h = {i : 0 for i in range(n_qubits)}
    # zi^2 = 1
    for i in range(n_qubits):
        # Q[i, i]xi^2 = Q[i, i](1 - 2zi + zi^2)/4 -> h[i] = - Q[i, i]/ 2, O += Q[i, i] / 2
        h[i] -= Q[i, i] / 2
        offset += Q[i, i] / 2
        # Calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Q[i, j]xi xj = Q[i, j] (1 - zi - zj + zi zj)/4 -> J[i, j] = Q[i, j] / 4, h[i], h[j] -= Q[i, j]/4, O+= Q[i, j] / 4
            J[(i, j)] = Q[i, j] / 4
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            offset += Q[i, j] / 4
    return h, J, offset


gammas = qu.randn(p, seed=seed)
betas = qu.randn(p, seed=seed)
opt = ctg.ReusableHyperOptimizer(
    methods=['kahypar', 'greedy'],
    optlib='nevergrad',
    max_time='equil:4',
    parallel=True,
    # make sure contractions fit onto GPU
    slicing_reconf_opts={"target_size": 2**28},
    max_repeats=32,
    progbar=True,
    directory=True
)

data = np.load('/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy', allow_pickle=True)
Q, offset, T, W = data

# Move terms to upper triangular part
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2
N_vars = Q.shape[0]

# Get Hamiltonian terms
h, J, offset = Q_to_Ising(Q, offset)


p = 16
gammas = qu.randn(p, seed=seed)
betas = qu.randn(p, seed=seed)

Z = qu.pauli('Z')
I = qu.pauli('I')
ZZ = qu.pauli('Z') & qu.pauli('Z')

eprint(f'Num of vars: {N_vars}')
eprint(f'Num interaction terms in Hamiltonian: {len(list(J.items()))}')
eprint(f'Terms: {list(h.items())} {list(J.items())}')

# Use 1 gpu for now ?
pool = ThreadPoolExecutor(1)


circ = qtn.circ_qaoa(h, J, p, gammas, betas)

# IZI = ikron(pauli('z'), [2, 2, 2], 1)
# This is possibly very slow
d = 2 ** N_vars
H = np.zeros((d, d), dtype=complex)
for edge, weight in list(J.items()):
    Z_op = np.diagflat([(-1) ** (i[edge[0]] + i[edge[1]]) for i in product([0, 1], repeat=N_vars)])
    H += weight * Z_op
for i, weight in list(h.items()):
    Z_op = np.diagflat([(-1) ** (j[i]) for j in product([0, 1], repeat=N_vars)])
    H += weight * Z_op
H = qu.qarray(H)

# Find a contraction tree
rehs = circ.local_expectation_rehearse(H, range(N_vars), optimize=opt)
eprint('Finished rehearsing')
tree = rehs['tree']

contract_core_jit = jax.jit(tree.contract_core)


def energy(x):
    p = len(x) // 2
    gammas = x[:p]
    betas = x[p:]
    circ = qtn.circ_qaoa(h, J, p, gammas, betas)

    exp_tn = circ.local_expectation_tn(H, range(N_vars))

    futures = [
            pool.submit(contract_core_jit, tree.slice_arrays(exp_tn.arrays, j)) 
            for j in range(tree.nslices)
        ] 
    
    # lazily gather all the slices in the main process
    slices = (np.array(f.result()) for f in futures)

    results = np.sum(list(slices))
    return results.real



eps = 1e-6
bounds = (
    [(0.0        + eps, qu.pi / 2 - eps)] * p + 
    [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
)

bopt = Optimizer(bounds, random_state=seed)

for i in tqdm.trange(400):
    x = bopt.ask()
    res = bopt.tell(x, energy(x))
    
print(res)
print(offset)

to_save = np.array([res, offset], dtype=object)
data = np.save(f'/lustre/scratch127/qpg/jc59/out/cotengra/non_local_exp_data_p_{p}_trivial.gfa.npy', to_save, allow_pickle=True)

