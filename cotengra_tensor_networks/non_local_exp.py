import jax
import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
import quimb as qu
from concurrent.futures import ThreadPoolExecutor
import sys
import tqdm
from functools import reduce


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
    
def Q_to_Ising(Q, offset):
    n_qubits = Q.shape[0]
    J = {(i, i) : 0 for i in range(n_qubits)}

    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        J[(i, i)] -= Q[i, i] / 2
        # Update the offset based on the diagonal element in Q
        offset += Q[i, i] / 2
        # Calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J[(i, j)] = Q[i, j] / 4
            # Update the magnetic fields for qubits i and j based on their interactions in Q
            J[(i, i)] -= Q[i, j] / 4
            J[(j, j)] -= Q[i, j] / 4
            # Update the offset based on the interaction strength between qubits i and j
            offset += Q[i, j] / 4
    del_keys = []
    for key in J.keys():
        if J[key] == 0:
            del_keys.append(key)
    for key in del_keys:
        J.pop(key)
    return J, offset


seed = 666

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
terms, offset = Q_to_Ising(Q, offset)


p = 4
gammas = qu.randn(p, seed=seed)
betas = qu.randn(p, seed=seed)

Z = qu.pauli('Z')
I = qu.pauli('I')
ZZ = qu.pauli('Z') & qu.pauli('Z')

eprint(f'Num of vars: {N_vars}')
eprint(f'Num terms in Hamiltonian: {len(list(terms.items()))}')
eprint(f'Terms: {list(terms.items())}')

# Use 1 gpu for now ?
pool = ThreadPoolExecutor(1)


circ = qtn.circ_qaoa(terms, p, gammas, betas)

# This is possibly very slow
d = 2 ** N_vars
H = np.zeros((d, d), dtype=complex)
for edge, weight in list(terms.items()):
    Z_op = np.diagflat(reduce(np.kron, [[1, 1] if not i in edge else [1,-1] for i in range(N_vars)]))
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
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    exp_tn = circ.local_expectation_tn(H, range(N_vars))

    futures = [
            pool.submit(contract_core_jit, tree.slice_arrays(exp_tn.arrays, j)) 
            for j in range(tree.nslices)
        ] 
    

    # lazily gather all the slices in the main process
    slices = (np.array(f.result()) for f in futures)

    results = np.sum(list(slices))
    return results.real


from skopt import Optimizer

eps = 1e-6
bounds = (
    [(0.0        + eps, qu.pi / 2 - eps)] * p + 
    [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
)

bopt = Optimizer(bounds, random_state=seed)

for i in tqdm.trange(200):
    x = bopt.ask()
    res = bopt.tell(x, energy(x))
    
print(res)
print(offset)

to_save = np.array([res, offset], dtype=object)
data = np.save('/lustre/scratch127/qpg/jc59/out/cotengra/non_local_exp_data_trivial.gfa.npy', to_save, allow_pickle=True)
