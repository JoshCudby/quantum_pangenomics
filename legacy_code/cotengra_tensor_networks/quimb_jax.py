import jax
import numpy as np
import cotengra as ctg
import quimb.tensor as qtn
import quimb as qu
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import tqdm
from functools import reduce


if len(sys.argv) > 1:
    out_dir = sys.argv[1]
else:
    out_dir = '/lustre/scratch127/qpg/jc59/out/cotengra'

if len(sys.argv) > 2:
    filepath = sys.argv[2]
else:
    filepath = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'
filename = os.path.basename(filepath)

if len(sys.argv) > 3:
    seed = int(sys.argv[3])
else:
    seed = 100

rng = np.random.default_rng(seed)


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

# N_vars = 15
# Q = qu.randn((N_vars, N_vars), scale=5, loc=-2.5, seed=seed)
# offset = 0

data = np.load(filepath, allow_pickle=True)
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

# Find a contraction tree for each local expectation
local_exp_rehs = [
    circ.local_expectation_rehearse(weight * ZZ, edge, optimize=opt)#, simplify_sequence='ADCRS', simplify_equalize_norms=False, backend="jax")
    if not edge[0] == edge[1]
    else circ.local_expectation_rehearse(weight * Z, edge[0], optimize=opt)#, simplify_sequence='ADCRS', simplify_equalize_norms=False, backend="jax")
    for edge, weight in list(terms.items())
]
eprint('Finished rehearsing')
trees = [rehs['tree'] for rehs in local_exp_rehs]

contract_cores_jit = [jax.jit(tree.contract_core) for tree in trees]


def energy(x):
    p = len(x) // 2
    gammas = x[:p]
    betas = x[p:]
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    local_exp_tns = [
        circ.local_expectation_tn(weight * ZZ, edge)#, simplify_sequence='ADCRS', simplify_equalize_norms=False, backend="jax")
        if not edge[0] == edge[1]
        else circ.local_expectation_tn(weight * Z, edge[0])#, simplify_sequence='ADCRS', simplify_equalize_norms=False, backend="jax")
        for edge, weight in list(terms.items())
    ]

    future_groups = [
        [
            pool.submit(contract_cores_jit[i], trees[i].slice_arrays(local_exp_tns[i].arrays, j)) 
            for j in range(trees[i].nslices)
        ] for i in range(len(trees))
    ]

    # lazily gather all the slices in the main process
    slices = [(np.array(f.result()) for f in fs) for fs in future_groups]

    results = [trees[i].gather_slices(slices[i], progbar=False) for i in range(len(slices))]
    return sum(results).real


from skopt import Optimizer

eps = 1e-6
bounds = (
    [(0.0        + eps, qu.pi / 2 - eps)] * p + 
    [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
)

bopt = Optimizer(bounds, random_state=seed)

for i in tqdm.trange(100):
    x = bopt.ask()
    res = bopt.tell(x, energy(x))
    
print(res)
print(offset)

to_save = np.array([res, offset], dtype=object)
data = np.save(f'{out_dir}/cotengra_data_p_{p}_{filename}.npy', to_save, allow_pickle=True)
