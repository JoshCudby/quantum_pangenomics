import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
from collections import Counter
from matplotlib.ticker import FormatStrFormatter

from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-N', '--nodes', nargs='?', type=int)
parser.add_argument('-T', '--time', type=int)
parser.add_argument('-m', '--memory', type=int, default=4000)
parser.add_argument('-M', '--method', type=str, default='')
parser.add_argument('-n', '--shots', type=int, default=2000)
parser.add_argument('-a', '--alpha', type=float, default=0.25)
parser.add_argument('--hardware', action='store_true', default=False)
parser.add_argument('--noisy', action='store_true', default=False)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps
hardware = args.hardware
shots = args.shots
noisy = args.noisy
init_type = args.init
alpha = args.alpha

data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'
Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)

if args.nodes is not None:
    filename_suffix = f'alpha{alpha}.p{p}.N{args.nodes}.shots{shots}.method{args.method}.hardware{hardware}.noisy{noisy}.init{init_type}'
elif args.method != '':
    filename_suffix = f'alpha{alpha}.p{p}.shots{shots}.method{args.method}.hardware{hardware}.noisy{noisy}.init{init_type}'
else:
    filename_suffix = f'p{p}.shots{shots}.hardware{hardware}.noisy{noisy}.init{init_type}'

# '/lustre/scratch127/qpg/jc59/out/qiskit/cvar/test_N2_W2_cvar.p4.shots256.methodCOBYLA.hardwareTrue.noisyFalse.initrandom.pkl'
with open('/lustre/scratch127/qpg/jc59/out/qiskit/hardware/test_N2_W2_cvar.p4.shots256.methodCOBYLA.max_iter120.initfixed.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
singles = data["singles"]
doubles = data["doubles"]
sat_map = data["sat_map"]
cost_op = data["cost_op"]
best_func_val = data["best_func_val"]
best_samples = data["best_samples"]

fig, ax = plt.subplots(1, 1, figsize=(3, 2))
ax.plot([hist[1] for hist in history])

ax.set_xlabel('Iteration', fontsize=8)
ax.set_ylabel('Objective value', fontsize=8)

ax.tick_params(axis='both', which='major', labelsize=7)

fig.subplots_adjust(
    left=0.16,
    right=0.98,
    bottom=0.2,
    top=0.99
)
fig.savefig('/nfs/users/nfs_j/jc59/quantumwork/pangenome/out/paper/qaoa_hardware_convergence.png', dpi=300)

min_val = 0
n = cost_op.num_qubits

start = time()
if best_func_val < np.inf and len(best_samples):
    print(f'Using best_func_val: {best_func_val}')
    counts = best_samples
else:
    counts = history[-1][3]
num_samples = sum(counts.values())
counter = Counter(counts)

evals = [[int(x) for x in s[::-1]] @ Q @ [int(x) for x in s[::-1]] + offset for s in counts.keys()]
# evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op) + ising_offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = np.array([x for xs in energies for x in xs])
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))


random_samples = np.random.choice(('0', '1'), (num_samples, n))
rand_samples = [''.join(sample) for sample in random_samples]
rand_vals = evaluate_sparse_pauli_samples(rand_samples, cost_op) + ising_offset

rand_counter = Counter(rand_vals)
rand_vals, rand_samples = zip(*sorted(zip(rand_vals, rand_samples), key=lambda e: e[0]))
rand_vals = np.array(rand_vals)



fig, ax = plt.subplots(1, 1, figsize=(3, 2))
print(f'Max val: {np.max(list(counter.keys()) + list(rand_counter.keys()))}')
print(f'Max sample: {np.max(sample_vals)}, {np.log10(np.max(sample_vals))}')
print(f'Max rand: {np.max(rand_vals)}')

bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)

ax.hist(sample_vals+1, bins=bins, label='QAOA samples', density=False, weights=[1/num_samples]*num_samples)
ax.hist(rand_vals+1, bins=bins, label='Random samples', density=False, alpha=0.5, weights=[1/num_samples]*num_samples)

ylims = ax.get_ylim()
ax.vlines(min_val+1, ylims[0], ylims[1], ls='--', color='k', label='Optimal solution')

ax.vlines(min(sample_vals)+1, ylims[0], ylims[1], ls=':', color='C0', label='Best QAOA')
ax.vlines(min(rand_vals)+1, ylims[0], ylims[1], ls=':', color='C1', label='Best random')

logger.info(f"QAOA gap: {min(sample_vals) - min_val}")
logger.info(f"Random gap: {min(rand_vals) - min_val}")

ax.legend(fontsize=6, bbox_to_anchor=(0.58, 0.99), bbox_transform=fig.transFigure,)
ax.set_xlabel("Quadratic program objective value", fontsize=7)
ax.set_ylabel("Sample density", fontsize=7)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.set_xscale('log')
ax.set_xlim(10**-.1, 500)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 dp
fig.subplots_adjust(
    left=0.16,
    right=0.98,
    bottom=0.2,
    top=0.99
)
fig.savefig('/nfs/users/nfs_j/jc59/quantumwork/pangenome/out/paper/qaoa_hardware_histogram.png', dpi=300)

