import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import Counter

from qiskit_qaoa.utils.argparser import get_parser
from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


logger = get_logger(__name__)
parser = get_parser()
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

if args.method != '':
    filename_suffix = f'alpha{alpha}.p{p}.shots{shots}.method{args.method}.hardware{hardware}.noisy{noisy}.init{init_type}'
else:
    filename_suffix = f'p{p}.shots{shots}.hardware{hardware}.noisy{noisy}.init{init_type}'

with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/{filename}_cvar.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
singles = data["singles"]
doubles = data["doubles"]
sat_map = data["sat_map"]
cost_op = data["cost_op"]
best_func_val = data["best_func_val"]
best_samples = data["best_samples"]

fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs.plot([hist[1] for hist in history])


axs.set_xlabel('Iteration')
axs.set_ylabel('Objective value')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/{filename}.convergence.{filename_suffix}.png')

min_val = 0


n = cost_op.num_qubits

start = time()
if best_func_val < np.inf and len(best_samples):
    print(f'Using best_func_val: {best_func_val}')
    counts = best_samples
else:
    counts = history[-1][3]
num_samples = sum(counts.values())


# int_samples = [np.array([int(x) for x in sample[::-1]]) for sample in counts.keys()]
# evals = np.array([
#     sample @ Q @ sample for sample in int_samples
# ]) + offset
# energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
# sample_vals = np.array([x for xs in energies for x in xs])

evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op) + ising_offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = np.array([x for xs in energies for x in xs])
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))

optimum = [1,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0,0] + [0,0,0,0,0,1,0,0,0] + [1,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0,0] + [0,0,0,0,0,0,1,0,0]
print(evaluate_sparse_pauli_samples([''.join([str(x) for x in optimum[::-1]])], hamiltonian) + ising_offset)

sample = ''.join([str(x) for x in optimum[::-1]])
new_sample = [''] * len(sample)
for x in range(len(sample)):
    new_sample[len(sample)-1- sat_map[len(sample)-1-x]] = sample[x]
new_sample = ''.join(new_sample)
print(evaluate_sparse_pauli_samples([new_sample], cost_op) + ising_offset)


print(np.array([
    sample @ Q @ sample for sample in [optimum]
]) + offset)


# print(len(evals), len(int_samples))
# evals, int_samples = zip(*sorted(zip(evals, int_samples), key=lambda e: e[0]))
# for x in range(7):
#     print(int_samples[x], evals[x], counts[''.join([str(y) for y in int_samples[x]])[::-1]], counter[evals[x]])


random_samples = np.random.choice(('0', '1'), (sum(history[-1][3].values()), n))
rand_vals = evaluate_sparse_pauli_samples([''.join(sample) for sample in random_samples], cost_op) + ising_offset

# random_samples = np.random.choice((0, 1), (num_samples, n))
# rand_vals = np.array([
#     sample @ Q @ sample for sample in random_samples
# ]) + offset
rand_counter = Counter(rand_vals)
rand_vals, random_samples = zip(*sorted(zip(rand_vals, random_samples), key=lambda e: e[0]))
rand_vals = np.array(rand_vals)
# for x in range(7):
#     print(random_samples[x], rand_vals[x], rand_counter[rand_vals[x]])

fig, axs = plt.subplots(1,1,figsize=(8, 5))
print(f'Max val: {np.max(list(counter.keys()) + list(rand_counter.keys()))}')
print(f'Max sample: {np.max(sample_vals)}, {np.log10(np.max(sample_vals))}')
print(f'Max rand: {np.max(rand_vals)}')

bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)
# bins = sorted(list(set(np.ceil(bins))))
# print(bins)

# hist_counts, hist_bins = np.histogram(sample_vals+1, bins, density=False)
# print(hist_counts, hist_bins)

# axs.set_ylim(0, max([counter.most_common(1)[0][1], rand_counter.most_common(1)[0][1]])/sum(counts.values()))
axs.hist(sample_vals+1, bins=bins, label='QAOA samples at last iter', density=False, weights=[1/num_samples]*num_samples)
axs.hist(rand_vals+1, bins=bins, label='Random samples', density=False, alpha=0.5, weights=[1/num_samples]*num_samples)



ylims = axs.get_ylim()
axs.vlines(min_val+1, ylims[0], ylims[1], ls='--', color='k', label='Optimal solution')

axs.vlines(min(sample_vals)+1, ylims[0], ylims[1], ls=':', color='C0', label='Best QAOA sample')
axs.vlines(min(rand_vals)+1, ylims[0], ylims[1], ls='-.', color='C1', label='Best random sample')

logger.info(f"QAOA gap: {min(sample_vals) - min_val}")
logger.info(f"Random gap: {min(rand_vals) - min_val}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")
axs.set_xscale('log')

fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/{filename}.histogram.{filename_suffix}.png')
