"""Plot CVaR-QAOA convergence and sample-energy histograms.

Loads a pickled CVaR-QAOA optimisation result and generates two plots:

1. **Convergence plot** — objective value vs. optimiser iteration, saved as a
   PNG to the experiments output directory.
2. **Histogram** — normalised density of sample energies (QAOA and random
   baseline) with log-scale x-axis, including post-processed variants where
   each bitstring is locally improved.  Saved as a PNG alongside the
   convergence plot.

Source pickle: ``experiments/<filename>_cvar.<suffix>.pkl``

CLI usage::

    python plot_cvar.py -f <filename> [-p <reps>] [-N <nodes>] [-T <time>]
                        [-m <memory>] [-M <method>] [-n <shots>] [-a <alpha>]
                        [--hardware] [--noisy] [--init {ramp,random,fixed}]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
from collections import Counter
from functools import reduce

from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.postprocess import postprocess


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

with open('/lustre/scratch127/qpg/jc59/out/qiskit/cvartest_N2_W2_cvar.p4.shots256.methodCOBYLA.hardwareTrue.noisyFalse.initrandom.pkl', 'rb') as f:
# with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/{filename}_cvar.{filename_suffix}.pkl', 'rb') as f:
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


evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op) + ising_offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = np.array([x for xs in energies for x in xs])
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))

# optimum = [1,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0,0] + [0,0,0,0,0,1,0,0,0] + [1,0,0,0,0,0,0,0,0] + [0,0,1,0,0,0,0,0,0] + [0,0,0,0,0,0,1,0,0]
# print(evaluate_sparse_pauli_samples([''.join([str(x) for x in optimum[::-1]])], hamiltonian) + ising_offset)

# sample = ''.join([str(x) for x in optimum[::-1]])
# new_sample = [''] * len(sample)
# for x in range(len(sample)):
#     new_sample[len(sample)-1- sat_map[len(sample)-1-x]] = sample[x]
# new_sample = ''.join(new_sample)
# print(evaluate_sparse_pauli_samples([new_sample], cost_op) + ising_offset)


# print(np.array([
#     sample @ Q @ sample for sample in [optimum]
# ]) + offset)


random_samples = np.random.choice(('0', '1'), (num_samples, n))
# random_ones = np.random.random_integers(0, 9-1, (num_samples, 6))
# vectors = [
#     ''.join(['0']*i + ['1'] + ['0']*(9-i-1)) for i in range(9)
# ]
# random_samples = [
#     reduce(
#         str.__add__,
#         [vectors[x] for x in random_ones[j, :]],
#         ''
#     )
#     for j in range(num_samples)
# ]
rand_samples = [''.join(sample) for sample in random_samples]
rand_vals = evaluate_sparse_pauli_samples(rand_samples, cost_op) + ising_offset

rand_counter = Counter(rand_vals)
rand_vals, rand_samples = zip(*sorted(zip(rand_vals, rand_samples), key=lambda e: e[0]))
rand_vals = np.array(rand_vals)




fig, axs = plt.subplots(1,1,figsize=(8, 5))
print(f'Max val: {np.max(list(counter.keys()) + list(rand_counter.keys()))}')
print(f'Max sample: {np.max(sample_vals)}, {np.log10(np.max(sample_vals))}')
print(f'Max rand: {np.max(rand_vals)}')

bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)

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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/test_random.{filename}.histogram.{filename_suffix}.png')


count_keys = list(counts.keys())
count_values = list(counts.values())    
post_process_samples = postprocess(count_keys, args.time)
post_process_counts = {}    
for idx, processed_samples in enumerate(post_process_samples):
    if len(processed_samples) > 1:
        print(f'Post-processed: {count_keys[idx]}')
    for sample in processed_samples:    
        post_process_counts[sample] = post_process_counts.get(sample, 0) + count_values[idx]
post_process_evals = evaluate_sparse_pauli_samples(list(post_process_counts.keys()), cost_op) + ising_offset
post_process_energies = [count * [post_process_evals[idx]] for idx, count in enumerate(post_process_counts.values())]
post_process_sample_vals = np.array([x for xs in post_process_energies for x in xs])


rand_counts = Counter(rand_samples)
rand_count_keys = list(rand_counts.keys())
rand_count_values = list(rand_counts.values())    
rand_post_process_samples = postprocess(rand_count_keys, args.time)
rand_post_process_counts = {}    
for idx, processed_samples in enumerate(rand_post_process_samples):
    if len(processed_samples) > 1:
        print(f'Post-processed: {rand_count_keys[idx]}')
    for sample in processed_samples:    
        rand_post_process_counts[sample] = rand_post_process_counts.get(sample, 0) + rand_count_values[idx]
rand_post_process_evals = evaluate_sparse_pauli_samples(list(rand_post_process_counts.keys()), cost_op) + ising_offset
rand_post_process_energies = [count * [rand_post_process_evals[idx]] for idx, count in enumerate(rand_post_process_counts.values())]
rand_post_process_sample_vals = np.array([x for xs in rand_post_process_energies for x in xs])



fig, axs = plt.subplots(1,1,figsize=(8, 5))
# axs.hist(sample_vals+1, bins=bins, label='QAOA circuit samples', density=False, weights=[1/num_samples]*num_samples)
# axs.hist(rand_vals+1, bins=bins, label='Random samples', density=False, alpha=0.75, weights=[1/num_samples]*num_samples)
axs.hist(post_process_sample_vals+1, bins=bins, label='QAOA post-process circuit samples', density=False, alpha=1, weights=[1/len(post_process_sample_vals)]*len(post_process_sample_vals))
axs.hist(rand_post_process_sample_vals+1, bins=bins, label='Random post-process samples', density=False, alpha=0.5, weights=[1/len(rand_post_process_sample_vals)]*len(rand_post_process_sample_vals))


ylims = axs.get_ylim()
axs.vlines(min_val+1, ylims[0], ylims[1], ls='--', color='k', label='Optimal solution')

axs.vlines(min(post_process_sample_vals)+1, ylims[0], ylims[1], ls=':', color='C0', label='Best post-process QAOA sample')
axs.vlines(min(rand_post_process_sample_vals)+1, ylims[0], ylims[1], ls='-.', color='C1', label='Best post-process random sample')

logger.info(f"QAOA post-process gap: {min(post_process_sample_vals) - min_val}")
logger.info(f"Random post-process gap: {min(rand_post_process_sample_vals) - min_val}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")
axs.set_xscale('log')

fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/test_random.{filename}.postprocess.histogram.{filename_suffix}.png')
