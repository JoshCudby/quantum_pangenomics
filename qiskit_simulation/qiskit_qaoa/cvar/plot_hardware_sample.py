import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import Counter
import argparse

from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.postprocess import postprocess

logger = get_logger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('-e', '--error', nargs='?', type=bool)
parser.add_argument('-T', '--time', type=int)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed', 'warm'], default='random')
args = parser.parse_args()

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init


data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'
Q, _, offset, ising_offset = get_Q_and_hamiltonian(data_file)

if args.error is not None:
    filename_suffix = f'error_miti{args.error}.p{p}.shots{shots}.init{init_type}'
else:
    filename_suffix = f'p{p}.shots{shots}.init{init_type}'


with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/hardware.{filename}_sample.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
counts = data["counts"]
cost_op = data["cost_op"]


min_val = 0


n = cost_op.num_qubits

start = time()
num_samples = sum(counts.values())
evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op) + ising_offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = np.array([x for xs in energies for x in xs])

elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')


counter = Counter(sample_vals)
    
    
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



random_samples = np.random.choice(('0', '1'), (num_samples, n))
rand_samples = [''.join(sample) for sample in random_samples]


rand_vals = evaluate_sparse_pauli_samples(rand_samples, cost_op) + ising_offset
rand_counter = Counter(rand_vals)

    

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
print(f'Max val: {np.max(list(counter.keys()) + list(rand_counter.keys()))}')
print(f'Max sample: {np.max(sample_vals)}, {np.log10(np.max(sample_vals))}')
print(f'Max rand: {np.max(rand_vals)}')

bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)


hist_counts, hist_bins = np.histogram(sample_vals+1, bins, density=False)

axs.hist(sample_vals+1, bins=bins, label='QAOA circuit samples', density=False, weights=[1/num_samples]*num_samples)
axs.hist(rand_vals+1, bins=bins, label='Random samples', density=False, alpha=0.5, weights=[1/num_samples]*num_samples)
# axs.hist(post_process_sample_vals+1, bins=bins, label='QAOA post-process circuit samples', density=False, alpha=0.5, weights=[1/len(post_process_sample_vals)]*len(post_process_sample_vals))
# axs.hist(rand_post_process_sample_vals+1, bins=bins, label='Random post-process samples', density=False, alpha=0.25, weights=[1/len(rand_post_process_sample_vals)]*len(rand_post_process_sample_vals))


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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/hardware.{filename}.sample_only.histogram.{filename_suffix}.png')


fig, axs = plt.subplots(1,1,figsize=(8, 5))
bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)
# bins = sorted(list(set(np.ceil(bins))))
# print(bins)

hist_counts, hist_bins = np.histogram(sample_vals+1, bins, density=False)

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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/experiments/hardware.{filename}.sample_only.postprocess.histogram.{filename_suffix}.png')
