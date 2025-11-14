import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from collections import Counter
import argparse

from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian


logger = get_logger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed', 'warm'], default='random')
args = parser.parse_args()

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init


data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'
Q, _, offset = get_Q_and_hamiltonian(data_file)

filename_suffix = f'p{p}.shots{shots}.init{init_type}'


with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/hardware/{filename}_sample.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
counts = data["counts"]
cost_op = data["cost_op"]


min_val = 0


n = cost_op.num_qubits

start = time()
num_samples = sum(counts.values())


int_samples = [np.array([int(x) for x in sample[::-1]]) for sample in counts.keys()]
evals = np.array([
    sample @ Q @ sample for sample in int_samples
]) + offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = np.array([x for xs in energies for x in xs])
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')


counter = Counter(sample_vals)
print(counter.most_common(10))
for x in sorted(evals)[:5]:
    print(x, counter[x])


random_samples = np.random.choice((0, 1), (num_samples, n))
rand_vals = np.array([
    sample @ Q @ sample for sample in random_samples
]) + offset
rand_counter = Counter(rand_vals)
for x in sorted(rand_vals)[:5]:
    print(x, rand_counter[x])

fig, axs = plt.subplots(1,1,figsize=(8, 5))
print(f'Max val: {np.max(list(counter.keys()) + list(rand_counter.keys()))}')
print(f'Max sample: {np.max(sample_vals)}, {np.log10(np.max(sample_vals))}')
print(f'Max rand: {np.max(rand_vals)}')

bins = np.logspace(0, np.log10(np.max([np.max(sample_vals+1), np.max(rand_vals+1)])), 50, base=10)
# bins = sorted(list(set(np.ceil(bins))))
# print(bins)

hist_counts, hist_bins = np.histogram(sample_vals+1, bins, density=False)
print(hist_counts, hist_bins)

axs.hist(sample_vals+1, bins=bins, label='QAOA circuit samples', density=False, weights=[1/num_samples]*num_samples)
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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/qubo/hardware/{filename}.sample_only.histogram.{filename_suffix}.png')
