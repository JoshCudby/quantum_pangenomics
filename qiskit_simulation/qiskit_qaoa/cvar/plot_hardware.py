"""Plot hardware CVaR-QAOA convergence and energy histogram.

Loads a pickled result from ``optimize_hardware.py`` and generates:

1. **Convergence plot** — CVaR objective value per COBYLA iteration.
2. **Energy histogram** — QAOA vs. random sample density on a log-scale
   x-axis, with vertical lines marking the optimal solution and best samples.

Source pickle:
    ``cvar_new/hardware/<filename>_cvar.error_miti.<suffix>.pkl``

CLI usage::

    python plot_hardware.py -f <filename> -p <reps> -n <shots> -i <max_iter>
                            [-M <method>] [-a <alpha>]
                            [--init {ramp,random,fixed}]
"""

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
parser.add_argument('-M', '--method', type=str, default='COBYLA')
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('-i', '--max_iter', type=int)
parser.add_argument('-a', '--alpha', type=float, default=0.25)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
args = parser.parse_args()

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init
max_iter = args.max_iter
alpha = args.alpha


data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'
Q, _, offset, _ = get_Q_and_hamiltonian(data_file)

filename_suffix = f'error_miti.alpha{alpha}.p{p}.shots{shots}.method{args.method}.max_iter{max_iter}.init{init_type}'

with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar_new/hardware/{filename}_cvar.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
cost_op = data["cost_op"]
best_func_val = data["best_func_val"]
best_samples = data["best_samples"]

fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs.plot([hist[1] for hist in history])


axs.set_xlabel('Iteration')
axs.set_ylabel('Objective value')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/qubo_new/hardware/{filename}.convergence.{filename_suffix}.png')

min_val = 0


n = cost_op.num_qubits

start = time()
if best_func_val < np.inf and len(best_samples):
    print(f'Using best_func_val: {best_func_val}')
    counts = best_samples
else:
    counts = history[-1][3]
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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/qubo_new/hardware/{filename}.histogram.{filename_suffix}.png')
