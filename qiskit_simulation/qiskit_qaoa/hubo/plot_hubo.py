"""Visualise HUBO QAOA hardware-run optimisation results.

Loads an optimisation pickle produced by ``hubo_optimisation.py`` (from the
``hubo_hardware/`` scratch directory) and generates two plots:

1. **Convergence plot**: objective value (CVaR energy) versus optimiser
   iteration, saved as ``<filename>.times<t>.convergence.png``.

2. **Energy histogram**: density histogram of bitstring energies evaluated at
   the *final* iteration of the optimiser, compared against a random baseline.
   Vertical lines mark the optimal energy (0) and the minimum energies
   achieved by QAOA and random sampling.  Saved as
   ``<filename>.times<t>.histogram.png``.

Plots are written to::

    /nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/hardware/

The CLI arguments mirror those of ``hubo_optimisation.py`` and are used to
reconstruct the exact pickle filename from the optimisation run parameters.

CLI arguments:
    -f / --filename:       GFA file stem of the original instance.
    -p / --reps:           Number of QAOA layers p.
    -d / --swap-depth:     SWAP-layer budget index used during optimisation.
    -m / --memory:         (Unused; kept for CLI consistency.)
    -M / --method:         Optimiser method name used.
    -n / --shots:          Shots per evaluation used.
    --init:                Initialisation strategy used.
    -e / --extra:          Extra SWAP layers used during compilation.
    --fraction-four:       4-body term fraction used.
    --fraction-six:        6-body term fraction used.
    --times-to-keep:       Timestep-transition indices used.
    -a / --alpha:          CVaR tail fraction used.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import Counter
from time import time
import argparse

from qiskit.quantum_info import SparsePauliOp
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples

from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-d', '--swap-depth', type=int, default=0)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-M', '--method', type=str)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random', 'warm'], default='ramp')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('--times-to-keep', help='delimited list input', 
    type=lambda s: tuple([int(item) for item in s.split(',') if len(item)]))
parser.add_argument('-a', '--alpha', type=float)

args = parser.parse_args()


logger.info(args)
basepath='/lustre/scratch127/qpg/jc59/hubo_hardware/'


filename='optimisation.{}.extra{}.times{}.four{}.six{}.method{}.cvar{}.p{}.shots{}.init{}.d{}'.format(
    args.filename,
    args.extra,
    ''.join([str(t) for t in args.times_to_keep]),
    args.fraction_four,
    args.fraction_six,
    args.method,
    args.alpha,
    args.reps,
    args.shots,
    args.init,
    args.swap_depth
)


filepath = basepath + filename + '.pkl'
with open(filepath, 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
remapped_full_hamiltonian: SparsePauliOp = data["remapped_full_hamiltonian"]
compiled_hamiltonian: SparsePauliOp = data['compiled_hamiltonian']

fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs.plot([hist[1] for hist in history])
axs.set_xlabel('Iteration')
axs.set_ylabel('Objective value')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/hardware/{filename}.times{"".join([str(t) for t in args.times_to_keep])}.convergence.png')

min_val = 0
# max_val = 100 (T-1)*lambda_G + T^2 + sum(weight**2)

n: int = remapped_full_hamiltonian.num_qubits

start = time()
counts = history[-1][3]
evals = evaluate_sparse_pauli_samples(list(counts.keys()), remapped_full_hamiltonian)
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = [x for xs in energies for x in xs]
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))
    

# (n = 4, T = 5. 16**5 paths. 2 optimal. 1/ 2**19 ~ 2e-06 )

def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx
print(cvar(sample_vals, 0.25))

random_samples = np.random.choice(('0', '1'), (sum(history[-1][3].values()), n))
rand_vals = evaluate_sparse_pauli_samples([''.join(sample) for sample in random_samples], remapped_full_hamiltonian)


# alpha_qaoa = (min(sample_vals) - max_val) / (min_val - max_val)
# alpha_rand = (min(rand_vals) - max_val) / (min_val - max_val)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='QAOA samples at last iter', density=True) # , approx. ratio {alpha_qaoa*100:.2f}%
axs.hist(rand_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='Random samples', density=True, alpha=0.5) # , approx. ratio {alpha_rand*100:.2f}%
# axs.hist(sample_vals_2, bins=100, label=f'Old ham sample', density=True)
# axs.hist(rand_vals_2, bins=100, label=f'Old ham rand', density=True, alpha=0.5)
ylims = axs.get_ylim()
axs.vlines(min_val, ylims[0], ylims[1], ls='--', color='k', label='Optimal solution')
axs.vlines(min(sample_vals), ylims[0], ylims[1], ls=':', color='C0', label='Best QAOA sample')
axs.vlines(min(rand_vals), ylims[0], ylims[1], ls='-.', color='C1', label='Best random sample')

logger.info(f"QAOA gap: {min(sample_vals) - min_val}")
logger.info(f"Random gap: {min(rand_vals) - min_val}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")
axs.xaxis.set_minor_locator(MultipleLocator(1))

fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/hardware/{filename}.times{"".join([str(t) for t in args.times_to_keep])}.histogram.png')
