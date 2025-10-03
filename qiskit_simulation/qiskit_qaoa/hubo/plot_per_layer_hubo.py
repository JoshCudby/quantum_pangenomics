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
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-M', '--method', type=str)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random', 'warm'], default='ramp')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('-a', '--alpha', type=float)
parser.add_argument('-C', '--coupling-map', choices=['line', 'grid'])

args = parser.parse_args()


logger.info(args)
basepath='/lustre/scratch127/qpg/jc59/hubo/'


filename='simulation.{}.per_layer.{}.extra{}.four{}.six{}.method{}.cvar{}.p{}.shots{}.init{}'.format(
    args.coupling_map,
    args.filename,
    args.extra,
    args.fraction_four,
    args.fraction_six,
    args.method,
    args.alpha,
    args.reps,
    args.shots,
    args.init,
)

filepath = basepath + filename + '.pkl'
with open(filepath, 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
full_hamiltonian: SparsePauliOp = data["full_hamiltonian"]

fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs.plot([hist[1] for hist in history])
axs.set_xlabel('Iteration')
axs.set_ylabel('Objective value')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/{filename}.convergence.png')

min_val = 0
# max_val = 100 (T-1)*lambda_G + T^2 + sum(weight**2)

n: int = full_hamiltonian.num_qubits

start = time()
counts = history[-1][3]
evals = evaluate_sparse_pauli_samples(list(counts.keys()), full_hamiltonian)
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = [x for xs in energies for x in xs]
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))
    

random_samples = np.random.choice(('0', '1'), (sum(history[-1][3].values()), n))
rand_vals = evaluate_sparse_pauli_samples([''.join(sample) for sample in random_samples], full_hamiltonian)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='QAOA samples at last iter', density=True) 
axs.hist(rand_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='Random samples', density=True, alpha=0.5) 
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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/{filename}.histogram.png')


start = time()
counts = history[0][3]
evals = evaluate_sparse_pauli_samples(list(counts.keys()), full_hamiltonian)
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = [x for xs in energies for x in xs]
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')
counter = Counter(sample_vals)
print(counter.most_common(10))
    

random_samples = np.random.choice(('0', '1'), (sum(history[-1][3].values()), n))
rand_vals = evaluate_sparse_pauli_samples([''.join(sample) for sample in random_samples], full_hamiltonian)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='QAOA samples at last iter', density=True) 
axs.hist(rand_vals, bins=np.arange(0, np.max(list(counter.keys()))+2)-0.5, label='Random samples', density=True, alpha=0.5) 
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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/{filename}.pre_qaoa.histogram.png')