import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse


from qiskit.quantum_info import SparsePauliOp
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples


from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
# parser.add_argument('-p', '--reps', type=int, default=4)
# parser.add_argument('-d', '--swap-depth', type=int, default=0)
# parser.add_argument('-m', '--memory', type=int, default=16000)
# parser.add_argument('-n', '--shots', type=int, default=1000)
# parser.add_argument('--init', choices=['ramp', 'random'], default='ramp')
# parser.add_argument('-e', '--extra', type=int, default=0)

args = parser.parse_args()


logger.info(args)

filename = args.filename
# p: int = args.reps
# shots = args.shots
# init_type = args.init
# swap_depth = args.swap_depth
alpha = 0.05



with open(f'/lustre/scratch127/qpg/jc59/hubo/simulation.optimisation.{filename}.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
hamiltonian: SparsePauliOp = data["hamiltonian"]

fig, axs = plt.subplots(1, 1, figsize=(8,5))
axs.plot([hist[1] for hist in history])
axs.set_xlabel('Iteration')
axs.set_ylabel('Objective value')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/{filename}.convergence.png')

min_val = 0
# max_val = 100

n: int = hamiltonian.num_qubits


start = time()
sample_vals = evaluate_sparse_pauli_samples(history[-1][3], hamiltonian)
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')

random_samples = np.random.choice(('0', '1'), (sum(history[-1][3].values()), n))
rand_vals = evaluate_sparse_pauli_samples([''.join(sample) for sample in random_samples], hamiltonian)


# alpha_qaoa = (min(sample_vals) - max_val) / (min_val - max_val)
# alpha_rand = (min(rand_vals) - max_val) / (min_val - max_val)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=100, label='QAOA samples at last iter', density=True) # , approx. ratio {alpha_qaoa*100:.2f}%
axs.hist(rand_vals, bins=100, label='Random samples', density=True, alpha=0.5) # , approx. ratio {alpha_rand*100:.2f}%
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

fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/hubo/{filename}.histogram.png')
