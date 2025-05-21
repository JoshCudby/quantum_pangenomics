import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger
from qiskit_prog_qaoa.utils.opt_utils import cost_function

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
method = args.method
max_iter = args.maxiter

filename_suffix = f'p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}'

with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/tangle/{filename}.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
graph = data["graph"]
n = data["n"]
T = data["T"]
lamda = data["lamda"]


fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].plot([hist[1] for hist in history])
xsamples = np.cumsum([sum(hist[3].values()) for hist in history])
axs[1].plot(xsamples, np.cumsum([hist[0] for hist in history]), label="QSim Time to generate samples")
axs[1].plot(xsamples, np.cumsum([hist[4] for hist in history]), label="Classical Time to process samples")


axs[1].legend()
axs[0].set_xlabel('Function Evaluation')
axs[0].set_ylabel('Energy')
axs[1].set_xlabel('Total number of samples')
axs[1].set_ylabel('Runtime')

fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/out/{filename}.convergence.{filename_suffix}.png')


min_val = 0
max_val = (T-1) * lamda + T ** 2 


start = time()
counts: dict = history[-1][3]
evals = [cost_function(sample, n, T, graph, lamda) for sample in counts.keys()]
print(evals)
sample_vals = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = [x for xs in sample_vals for x in xs]
print(sample_vals)
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')

rng = np.random.default_rng()
random_samples = rng.integers(1, n+1, (sum(counts.values()), T), endpoint=True)

random_samples = [
    [np.binary_repr(x, int(np.ceil(np.log2(n+2)))) for x in sample] for sample in random_samples
]
random_samples = [
    ''.join(sample) for sample in random_samples
]
rand_vals = [cost_function(sample, n, T, graph, lamda) for sample in random_samples]

alpha_qaoa = (min(sample_vals) - max_val) / (min_val - max_val)
alpha_rand = (min(rand_vals) - max_val) / (min_val - max_val)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=100, label=f'QAOA samples at last iter, approx. ratio {alpha_qaoa*100:.2f}%', density=True)
axs.hist(rand_vals, bins=100, label=f'Random samples, approx. ratio {alpha_rand*100:.2f}%', density=True, alpha=0.5)
ylims = axs.get_ylim()
axs.vlines(min_val, ylims[0], ylims[1], ls='--', color='k', label='Optimal solution')
axs.vlines(min(sample_vals), ylims[0], ylims[1], ls=':', color='C0', label='QAOA best sample')
axs.vlines(min(rand_vals), ylims[0], ylims[1], ls='-.', color='C1', label='Random best solution')

logger.info(f"QAOA gap: {min(sample_vals) - min_val}")
logger.info(f"Random gap: {min(rand_vals) - min_val}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")

fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/out/{filename}.histogram.{filename_suffix}.png')
