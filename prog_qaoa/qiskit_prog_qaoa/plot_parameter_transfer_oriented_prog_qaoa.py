import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger
from qiskit_prog_qaoa.utils.opt_utils import oriented_cost_function

logger = get_logger(__name__)
parser = get_parser()
parser.add_argument('-t', '--transfer')
args = parser.parse_args()

transfer = args.transfer
filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init
method = args.method
max_iter = args.maxiter

filename_suffix = f'p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}'

# with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/oriented/{filename}.{filename_suffix}.pkl', 'rb') as f:
with open(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/out/oriented/transfer.{transfer}.{filename}.{filename_suffix}.pkl', 'rb') as f:
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
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/out/transfer.oriented.{transfer}.{filename}.convergence.{filename_suffix}.png')

nodes_weights = list(graph.nodes(data="weight", default=0)) # type: ignore

min_val = 0
# Assume there is a 0-weight node without a self-edge. Then this max val is T visits to that node.
max_val = (T-1) * lamda + T ** 2 + int(sum(x[1] ** 2 for x in nodes_weights[::2]))
print(max_val)

start = time()
counts: dict = history[-1][3]
evals = [oriented_cost_function(sample, n, T, graph, lamda) for sample in counts.keys()]
sample_vals = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
sample_vals = [x for xs in sample_vals for x in xs]
sample_vals = [x for x in sample_vals if not x == 10000]
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')

rng = np.random.default_rng()
random_samples = rng.integers(2, 2*(n+1), (sum(counts.values()), T), endpoint=True)

random_samples = [
    [np.binary_repr(x, 1+int(np.ceil(np.log2(n+2)))) for x in sample] for sample in random_samples
]
random_samples = [
    ''.join(sample) for sample in random_samples
]
rand_vals = [oriented_cost_function(sample, n, T, graph, lamda) for sample in random_samples]

alpha_qaoa = (min(sample_vals) - max_val) / (min_val - max_val)
alpha_rand = (min(rand_vals) - max_val) / (min_val - max_val)

print(len(sample_vals))
print(len(rand_vals))
fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=max_val-min_val, label=f'QAOA samples at last iter, approx. ratio {alpha_qaoa*100:.2f}%', density=True)
axs.hist(rand_vals, bins=max_val-min_val, label=f'Random samples, approx. ratio {alpha_rand*100:.2f}%', density=True, alpha=0.5)
ylims = axs.get_ylim()
axs.vlines(min_val, ylims[0], ylims[1], ls='--', color='k', label='Optimal energy')
axs.vlines(max_val, ylims[0], ylims[1], ls='-', color='k', label='Maximum energy')
axs.vlines(min(sample_vals), ylims[0], ylims[1], ls=':', color='C0', label='QAOA best energy')
axs.vlines(min(rand_vals), ylims[0], ylims[1], ls='-.', color='C1', label='Random best energy')

logger.info(f"QAOA gap: {min(sample_vals) - min_val}")
logger.info(f"Random gap: {min(rand_vals) - min_val}")

axs.legend()
axs.set_xlabel("Objective value")
axs.set_ylabel("Sample density")

fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/out/transfer.oriented.{transfer}.{filename}.histogram.{filename_suffix}.png')
