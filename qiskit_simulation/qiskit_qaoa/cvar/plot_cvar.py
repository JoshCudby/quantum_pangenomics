import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.problems.quadratic_objective import ObjSense

from qiskit_qaoa.utils.argparser import get_parser
from qiskit_qaoa.utils.logging import get_logger
from qiskit_qaoa.utils.hamiltonian_utils import get_ising_offset

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

data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.npy'

filename_suffix = f'p{p}.shots{shots}.hardware{hardware}.noisy{noisy}.init{init_type}'

with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar/{filename}_cvar.{filename_suffix}.pkl', 'rb') as f:
    data = pickle.load(f)
    
history = data["history"]
singles = data["singles"]
doubles = data["doubles"]
sat_map = data["sat_map"]


fig, axs = plt.subplots(1, 2, figsize=(12,4))
axs[0].plot([hist[1] for hist in history])
xsamples = np.cumsum([sum(hist[3].values()) for hist in history])
axs[1].plot(xsamples, np.cumsum([hist[0] for hist in history]), label="QSim Time to generate samples")
axs[1].plot(xsamples, np.cumsum([hist[4] for hist in history]), label="Classical Time to process samples")


axs[1].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Energy')
axs[1].set_xlabel('Total number of samples')
axs[1].set_ylabel('Runtime')


# For time without queueing
# delta = np.median([hist[0] for hist in history])
# logger.info(f'Median time for {sum(history[0][3].values())} samples: {delta:.2f} seconds')

# cumulative_qpu = [delta]
# for idx, hist in enumerate(history[1:]):
#     t_qpu = hist[0]
#     to_add = t_qpu if t_qpu < 10 * delta else delta
#     cumulative_qpu.append(t_qpu + cumulative_qpu[-1])

# delta = np.median([hist[4] for hist in history])
# cumulative_cpu = [delta]
# for idx, hist in enumerate(history[1:]):
#     t_qpu = hist[4]
#     cumulative_cpu.append(t_qpu + cumulative_cpu[-1])

# axs[2].plot(xsamples, cumulative_qpu, label="QSim time to generate samples")
# axs[2].plot(xsamples, cumulative_cpu, label="CPU time to process samples")
# axs[2].legend()
# axs[2].set_xlabel('Total number of samples')
# axs[2].set_ylabel('Runtime without queueing (s)')
# axs[2].set_yscale('log')


fig.tight_layout()
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/{filename}.convergence.{filename_suffix}.png')


qp = QuadraticProgram()
qp.from_ising(singles + doubles, offset = get_ising_offset(data_file))
cplex_res = CplexOptimizer().solve(qp)
reordered = [sat_map[x] for x in range(len(cplex_res.x))]
logger.info(f'Optimal: {cplex_res.x[reordered]}, Fval: {cplex_res.fval}')

qp_max = QuadraticProgram()
qp_max.from_ising(singles + doubles, offset = get_ising_offset(data_file))
qp_max.objective._sense = ObjSense.MAXIMIZE
cplex_res_max = CplexOptimizer().solve(qp_max)

# original_qp = get_qp(data_file)
# original_cplex_res = CplexOptimizer().solve(original_qp)
# logger.info(f'Optimal: {original_cplex_res.x}, Fval: {original_cplex_res.fval}')

n = qp.get_num_vars()

last_counts = [[int(val) for val in bit_str[::-1]] for bit_str in history[-1][3]]
start = time()
sample_vals = [qp.objective.evaluate(sample) for sample in last_counts]
elapsed = time() - start
logger.info(f'Time to compute energies {elapsed}')

random_samples = np.random.choice((0, 1), (sum(history[-1][3].values()), n))
rand_vals = [qp.objective.evaluate(sample) for sample in random_samples]

alpha_qaoa = (min(sample_vals) - cplex_res_max.fval) / (cplex_res.fval - cplex_res_max.fval)
alpha_rand = (min(rand_vals) - cplex_res_max.fval) / (cplex_res.fval - cplex_res_max.fval)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=100, label=f'QAOA samples at last iter, approx. ratio {alpha_qaoa*100:.2f}%', density=True)
axs.hist(rand_vals, bins=100, label=f'Random samples, approx. ratio {alpha_rand*100:.2f}%', density=True, alpha=0.5)
ylims = axs.get_ylim()
axs.vlines(cplex_res.fval, ylims[0], ylims[1], ls='--', color='k', label='CPLEX solution')
axs.vlines(min(sample_vals), ylims[0], ylims[1], ls=':', color='C0', label='QAOA best sample')
axs.vlines(min(rand_vals), ylims[0], ylims[1], ls='-.', color='C1', label='Random best solution')

logger.info(f"QAOA gap: {min(sample_vals) - cplex_res.fval}")
logger.info(f"Random gap: {min(rand_vals) - cplex_res.fval}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")

fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/out/{filename}.histogram.{filename_suffix}.png')
