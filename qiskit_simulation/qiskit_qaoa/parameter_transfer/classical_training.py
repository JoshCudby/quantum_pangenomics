"""Classical pre-training of QAOA parameters using MPS simulation.

Pre-trains QAOA variational parameters on a classical simulator before
transferring them to quantum hardware execution.  The training uses an MPS
(matrix product state) evaluator from the ``qaoa_training_pipeline`` library
with a COBYLA inner optimiser (up to 1000 iterations).  After training:

1. A parameter-trajectory plot is saved (beta_j and gamma_j vs. iteration).
2. The optimised parameters are bound to a QAOAAnsatz circuit, which is
   transpiled and executed on a GPU statevector Aer backend (10,000 shots).
3. The resulting sample distribution is plotted as a histogram (QAOA vs.
   random) with approximation ratios annotated in the legend.
4. The optimised parameters, final energy, sample counts, and Ising offset are
   pickled for downstream analysis or hardware upload.

CLI usage::

    python classical_training.py -f <filename> [-p <reps>]

Args:
    -f / --filename (str): Base name of the QUBO data ``.pkl`` file under
        ``/lustre/.../tangle/qubo_data_<filename>.gfa.pkl``.
    -p / --reps (int): QAOA circuit depth (default: 2).

Output:
    - Parameter trajectory PNG at
      ``plots/<filename>.depth<p>.trained_parameters.png``.
    - Sample histogram PNG at
      ``plots/<filename>.depth<p>.histogram.png``.
    - Results pickle at
      ``plots/<filename>.depth<p>.results_mps.pkl`` containing
      ``optimized_params``, ``energy``, ``result_samples``, and
      ``ising_offset``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

import argparse
from qaoa_training_pipeline.evaluation import MPSEvaluator
from qaoa_training_pipeline.training import ScipyTrainer

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator

from qiskit_optimization.problems.quadratic_objective import ObjSense
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer

from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

rng = np.random.default_rng(seed=1)
logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', default=2, type=int)
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = int(args.reps)


data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.pkl'

objective, cost_op, ising_offset = get_objective_and_hamiltonian(data_file)

parameters = list(rng.uniform(-1, 1, 2*p))

mps_evaluator = MPSEvaluator(
    use_vidal_form=True,
    threshold_circuit=1.0E-3,
    store_schmidt_values=True
)
trainer_mps = ScipyTrainer(mps_evaluator, energy_minimization=True, minimize_args={"method": "COBYLA", "options": {"maxiter": 1000, "maxfev": 1000}})
result_mps = trainer_mps.train(cost_op, parameters)
logger.info(result_mps)

x_values = range(len(result_mps["parameter_history"]))
parameter_values = []

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
for j in range(int(len(parameters)/2)):
    ax.plot(x_values, [i[j] for i in result_mps["parameter_history"]], label=f"beta_{j}")
for j in range(int(len(parameters)/2)):
    ax.plot(x_values, [i[j+int(len(parameters)/2)] for i in result_mps["parameter_history"]], label=f"gamma_{j}")


ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameter value")

fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/parameter_transfer/plots/{filename}.depth{p}.trained_parameters.png')

    
optimal_circuit = QAOAAnsatz(cost_op, reps=p).decompose()
logger.info(optimal_circuit.parameters)

optimal_circuit.assign_parameters(result_mps["optimized_params"], inplace=True)
optimal_circuit.measure_all()

backend_options = dict(
    method='statevector',
    device='GPU',
    cuStateVec_enable=True,
    blocking_enable=False,
    precision='single'
)

backend = AerSimulator(**backend_options)
transpile_circuit = transpile(optimal_circuit, backend, optimization_level=3)

res = backend.run(transpile_circuit, shots=10000).result()
logger.info(res.get_counts())

counts: dict = res.get_counts()
fig, ax = plt.subplots(1, 1, figsize=(12, 4))


n: int = cost_op.num_qubits
objective_max = objective.evaluate([1] * n)

last_counts = [[int(val) for val in key] for key in counts.keys()]
sample_vals = [objective.evaluate(sample) for sample in last_counts]


random_samples = np.random.choice([0, 1], (sum(counts.values()), n))
rand_vals = [objective.evaluate(sample) for sample in random_samples]

alpha_qaoa = (min(sample_vals)- objective_max) / (- objective_max)
alpha_rand = (min(rand_vals) - objective_max) / (- objective_max)

fig, axs = plt.subplots(1,1,figsize=(8, 5))
axs.hist(sample_vals, bins=100, label=f'QAOA samples at last iter, approx. ratio {alpha_qaoa*100:.2f}%', density=True)
axs.hist(rand_vals, bins=100, label=f'Random samples, approx. ratio {alpha_rand*100:.2f}%', density=True, alpha=0.5)
ylims = axs.get_ylim()
axs.vlines(0, ylims[0], ylims[1], ls='--', color='k', label='CPLEX solution')
axs.vlines(min(sample_vals), ylims[0], ylims[1], ls=':', color='C0', label='QAOA best sample')
axs.vlines(min(rand_vals), ylims[0], ylims[1], ls='-.', color='C1', label='Random best solution')

logger.info(f"QAOA gap: {min(sample_vals)}")
logger.info(f"Random gap: {min(rand_vals)}")

axs.legend()
axs.set_xlabel("Quadratic program objective value")
axs.set_ylabel("Sample density")

fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/parameter_transfer/plots/{filename}.depth{p}.histogram.png')

to_save = {
    "optimized_params": result_mps['optimized_params'],
    "energy": result_mps['energy'],
    "result_samples": res.get_counts(),
    "ising_offset": ising_offset
}
with open(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/parameter_transfer/plots/{filename}.depth{p}.results_mps.pkl', 'wb') as f:
    pickle.dump(to_save, f)