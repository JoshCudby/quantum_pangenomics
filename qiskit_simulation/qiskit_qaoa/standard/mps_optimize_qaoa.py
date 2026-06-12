"""MPS (matrix product state) simulator specialisation for QAOA optimisation.

Builds a QAOAAnsatz from an explicitly loaded QUBO matrix (rather than a
pre-built Hamiltonian utility), configures Qiskit-Aer's
``matrix_product_state`` method with a capped bond dimension for memory
efficiency, then optimises the variational parameters using either
basin-hopping (scipy) or Bayesian (scikit-optimize) search.  Intended for
larger circuits where a full statevector would exceed available memory.

CLI usage::

    python mps_optimize_qaoa.py <seed> [data_file] [p] [cpu_mem] [optimization_method]

Args:
    seed (int): Random seed (required, positional argv[1]).
    data_file (str): Path to the ``.npy`` QUBO data file
        (default: ``qubo_data_trivial.gfa.npy``).
    p (int): QAOA circuit depth (default: 4).
    cpu_mem (int): CPU memory limit in MB (default: 4000).
    optimization_method (str): ``"scipy"`` (basin-hopping) or ``"scikit"``
        (Bayesian optimisation) (default: ``"scipy"``).

Output:
    Saves the optimised parameter vector to
    ``qaoa_params_mps_n<qubits>_p<p>_seed<seed>.npy``.
"""

import numpy as np
import sys
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_qaoa.utils.qaoa_utils import basinhopping_optimize_qaoa_parameters, bayesian_optimize_qaoa_parameters

from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
else:
    raise Exception('No seed provided')

if len(sys.argv) > 2:
    data_file = sys.argv[2]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'

if len(sys.argv) > 3:
    p = int(sys.argv[3])
else:
    p = 4

if len(sys.argv) > 4:
    cpu_mem = int(sys.argv[4])
else:
    cpu_mem = 4000

if len(sys.argv) > 5:
    optimization_method = sys.argv[5]
else:
    optimization_method = 'scipy'

np.random.seed(seed)
rng = np.random.default_rng(seed)



data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2

normalisation = np.max(np.abs(Q))
Q = Q / normalisation
offset = offset / normalisation


mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, offset = mod.to_ising()
hamiltonian = hamiltonian.sort(weight=True)


circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
circuit.measure_all()
logger.info(f'Num qubit: {circuit.num_qubits}')


try:
    ideal_aer = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=5,
        max_memory_mb=cpu_mem*0.9,
    )
except AerError as error:
    logger.error(error)

logger.info(ideal_aer.available_devices())

# Create pass manager for transpilation
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
compiled_circuit = pass_manager.run(circuit)

# init_params = np.zeros((2*p,))
init_params = rng.random((2*p,)) * np.array([np.pi/2, np.pi] * p)

parameter_binding = {
    compiled_circuit.parameters[i]: [init_params[i]] for i in range(len(init_params))
}

match optimization_method:
    case 'scikit':
        opt_result = bayesian_optimize_qaoa_parameters(
            ideal_aer,
            init_params,
            compiled_circuit,
            hamiltonian,
            p,
            estimator_shots=1e5
        )
    case 'scipy':
        opt_result = basinhopping_optimize_qaoa_parameters(
            ideal_aer,
            init_params,
            compiled_circuit,
            hamiltonian,
            p,
            estimator_shots=1e5
        )
        logger.info(opt_result)

optimized_params = [float(param) for param in opt_result.x] 
logger.info(f'Optimized params: {optimized_params}')

to_save = f'/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_params_mps_n{circuit.num_qubits}_p{p}_seed{seed}.npy'
np.save(to_save, optimized_params)