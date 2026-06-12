"""Standard QUBO QAOA optimisation using Qiskit-Aer.

Builds a QAOAAnsatz circuit from a QUBO cost Hamiltonian loaded from a ``.npy``
data file and optimises the variational parameters (beta, gamma) using either
basin-hopping (scipy) or Bayesian (scikit-optimize) search.  The simulator
backend can be a GPU statevector, multi-GPU statevector with cache-blocking, or
an MPS (matrix product state) backend.

CLI usage::

    python optimize_qaoa.py <seed> [data_file] [method] [p] [use_gpu] [cpu_mem] [optimization_method]

Args:
    seed (int): Random seed (required, positional argv[1]).
    data_file (str): Path to the ``.npy`` QUBO data file
        (default: ``qubo_data_trivial.gfa.npy``).
    method (str): Aer simulation method, e.g. ``"automatic"``,
        ``"statevector"``, ``"matrix_product_state"`` (default:
        ``"automatic"``).
    p (int): QAOA circuit depth / number of repetitions (default: 4).
    use_gpu (int): Non-zero to enable GPU device (default: 0 / CPU).
    cpu_mem (int): CPU memory limit in MB for the simulator (default: 4000).
    optimization_method (str): Parameter search strategy — ``"scipy"`` for
        basin-hopping or ``"scikit"`` for Bayesian optimisation
        (default: ``"scipy"``).

Output:
    Saves the optimised parameter vector as a ``.npy`` file at
    ``qaoa_params_n<qubits>_p<p>_seed<seed>.npy``.
"""

import numpy as np
import sys
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_qaoa.utils.estimator_with_history import EstimatorWithHistory
from qiskit_qaoa.utils.qaoa_utils import basinhopping_optimize_qaoa_parameters, bayesian_optimize_qaoa_parameters
from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
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
    method = str(sys.argv[3])
else:
    method = 'automatic'

if len(sys.argv) > 4:
    p = int(sys.argv[4])
else:
    p = 4

if len(sys.argv) > 5:
    use_gpu = int(sys.argv[5])
else:
    use_gpu = False

if len(sys.argv) > 6:
    cpu_mem = int(sys.argv[6])
else:
    cpu_mem = 4000

if len(sys.argv) > 7:
    optimization_method = sys.argv[7]
else:
    optimization_method = 'scipy'


np.random.seed(seed)
rng = np.random.default_rng(seed)

_, hamiltonian = get_objective_and_hamiltonian(data_file)


circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
circuit.measure_all()
logger.info(f'Num qubit: {circuit.num_qubits}')
logger.info(f'(Possibly Unordered?) parameters: {circuit.parameters}')

cacheblocking_required = 16 * 2 ** circuit.num_qubits / (1024 ** 2) > 81559
logger.info(f'Cacheblocking required: {cacheblocking_required}')

try:
    ideal_aer = AerSimulator(
        method=method,
        matrix_product_state_max_bond_dimension=5,
        device='GPU' if use_gpu else 'CPU',
        blocking_enable=cacheblocking_required, blocking_qubits=23,
        max_memory_mb=cpu_mem*0.9,
    )
except AerError as error:
    logger.error(error)

estimator = EstimatorWithHistory.from_backend(ideal_aer)
estimator.options.default_shots = 1e4
estimator.options.default_precision = 0

# Create pass manager for transpilation
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
compiled_circuit = pass_manager.run(circuit)

# init_params = np.zeros((2*p,))
beta_bounds = (-np.pi/2, np.pi/2)
gamma_bounds = (-np.pi, np.pi)

init_params = rng.random((2*p,)) \
    * np.array([beta_bounds[1] - beta_bounds[0]] * p + [gamma_bounds[1] - gamma_bounds[0]] * p) \
    + np.array([beta_bounds[0]] * p + [gamma_bounds[0]] * p)

match optimization_method:
    case 'scikit':
        opt_result = bayesian_optimize_qaoa_parameters(
            estimator,
            init_params,
            compiled_circuit,
            hamiltonian,
            p,
            bounds=[gamma_bounds, beta_bounds],
            estimator_shots=1e5,
            
        )
    case 'scipy':
        opt_result = basinhopping_optimize_qaoa_parameters(
            estimator,
            init_params,
            compiled_circuit,
            hamiltonian,
            p,
            bounds=[gamma_bounds, beta_bounds],
            estimator_shots=1e5
        )
        logger.info(opt_result)

optimized_params = [float(param) for param in opt_result.x] 
logger.info(f'Optimized params: {optimized_params}')

to_save = f'/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_params_n{circuit.num_qubits}_p{p}_seed{seed}.npy'
np.save(to_save, optimized_params)