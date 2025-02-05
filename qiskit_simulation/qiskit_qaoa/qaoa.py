import numpy as np
import sys
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_qaoa.utils.qaoa_utils import optimize_qaoa_parameters, bayesian_optimize_qaoa_parameters
from qiskit_qaoa.utils.sample_utils import sample_optimized_circuit
from qiskit_qaoa.utils.string_utils import bitstring_to_energy, print_optimal_solution_properties
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

seed = 10

np.random.seed(seed)
rng = np.random.default_rng(seed)

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'

if len(sys.argv) > 2:
    method = str(sys.argv[2])
else:
    method = 'automatic'

if len(sys.argv) > 3:
    p = int(sys.argv[3])
else:
    p = 4

if len(sys.argv) > 4:
    use_gpu = int(sys.argv[4])
else:
    use_gpu = False

if len(sys.argv) > 5:
    cpu_mem = int(sys.argv[5])
else:
    cpu_mem = 4000

if len(sys.argv) > 6:
    optimization_method = sys.argv[6]
else:
    optimization_method = 'scipy'


data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2


mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, offset = mod.to_ising()
hamiltonian = hamiltonian.sort(weight=True)


circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
circuit.measure_all()
logger.info(f'Num qubit: {circuit.num_qubits}')

cacheblocking_required = 16 * 2 ** Q.shape[0] / (1024 ** 2) > 81559
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

logger.info(ideal_aer.available_devices())

# Create pass manager for transpilation
ideal_pm = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
ideal_circuit = ideal_pm.run(circuit)
# ideal_circuit.measure_all()

# init_params = np.zeros((2*p,))
init_params = rng.random((2*p,)) * np.array([np.pi/2, np.pi] * p)

parameter_binding = {
    ideal_circuit.parameters[i]: [init_params[i]] for i in range(len(init_params))
}

match optimization_method:
    case 'scikit':
        opt_result = bayesian_optimize_qaoa_parameters(
            ideal_aer,
            init_params,
            ideal_circuit,
            hamiltonian,
            p,
            estimator_shots=1e5
        )
    case 'scipy':
        opt_result = optimize_qaoa_parameters(
            ideal_aer,
            init_params,
            ideal_circuit,
            hamiltonian,
            p,
            estimator_shots=1e5
        )

logger.info(opt_result)
optimized_params = [float(param) for param in opt_result.x] 

sample = sample_optimized_circuit(
    ideal_circuit,
    optimized_params
)

keys = list(sample.keys())
values = list(sample.values())
most_likely_bitstring = [int(x) for x in keys[np.argmax(np.abs(values))]]
most_likely_bitstring.reverse()

logger.info(f'Model offset: {offset}')

logger.info(f'Most likely bitstring: {most_likely_bitstring}')
logger.info(f'Prob of most likely: {np.max(np.abs(values))}')
logger.info(f'Most likely energy: {bitstring_to_energy(most_likely_bitstring, hamiltonian)}')

logger.info(f'Uniform random prob: {2 ** -Q.shape[0]}')

if data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy":
    optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
    print_optimal_solution_properties(optimal, hamiltonian, sample)
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_small_test.gfa.npy":
    optimal = [
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample)
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_test.gfa.npy":
    optimal = [
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,0,0,1,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample)

    optimal = [
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,1,0,0,0,0,
        0,0,0,1,0,0,
        0,0,1,0,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample)
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_test_N4_W5.gfa.npy":
    optimal = [
        1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        1,0,0,0,0,
        0,0,0,1,0,
        0,0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample)
