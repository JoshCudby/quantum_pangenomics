from datetime import datetime
from pytket.utils.operators import QubitPauliOperator
from pytket.pauli import Pauli, QubitPauliString
from pytket.circuit import Qubit
from scipy.optimize import minimize
import numpy as np
import sys
import qnexus as qnx
from pytket_qaoa.utils.logging import get_logger
from pytket_qaoa.utils.qaoa_utils import Q_to_Ising, qaoa_circuit
from pytket_qaoa.utils.objective import Objective


logger = get_logger(__name__)
seed = 13
rng = np.random.default_rng(seed)
jobname_suffix = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')


# set up the project
project_ref = qnx.projects.create(
    name=f'QAOA_nexus_{str(datetime.now())}',
    description='QAOA done with qnexus',
)

# set this in the context
qnx.context.set_active_project(project_ref)

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'

if len(sys.argv) > 2:
    p = int(sys.argv[2])
else:
    p = 4


data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2
n_qubits = Q.shape[0]

h, J, offset = Q_to_Ising(Q, offset)

terms = list(J.items()) + [((key,), val) for key, val in h.items()]
symbolic_circuit, parameters = qaoa_circuit(n_qubits, p, terms)


J_terms = {
    QubitPauliString({Qubit(key[0]): Pauli.Z, Qubit(key[1]): Pauli.Z}) : val for key, val in J.items()
}
h_terms = {
    QubitPauliString({Qubit(key): Pauli.Z}) : val for key, val in h.items()
}
term_sum = {}
term_sum.update(J_terms)
term_sum.update(h_terms)
hamiltonian = QubitPauliOperator(term_sum)


qnx.projects.add_property(
    name='iteration', 
    property_type='int', 
    description='The iteration number in QAOA',
)

# Set up the properties for the symbolic circuit parameters
for sym in symbolic_circuit.free_symbols():
    qnx.projects.add_property(
        name=str(sym), 
        property_type='float',
        description=f'QAOA {str(sym)} parameter', 
    )


ansatz_ref = qnx.circuits.upload(
    circuit=symbolic_circuit,
    name='QAOA ansatz',
    description='The ansatz circuit for QAOA',
)


n_shots_per_circuit = 1000
n_iterations = 50

objective = Objective(
    symbolic_circuit=ansatz_ref,
    problem_hamiltonian=hamiltonian,
    n_shots_per_circuit=n_shots_per_circuit,
    n_iterations=n_iterations,
    target=qnx.QuantinuumConfig(device_name='H1-1LE')
)
init_params = rng.random((2*p,)) * np.array([np.pi, np.pi/2] * p)

def callback(xk):
    logger.info(f'Current x: {xk}')

logger.info('Starting minimize')
result = minimize(
    objective,
    init_params,
    method='COBYLA',
    bounds=[(0, np.pi), (0, np.pi/2)] * p,
    callback=callback,
    options={'disp': True, 'maxiter': objective._niters, 'rhobeg': 0.01 / n_qubits},
    tol=1e-2,
)
logger.info(result)

logger.info(f'Optimized cost: {result.fun}')
logger.info(f'Optimized params: {result.x}')

symbol_circ = symbolic_circuit.copy()
optimised_symbol_map = {parameters[i]: result.x[i] for i in range(len(parameters))}
symbolic_circuit.symbol_substitution(optimised_symbol_map)
symbolic_circuit.measure_all()
ref = qnx.circuits.upload(circuit=symbolic_circuit, name=f'Optimized_QAOA_Circuit_n{n_qubits}_p{p}_{jobname_suffix}')


logger.info('Starting nexus compile job')
ref_compile_job = qnx.start_compile_job(
    circuits=[ref],
    backend_config=qnx.QuantinuumConfig(device_name='H1-1LE'),
    optimisation_level=2,
    name=f'compilation-job-{jobname_suffix}'
)

qnx.jobs.wait_for(ref_compile_job)
ref_compiled_circuit = qnx.jobs.results(ref_compile_job)[0].get_output()
logger.info('ref compiled circuit')
logger.info(ref_compiled_circuit)

logger.info('Starting nexus execute job')
shots=500
ref_execute_job = qnx.start_execute_job(
    circuits=[ref_compiled_circuit],
    n_shots=[shots],
    backend_config=qnx.QuantinuumConfig(device_name='H1-1LE'),
    name=f'execution-job-shots{shots}-{jobname_suffix}'
)
qnx.jobs.wait_for(ref_execute_job)
ref_result = qnx.jobs.results(ref_execute_job)[0]
backend_result = ref_result.download_result()
distribution = backend_result.get_empirical_distribution()
logger.info(distribution)