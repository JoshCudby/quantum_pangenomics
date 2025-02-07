import numpy as np
from jax import numpy as jnp, value_and_grad, jit
import matplotlib.pyplot as plt
import qujax
import sys
from pytket.extensions.qujax import tk_to_qujax
from pytket_qaoa.utils.qaoa_utils import qaoa_circuit, Q_to_Ising
from pytket_qaoa.utils.logging import get_logger
from pytket.transform import Transform
from pytket.circuit.display import get_circuit_renderer
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
import qnexus as qnx
import datetime

jobname_suffix = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

project = qnx.projects.get_or_create('Pytket QAOA simulation')
qnx.context.set_active_project(project)
config = qnx.QuantinuumConfig(device_name='H1-Emulator')


logger = get_logger(__name__)
seed = 13
rng = np.random.default_rng(seed)

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
hamiltonian_qubit_inds = list(J.keys()) + [(key,) for key in h.keys()]
coefficients = list(J.values()) + list(h.values())
hamiltonian_gates = [
    ['Z'] * len(x) for x in hamiltonian_qubit_inds
]

logger.info(f'Num qubits: {n_qubits}')
logger.info(f'Gates:\t {hamiltonian_gates}')
logger.info(f'Qubits:\t {hamiltonian_qubit_inds}')
logger.info(f'Coefficients:\t {coefficients}')
logger.info(terms)

circuit, parameters = qaoa_circuit(n_qubits, p, terms)

symbol_map = {parameters[i]: i for i in range(len(parameters))}

param_to_st = tk_to_qujax(circuit, symbol_map=symbol_map)
st_to_expectation = qujax.get_statetensor_to_expectation_func(
    hamiltonian_gates, hamiltonian_qubit_inds, coefficients
)

def param_to_expectation(param): 
    return st_to_expectation(param_to_st(param))

logger.info('Compiling value and grad')
cost_and_grad = jit(value_and_grad(param_to_expectation))


init_params = rng.random((2*p,)) * np.array([np.pi/2, np.pi] * p)
init_cost = param_to_expectation(init_params)
n_steps = 150
stepsize = 0.01 / n_qubits
params = init_params
best_cost = init_cost
best_params = init_params
cost_vals = jnp.zeros(n_steps)
cost_vals = cost_vals.at[0].set(param_to_expectation(params))

# Simple gradient descent
for step in range(1, n_steps):
    cost_val, cost_grad = cost_and_grad(params)
    if cost_val < best_cost:
        best_cost = cost_val
        best_params = params
    cost_vals = cost_vals.at[step].set(cost_val)
    params = params - stepsize * cost_grad
    logger.info(f'Iteration: {step}, Cost: {cost_val}')

logger.info(f'Best cost: {best_cost}')
logger.info(f'Best params: {best_params}')

fig, ax = plt.subplots()
ax.plot(cost_vals)
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
fig.savefig(f'/lustre/scratch127/qpg/jc59/out/pytket/qaoa_cost_n{n_qubits}_p{p}')

symbol_circ = circuit.copy()
optimised_symbol_map = {parameters[i]: best_params[i] for i in range(len(parameters))}
circuit.symbol_substitution(optimised_symbol_map)
logger.info('Optimising circuit')
Transform.OptimiseStandard().apply(circuit)

logger.info('Rendering circuits')
renderer = get_circuit_renderer()
html = renderer.render_circuit_as_html([symbol_circ, circuit], orient='column')
with open(f'/lustre/scratch127/qpg/jc59/out/pytket/qaoa_circuit_n{n_qubits}_p{p}.html', 'w') as file:
    file.write(html)



logger.info('Uploading circuit to nexus')
ref = qnx.circuits.upload(circuit=circuit, name=f'Optimized_QAOA_Circuit_n{n_qubits}_p{p}_{jobname_suffix}')

logger.info('Starting nexus compile job')
ref_compile_job = qnx.start_compile_job(
    circuits=[ref],
    backend_config=config,
    optimisation_level=2,
    name=f'compilation-job-{jobname_suffix}'
)

qnx.jobs.wait_for(ref_compile_job)
ref_compiled_circuit = qnx.jobs.results(ref_compile_job)[0].get_output()
logger.info('ref compiled circuit')
logger.info(ref_compiled_circuit)

logger.info('Starting nexus execute job')
shots=100
ref_execute_job = qnx.start_execute_job(
    circuits=[ref_compiled_circuit],
    n_shots=[shots],
    backend_config=config,
    name=f'execution-job-shots{shots}-{jobname_suffix}'
)
qnx.jobs.wait_for(ref_execute_job)
ref_result = qnx.jobs.results(ref_execute_job)[0]
backend_result = ref_result.download_result()
distribution = backend_result.get_empirical_distribution()
logger.info(distribution)



api_offline = QuantinuumAPIOffline()
backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
backend.default_compilation_pass().apply(circ)
