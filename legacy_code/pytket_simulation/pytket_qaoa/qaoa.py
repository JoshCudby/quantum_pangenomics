import numpy as np
from jax import numpy as jnp, value_and_grad, jit, devices
import matplotlib.pyplot as plt
import qujax
import sys
from pytket.extensions.qujax import tk_to_qujax
from pytket_qaoa.utils.qaoa_utils import qaoa_circuit, Q_to_Ising
from pytket_qaoa.utils.logging import get_logger
from pytket.transform import Transform
from pytket.circuit.display import get_circuit_renderer
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from scipy.optimize import minimize


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
Q = Q / np.max(np.abs(Q))
n_qubits = Q.shape[0]

h, J, offset = Q_to_Ising(Q, offset)
terms = list(J.items()) + [((key,), val) for key, val in h.items()]
hamiltonian_qubit_inds = list(J.keys()) + [(key,) for key in h.keys()]
coefficients = list(J.values()) + list(h.values())
hamiltonian_gates = [
    ['Z'] * len(x) for x in hamiltonian_qubit_inds
]

logger.info(devices())
logger.info(f'Num qubits: {n_qubits}')
logger.info(f'Gates:\t {hamiltonian_gates}')
logger.info(f'Qubits:\t {hamiltonian_qubit_inds}')
logger.info(f'Coefficients:\t {coefficients}')
logger.info(terms)

circuit, parameters = qaoa_circuit(n_qubits, p, terms)

symbol_map = {parameters[i]: i for i in range(len(parameters))}

logger.info(f'Num 1 qubit gates: {circuit.n_1qb_gates()}')
logger.info(f'Num 2 qubit gates: {circuit.n_2qb_gates()}')

param_to_st = tk_to_qujax(circuit, symbol_map=symbol_map)
st_to_expectation = qujax.get_statetensor_to_expectation_func(
    hamiltonian_gates, hamiltonian_qubit_inds, coefficients
)

def param_to_expectation(param): 
    return st_to_expectation(param_to_st(param))


bounds = [np.pi/2, np.pi] * p
init_params = rng.random((2*p,)) * np.array(bounds)
logger.info(f'Init params: {init_params}')
init_cost = param_to_expectation(init_params)
n_steps = 20
stepsize = 1 / n_qubits
params = init_params
best_cost = init_cost
best_params = init_params
cost_vals = jnp.zeros(n_steps)
cost_vals = cost_vals.at[0].set(param_to_expectation(params))



# logger.info('Compiling value and grad')
# cost_and_grad = jit(value_and_grad(param_to_expectation))
# logger.info('Starting gradient descent')
# # Simple gradient descent
# for step in range(1, n_steps):
#     cost_val, cost_grad = cost_and_grad(params)
#     if cost_val < best_cost:
#         best_cost = cost_val
#         best_params = params
#     cost_vals = cost_vals.at[step].set(cost_val)
#     params = params - stepsize * cost_grad
#     logger.info(f'Iteration: {step}, Cost: {cost_val}')


iteration = 0
logger.info('Compiling value')
cost = jit(param_to_expectation)

def callback(xk):
    global iteration
    global cost_vals
    iteration += 1
    logger.info(f'Iteration {iteration}. x: {xk}')
    cost_vals = cost_vals.at[iteration].set(cost(xk))

eps = 1e-6
logger.info('Starting COBYLA')
opt_result = minimize(
    cost,
    init_params,
    method='COBYLA',
    bounds=list(zip([eps]*2*p, np.array(bounds) - eps)),
    callback=callback,
    options={'disp': True, 'maxiter': n_steps }, #, 'rhobeg': stepsize},
    tol=1e-2,
)
logger.info(opt_result)
best_params = opt_result.x
best_cost = opt_result.fun



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
circuit.measure_all()

logger.info('Rendering circuits')
renderer = get_circuit_renderer()
html = renderer.render_circuit_as_html([symbol_circ, circuit], orient='column')
with open(f'/lustre/scratch127/qpg/jc59/out/pytket/qaoa_circuit_n{n_qubits}_p{p}.html', 'w') as file:
    file.write(html)


logger.info(f'Num 1 qubit gates: {circuit.n_1qb_gates()}')
logger.info(f'Num 2 qubit gates: {circuit.n_2qb_gates()}')
api_offline = QuantinuumAPIOffline()
backend = QuantinuumBackend(device_name="H2-2LE", api_handler=api_offline)
logger.info(backend.backend_info)

logger.info('Compiling circuit')
compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=3)

logger.info(f'Num 1 qubit gates: {circuit.n_1qb_gates()}')
logger.info(f'Num 2 qubit gates: {circuit.n_2qb_gates()}')

handle = backend.process_circuit(compiled_circuit, n_shots=10)

result = backend.get_result(handle)


dist = result.get_empirical_distribution()
logger.info(dist)

if data_file == '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy':
    optimal = [1,0,0,0,0,1,0,0,0,0,1,0]

    dist_forward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    optimal.reverse()
    dist_backward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    logger.info(dist_forward_opt)
    logger.info(dist_backward_opt)

if data_file == '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_small_test.gfa.npy':
    optimal = [
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1,
        ]
    
    dist_forward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    optimal.reverse()
    dist_backward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    logger.info(dist_forward_opt)
    logger.info(dist_backward_opt)


    optimal = [
        1,0,0,0,
        0,1,0,0,
        1,0,0,0,
        0,0,1,0,
        0,1,0,0,
        0,0,0,1,
    ]
    dist_forward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    optimal.reverse()
    dist_backward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
    logger.info(dist_forward_opt)
    logger.info(dist_backward_opt)

