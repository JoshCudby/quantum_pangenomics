import numpy as np
import sys
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, dst, idst
from scipy.optimize import minimize

from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

logger.info('Starting multi-level experiment')

filename = sys.argv[1]
data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.npy'

func_tol = 0.01
standard_error_tol = 0.1
step_tol = 0.01

seed = np.random.randint(10000)
logger.info(f'Seed: {seed}')
np.random.seed(seed)
rng = np.random.default_rng(seed)


objective, hamiltonian = get_objective_and_hamiltonian(data_file)
v_objective_evaluate = np.vectorize(objective.evaluate, signature='(i)->()')


def mean_cost(samples):
    return np.mean(v_objective_evaluate(samples))


def cumulative_standard_error(samples):
    # TODO: can reduce number of function calls around here
    M = samples.shape[0]
    mean = mean_cost(samples)
    return (1 / (M * (M-1)) * np.sum((v_objective_evaluate(samples) - mean) ** 2)) ** 0.5


def parse_data(data: list[str]):
    return np.array([[int(y) for y in list(x)] for x in data])


def sample_circuit(parameters: np.ndarray, backend: AerSimulator, circuit: QuantumCircuit):
    global samples_history
    global costs_history
    parameter_binding = {
        circuit.parameters[i]: [parameters[i]] for i in range(len(parameters))
    }
    standard_error = 1
    iters = 0
    samples = None
    while standard_error > standard_error_tol and iters < max_iters:
        job = backend.run(compiled_circuit, parameter_binds=[parameter_binding], shots=shots, memory=True)
        result = job.result()
        new_samples = parse_data(result.get_memory())
        samples = np.vstack((samples, new_samples)) if samples is not None else new_samples
        standard_error = cumulative_standard_error(samples)
        iters += 1
    samples_history.extend(samples)
    costs = v_objective_evaluate(samples)
    costs_history.extend(costs)
    return np.mean(costs)


def momentum_to_position_params(momentum_params, p):
    """For testing only"""
    q = int(len(momentum_params) / 2)
    u = momentum_params[:q]
    v = momentum_params[q:]
    beta = np.dot(v, np.cos(np.outer(np.arange(1, q+1)-0.5, np.arange(1, p+1)-0.5) * np.pi / p))
    gamma = np.dot(u, np.sin(np.outer(np.arange(1, q+1)-0.5, np.arange(1, p+1)-0.5) * np.pi / p))
    return [beta, gamma]


def get_next_layer_params(params):
    p = int(len(params) / 2)
    beta, gamma = params[:p], params[p:]
    u, v = idst(gamma, type=4), idct(beta, type=4)

    new_u, new_v = np.array([*u, 0]), np.array([*v, 0])
    new_beta, new_gamma = dct(new_v, type=4), dst(new_u, type=4)
    return np.array([*new_beta, *new_gamma])


def get_perturbed_next_layer_params(params, R, alpha):
    p = int(len(params) / 2)
    beta, gamma = params[:p], params[p:]
    u, v = idst(gamma, type=4), idct(beta, type=4)
    new_u, new_v = np.zeros((R+1, p+1)), np.zeros((R+1, p+1))
    new_u[0, :-1], new_v[0, :-1] = u, v
    new_u[1:, :-1], new_v[1:, :-1] = u + alpha * rng.normal(np.zeros((R, p)), np.abs(u)), v + alpha * rng.normal(np.zeros((R, p)), np.abs(v))
    new_beta, new_gamma = dct(new_v, type=4), dst(new_u, type=4)
    return np.hstack((new_beta, new_gamma))


def callback(intermediate_result):
    logger.info(f'Inter result: {intermediate_result.fun}')
    if np.min(costs_history) < 1e-6 == 0:
        raise StopIteration

try:
    ideal_aer = AerSimulator(
        method="statevector",
        device='GPU',
        max_memory_mb=16000*0.9,
    )
except AerError as error:
    logger.error(error)

max_iters = 5
shots = 500
R = 5
alpha = 0.6
logger.info(f'Max iters in circuit sample: {max_iters}')
logger.info(f'Shots per iter: {shots}')
logger.info(f'Perturbed points: {R}')
logger.info(f'Perturbation strength: {alpha}')

init_params = {}
opt_params = {}

beta_bounds = (-np.pi/2, np.pi/2)
gamma_bounds = (-np.pi, np.pi)
p = 1
params = rng.random((2*p,)) \
    * np.array([beta_bounds[1] - beta_bounds[0]] * p + [gamma_bounds[1] - gamma_bounds[0]] * p) \
    + np.array([beta_bounds[0]] * p + [gamma_bounds[0]] * p)

# p = 1
# params = np.array([ 0.52418733, -0.94098534])

# p = 4
# params = np.array([ 0.6002043 ,  0.38650917,  0.138396  ,  0.13908375, -0.33913677,
#        -0.69919019, -1.27376677, -1.7888268 ])

init_params[p] = params

samples_history = []
costs_history = []

found_opt = False

while not found_opt and p < 11:
    circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
    circuit.measure_all()

    logger.info(f'Transpiling circuit for p={p}')
    # Create pass manager for transpilation
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
    compiled_circuit = pass_manager.run(circuit)

    # TODO: multithread the multi-opt?
    logger.info('Starting minimize')
    if len(init_params[p].shape) == 2:
        opt_x = np.zeros((init_params[p].shape[0], 2 * p))
        opt_f = np.zeros(init_params[p].shape[0])
        for i in range(init_params[p].shape[0]):
            # if len(costs_history) and np.min(costs_history) < 1e-6:
            #     break
            opt = minimize(
                sample_circuit,
                init_params[p][i],
                args=(ideal_aer, compiled_circuit),
                method='powell',
                bounds=[beta_bounds] * p + [gamma_bounds] * p,
                options={'ftol': func_tol, 'disp': True},
                callback=callback
            )
            logger.info(opt)
            opt_x[i,:] = opt.x
            opt_f[i] = opt.fun
        opt_params[p] = opt_x[np.argmin(opt_f), :]
    elif len(init_params[p].shape) == 1:
        # if len(costs_history) and np.min(costs_history) < 1e-6:
        #     break
        opt = minimize(
            sample_circuit,
            init_params[p],
            args=(ideal_aer, circuit),
            method='powell',
            bounds=[beta_bounds] * p + [gamma_bounds] * p,
            options={'ftol': func_tol, 'disp': True},
            callback=callback
        )
        logger.info(opt)
        opt_params[p] = opt.x
    else:
        raise Exception('Params should be 1D or 2D')

    # found_opt = len(costs_history) and np.min(costs_history) < 1e-6
    if not found_opt:
        params = get_perturbed_next_layer_params(opt_params[p], R, alpha)
        p += 1
        init_params[p] = params


fig, ax = plt.subplots()
ax.plot(np.minimum.accumulate(costs_history))
ax.set_xscale('log')
ax.set_xlabel('Number of measurements')
ax.set_ylabel('Minimum Cost')
fig.savefig(f'/lustre/scratch127/qpg/jc59/out/qiskit/multilevel/qaoa_costs_{filename}_seed{seed}.png', format='png')


logger.info(f'Init params: {init_params}')
logger.info(f'Opt params: {opt_params}')
logger.info(f'Opt fun: {opt.fun}')
logger.info(f'Min cost: {np.min(costs_history)}')
