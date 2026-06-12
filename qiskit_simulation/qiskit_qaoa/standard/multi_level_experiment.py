"""Hierarchical (multi-level) QAOA using DCT/DST parameter extrapolation.

Implements the multi-level QAOA strategy: parameters are optimised at circuit
depth p=1 using Powell's method, then extrapolated to depth p+1 via inverse/
forward DCT-IV and DST-IV transforms (the "momentum-space" representation of
the beta and gamma schedules).  A set of ``R`` perturbed candidates is also
generated at each step to avoid local optima.  The outer loop increments p up
to a maximum depth or until a convergence criterion is met.

CLI usage::

    python multi_level_experiment.py <filename>

Args:
    filename (str): Base name (without path or extension) of the QUBO data
        file under ``/lustre/.../qubo_data_<filename>.gfa.npy``.

Output:
    A PNG plot of cumulative minimum cost vs. number of measurements saved to
    the multilevel output directory.  Optimised parameters are logged at each
    level.
"""

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
    """Compute the mean QUBO objective value over a batch of bitstring samples.

    Args:
        samples: Array of shape ``(M, n_qubits)`` containing binary samples.

    Returns:
        float: Mean objective value across all samples.
    """
    return np.mean(v_objective_evaluate(samples))


def cumulative_standard_error(samples):
    """Estimate the standard error of the mean cost over accumulated samples.

    Args:
        samples: Array of shape ``(M, n_qubits)`` containing binary samples.

    Returns:
        float: Standard error of the mean cost estimate.
    """
    # TODO: can reduce number of function calls around here
    M = samples.shape[0]
    mean = mean_cost(samples)
    return (1 / (M * (M-1)) * np.sum((v_objective_evaluate(samples) - mean) ** 2)) ** 0.5


def parse_data(data: list[str]):
    """Convert a list of measurement bitstrings to a binary integer array.

    Args:
        data: List of bitstrings as returned by ``job.result().get_memory()``.

    Returns:
        np.ndarray: Integer array of shape ``(len(data), n_bits)``.
    """
    return np.array([[int(y) for y in list(x)] for x in data])


def sample_circuit(parameters: np.ndarray, backend: AerSimulator, circuit: QuantumCircuit):
    """Run the QAOA circuit and return the mean cost, accumulating samples.

    Runs the circuit repeatedly until the cumulative standard error of the mean
    cost falls below ``standard_error_tol`` or ``max_iters`` shots batches have
    been taken.  All samples and costs are appended to the module-level
    ``samples_history`` and ``costs_history`` lists.

    Args:
        parameters: 1-D array of variational parameters (betas then gammas).
        backend: Configured AerSimulator backend to execute the circuit.
        circuit: Transpiled and compiled QAOA circuit with bound parameters.

    Returns:
        float: Mean QUBO objective value over all accumulated samples.
    """
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
    """Convert DCT/DST frequency-domain parameters to position-domain (beta, gamma).

    Used for testing the DCT-IV / DST-IV basis representation.  The momentum
    (frequency) coefficients ``v`` and ``u`` are mapped back to ``beta`` and
    ``gamma`` schedules of length ``p`` via the cosine and sine series defined
    in the Fourier interpolation scheme.

    Note:
        This function is provided for testing purposes only.

    Args:
        momentum_params: 1-D array of length ``2*q`` containing ``q``
            cosine-basis coefficients (for beta) followed by ``q`` sine-basis
            coefficients (for gamma).
        p: Target circuit depth (number of QAOA layers).

    Returns:
        list: ``[beta, gamma]`` where each element is a 1-D array of length
        ``p``.
    """
    q = int(len(momentum_params) / 2)
    u = momentum_params[:q]
    v = momentum_params[q:]
    beta = np.dot(v, np.cos(np.outer(np.arange(1, q+1)-0.5, np.arange(1, p+1)-0.5) * np.pi / p))
    gamma = np.dot(u, np.sin(np.outer(np.arange(1, q+1)-0.5, np.arange(1, p+1)-0.5) * np.pi / p))
    return [beta, gamma]


def get_next_layer_params(params):
    """Extrapolate optimised p-layer parameters to p+1 layers via DCT/DST.

    Converts the current (beta, gamma) schedule to frequency-domain
    coefficients using inverse DCT-IV (for beta) and inverse DST-IV (for
    gamma), zero-pads by one coefficient, then transforms back to produce a
    p+1 layer schedule.

    Args:
        params: 1-D array of length ``2*p`` containing ``p`` beta values
            followed by ``p`` gamma values.

    Returns:
        np.ndarray: 1-D array of length ``2*(p+1)`` with the extrapolated
        beta and gamma schedules concatenated.
    """
    p = int(len(params) / 2)
    beta, gamma = params[:p], params[p:]
    u, v = idst(gamma, type=4), idct(beta, type=4)

    new_u, new_v = np.array([*u, 0]), np.array([*v, 0])
    new_beta, new_gamma = dct(new_v, type=4), dst(new_u, type=4)
    return np.array([*new_beta, *new_gamma])


def get_perturbed_next_layer_params(params, R, alpha):
    """Generate R+1 candidate parameter sets for the next QAOA layer.

    Extrapolates the current parameters to depth p+1 (zero-padded DCT/DST as
    in :func:`get_next_layer_params`) and additionally creates ``R`` randomly
    perturbed variants.  Perturbation magnitude is proportional to the
    absolute value of each frequency coefficient, scaled by ``alpha``.

    Args:
        params: 1-D array of length ``2*p`` (betas then gammas).
        R: Number of perturbed candidate initialisation points to generate.
        alpha: Perturbation strength relative to the frequency-coefficient
            magnitude.

    Returns:
        np.ndarray: Array of shape ``(R+1, 2*(p+1))`` where row 0 is the
        unperturbed extrapolation and rows 1..R are perturbed variants.
    """
    p = int(len(params) / 2)
    beta, gamma = params[:p], params[p:]
    u, v = idst(gamma, type=4), idct(beta, type=4)
    new_u, new_v = np.zeros((R+1, p+1)), np.zeros((R+1, p+1))
    new_u[0, :-1], new_v[0, :-1] = u, v
    new_u[1:, :-1], new_v[1:, :-1] = u + alpha * rng.normal(np.zeros((R, p)), np.abs(u)), v + alpha * rng.normal(np.zeros((R, p)), np.abs(v))
    new_beta, new_gamma = dct(new_v, type=4), dst(new_u, type=4)
    return np.hstack((new_beta, new_gamma))


def callback(intermediate_result):
    """Log the current objective value and raise StopIteration if converged.

    Passed to ``scipy.optimize.minimize`` as the ``callback`` argument.
    Raises ``StopIteration`` if the global minimum cost has effectively
    reached zero (``< 1e-6``), which scipy interprets as a signal to halt.

    Args:
        intermediate_result: Optimisation result object with a ``.fun``
            attribute holding the current objective value.
    """
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
