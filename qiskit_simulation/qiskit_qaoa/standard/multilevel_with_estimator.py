"""Multi-level QAOA using EstimatorWithHistory for convergence tracking.

Variant of the hierarchical QAOA approach (see ``multi_level_experiment.py``)
that replaces the raw AerSimulator shot-based objective with a Qiskit
``EstimatorWithHistory`` primitive, enabling direct energy-expectation
evaluation and recording of the full convergence history.  Parameters are
extrapolated from depth p to p+1 using DCT-IV/DST-IV frequency-domain
interpolation, and ``R`` perturbed candidates are seeded at each step.

CLI usage::

    python multilevel_with_estimator.py <filename>

Args:
    filename (str): Base name (without path or extension) of the QUBO data
        file under ``/lustre/.../qubo_data_<filename>.gfa.npy``.

Output:
    A PNG convergence plot (cost vs. number of measurements) saved to the
    multilevel_estimator output directory.
"""

import numpy as np
import sys
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import matplotlib.pyplot as plt
from scipy.fft import dct, idct, dst, idst
from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
from qiskit_qaoa.utils.qaoa_utils import local_optimize_qaoa_parameters
from qiskit_qaoa.utils.estimator_with_history import EstimatorWithHistory
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

logger.info('Starting multi-level experiment with qiskit estimator')

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


def get_perturbed_next_layer_params(params, R, alpha):
    """Generate R+1 candidate parameter sets for the next QAOA layer.

    Extrapolates the current parameters to depth p+1 via zero-padded DCT-IV/
    DST-IV frequency interpolation and produces ``R`` additional perturbed
    candidates by adding Gaussian noise scaled to the frequency-coefficient
    magnitudes.

    Args:
        params: 1-D array of length ``2*p`` (betas then gammas).
        R: Number of perturbed candidate initialisation points to generate.
        alpha: Perturbation strength relative to each frequency-coefficient
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
    """Log the current objective value and signal convergence if near zero.

    Passed to the local optimiser as its ``callback`` argument.  Raises
    ``StopIteration`` when the global minimum cost history drops below 1e-6,
    causing scipy to terminate the optimisation early.

    Args:
        intermediate_result: Optimisation result object exposing a ``.fun``
            attribute with the current objective value.
    """
    logger.info(f'Inter result: {intermediate_result.fun}')
    if np.min(costs_history) < 1e-6 == 0:
        raise StopIteration

try:
    ideal_aer = AerSimulator(
        method="statevector",
        device='GPU',
        cuStateVec_enable=True,
        max_memory_mb=16000*0.9,
    )
except AerError as error:
    logger.error(error)

estimator = EstimatorWithHistory.from_backend(ideal_aer)
estimator.options.default_shots = 1e5
estimator.options.default_precision = 0


R = 5
alpha = 0.6
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


init_params[p] = params

samples_history = []
costs_history = []

found_opt = False

while not found_opt and p < 2:
    circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
    circuit.measure_all()

    logger.info(f'Transpiling circuit for p={p}')
    # Create pass manager for transpilation
    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
    compiled_circuit = pass_manager.run(circuit)


    # TODO: multithread the multi-opt?
    logger.info('Starting local minimize')
    if len(init_params[p].shape) == 2:
        opt_x = np.zeros((init_params[p].shape[0], 2 * p))
        opt_f = np.zeros(init_params[p].shape[0])
        for i in range(init_params[p].shape[0]):
            # if len(costs_history) and np.min(costs_history) < 1e-6:
            #     break
            opt = local_optimize_qaoa_parameters(
                estimator,
                init_params[p][i],
                compiled_circuit,
                hamiltonian,
                [beta_bounds] * p + [gamma_bounds] * p,
                costs_history,
                v_objective_evaluate,
                ftol=func_tol,
            )
            logger.info(opt)
            opt_x[i,:] = opt.x
            opt_f[i] = opt.fun
        opt_params[p] = opt_x[np.argmin(opt_f), :]
    elif len(init_params[p].shape) == 1:
        # if len(costs_history) and np.min(costs_history) < 1e-6:
        #     break
        opt = local_optimize_qaoa_parameters(
            estimator,
            init_params[p],
            compiled_circuit,
            hamiltonian,
            [beta_bounds] * p + [gamma_bounds] * p,
            costs_history,
            v_objective_evaluate,
            ftol=func_tol
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
# ax.plot(np.minimum.accumulate(costs_history))
ax.plot(costs_history)

ax.set_xscale('log')
ax.set_xlabel('Number of measurements')
ax.set_ylabel('Minimum Cost')
fig.savefig(f'/lustre/scratch127/qpg/jc59/out/qiskit/multilevel_estimator/qaoa_costs_{filename}_seed{seed}.png', format='png')


logger.info(f'Init params: {init_params}')
logger.info(f'Opt params: {opt_params}')
logger.info(f'Opt fun: {opt.fun}')
logger.info(f'Min cost: {np.min(costs_history)}')