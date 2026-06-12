"""QAOA parameter optimisation strategies.

Provides three classical optimisers for finding the QAOA variational parameters
(γ, β) that minimise the Hamiltonian expectation value:

- ``bayesian_optimize_qaoa_parameters``: scikit-optimise Bayesian optimisation.
- ``basinhopping_optimize_qaoa_parameters``: scipy basin-hopping with COBYLA.
- ``local_optimize_qaoa_parameters``: scipy Powell method with optional early
  stopping when the sampled cost reaches zero.
"""

import numpy as np
from scipy.optimize import minimize, basinhopping
from skopt import Optimizer
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from .estimator_with_history import EstimatorWithHistory
from .logging import get_logger

logger = get_logger(__name__)


def _parse_strings_to_binary_array(data: list[str]):
    return np.array([[int(y) for y in list(x)] for x in data])


def _cost_func_estimator(
    params: list,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: EstimatorWithHistory,
    costs_history: list,
    cost_fun: np.vectorize
):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()
    outcomes = _parse_strings_to_binary_array(list(results[0].data.counts.keys()))
    costs_history.extend(cost_fun(outcomes))

    return float(results[0].data.evs)


def bayesian_optimize_qaoa_parameters(
    estimator: EstimatorWithHistory,
    init_params: list[float],
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    reps: int,
    bounds: list[tuple]
):
    """Optimise QAOA parameters using Bayesian optimisation (scikit-optimise).

    Runs 200 iterations of a Gaussian-process surrogate model.  Each iteration
    evaluates the Hamiltonian expectation value via the estimator and updates
    the model.

    Args:
        estimator: An ``EstimatorWithHistory`` used to evaluate the circuit.
        init_params: Initial parameter vector (length ``2 * reps`` for
            standard QAOA with ``reps`` layers).
        circuit: The transpiled QAOA ansatz circuit.
        hamiltonian: The Ising Hamiltonian as a ``SparsePauliOp``.
        reps: Number of QAOA layers; used to replicate the per-layer bounds.
        bounds: Per-layer parameter bounds as a list of ``(low, high)`` tuples
            (replicated ``reps`` times internally).

    Returns:
        The scikit-optimise ``OptimizeResult`` from the final ``tell`` call.
    """
    bounds = bounds * reps

    bopt = Optimizer(bounds, random_state=10)
    
    x = init_params
    logger.info('Starting Bayesian optimization')
    for _ in range(200):
        cost = _cost_func_estimator(x, circuit, hamiltonian, estimator)
        res = bopt.tell(list(x), cost)
        logger.debug(f'Found value: {cost}')
        x = bopt.ask()
    return res


def basinhopping_optimize_qaoa_parameters(
        estimator: EstimatorWithHistory,
        init_params: list[float],
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        reps: int,
        bounds: list[tuple],
):
    """Optimise QAOA parameters using basin-hopping with COBYLA local search.

    Applies the observable layout transformation to map the virtual-qubit
    Hamiltonian to the physical layout before optimisation.  Runs 100
    basin-hopping steps with ``niter_success=20`` and a temperature of 0.01.

    Args:
        estimator: An ``EstimatorWithHistory`` used to evaluate the circuit.
        init_params: Initial parameter vector.
        circuit: The transpiled QAOA ansatz circuit (must have a ``layout``
            attribute set by the transpiler).
        hamiltonian: The Ising Hamiltonian as a ``SparsePauliOp`` defined on
            virtual qubits.
        reps: Number of QAOA layers; used to replicate the per-layer bounds.
        bounds: Per-layer parameter bounds as a list of ``(low, high)`` tuples.

    Returns:
        The scipy ``OptimizeResult`` from ``basinhopping``.
    """
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
    
    def _callback(_, f, accept):
        logger.info(f'Found minimum: {f}, accepted: {accept}')

    logger.info('Starting minimization')
    return basinhopping(
        _cost_func_estimator,
        x0=init_params,
        niter=100,
        T = 0.01,
        niter_success=20,
        callback=_callback,
        minimizer_kwargs={
            'args':(circuit, isa_hamiltonian, estimator), 
            'method':'COBYLA', 
            'bounds': bounds * reps,
            'tol':1e-4,
            }
    )


def local_optimize_qaoa_parameters(
        estimator: Estimator,
        init_params: list[float],
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        bounds: list[tuple],
        costs_history: list,
        cost_fun: np.vectorize,
        ftol=1e-2,
):
    """Optimise QAOA parameters using Powell's method with optional early stopping.

    Applies the observable layout transformation then minimises the Hamiltonian
    expectation value with scipy's Powell algorithm.  At each step the raw
    measurement outcomes are scored by ``cost_fun`` and appended to
    ``costs_history``; if the minimum observed cost drops below 1e-6 the
    callback raises ``StopIteration`` to halt early.

    Args:
        estimator: An Aer ``EstimatorV2`` (or ``EstimatorWithHistory``) used
            to evaluate the circuit.
        init_params: Initial parameter vector.
        circuit: The transpiled QAOA ansatz circuit (must have a ``layout``
            attribute).
        hamiltonian: The Ising Hamiltonian as a ``SparsePauliOp`` defined on
            virtual qubits.
        bounds: Parameter bounds as a list of ``(low, high)`` tuples.
        costs_history: A mutable list that is extended with the raw sample
            costs at each function evaluation.  Also used for early-stopping.
        cost_fun: A ``numpy.vectorize``-d callable that maps a 2-D binary
            outcome array (shape ``(n_samples, n_qubits)``) to a 1-D cost
            array.
        ftol: Function-value tolerance for Powell convergence (default 1e-2).

    Returns:
        The scipy ``OptimizeResult`` from ``minimize``.
    """
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
    
    def _callback(intermediate_result):
        logger.info(f'Inter result: {intermediate_result.fun}')
        if len(costs_history) and np.min(costs_history) < 1e-6 == 0:
            raise StopIteration

    logger.info('Starting minimization')

    return minimize(
        _cost_func_estimator,
        init_params,
        args=(circuit, isa_hamiltonian, estimator, costs_history, cost_fun),
        method='powell',
        bounds=bounds,
        options={'ftol': ftol, 'disp': True},
        callback=_callback
    )