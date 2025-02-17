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