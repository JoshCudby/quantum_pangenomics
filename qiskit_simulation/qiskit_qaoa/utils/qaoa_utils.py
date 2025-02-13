from scipy.optimize import minimize, basinhopping
from skopt import Optimizer
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from .logging import get_logger

logger = get_logger(__name__)


def _cost_func_estimator(
    params: list,
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: Estimator,
):
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()
    cost = results[0].data.evs

    return float(cost)


def bayesian_optimize_qaoa_parameters(
    backend,
    init_params: list[float],
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    reps: int,
    bounds: list[tuple],
    estimator_shots=10000
):
    bounds = bounds * reps

    bopt = Optimizer(bounds, random_state=10)
    estimator = Estimator.from_backend(backend)
    estimator.options.default_shots = estimator_shots
    estimator.options.default_precision = 0
    
    x = init_params
    logger.info('Starting Bayesian optimization')
    for _ in range(200):
        cost = _cost_func_estimator(x, circuit, hamiltonian, estimator)
        res = bopt.tell(list(x), cost)
        logger.debug(f'Found value: {cost}')
        x = bopt.ask()
    return res


def optimize_qaoa_parameters(
        backend,
        init_params: list[float],
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        reps: int,
        bounds: list[tuple],
        estimator_shots=10000,
):
    estimator = Estimator.from_backend(backend)
    estimator.options.default_shots = estimator_shots
    estimator.options.default_precision = 0

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
    # return minimize(
    #     _cost_func_estimator,
    #     init_params,
    #     args=(circuit, hamiltonian, estimator),
    #     method='COBYLA',
    #     bounds=[(0, np.pi/2), (0, np.pi)] * reps,
    #     options={'rhobeg': 0.01 / circuit.num_qubits,},
    #     tol=1e-3,
    # )