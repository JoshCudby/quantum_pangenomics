import numpy as np
from scipy.optimize import minimize, basinhopping
from skopt import Optimizer
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
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
    estimator_shots=10000
):
    eps = 1e-6
    bounds = (
        [(0.0 + eps, np.pi / 2 - eps)] * reps + 
        [(0.0 + eps, np.pi     - eps)] * reps
    )

    bopt = Optimizer(bounds, random_state=10)
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = estimator_shots
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
        estimator_shots=10000
):
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = estimator_shots

        # transform the observable defined on virtual qubits to
        # an observable defined on all physical qubits
        isa_hamiltonian = hamiltonian.apply_layout(circuit.layout)
        
        def _callback(_, f, accept):
            logger.info(f'Found minimum: {f}, accepted: {accept}')

        logger.info('Starting minimization')
        eps = 1e-6
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
                'bounds':[(eps, np.pi/2 - eps), (eps, np.pi - eps)] * reps,
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