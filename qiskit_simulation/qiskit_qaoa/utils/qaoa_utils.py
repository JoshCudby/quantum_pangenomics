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
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()
    try:
        logger.debug(results)
        logger.debug(results[0].metadata)
        # print(results.to_dict()['results'][0]['metadata']['cacheblocking'])
    except Exception as error:
        logger.error(error)

    cost = results[0].data.evs

    return cost


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
        [(0.0        + eps, np.pi / 2 - eps)] * reps + 
        [(-np.pi / 4 + eps, np.pi / 4 - eps)] * reps
    )

    bopt = Optimizer(bounds, random_state=10)
    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = estimator_shots
        x = init_params
        for _ in range(100):
            res = bopt.tell(x, _cost_func_estimator(x, circuit, hamiltonian, estimator))
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


        # Set simple error suppression/mitigation options
        # estimator.options.dynamical_decoupling.enable = True
        # estimator.options.dynamical_decoupling.sequence_type = "XY4"
        # estimator.options.twirling.enable_gates = True
        # estimator.options.twirling.num_randomizations = "auto"
        
        def _callback(_, f, accept):
            logger.info(f'Found minimum: {f}, accepted: {accept}')

        logger.info('Starting minimization')
        return basinhopping(
            _cost_func_estimator,
            x0=init_params,
            niter=10,
            niter_success=5,
            callback=_callback,
            minimizer_kwargs={
                'args':(circuit, hamiltonian, estimator), 
                'method':'COBYLA', 
                'bounds':[(0, np.pi/2), (0, np.pi)] * reps,
                'options':{'rhobeg': 0.01 / circuit.num_qubits,},
                'tol':1e-3,
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