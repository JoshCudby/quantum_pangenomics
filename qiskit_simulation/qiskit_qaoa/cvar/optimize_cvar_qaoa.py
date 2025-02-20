import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, AerError
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit.circuit import ParameterVector
from qiskit_qaoa.utils.optimal_qaoa_pass_manager import get_optimal_pass_manager
from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
from qiskit_qaoa.utils.backend_evaluator import BackendEvaluator
from qiskit.transpiler import CouplingMap
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=np.complex128)


def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])


def aggregate(alpha: float, measurements: list[tuple[float, complex]]):
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    sorted_measurements: list[tuple[float, complex]] = sorted(measurements, key=lambda x: x[1])
    accumulated_percent = 0.0
    cvar = 0.0

    while accumulated_percent < alpha and len(sorted_measurements):
        probability, value = sorted_measurements.pop(0)
        cvar += value * min(probability, alpha - accumulated_percent)
        accumulated_percent += probability

    return np.real(cvar / alpha)


def cost_func_cvar_sampler(
    params: np.ndarray, 
    ansatz: QuantumCircuit, 
    hamiltonian: SparsePauliOp, 
    sampler: Sampler, 
    aggregation: float,
    optimal: str | None = None
):
    pub = [ansatz, params]
    job = sampler.run([pub])
    sampler_result = job.result()
    counts = sampler_result[0].data.meas.get_counts()
    v1_format = {int(key, 2): val/n_shots for key, val in counts.items()}

    if optimal is not None:
        try:
            v1_format[int(optimal, 2)]
            logger.info(f'Sampled optimal with params: {params}')
        except KeyError:
            pass
    measurement_values = [(probability, evaluate_sparse_pauli(state, hamiltonian)) for state, probability in v1_format.items()]

    return aggregate(aggregation, measurement_values)


def get_transpiled_circuit(cost_operator: SparsePauliOp, qaoa_layers: int):
    num_qubits = cost_operator.num_qubits

    # Initial state = equal superposition
    initial_state = QuantumCircuit(num_qubits)
    initial_state.h(range(num_qubits))

    # Mixer operator = rx rotations
    betas = ParameterVector("β", qaoa_layers)
    mixer_operator = QuantumCircuit(num_qubits)
    mixer_operator.rx(-2*betas[0], range(num_qubits))

    try:
        distance = 1
        while (5 * distance ** 2 - 2 * distance -1) / 2 < num_qubits:
            distance += 2
        num_virtual_qubits = int((5 * distance ** 2 - 2 * distance -1) / 2)
        generic_backend = GenericBackendV2(num_qubits = num_virtual_qubits, coupling_map = CouplingMap.from_heavy_hex(distance=distance), basis_gates = ["x", "sx", "cz", "id", "rz"], seed=0)
        backend = AerSimulator.from_backend(generic_backend)
    except AerError as error:
        logger.error(error)
        exit(1)
    naive_pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    qaoa_ansatz = QAOAAnsatz(
        cost_operator,
        initial_state = initial_state,
        mixer_operator = mixer_operator,
        reps = qaoa_layers,
        flatten=True
    )
    qaoa_ansatz.measure_all()
    naively_transpiled_qaoa = naive_pm.run(qaoa_ansatz)
    logger.info(f'Naive counts: {naively_transpiled_qaoa.count_ops()}')
    logger.info(f'Naive 2Q depth: {naively_transpiled_qaoa.depth(filter_function=lambda x: x.operation.name == "cz")}')

    dummy_initial_state = QuantumCircuit(num_qubits)  # the real initial state is defined later
    dummy_mixer_operator = QuantumCircuit(num_qubits)  # the real mixer is defined later

    cost_layer = QAOAAnsatz(
        cost_operator,
        reps=1,
        initial_state=dummy_initial_state,
        mixer_operator=dummy_mixer_operator,
        name="QAOA cost block",
    )

    path_finder = BackendEvaluator(backend)
    path, fidelity, _ = path_finder.evaluate(num_qubits)
    logger.info([path, fidelity])
    
    initial_layout = Layout.from_intlist(path, cost_layer.qregs[0])
    staged_pm = get_optimal_pass_manager(num_qubits, backend, initial_layout, betas)
    optimally_transpiled_qaoa = staged_pm.run(cost_layer)
    logger.info(optimally_transpiled_qaoa.count_ops())
    logger.info(optimally_transpiled_qaoa.depth(filter_function=lambda x: x.operation.name == "cz"))
    return optimally_transpiled_qaoa



backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=16000*0.9,
)

beta_bounds = (-np.pi/2, np.pi/2)
gamma_bounds = (-np.pi, np.pi)


if __name__ == "__main__":
    seed = 1000
    rng = np.random.default_rng(seed)
    p = 4
    n_shots = 2000
    sample_shots = int(1e3)

    filename = 'trivial'
    match filename:
        case 'trivial':
            optimal = '100001000010'
        case 'small_test':
            optimal = '100001000010100001000001'
        case _:
            optimal = None
    
    logger.info(filename)
    logger.info(f'Seed={seed}')
    logger.info(f'p={p}')
    logger.info(f'shots: {n_shots}')
    logger.info(f'Final sample shots: {sample_shots}')

    sampler = Sampler(default_shots=n_shots, seed=seed, options={'backend_options': backend_options})

    data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.npy'
    _, hamiltonian = get_objective_and_hamiltonian(data_file)

    ansatz = get_transpiled_circuit(cost_operator=hamiltonian, qaoa_layers=p)

    init_params = rng.random((2*p,)) \
        * np.array([beta_bounds[1] - beta_bounds[0]] * p + [gamma_bounds[1] - gamma_bounds[0]] * p) \
        + np.array([beta_bounds[0]] * p + [gamma_bounds[0]] * p)
    
    result = minimize(
        cost_func_cvar_sampler,
        init_params,
        args=(ansatz, hamiltonian, sampler, 0.05, optimal),
        options={'rhobeg': 0.01, 'maxfev': 100},
        method="COBYLA",
    )
    logger.info(result)

    qc = ansatz.assign_parameters(result.x)
    samp_counts = sampler.run([qc], shots=sample_shots).result()[0].data.meas.get_counts()

    # optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
    try:
        logger.info(f'Empirical prob of optimal: {samp_counts[optimal] / sample_shots}')
    except KeyError:
        logger.info('Did not sample optimal')
    logger.info(f'Uniform random probability: {2 ** -qc.num_qubits}')
