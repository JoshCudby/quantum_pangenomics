
import numpy as np
import numpy.typing as npt
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
from qiskit_aer.primitives import SamplerV2 as Sampler


from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger


def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    
    
properties = {}
def get_permutation(pass_, dag, time, property_set, count):
    properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]
    

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random'], default='ramp')
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])

args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
shots: int = args.shots
init_type: str = args.init


seed = 1
rng = np.random.default_rng(seed=seed)

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates=basis_gates,
)

filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{args.filename}.gfa'

graph, n, N, T = gfa_file_to_graph(filepath, args.copy_numbers)
hamiltonian = graph_to_hubo_hamiltonian(graph, n, N, T, lamda=10)


config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = hamiltonian.num_qubits
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, **backend_options)
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')
sampler = Sampler(seed=1).from_backend(backend)



qaoa_circ = QAOAAnsatz(hamiltonian, reps=p)
# Now transpile to basis gates
t_qaoa_circ = transpile(qaoa_circ, basis_gates=basis_gates)
t_qaoa_circ.measure_all()

print_circuit_info(t_qaoa_circ, 'QAOA circuit')
logger.info(f'QAOA circuit has {t_qaoa_circ.num_qubits} qubits')


qaoa_depth = len(t_qaoa_circ.parameters) // 2
if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
else:
    init_params = rng.uniform(0, 0.9 * np.pi, qaoa_depth).tolist() + rng.uniform(0, 0.5 * np.pi, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')


logger.info(f'Noise model: {getattr(sampler._backend.options, "noise_model", "Ideal noise")}')

history = []
alpha = 0.25

def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: npt.NDArray[np.float64]):
    start = time()
    assigned_circuit = t_qaoa_circ.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.meas.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = evaluate_sparse_pauli_samples(counts.keys(), hamiltonian)
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, alpha)

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy


def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    if intermediate_result.fun == -1:
        raise StopIteration
    

def callback_cobyla(xk: np.ndarray):
    logger.info(f'Current params: {xk}.')
    
    
method="Powell"
result = minimize(
    objective, 
    x0=init_params, 
    method=method, 
    bounds=tuple((0,1) for _ in range(2 * p)), 
    options={"maxiter": 100, "maxfev": 1000, "rhobeg": 0.01},  # , "ftol": 1e-7
    callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, hamiltonian=hamiltonian, t_qaoa_circ=t_qaoa_circ
)
with open(f'/lustre/scratch127/qpg/jc59/hubo/simulation.optimisation.default_{filename}.cvar{alpha}.p{p}.shots{shots}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
