
import numpy as np
import networkx as nx
from itertools import combinations
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, CXGate, SwapGate
from qiskit.transpiler import PassManager
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.passes import (
    HighLevelSynthesis, 
    InverseCancellation
)
from qiskit.circuit import Parameter

from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
from qiskit_aer.primitives import SamplerV2 as Sampler

from qiskit_qaoa.utils.qaoa_circuit_utils import get_mixer_operator, state_prep
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, CommutingGateRouter, FindCommutingPauliEvolutionsMulti, DecomposePauliZEvolution
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
parser.add_argument('-d', '--swap-depth', type=int, default=0)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random'], default='ramp')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('--fraction-constraint', type=float)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-T', '--time', type=int)
parser.add_argument('-C', '--coupling-map', choices=['line', 'grid'])


args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
shots: int = args.shots
init_type: str = args.init
swap_depth: int = args.swap_depth
N: int = args.nodes
T: int = args.time
n = int(np.ceil(np.log2(2*N+1)))

seed = 1
rng = np.random.default_rng(seed=seed)

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

basepath = '/lustre/scratch127/qpg/jc59/hubo/'
filename = 'simulation.{}.compilation.{}.extra{}.constraint{}.four{}.six{}'.format(
    args.coupling_map,
    args.filename,
    args.extra,
    args.fraction_constraint,
    args.fraction_four,
    args.fraction_six
)
results_file = basepath + filename + '.pkl'

with open(results_file, 'rb') as f:
    data = pickle.load(f)
    compiled_hamiltonian: SparsePauliOp = data['compiled_hamiltonian']
    full_hamiltonian: SparsePauliOp = data['full_hamiltonian']
    edge_map: dict[int, int] = data[swap_depth]

num_qubits: int = full_hamiltonian.num_qubits if full_hamiltonian.num_qubits is not None else max(edge_map.keys())

if args.coupling_map == 'line':
    extended_swap_strat = ExtendedSwapStrategy.from_line(list(range(num_qubits)), num_swap_layers=1000)
elif args.coupling_map == 'grid':
    extended_swap_strat = ExtendedSwapStrategy.from_grid(n, T)
else:
    raise Exception('Invalid coupling map type')

num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map

remapped_full_hamiltonian = full_hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)


basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]


logger.info(f'Physical qubits: {num_physical_qubits}')

coupling_map_edge = list(coupling_map)
physical_qubits = list(coupling_map.physical_qubits)
dual_coupling_map = nx.Graph()

for qubit in physical_qubits:
    edges = [edge for edge in coupling_map_edge if edge[0]==qubit]
    for edge1, edge2 in combinations(edges, 2):
        dual_coupling_map.add_edge(tuple(sorted(edge1)), tuple(sorted(edge2)))
edge_colouring = nx.greedy_color(dual_coupling_map, interchange=True)


pm = PassManager(
    [
        HighLevelSynthesis(basis_gates=["PauliEvolution"]), # Not needed if set up circuit as PauliEvolutionGate
        FindCommutingPauliEvolutionsMulti(), 
        CommutingGateRouter(
            extended_swap_strat,
            edge_colouring,
            max_layers=swap_depth,
            perform_extra_swaps=bool(args.extra)
        ),
        SwapToFinalMapping(),
        DecomposePauliZEvolution(extended_swap_strat._coupling_map),
        HighLevelSynthesis(
            basis_gates=["sx", "x", "rz", "rzz", "cx", "id", "swap"], 
        ),
        InverseCancellation(gates_to_cancel=[CXGate(), SwapGate()]),
    ]
)

cost_qc = QuantumCircuit(num_physical_qubits)
cost_qc.append(PauliEvolutionGate(compiled_hamiltonian, time=Parameter("c")), [edge_map[i] for i in range(len(edge_map))])
tcost_qc = pm.run(cost_qc, callback=get_permutation)

print_circuit_info(tcost_qc, 'Transpiled cost hamiltonian circuit')

print(tcost_qc.count_ops())
logger.info(f'Cost hamiltonian circuit has {tcost_qc.num_qubits} qubits')


backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates=basis_gates,
)


config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = num_physical_qubits
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map, **backend_options)
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')
sampler = Sampler(seed=1).from_backend(backend)


# TODO: instead of using construction pass, use p different cost hamiltonians with different mappings
# Can't do different mappings since the qubit locations are now set.. but could do the next N layers of SWAP strat
# Which would allow for a different subset of interactions to be used

if not 2*N+1 == 2**(int(np.log2(2*N+1))):
    sp = state_prep(N,T)
    mixer = get_mixer_operator(N,T)
    logger.info('Using Grover mixer and state prep')
else:
    sp = None
    mixer = None
    logger.info('Using X mixer and Hadamard state prep')
    
    
construction_pass = QAOAConstructionPass(p, init_state=sp, mixer_layer=mixer)
construction_pass.property_set = properties
qaoa_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(tcost_qc)))

# Now transpile to basis gates
t_qaoa_circ = transpile(qaoa_circ, basis_gates=basis_gates)

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


def objective(x: np.ndarray):
    start = time()
    assigned_circuit = t_qaoa_circ.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = evaluate_sparse_pauli_samples(counts.keys(), remapped_full_hamiltonian)
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
    
    
method="COBYLA"
result = minimize(
    objective, 
    x0=init_params, 
    method=method, 
    bounds=tuple((0,1) for _ in range(2 * p)), 
    options={"maxiter": 100, "maxfev": 1000, "rhobeg": 0.05},  # , "ftol": 1e-7
    callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, remapped_full_hamiltonian=remapped_full_hamiltonian, t_qaoa_circ=t_qaoa_circ, compiled_hamiltonian=compiled_hamiltonian
)

dump_file = basepath + filename.replace('compilation', 'optimisation') + '.cvar{}.p{}.shots{}.init{}.d{}'.format(
    alpha, p,shots, init_type, swap_depth
) + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(obj_to_dump, f)
