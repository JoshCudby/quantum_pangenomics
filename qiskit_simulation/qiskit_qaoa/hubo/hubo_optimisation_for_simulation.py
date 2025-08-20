
import numpy as np
import networkx as nx
from itertools import combinations
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, CXGate, SwapGate, QAOAAnsatz
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
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-T', '--time', type=int)


args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
shots: int = args.shots
init_type: str = args.init
swap_depth: int = args.swap_depth
N: int = args.nodes
T: int = args.time


seed = 1
rng = np.random.default_rng(seed=seed)

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

results_file = f'/lustre/scratch127/qpg/jc59/hubo/simulation_results_{filename}.pkl'
with open(results_file, 'rb') as f:
    data = pickle.load(f)
    old_hamiltonian: SparsePauliOp = data['old_hamiltonian']
    hamiltonian: SparsePauliOp = data[swap_depth]['hamiltonian']
    edge_map = data[swap_depth]['layout']
    

num_qubits = hamiltonian.num_qubits
extended_swap_strat = ExtendedSwapStrategy.from_line(range(num_qubits))
num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map

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
# cost_qc.append(PauliEvolutionGate(hamiltonian, time=Parameter("c")), range(num_physical_qubits))
cost_qc.append(PauliEvolutionGate(old_hamiltonian, time=Parameter("c")), [edge_map[i] for i in range(len(edge_map))])
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

def uniform_over_range(num_qubits: int, M: int):
    """
    Returns a circuit that prepares a uniform superposition over |0>,|1>,...,|M-1> on num_qubits qubits.
    Uses a Hadamard layer if M is a power of 2, else uses the method of Shukla and Vedula.
    """
    if M not in range(2 ** num_qubits +1):
        print(M)
        print(num_qubits)
        raise Exception('Bad M: out of range')
    for i in range(num_qubits+1):
        if M == 2 ** i:
            print(f'M={M} a power of 2. Use Hadamard circuit.')
            circuit = QuantumCircuit(num_qubits)
            for j in range(i):
                circuit.h(j)
                
            return circuit
    
    circuit = QuantumCircuit(num_qubits)

    try:
        M_binary = np.binary_repr(M, num_qubits)
    except Exception as e:
        print(M)
        print(num_qubits)
        raise e
    M_binary = M_binary[::-1]
    ran = np.arange(len(M_binary))
    mask = [M_binary[x] == '1' for x in range(len(M_binary))]
    l = ran[mask]
    
    for i in range(1, len(l)):
        circuit.x(l[i])
    if l[0] > 0:
        for i in range(l[0]):
            circuit.h(i)

    MM = 2 ** l[0]

    circuit.ry(-2 * np.arccos(np.sqrt(MM/M)), l[1])

    for i in range(l[0], l[1]):
        circuit.ch(l[1], i, ctrl_state=0)

    for m in range(1, len(l)-1):
        circuit.cry(
            -2 * np.arccos(np.sqrt(2 ** l[m] / (M - MM) )), 
            l[m], l[m+1], ctrl_state=0
        )
        for i in range(l[m], l[m+1]):
            circuit.ch(l[m+1], i, ctrl_state=0)
        MM += 2 ** l[m]

    return circuit


def state_prep(N: int, T: int) -> QuantumCircuit:
    n = int(np.ceil(np.log2(2*N+1)))
    uni = uniform_over_range(n, N+1)
    circuit = QuantumCircuit(n * T)
    for t in range(T):
        circuit.append(
            uni,
            list(range(t * n, (t+1) * n))   
        )
    return circuit


def get_mixer_operator(N: int, T: int, parameter=Parameter('beta')) -> QuantumCircuit:
    # TODO: use ancillas to reduce depth of mcp?
    num_qubits = int(np.ceil(np.log2(2*N+1))) * T
    state_prep_circuit = state_prep(N, T)
    mixer = QuantumCircuit(num_qubits)
    mixer.append(
        state_prep_circuit.inverse(),
        range(num_qubits)
    )
    # mixer.save_statevector('after_prep')
    mixer.x(-1)
    mixer.mcp(-parameter, list(range(num_qubits - 1)), -1, ctrl_state=0)
    mixer.x(-1)
    # mixer.save_statevector('after_phase')
    mixer.append(
        state_prep_circuit,
        range(num_qubits)
    )
    # mixer.save_statevector('after_unprep')
    return mixer


if not N == 2**(int(np.log2(N))):
    sp = state_prep(N,T)
    mixer = get_mixer_operator(N,T)
    logger.info(f'SP qubit: {sp.num_qubits}. Mixer qubit: {mixer.num_qubits}')
else:
    sp = None
    mixer = None
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
    
    
method="COBYLA"
result = minimize(
    objective, 
    x0=init_params, 
    method=method, 
    bounds=tuple((0,1) for _ in range(2 * p)), 
    options={"maxiter": 100, "maxfev": 10000, "rhobeg": 0.01},  # , "ftol": 1e-7
    callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, hamiltonian=hamiltonian, t_qaoa_circ=t_qaoa_circ, old_hamiltonian=old_hamiltonian
)
with open(f'/lustre/scratch127/qpg/jc59/hubo/simulation.optimisation_{filename}.cvar{alpha}.p{p}.shots{shots}.init{init_type}.d{swap_depth}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
