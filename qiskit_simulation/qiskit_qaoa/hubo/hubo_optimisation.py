
import numpy as np
import networkx as nx
import copy
from itertools import combinations
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, CXGate, SwapGate
from qiskit.transpiler import PassManager, Layout
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.passes import (
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SetLayout,
    HighLevelSynthesis, 
    InverseCancellation
)
from qiskit.circuit import Parameter

from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping
from qopt_best_practices.qubit_selection import BackendEvaluator

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
    
    
def count_gates(qc: QuantumCircuit):
    gate_count = { qubit: 0 for qubit in qc.qubits }
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_wires(qc: QuantumCircuit):
    qc_out = qc.copy()
    gate_count = count_gates(qc_out)
    logger.info(gate_count)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out
    

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-d', '--swap-depth', type=int, default=0)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random'], default='ramp')
parser.add_argument('-R', '--grid-rows', type=int)
parser.add_argument('-C', '--grid-cols', type=int)
parser.add_argument('-e', '--extra', type=int, default=1)

args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
shots: int = args.shots
init_type: str = args.init
swap_depth: int = args.swap_depth


seed = 1
rng = np.random.default_rng(seed=seed)

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(args.grid_rows, args.grid_cols)
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


results_file = f'/lustre/scratch127/qpg/jc59/hubo/{filename}'
with open(results_file, 'rb') as f:
    data = pickle.load(f)
    old_hamiltonian: SparsePauliOp = data['old_hamiltonian']
    hamiltonian: SparsePauliOp = data[swap_depth]['hamiltonian']
    edge_map = data[swap_depth]['layout']
    
    
    
current_virtual_qubit_locations = [j for j in edge_map.values()]
used_physical_qubits = set(current_virtual_qubit_locations)
swap_layers: list[tuple[tuple[int, int], ...]] = []
for idx in range(10):
    swaps_to_perform = [x for x in extended_swap_strat.swap_layer(idx) if any(j in x for j in edge_map.values())]
    swap_layers.append(tuple(swaps_to_perform))
    new_list = copy.copy(current_virtual_qubit_locations)
    for i, j in swaps_to_perform:
        if i in current_virtual_qubit_locations:
            new_list[current_virtual_qubit_locations.index(i)] = j
        if j in current_virtual_qubit_locations:
            new_list[current_virtual_qubit_locations.index(j)] = i
    current_virtual_qubit_locations = new_list
    used_physical_qubits = used_physical_qubits.union(current_virtual_qubit_locations)
new_ess = ExtendedSwapStrategy(coupling_map, tuple(swap_layers))

    
pm = PassManager(
    [
        HighLevelSynthesis(basis_gates=["PauliEvolution"]), # Not needed if set up circuit as PauliEvolutionGate
        FindCommutingPauliEvolutionsMulti(), 
        CommutingGateRouter(
            new_ess,
            edge_colouring,
            max_layers=swap_depth,
            perform_extra_swaps=bool(args.extra)
        ),
        SwapToFinalMapping(),
        DecomposePauliZEvolution(new_ess._coupling_map),
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
tcost_qc = remove_idle_wires(tcost_qc)
print(f'Qubits after removing idle wires: {tcost_qc.qubits}')

print(tcost_qc.count_ops())


backend_options = dict(
    method='statevector',
    device='CPU',
    precision='single',
    basis_gates=basis_gates,
)


config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = len(used_physical_qubits)
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, coupling_map=new_ess._coupling_map, **backend_options)
sampler = Sampler(seed=1).from_backend(backend)



backend_cost_qc = transpile(tcost_qc, optimization_level=0, backend=backend)
construction_pass = QAOAConstructionPass(p)
construction_pass.property_set = properties
qaoa_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(backend_cost_qc)))



# path_finder = BackendEvaluator(backend)

# # the Backend Evaluator accepts custom subset definitions and metrics,
# # but defaults to finding the line with the best fidelity
# num_active_qubits = np.sum(
#     [np.any([pauli.z[i] for pauli in hamiltonian.paulis]).astype(int) for i in range(len(hamiltonian.paulis[0].z))]
# )
# logger.info(f'Num active qubits: {num_active_qubits}')
# path, fidelity, num_subsets = path_finder.evaluate(num_active_qubits)

# print("Best path: ", path)
# print("Best path fidelity", fidelity)
# print("Num. evaluated paths", num_subsets)
# initial_layout = Layout.from_intlist(path, qaoa_circ.qregs[0])  # needs qaoa_circ

# pass_manager_post = PassManager(
#     [
#         SetLayout(initial_layout),
#         FullAncillaAllocation(coupling_map),
#         EnlargeWithAncilla(),
#         ApplyLayout(),
#     ]
# )

# # Map to initial_layout and finally enlarge with ancilla.
# ancilla_qaoa_circ = pass_manager_post.run(qaoa_circ)

# Now transpile to basis gates
t_qaoa_circ = transpile(qaoa_circ, basis_gates=basis_gates)

print_circuit_info(t_qaoa_circ, 'QAOA circuit')


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
alpha = 0.05

def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
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
    
    
method="COBYLA"
result = minimize(
    objective, 
    x0=init_params, 
    method=method, 
    bounds=tuple((0,1) for _ in range(2 * p)), 
    options={"maxiter": 100, "maxfev": 100},  # "rhobeg": 0.01, "ftol": 1e-7
    callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, hamiltonian=hamiltonian, t_qaoa_circ=t_qaoa_circ
)
with open(f'/lustre/scratch127/qpg/jc59/hubo/optimisation_{filename}.cvar{alpha}.p{p}.shots{shots}.init{init_type}.d{swap_depth}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
