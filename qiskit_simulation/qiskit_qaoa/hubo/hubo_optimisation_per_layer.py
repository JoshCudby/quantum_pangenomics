import numpy as np
import networkx as nx
import pickle
import re
import argparse
from itertools import combinations
from time import time
from collections import Counter
from scipy.optimize import minimize, OptimizeResult, basinhopping

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler import Layout
from qiskit.transpiler.passes import LayoutTransformation
from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration
from qiskit_aer.primitives import SamplerV2 as Sampler

from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.pass_managers import get_hubo_pass_manager
from qiskit_qaoa.utils.layout_utils import swap_between_circuit_layouts
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('-t', '--timeout', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
parser.add_argument('-C', '--coupling-map', choices=['line', 'grid'])
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-M', '--method', type=str)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random', 'warm'], default='ramp')
parser.add_argument('-a', '--alpha', type=float)
args = parser.parse_args()
logger.info(args)
p = args.reps


def two_qubit_count(qc: QuantumCircuit):
    ops: dict[str, int] = qc.count_ops()
    return ops.get("cz", 0) + ops.get("rzz", 0) + ops.get("cx", 0) + ops.get("swap", 0)
   
    
def print_circuit_info(qc: QuantumCircuit, circuit_name: str):
    logger.info(f'{circuit_name} has {two_qubit_count(qc)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    
    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{args.filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
num_qubits = n*T
logger.info(f'Virtual qubits: {num_qubits}')


if args.coupling_map == 'line':
    extended_swap_strat = ExtendedSwapStrategy.from_line(list(range(num_qubits)), num_swap_layers=1000)
elif args.coupling_map == 'grid':
    extended_swap_strat = ExtendedSwapStrategy.from_grid(n, T)
else:
    raise Exception('Invalid coupling map type')

num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map
    
coupling_map_edge = list(coupling_map)
physical_qubits = list(coupling_map.physical_qubits)
dual_coupling_map = nx.Graph()

for qubit in physical_qubits:
    edges = [edge for edge in coupling_map_edge if edge[0]==qubit]
    for edge1, edge2 in combinations(edges, 2):
        dual_coupling_map.add_edge(tuple(sorted(edge1)), tuple(sorted(edge2)))
edge_colouring = nx.greedy_color(dual_coupling_map, interchange=True)



logger.info(f'Physical qubits: {num_physical_qubits}')

basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "cx"]

backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates=basis_gates
)


config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = num_physical_qubits
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map, **backend_options)
backend.set_option("n_qubits", num_physical_qubits)
sampler = Sampler().from_backend(backend)
logger.info(backend.configuration().to_dict()["n_qubits"])

# full_hamiltonian = args.reps * graph_to_hubo_hamiltonian(graph, n, T, lamda=10/args.reps, constraint_terms=1.0)
full_hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
terms_to_keep = [tuple(x) for x in np.array_split(np.arange(T-1), p)]


try:
    with open(f'/lustre/scratch127/qpg/jc59/hubo/per_layer_results.{args.filename}.reps{args.reps}.pkl', 'rb') as f:
        data = pickle.load(f) 
        
    hamiltonians = data["hamiltonians"]
    swap_depths = data["swap_depths"]
    layouts = data["layouts"]
    compiled_circuits = data["compiled_circuits"]
except FileNotFoundError:
    hamiltonians, swap_depths, layouts, compiled_circuits = {}, {}, {}, {}
    for layer in range(args.reps):
        logger.info(f'Getting hamiltonian for layer {layer}')
        hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=terms_to_keep[layer])
        hamiltonians[layer] = hamiltonian
        
        all_pauli_z = np.array(
            [i.paulis[0].z for i in hamiltonian]
        )
        logger.info(f'Hamiltonian: {len(hamiltonian)}')
        logger.info(f'Orders: {Counter(np.sum(all_pauli_z, axis=1))}')
        
        program_interactions = hamiltonian_to_interactions(hamiltonian, args.fraction_four, args.fraction_six)
        lengths = Counter([len(interaction) for interaction in program_interactions])

        logger.info(f'Program interactions: {len(program_interactions)}')
        logger.info(f'Orders: {Counter([len(interaction) for interaction in program_interactions])}')
        
        mapper = HigherOrderSatMapper(timeout=args.timeout)

        best_circuit_depth, best_swap_depth, best_layout, best_circuit = np.inf, 0, Layout(), QuantumCircuit(num_physical_qubits)
        depths = sorted(list(set([int(x) for x in np.linspace(0, len(extended_swap_strat._swap_layers), 10)])))
        for depth in depths:
            logger.info('--------------------------------------------------')
            sat_results = mapper.hubo_max_sat(
                num_qubits, program_interactions, extended_swap_strat, depth
            )
            if sat_results is None:
                logger.info('No results')
                continue
            mapping = sat_results[depth][1]
            edge_map = dict(mapping)
            donor_qc = QuantumCircuit(num_qubits)
            layout = Layout({donor_qc.qubits[key]: val for key, val in edge_map.items()})
            
            logger.info(f'Cost: {sat_results[depth][0]}')
            logger.info(layout)

            pm = get_hubo_pass_manager(extended_swap_strat, depth, args.extra)

            new_cost_circ = QuantumCircuit(num_physical_qubits)
            new_cost_circ.append(PauliEvolutionGate(hamiltonian, time=Parameter("γ")), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])
            new_tcost = pm.run(new_cost_circ)
            
            print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
            print(new_tcost.count_ops())
            
            circuit_depth = new_tcost.depth(lambda instr: len(instr.qubits) > 1)
            if circuit_depth < best_circuit_depth:
                best_circuit_depth = circuit_depth
                best_swap_depth = depth
                best_layout = layout
                best_circuit = new_tcost
                
            if sat_results[depth][0] == 0:
                break
        
        swap_depths[layer] = best_swap_depth
        layouts[layer] = best_layout
        compiled_circuits[layer] = best_circuit
        
    to_save = dict(
        hamiltonians=hamiltonians, swap_depths=swap_depths, layouts=layouts, compiled_circuits=compiled_circuits
    )
    with open(f'/lustre/scratch127/qpg/jc59/hubo/per_layer_results.{args.filename}.reps{args.reps}.pkl', 'wb') as f:
        pickle.dump(to_save, f)


donor_qc = QuantumCircuit(num_qubits)
qaoa_circuit = QuantumCircuit(0, num_qubits)
qaoa_circuit.add_register([layouts[0].get_physical_bits()[i] for i in range(num_qubits)])

mixer_layer = QuantumCircuit(num_qubits)
beta = Parameter("β")
mixer_layer.rx(2 * beta, range(num_qubits))

gammas = ParameterVector("γ", args.reps)
betas = ParameterVector("β", args.reps)

for i in range(num_qubits):
    qaoa_circuit.h(i)


for layer in range(0, args.reps):
    swap_circuit = swap_between_circuit_layouts(layer-1, compiled_circuits, layouts, extended_swap_strat._coupling_map)
    qaoa_circuit.compose(swap_circuit, range(num_qubits), inplace=True)

    bind_dict = {compiled_circuits[layer].parameters[0]: gammas[layer]}
    bound_cost_layer = compiled_circuits[layer].assign_parameters(bind_dict)
    qaoa_circuit.compose(bound_cost_layer, range(num_qubits), inplace=True)
   
    bind_dict = {mixer_layer.parameters[0]: betas[layer]}
    bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)
    qaoa_circuit.compose(bound_mixer_layer, range(num_qubits), inplace=True)


layer = args.reps - 1
final_layout = layouts[layer].copy()
for instruction in compiled_circuits[layer].data:
    if instruction.operation.name == 'swap':
        qubits_str = str(instruction.qubits)
        matches = re.findall('index=([0-9]+)', qubits_str)
        if len(matches) == 2:
            final_layout.swap(int(matches[0]), int(matches[1]))
        else:
            raise Exception('Did not find 2 swap indices')


for physical, virtual in (
    final_layout.get_physical_bits().items()
):
    qaoa_circuit.measure(physical, donor_qc.find_bit(virtual).index)

# TODO: temporary hack to check that measurements work as expected
# to_layout = Layout({i: donor_qc.qubits[i] for i in range(num_qubits)})
# transformation_pass = LayoutTransformation(coupling_map, final_layout, to_layout)
# swap_qc = QuantumCircuit(num_qubits)
# swap_qc = dag_to_circuit(transformation_pass.run(circuit_to_dag(swap_qc)))
# qaoa_circuit.compose(swap_qc, range(num_qubits), inplace=True)
# for physical, virtual in (
#     to_layout.get_physical_bits().items()
# ):
#     print(physical, virtual, donor_qc.find_bit(virtual).index)
#     qaoa_circuit.measure(physical, donor_qc.find_bit(virtual).index)  
    
# Now transpile to basis gates
# generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, basis_gates=basis_gates)
# init  = generic_pm.init
# init.remove(3)
# generic_pm.init = init
# generic_pm.layout = None
# t_qaoa_circ = generic_pm.run(qaoa_circuit)

print_circuit_info(qaoa_circuit, 'QAOA circuit')
print(qaoa_circuit.count_ops())

qaoa_depth = len(qaoa_circuit.parameters) // 2
if args.init == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
elif args.init == 'warm':
    raise Exception('Warm start not implemented yet')
else:
    init_params = rng.uniform(0, 1, qaoa_depth).tolist() + rng.uniform(0, 1, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')


history = []

def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
    start = time()
    assigned_circuit = qaoa_circuit.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=args.shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = evaluate_sparse_pauli_samples(counts.keys(), full_hamiltonian)
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, args.alpha)

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy


def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    if intermediate_result.fun == -1:
        raise StopIteration
    

def callback_cobyla(xk: np.ndarray):
    logger.info(f'Current params: {xk}.')
    

def callback_basinhopping(x: np.ndarray, f: float, accept: bool):
    logger.info(f'Current params: {x}. Current func value: {f}')
    
logger.info(f'Using method: {args.method}.')
if args.method == 'basinhopping':
    result = basinhopping(
        objective, 
        x0=init_params, 
        niter=100,
        minimizer_kwargs=dict(bounds=tuple((0,1) for _ in range(2 * p)),method="Powell",options={"maxiter":100, "maxfev":1000}),
        callback=callback_basinhopping,
        disp=True
    )
else:
    result = minimize(
        objective, 
        x0=init_params, 
        method=args.method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": 100, "maxfev": 10000, "rhobeg": 0.05, "ftol": 1e-7},
        callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )
    logger.info(result)


obj_to_dump = dict(
    result=result, history=history, full_hamiltonian=full_hamiltonian, qaoa_circuit=qaoa_circuit, 
    hamiltonians=hamiltonians, swap_depths=swap_depths, layouts=layouts, compiled_circuits=compiled_circuits
)

basepath = '/lustre/scratch127/qpg/jc59/hubo/'
dump_file = basepath + f'simulation.{args.coupling_map}.per_layer.'  + '{}.extra{}.four{}.six{}.method{}.cvar{}.p{}.shots{}.init{}'.format(
    args.filename,
    args.extra,
    args.fraction_four,
    args.fraction_six,
    args.method, 
    args.alpha, 
    p,
    args.shots,
    args.init
) + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(obj_to_dump, f)
