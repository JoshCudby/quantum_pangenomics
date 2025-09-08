import numpy as np
import networkx as nx
import pickle
import argparse
from itertools import combinations
from collections import Counter

from qiskit import QuantumCircuit, transpile, generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.quantum_info import SparsePauliOp

from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.pass_managers import get_hubo_pass_manager
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('--fraction-constraint', type=float)
parser.add_argument('-t', '--timeout', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
parser.add_argument('-C', '--coupling-map', choices=['line', 'grid'])

args = parser.parse_args()
logger.info(args)

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
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map)
logger.info(backend.configuration().to_dict()["n_qubits"])

full_hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, fraction_terms=1.0)
hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, fraction_terms=args.fraction_constraint)

logger.info(f'Number of hamiltonian terms: {len(hamiltonian)}')

logger.info('------------------------------------')
logger.info('------------------------------------')

qaoa_cost_op = QAOAAnsatz(
    hamiltonian,
    mixer_operator=QuantumCircuit(num_qubits),
    initial_state=QuantumCircuit(num_qubits)
)
backend_tqaoa = transpile(qaoa_cost_op, optimization_level=3, backend=backend, basis_gates=basis_gates)

print_circuit_info(backend_tqaoa, 'Default qaoa cost layer on backend')
logger.info(backend_tqaoa.count_ops())

logger.info('------------------------------------')
logger.info('------------------------------------')


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
results: dict[str | int, SparsePauliOp | dict] = {
    'full_hamiltonian': full_hamiltonian,
    'compiled_hamiltonian': hamiltonian
}

layers = sorted(list(set([int(x) for x in np.linspace(0, len(extended_swap_strat._swap_layers), 10)])))
for num_layers in layers:
# for num_layers in range(0, len(extended_swap_strat._swap_layers), 100):
    logger.info('--------------------------------------------------')
    sat_results = mapper.hubo_max_sat(
        num_qubits, program_interactions, extended_swap_strat, num_layers
    )
    if sat_results is None:
        logger.info('No results')
        continue
    mapping = sat_results[num_layers][1]
    edge_map = dict(mapping)
    all_qubits = set(range(num_qubits))
    keys = set(edge_map.keys())
    vals = set(edge_map.values())
    missing_keys, missing_vals = all_qubits.difference(keys), all_qubits.difference(vals)
    for key, val in zip(missing_keys, missing_vals):
        logger.info(f'Manually assigning: {key}:{val}')
        edge_map[key] = val
    logger.info(f'Cost: {sat_results[num_layers][0]}')
    logger.info(edge_map)

    pm = get_hubo_pass_manager(extended_swap_strat, num_layers, args.extra)

    new_cost_circ = QuantumCircuit(num_physical_qubits)
    new_cost_circ.append(PauliEvolutionGate(hamiltonian), [edge_map[i] for i in range(num_physical_qubits)])
    new_tcost = pm.run(new_cost_circ)
    
    print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
    print(new_tcost.count_ops())
    
    
    generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, basis_gates=basis_gates)
    init  = generic_pm.init
    init.remove(3)
    generic_pm.init = init
    generic_pm.layout = None
    backend_new_tcost = generic_pm.run(new_tcost)
    
    print_circuit_info(backend_new_tcost, 'Remapped, commuting gate routed circuit on backend')
    print(backend_new_tcost.count_ops())
    results[num_layers] = edge_map
    
    
    
basepath = '/lustre/scratch127/qpg/jc59/hubo/'
filename = 'simulation.{}.compilation.{}.extra{}.constraint{}.four{}.six{}'.format(
    args.coupling_map,
    args.filename,
    args.extra,
    args.fraction_constraint,
    args.fraction_four,
    args.fraction_six
)
dump_file = basepath + filename + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(results, f)
    
