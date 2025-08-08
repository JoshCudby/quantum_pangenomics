import numpy as np
import networkx as nx
import pickle
import gfapy
import argparse
from sympy import Poly, Symbol
from itertools import combinations
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate, CXGate, SwapGate

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration



from qiskit.quantum_info import SparsePauliOp


from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_doubles_graph, hamiltonian_to_interactions, monomial_to_pauli
from qiskit_qaoa.utils.string_utils import bin_rep
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, CommutingGateRouter, FindCommutingPauliEvolutionsMulti, DecomposePauliZEvolution
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('-t', '--timeout', type=int)
parser.add_argument('-R', '--grid-rows', type=int)
parser.add_argument('-C', '--grid-cols', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
args = parser.parse_args()

class Binary(Symbol):
    def _eval_power(self, other):
        return self
    

def two_qubit_count(qc: QuantumCircuit):
    ops = qc.count_ops()
    return ops.get("cz", 0) + ops.get("rzz", 0) + ops.get("cx", 0) + ops.get("swap", 0)
   
    
def print_circuit_info(qc: QuantumCircuit, circuit_name: str):
    logger.info(f'{circuit_name} has {two_qubit_count(qc)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    

extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(args.grid_rows, args.grid_cols)
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

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

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

# qc = QuantumCircuit(num_physical_qubits)
# for layer in range(18):
#     for swap in extended_swap_strat.swap_layer(layer):
#         qc.swap(swap[0], swap[1])
# print_circuit_info(qc, 'Swaps only, 18 layers')


filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{args.filename}.gfa'

gfa = gfapy.Gfa.from_file(filepath, vlevel=0)
copy_numbers = args.copy_numbers


graph = nx.DiGraph()
for index, segment_line in enumerate(gfa.segments):
    graph.add_node(f'{segment_line.name}_+', weight=copy_numbers[index], start=segment_line.st)
    graph.add_node(f'{segment_line.name}_-', weight=copy_numbers[index], start=segment_line.st)
for edge_line in gfa.edges:
    v1 = edge_line.sid1
    v2 = edge_line.sid2
    graph.add_edges_from([
        (f'{v1.name}_{v1.orient}', f'{v2.name}_{v2.orient}'),
    ])
    v1.invert()
    v2.invert()
    graph.add_edges_from([
        (f'{v2.name}_{v2.orient}', f'{v1.name}_{v1.orient}'),
    ])

nodes = list(graph.nodes)
N = len(nodes)
n = int(np.ceil(np.log2(N+1)))
total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
T = int(1.1 * total_weight)

x = [[Binary(f'x[{t}][{i}]') for i in range(n)] for t in range(T)]


constraint = sum([
    1 - sum([
        np.prod([
            1 - x[t][k] - bin_rep(i, n)[k] + 2 * x[t][k] * bin_rep(i, n)[k]
        for k in range(n)]) * sum([
            np.prod([
                1 - x[t+1][k] - bin_rep(j, n)[k] + 2 * x[t+1][k] * bin_rep(j, n)[k]
            for k in range(n)])
        for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])] + [N+1] ])
    for i in range(N)])
for t in range(T-1)])

obj = sum([
    (
        sum([
            np.prod([
                1 - x[t][k] - bin_rep(i, n)[k] + 2 * x[t][k] * bin_rep(i, n)[k]
            for k in range(n)])
            + np.prod([
                1 - x[t][k] - bin_rep(i+1, n)[k] + 2 * x[t][k] * bin_rep(i+1, n)[k]
            for k in range(n)])
        for t in range(T)])
        - graph.nodes[nodes[i]]["weight"]
    ) ** 2
for i in range(0,N,2)])


lamda = 10
total = lamda * constraint + obj

Z = [Binary(f"Z[{i}]") for i in range(n*T)]
ising = total.subs(zip([item for row in x for item in row], [0.5 - z/2 for z in Z]))

ising = Poly(ising, Z)

ising_expr_coeffs = ising.as_expr().as_coefficients_dict()

num_qubits = n*T
logger.info(f'Virtual qubits: {num_qubits}')

hamiltonian = SparsePauliOp('I' * num_qubits, ising_expr_coeffs[1])
for (monomial, coeff) in ising_expr_coeffs.items():
    if monomial == 1:
        continue
    hamiltonian += SparsePauliOp(monomial_to_pauli(monomial, n * T), coeff)
hamiltonian = hamiltonian.sort(weight=True)

logger.info(f'Number of hamiltonian terms: {len(hamiltonian)}')

logger.info('------------------------------------')
logger.info('------------------------------------')

qaoa_cost_op = QAOAAnsatz(
    hamiltonian,
    mixer_operator=QuantumCircuit(num_qubits),
    initial_state=QuantumCircuit(num_qubits)
)
# tqaoa = transpile(qaoa_cost_op, basis_gates=["sx", "rz", "cz"])
backend_tqaoa = transpile(qaoa_cost_op, optimization_level=3, backend=backend, basis_gates=basis_gates)

print_circuit_info(backend_tqaoa, 'Default qaoa circuit on backend')
logger.info(backend_tqaoa.count_ops())

logger.info('------------------------------------')
logger.info('------------------------------------')


# program_graph = hamiltonian_to_doubles_graph(hamiltonian)
# sat_results = SATMapper(timeout=60).find_initial_mappings(
#     program_graph, extended_swap_strat, 0, len(extended_swap_strat)
# )
# solutions = [k for k, v in sat_results.items() if v.satisfiable]
# if len(solutions):
#     min_k = min(solutions)
#     logger.info(f'Min SWAP layers to satisfy doubles: {min_k}')
#     edge_map = dict(sat_results[min_k].mapping)
#     print(f'Doubles edge map: {edge_map}')

#     new_hamiltonian = hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)
    
#     new_cost_circ = QuantumCircuit(num_physical_qubits)
#     new_cost_circ.append(PauliEvolutionGate(new_hamiltonian), range(num_physical_qubits))
#     new_tcost = pm.run(new_cost_circ)
#     print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
#     backend_new_tcost = transpile(new_tcost, optimization_level=3, backend=backend, basis_gates=basis_gates)
    
#     print_circuit_info(backend_new_tcost, 'Remapped, commuting gate routed circuit on backend')
#     logger.info(backend_new_tcost.count_ops())

# else:
#     logger.info('Could not find graph remapping')
    
    
# logger.info('------------------------------------')
# logger.info('------------------------------------')    

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
results = {}
for num_layers in range(0, 101, 10):
    logger.info('--------------------------------------------------')
    sat_results = mapper.hubo_max_sat(
        program_interactions, extended_swap_strat, num_layers
    )
    if sat_results is None:
        logger.info('No results')
        continue
    mapping = sat_results[num_layers][1]
    edge_map = dict(mapping)
    logger.info(f'Cost: {sat_results[num_layers][0]}')
    logger.info(edge_map)

    pm = PassManager(
        [
            HighLevelSynthesis(basis_gates=["PauliEvolution"]), # Not needed if set up circuit as PauliEvolutionGate
            FindCommutingPauliEvolutionsMulti(), 
            CommutingGateRouter(
                extended_swap_strat,
                edge_colouring,
                max_layers=num_layers,
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


    new_hamiltonian = hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)
    new_cost_circ = QuantumCircuit(num_physical_qubits)
    new_cost_circ.append(PauliEvolutionGate(new_hamiltonian), range(num_physical_qubits))
    new_tcost = pm.run(new_cost_circ)
    
    print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
    print(new_tcost.count_ops())
    
    backend_new_tcost = transpile(new_tcost, optimization_level=3, backend=backend, basis_gates=basis_gates)
    
    print_circuit_info(backend_new_tcost, 'Remapped, commuting gate routed circuit on backend')
    print(backend_new_tcost.count_ops())
    results[num_layers] = (new_tcost, backend_new_tcost)
    
    
with open(f'/lustre/scratch127/qpg/jc59/hubo/results_{args.filename}_extra{args.extra}_four{args.fraction_four}_six{args.fraction_six}.pkl', 'wb') as f:
    pickle.dump(results, f)
    

# sat_results = HigherOrderSatMapper(timeout=60).find_hubo_mappings(
#     program_interactions, extended_swap_strat, 0, len(extended_swap_strat)
# )
# solutions = [k for k, v in sat_results.items() if v.satisfiable]
# if len(solutions):
#     # Problem:
#     # To achieve high order connections, might end up using very many swap layers
#     # Might be more efficient to find a good mapping for low-order ones and dump high order gates at the end with requisite swaps
#     min_k = min(solutions)
#     logger.info(f'Min SWAP layers to satisfy HUBO: {min_k}')
#     edge_map = dict(sat_results[min_k].mapping)
#     print(f'HUBO edge map: {edge_map}')

#     new_hamiltonian = hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)
    
#     new_cost_circ = QuantumCircuit(num_physical_qubits)
#     new_cost_circ.append(PauliEvolutionGate(new_hamiltonian), range(num_physical_qubits))
#     new_tcost = pm.run(new_cost_circ)
#     print_circuit_info(new_tcost, 'Remapped HUBO, commuting gate routed circuit')
#     backend_new_tcost = transpile(new_tcost, optimization_level=3, backend=backend, basis_gates=basis_gates)
    
#     print_circuit_info(backend_new_tcost, 'Remapped HUBO, commuting gate routed circuit on backend')
#     logger.info(backend_new_tcost.count_ops())

# else:
#     logger.info('Could not find graph remapping for HUBO')