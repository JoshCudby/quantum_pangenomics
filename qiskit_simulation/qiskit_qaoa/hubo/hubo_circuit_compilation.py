import numpy as np
import networkx as nx
import re
import gfapy
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from sympy import Poly, Symbol

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate, CXGate, SwapGate

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qopt_best_practices.sat_mapping import SATMapper

from qiskit.quantum_info import SparsePauliOp


from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import HighLevelSynthesis, InverseCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, CommutingGateRouter, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)


class Binary(Symbol):
    def _eval_power(self, other):
        return self
    
    
def monomial_to_pauli(monomial, size):
    indices = [int(re.search(r'[0-9]+', atom.name).group(0)) for atom in monomial.atoms()]
    pauli_str = ['I'] * size
    for i in indices:
        pauli_str[i] = 'Z'
    return ''.join(pauli_str)


def two_qubit_count(qc):
    return qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)


def depth(qc):
    return qc.depth(lambda instr: len(instr.qubits) > 1)


def bin_rep(k, n):
    return [int(x) for x in np.binary_repr(k, n)[::-1]]


def callback_func(**kwargs):
    pass_ = kwargs['pass_']
    dag = kwargs['dag']
    logger.info(pass_, dag.properties())
    
    
def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    
    
def hamiltonian_to_doubles_graph(hamiltonian: SparsePauliOp) -> nx.Graph:
    edges = []
    weights = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) == 2:
            edge = np.nonzero(t.paulis[0].z)[0]
            edges.append(edge)
            weights.append(t.coeffs[0])
            
    program_graph = nx.Graph()
    for i in range(hamiltonian.num_qubits):
        program_graph.add_node(i)
    for idx in range(len(weights)):
        program_graph.add_edge(edges[idx][0],edges[idx][1],weight=weights[idx])
    return program_graph


def hamiltonian_to_interactions(hamiltonian: SparsePauliOp) -> list[tuple]:
    interactions = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) < 2 or np.sum(t.paulis[0].z) > 5:
            continue
        if np.sum(t.paulis[0].z) == 2:
            edge = np.nonzero(t.paulis[0].z)[0]
            interactions.append(edge)
        if rng.random() > 0.01:
            edge = np.nonzero(t.paulis[0].z)[0]
            interactions.append(edge)
    return interactions


extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(2, 2)
num_physical_qubits = extended_swap_strat._num_vertices

basis_gates=["sx", "x", "rz", "rzz", "cz", "id"]

backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates=basis_gates
)
# fake_fez = FakeFez()
# fake_algiers = FakeAlgiers()

config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = num_physical_qubits
config = AerBackendConfiguration.from_dict(AerSimulator._DEFAULT_CONFIGURATION)
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map)

# backend = AerSimulator.from_backend(fake_algiers, **backend_options)


filename = 'trivial'
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'

gfa = gfapy.Gfa.from_file(filepath, vlevel=0)
copy_numbers = [1,1,1]


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
hamiltonian = SparsePauliOp('I' * num_qubits, ising_expr_coeffs[1])
for (monomial, coeff) in ising_expr_coeffs.items():
    if monomial == 1:
        continue
    hamiltonian += SparsePauliOp(monomial_to_pauli(monomial, n * T), coeff)
hamiltonian = hamiltonian.sort(weight=True)

logger.info('------------------------------------')
logger.info('------------------------------------')

qaoa_cost_op = QAOAAnsatz(
    hamiltonian,
    mixer_operator=QuantumCircuit(num_qubits),
    initial_state=QuantumCircuit(num_qubits)
)
tqaoa = transpile(qaoa_cost_op, basis_gates=["sx", "rz", "cz"])
backend_tqaoa = transpile(tqaoa, backend=backend, basis_gates=basis_gates)

print_circuit_info(backend_tqaoa, 'Default qaoa circuit on backend')
logger.info(backend_tqaoa.count_ops())

logger.info('------------------------------------')
logger.info('------------------------------------')


pm = PassManager(
    [
        HighLevelSynthesis(basis_gates=["PauliEvolution"]), # Not needed if set up circuit as PauliEvolutionGate
        FindCommutingPauliEvolutionsMulti(), 
        CommutingGateRouter(
            extended_swap_strat,
        ),
        SwapToFinalMapping(),
        InverseCancellation(gates_to_cancel=[SwapGate()]),
        HighLevelSynthesis(basis_gates=["sx", "x", "rz", "rzz", "cx", "id"]),
        InverseCancellation(gates_to_cancel=[CXGate()]),
    ]
)


# cost_circ = QuantumCircuit(num_qubits)
# cost_circ.append(PauliEvolutionGate(hamiltonian), range(num_qubits))
# tcost = pm.run(cost_circ)
# ttcost = transpile(tcost, basis_gates=["h", "rz", "cz"])
# tttcost = transpile(ttcost, basis_gates=["sx", "rz", "cz"])
# backend_tcost = transpile(tttcost, backend=backend, basis_gates=basis_gates)


# print_circuit_info(backend_tcost, 'Commuting gate routed circuit on backend')
# logger.info(backend_tcost.count_ops())

logger.info('------------------------------------')
logger.info('------------------------------------')


program_graph = hamiltonian_to_doubles_graph(hamiltonian)
sat_results = SATMapper(timeout=60).find_initial_mappings(
    program_graph, extended_swap_strat, 0, len(extended_swap_strat)
)
solutions = [k for k, v in sat_results.items() if v.satisfiable]
if len(solutions):
    min_k = min(solutions)
    logger.info(f'Min SWAP layers to satisfy doubles: {min_k}')
    edge_map = dict(sat_results[min_k].mapping)
    print(f'Doubles edge map: {edge_map}')

    new_hamiltonian = hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)
    
    new_cost_circ = QuantumCircuit(num_physical_qubits)
    new_cost_circ.append(PauliEvolutionGate(new_hamiltonian), range(num_physical_qubits))
    new_tcost = pm.run(new_cost_circ)
    print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
    backend_new_tcost = transpile(new_tcost, optimization_level=3, backend=backend, basis_gates=basis_gates)
    
    print_circuit_info(backend_new_tcost, 'Remapped, commuting gate routed circuit on backend')
    logger.info(backend_new_tcost.count_ops())

else:
    logger.info('Could not find graph remapping')
    
    
logger.info('------------------------------------')
logger.info('------------------------------------')    

program_interactions = hamiltonian_to_interactions(hamiltonian)
sat_results = HigherOrderSatMapper(timeout=60).find_hubo_mappings(
    program_interactions, extended_swap_strat, 0, len(extended_swap_strat)
)
solutions = [k for k, v in sat_results.items() if v.satisfiable]
if len(solutions):
    # Problem:
    # To achieve high order connections, might end up using very many swap layers
    # Might be more efficient to find a good mapping for low-order ones and dump high order gates at the end with requisite swaps
    min_k = min(solutions)
    logger.info(f'Min SWAP layers to satisfy HUBO: {min_k}')
    edge_map = dict(sat_results[min_k].mapping)
    print(f'HUBO edge map: {edge_map}')

    new_hamiltonian = hamiltonian.apply_layout([edge_map[i] for i in range(num_qubits)], num_physical_qubits)
    
    new_cost_circ = QuantumCircuit(num_physical_qubits)
    new_cost_circ.append(PauliEvolutionGate(new_hamiltonian), range(num_physical_qubits))
    new_tcost = pm.run(new_cost_circ)
    print_circuit_info(new_tcost, 'Remapped HUBO, commuting gate routed circuit')
    backend_new_tcost = transpile(new_tcost, optimization_level=3, backend=backend, basis_gates=basis_gates)
    
    print_circuit_info(backend_new_tcost, 'Remapped HUBO, commuting gate routed circuit on backend')
    logger.info(backend_new_tcost.count_ops())

else:
    logger.info('Could not find graph remapping for HUBO')