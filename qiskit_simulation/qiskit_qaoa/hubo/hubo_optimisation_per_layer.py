import numpy as np
import networkx as nx
import pickle
import argparse
from itertools import combinations
from collections import Counter

from qiskit import QuantumCircuit, transpile, generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate, CXGate

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.quantum_info import SparsePauliOp

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router import CommutingGateRouter
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
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-d', '--swap-depth', type=int, default=0)
parser.add_argument('-M', '--method', type=str)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random', 'warm'], default='ramp')
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

hamiltonians = {}
for layer in range(args.reps):
    hamiltonians[layer] = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, fraction_terms=(layer/args.reps, (layer+1)/args.reps))
