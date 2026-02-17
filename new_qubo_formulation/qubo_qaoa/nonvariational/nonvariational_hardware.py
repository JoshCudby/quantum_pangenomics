
import numpy as np
import numpy.typing as npt
import networkx as nx
from itertools import product
import pickle
import argparse
from typing import Optional
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions, DynamicalDecouplingOptions

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from qopt_best_practices.sat_mapping import SATMapper

from qubo_qaoa.utils.swap_strategy import QUBOSwapStrategy
from qubo_qaoa.utils.iterative_qaoa_utils import IterativeQAOAData, iteration, get_beta_T
from qubo_qaoa.utils.lr_qaoa import get_hardware_LR_qaoa_circuit

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('--simulation', action='store_true')
parser.add_argument('--error-mitigation', action='store_true')
parser.add_argument('--heavy-hex', action='store_true')
args = parser.parse_args()

filename: str = args.filename
N: int = args.nodes
shots: int = args.shots
error_mitigation = args.error_mitigation
simulation = args.simulation

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits

# service = QiskitRuntimeService(name='eu_test_instance')
# backend = service.least_busy(min_num_qubits=num_qubits, operational=True, simulator=False) 
service = QiskitRuntimeService(name='us_instance')
backend = service.backend(name='ibm_boston')

if simulation:
    backend_options = dict(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension='32', 
        device='CPU',
        precision='single',
        basis_gates = backend.configuration().basis_gates
    )
    simulator = AerSimulator.from_backend(backend, **backend_options)
    sampler = AerSampler.from_backend(simulator)
else:
    # ddOptions = DynamicalDecouplingOptions(enable=False, sequence_type="XX")
    # shots_per_randomizations >= 100 per randomization, shot budget for experiment 
    twirlingOptions = TwirlingOptions(enable_gates=error_mitigation, enable_measure=error_mitigation, num_randomizations='auto', shots_per_randomization=100, strategy="active-accum")
    samplerOptions = SamplerOptions(twirling=twirlingOptions)
    sampler = Sampler(mode=backend, options=samplerOptions)

logger.info(f'Backend: {backend}')
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')

if args.heavy_hex:
    print('Compiling with heavy-hex SWAP strategy')
    rows, cols = 1, 1
    while 4 * (rows + cols + rows * cols) < num_qubits:
        if rows < cols:
            rows += 1
        else:
            cols += 1
    print(f'Min size to support virtual qubits: {(rows, cols)}')

    swap_strat = QUBOSwapStrategy.from_heavy_hex(rows, cols)
    coupling_map = swap_strat._coupling_map
    coupling_map_edge = list(coupling_map)
    physical_qubits = list(coupling_map.physical_qubits)

    dual_coupling_map = nx.Graph()

    for qubit in physical_qubits:
        edges = [edge for edge in coupling_map_edge if edge[0]==qubit]
        for edge1, edge2 in combinations(edges, 2):
            dual_coupling_map.add_edge(tuple(sorted(edge1)), tuple(sorted(edge2)))
    edge_colouring = nx.greedy_color(dual_coupling_map, interchange=True)
    edge_colouring_copy = {}
    for k, v in edge_colouring.items():
        edge_colouring_copy[k] = v
        edge_colouring_copy[k[::-1]] = v
    edge_colouring = edge_colouring_copy
else:
    print('Compiling with line SWAP strategy')
    swap_strat = QUBOSwapStrategy.from_line(range(num_qubits))
    edge_colouring = {(i, i+1): i % 2 for i in range(num_qubits)}
    edge_colouring.update({(i+1, i): i % 2 for i in range(num_qubits)})

qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = 1,
    flatten=True
)
graph = circuit_to_graph(qc, qc.parameters[1])

remapped_g, sat_map, min_sat_layers = SATMapper(timeout=30).remap_graph_with_sat(
    graph=graph, swap_strategy=swap_strat, max_layers = int(num_qubits + np.sqrt(num_qubits) + 61)
)
if remapped_g is None or sat_map is None:
    raise Exception('Failed to find initial layout')

cost_op = graph_to_operator(remapped_g, swap_strat._num_vertices)


def warm_start(
    p: int, 
    delta_b: float, 
    delta_g: float, 
    circ: Optional[QuantumCircuit]=None
) -> tuple[float, list[list[str]], QuantumCircuit, list[np.ndarray]]:
    phis = ParameterVector('ϕ', num_qubits)
    
    fixed_qc, circuit = get_hardware_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        cost_op, sat_map, backend, edge_colouring, swap_strat,
        circ, phis=phis,
    )
    
    history = []
    angles_history = [init_angles]
    angles = init_angles
    iters = 5

    for i in range(iters):
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T), data, history)
        angles_history.append(angles)
        

    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit, angles_history
     
        

delta_b_fixed = 0.63
delta_g_fixed = 0.16
        
eta = 1
eps = 0.15
max_beta_T =  0.15
alpha = 0.025

data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    ising_offset=ising_offset,
    eta=eta,
    eps=eps,
    alpha=alpha
)

prob = 1 / (2 * N)
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles: npt.NDArray = theta * np.ones((num_qubits,))


rescaling = np.array([1,])
ps = [1]

energies = {}
samples_dict = {}

# MAIN
energies = np.zeros((len(ps), len(rescaling)))
samples_dict: dict[tuple[int, float], list[list[str]]] = {}
angles_dict: dict[tuple[int, float], list[np.ndarray]] = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit, angles = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], np.round(rescaling[j], 3))] = samples
    angles_dict[(ps[i], np.round(rescaling[j], 3))] = angles
    
to_save=dict(energies=energies, delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict, angles_dict=angles_dict)    
append_str = f'.{filename}{".error_mit" if error_mitigation else ""}{".simulation" if simulation else ""}.backend{backend.name}.db{delta_b_fixed}.dg{delta_g_fixed}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}'
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/hardware/hardware{append_str}.pkl', 'wb') as f:
    pickle.dump(to_save, f)