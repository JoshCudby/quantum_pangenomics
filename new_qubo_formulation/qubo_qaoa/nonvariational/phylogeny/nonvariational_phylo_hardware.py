
import numpy as np
import numpy.typing as npt
import networkx as nx
from itertools import product
import pickle
import argparse
from typing import Optional
from itertools import combinations
import re

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
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vertices', type=str)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('--simulation', action='store_true')
parser.add_argument('--error-mitigation', action='store_true')
args = parser.parse_args()

shots: int = args.shots
error_mitigation = args.error_mitigation
simulation = args.simulation

data_file = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational/phylogeny/{args.vertices}v_pauli.pickle'
with open(data_file, 'rb') as f:
    hamiltonian = pickle.load(f)
num_qubits: int = hamiltonian.num_qubits

# service = QiskitRuntimeService(name='eu_test_instance')
# backend = service.least_busy(min_num_qubits=num_qubits, operational=True, simulator=False) 
service = QiskitRuntimeService(name='us_instance')
backend = service.backend('ibm_boston')

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


logger.info('Compiling with line SWAP strategy')
swap_strat = QUBOSwapStrategy.from_line(range(num_qubits))
edge_colouring = {(i, i+1): i % 2 for i in range(num_qubits)}
edge_colouring.update({(i+1, i): i % 2 for i in range(num_qubits)})

qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = 1,
    flatten=True
)
graph = circuit_to_graph(qc, qc.parameters[1])

remapped_g, sat_map, min_sat_layers = SATMapper(timeout=60).remap_graph_with_sat(
    graph=graph, swap_strategy=swap_strat, max_layers = int(num_qubits + 1)
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
    logger.info(f'Circuit ops: {fixed_qc.count_ops()}')
    
    history = []
    angles_history = [init_angles]
    angles = init_angles
    iters = 5

    for i in range(iters):
        logger.info(f'Iter: {i+1}')
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T), data, history)
        logger.info(f'Energy: {history[-1][2]}')
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
alpha = 0.05

ising_offset= 726/3 if int(args.vertices) == 64 else 812/3
data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    ising_offset=ising_offset,
    eta=eta,
    eps=eps,
    alpha=alpha
)

prob = 1 / 2
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
append_str = f'.{args.vertices}v{".error_mit" if error_mitigation else ""}{".simulation" if simulation else ""}.backend{backend.name}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}'
with open(f'/lustre/scratch127/qpg/jc59/phylogeny/iter_qaoa.hardware{append_str}.pkl', 'wb') as f:
    pickle.dump(to_save, f)