
import numpy as np
import networkx as nx
import numpy.typing as npt
import pickle
import argparse
from collections import Counter
from typing import Optional
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions, DynamicalDecouplingOptions


from qopt_best_practices.sat_mapping import SATMapper

from qubo_qaoa.utils.swap_strategy import ExtendedSwapStrategy
from qubo_qaoa.utils.circuit_construction import circuit_construction

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)


def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("ecr", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-n', '--shots', type=int)
args = parser.parse_args()

filename: str = args.filename
N: int = args.nodes
shots: int = args.shots

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits

service = QiskitRuntimeService(name='eu_test_instance')
backend = service.least_busy(min_num_qubits=num_qubits, operational=True, simulator=False) 
# backend = service.backend(name='ibm_aachen')
error_mit = True
ddOptions = DynamicalDecouplingOptions(enable=error_mit, sequence_type="XX")
twirlingOptions = TwirlingOptions(enable_gates=error_mit, enable_measure=error_mit, num_randomizations='auto', shots_per_randomization='auto', strategy="active-accum")
samplerOptions = SamplerOptions(dynamical_decoupling=ddOptions, twirling=twirlingOptions)
sampler = Sampler(mode=backend, options=samplerOptions)
logger.info(f'Backend: {backend}')
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')


rows, cols = 1, 1
while 4 * (rows + cols + rows * cols) < num_qubits:
    if rows < cols:
        rows += 1
    else:
        cols += 1
print(f'Min size to support virtual qubits: {(rows, cols)}')

swap_strat = ExtendedSwapStrategy.from_heavy_hex(rows, cols)
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


eta = 1
eps = 0.005

def normalisation(energies: npt.NDArray, beta_T: float) -> float:
    return sum([np.exp(-beta_T * E) for E in energies])

def boltzmann(energies: npt.NDArray, Z: float, beta_T: float) -> npt.NDArray:
    return np.exp(- beta_T * energies) / Z

def bias(boltzmanns: npt.NDArray, q: int, samples: list[str]) -> float:
    return sum([boltzmanns[i] * (-1 if samples[i][q] == '1' else 1) for i in range(len(samples))])

def get_biases(samples: list[str], energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    Z = normalisation(energies, beta_T)
    boltzmanns = boltzmann(energies, Z, beta_T)
    return np.array([bias(boltzmanns, q, samples) for q in range(num_qubits)][::-1])

def get_angles(samples: list[str], energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    biases = get_biases(samples, energies, beta_T)
    probabilities = 0.5 * (1 - eta * biases)
    probabilities[probabilities < eps] = eps
    probabilities[probabilities > 1 - eps] = 1 - eps
    
    angles = 2 * np.arcsin(np.sqrt(probabilities))
    return angles


def iteration(qc, angles, beta_T, history):
    sample_circuit = qc.assign_parameters(angles, inplace=False)
    sampler_job = sampler.run([sample_circuit],shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    
    evals = evaluate_sparse_pauli_samples(counts.keys(), hamiltonian) + ising_offset
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
    total_energy = np.mean(energies)
    
    
    history.append([samples, energies, total_energy])
    new_angles = get_angles(samples, np.array(energies), beta_T)
    return new_angles


def warm_start(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]=None):
    betas = [(1-k/p) * delta_b for k in range(p)]
    gammas = [(k+1) / p * delta_g for k in range(p)]
    fixed_params = betas + gammas
    
    if circ is None:
        phis = ParameterVector('ϕ', num_qubits)
        betas = ParameterVector('β', p)


        init = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            init.ry(phis[i], i)
            
        mixer = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            mixer.ry(-phis[i], i)
            mixer.rz(-2*betas[0], i)
            mixer.ry(phis[i], i)
            
        # TODO: need to feed in the init and mixer to the circuit construction
        circ_dict = circuit_construction(hamiltonian.num_qubits, cost_op, sat_map, p, backend, edge_colouring_copy, swap_strat)
        circuit = circ_dict["backend"]
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}. Circuit counts: {circuit.count_ops()}')
    else:
        circuit = circ
    

    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)


    history = []
    angles = [init_angles]
    iters = 10

    for i in range(iters):
        beta_T = (i ** 2) * 0.9 / (iters - 1) ** 2 + 0.1
        angles.append(iteration(fixed_qc, angles[-1], beta_T, history))
        
        
    for i in range(iters):
        logger.info(i)
        counter = Counter(history[i][0])
        energy_counter = Counter(history[i][1])
        energy = history[i][2]
        logger.info(energy)
        logger.info(counter.most_common(5))
        logger.info(evaluate_sparse_pauli_samples([e[0] for e in counter.most_common(5)], hamiltonian) + ising_offset)
        
        logger.info(energy_counter.most_common(5))
        logger.info('------------------------------')
    energy = history[-1][2]
    samples = (history[0][0], history[iters // 2][0], history[-1][0])
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit
        

delta_b = 0.1
delta_g = 4.0
ps = [1, 3, 6]

prob = 1 / (2 * N)
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles = theta * np.ones((num_qubits,))

energies = {}
samples_dict = {}

circuit = None
for p in ps:
    e, samples, _ = warm_start(p, delta_b, delta_g, None)
    energies[p] = e
    samples_dict[p] = 0
    
to_save=dict(energies=energies, delta_b=delta_b, delta_g=delta_g, ps=ps, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/hardware.error_mit{error_mit}.{filename}.db{delta_b}.dg{delta_g}.shots{shots}.pkl', 'wb') as f:
    pickle.dump(to_save, f)