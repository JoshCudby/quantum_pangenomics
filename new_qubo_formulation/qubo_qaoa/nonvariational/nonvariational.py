
import numpy as np
import numpy.typing as npt
import pickle
import argparse
from collections import Counter
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
# from qiskit_ibm_runtime.fake_provider import FakeFez

# from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

backend_options = dict(
    method='matrix_product_state',
    matrix_product_state_max_bond_dimension='32', 
    device='CPU',
    precision='single',
    basis_gates = ['rx', 'ry', 'rz', 'cx']
)
# fake_fez = FakeFez()
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-N', '--nodes', type=int)
# parser.add_argument('-p', '--reps', type=int)
parser.add_argument('-n', '--shots', type=int)
args = parser.parse_args()

filename: str = args.filename
N: int = args.nodes
# p: int = args.reps
shots: int = args.shots

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits

   
swap_strat = SwapStrategy.from_line(list(range(num_qubits)))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}

singles = hamiltonian[hamiltonian.paulis.z.sum(axis=-1) == 1]
doubles = hamiltonian[hamiltonian.paulis.z.sum(axis=-1) == 2]


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


# init_angles = np.pi/2 * np.ones((num_qubits,))

prob = 1 / (2 * N)
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles = theta * np.ones((num_qubits,))

# init_angles = np.pi * np.array([1,0,0,0,0,0,1,0])


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
            
        circ_dict = circuit_construction(singles, doubles, None, swap_strat, edge_coloring, {}, p, init, mixer)
        circuit = circ_dict["circuit_to_sample"]
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}. Circuit counts: {circuit.count_ops()}')
    else:
        circuit = circ
    

    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)


    history = []
    angles = [init_angles]
    iters = 10

    for i in range(iters):
        beta_T = (i ** 2) * 0.9 / (iters - 1)**2 + 0.1
        angles.append(iteration(fixed_qc, angles[-1], beta_T, history))
        
        
    # for i in range(iters):
        # print(i)
        # counter = Counter(history[i][0])
        # energy_counter = Counter(history[i][1])
        # energy = history[i][2]
        # print(energy)
        # print(counter.most_common(5))
        # print(evaluate_sparse_pauli_samples([e[0] for e in counter.most_common(5)], hamiltonian) + ising_offset)
        
        # print(energy_counter.most_common(5))
        # print()
    energy = history[-1][2]
    samples = (history[0][0], history[iters // 2][0], history[-1][0])
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit
        

delta_bs = np.linspace(0.1, 0.5, 5)
delta_gs = np.linspace(3.75, 4.25, 5)
ps = range(1, 12, 5)

# deltas = np.linspace(0.01, 1, 3)
# ps = range(1, 3)

energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
samples_dict = {}

circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, samples, circuit = warm_start(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e
    samples_dict[(ps[i], delta_bs[j], delta_gs[k])] = samples
    
to_save=dict(energies=energies, delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational.{filename}.db{delta_bs[-1]}.dg{delta_gs[-1]}.shots{shots}.pkl', 'wb') as f:
    pickle.dump(to_save, f)