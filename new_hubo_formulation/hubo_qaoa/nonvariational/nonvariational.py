
import numpy as np
import numpy.typing as npt
import pickle
import argparse
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
# from qiskit_ibm_runtime.fake_provider import FakeFez

# from qopt_best_practices.sat_mapping import SATMapper

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.parameterise_circuit import parameterise_circuit
from hubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit

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
# parser.add_argument('-p', '--reps', type=int)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
args = parser.parse_args()

filename: str = args.filename
# p: int = args.reps
shots: int = args.shots

rng = np.random.default_rng()

data_file = '/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.couplingall.precompute.0.pkl'
with open(data_file, 'rb') as f:
    res = pickle.load(f)
cost_circuit = parameterise_circuit(res[filename]['rzz']['circuit'], parameter=Parameter('γ'))
num_qubits: int = cost_circuit.num_qubits    
    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
full_hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)


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
    
    evals = evaluate_sparse_pauli_samples(counts.keys(), full_hamiltonian * norm)
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
    total_energy = np.mean(energies)
    
    
    history.append([samples, energies, total_energy])
    new_angles = get_angles(samples, np.array(energies), beta_T)
    return new_angles



prob = 1 / 2
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles = theta * np.ones((num_qubits,))



def warm_start(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]=None):
    phis = ParameterVector('ϕ', num_qubits)
    fixed_qc, circuit = get_LR_qaoa_circuit(p, delta_b, delta_g, num_qubits, cost_circuit, circ, phis, True)
    history = []
    angles = [init_angles]
    iters = 10

    for i in range(iters):
        # beta_T = (i ** 2) * 0.9 / (iters - 1)**2 + 0.1
        beta_T = (i ** 2) * 0.4 / (iters - 1)**2 + 0.1
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
        
# delta_b_fixed = 0.45
# delta_g_fixed = 0.26
delta_b_fixed = 0.33
delta_g_fixed = 0.19

rescaling = np.logspace(-0.5, 0.2, 8, base=10)
ps = sorted(set([int(x) for x in np.logspace(0, 1.5, 3, base=10)]))
# ps = [1, 5, 10]
# ps = [1, 6, 11, 16, 21]

energies = np.zeros((len(ps), len(rescaling)))
samples_dict = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], rescaling[j])] = samples
    
to_save=dict(energies=energies, delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/nonvariational.{filename}.db{delta_b_fixed}.dg{delta_g_fixed}.shots{shots}.pkl', 'wb') as f:
    pickle.dump(to_save, f)