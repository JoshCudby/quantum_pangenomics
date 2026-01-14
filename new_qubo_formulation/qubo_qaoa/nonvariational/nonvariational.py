
import numpy as np
import numpy.typing as npt
import pickle
import argparse
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit

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
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-n', '--shots', type=int)
args = parser.parse_args()

filename: str = args.filename
N: int = args.nodes
shots: int = args.shots

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

_, hamiltonian, _, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits


def get_beta_T(i: int):
    # A quadratic ramp from 0.1 to 1 over 10 iterations, scaled by max_beta_T
    return ((i ** 2)/9 + 1) / 10 * max_beta_T

def boltzmann(energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    B = np.exp(- beta_T * energies ** 2)
    return B / np.sum(B)

def bias(boltzmanns: npt.NDArray, q: int, samples: list[str]) -> float:
    return sum([boltzmanns[i] * (-1 if samples[i][q] == '1' else 1) for i in range(len(samples))])

def get_biases(samples: list[str], energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    boltzmanns = boltzmann(energies, beta_T)
    return np.array([bias(boltzmanns, q, samples) for q in range(num_qubits)][::-1])

def get_angles(samples: list[str], energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    biases = get_biases(samples, energies, beta_T)
    probabilities = 0.5 * (1 - eta * biases)
    probabilities[probabilities < eps] = eps
    probabilities[probabilities > 1 - eps] = 1 - eps
    
    angles = 2 * np.arcsin(np.sqrt(probabilities))
    return angles

def subsample(samples, energies, alpha=1.0) -> tuple[list[str], npt.NDArray]:
    idx = np.argsort(energies)
    sorted_energies = energies[idx]
    sorted_samples = [samples[i] for i in idx]
    end_idx = int(alpha * len(energies))
    return sorted_samples[:end_idx],sorted_energies[:end_idx]


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
    energies = np.array(energies)
    total_energy = np.mean(energies)
    
    subsamples, subenergies = subsample(samples, energies, alpha)
    history.append([samples, energies, total_energy])
    new_angles = get_angles(subsamples, subenergies, beta_T)
    return new_angles


def warm_start(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]=None) -> tuple[float, list[list[str]], QuantumCircuit]:
    phis = ParameterVector('ϕ', num_qubits)
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=phis, measure=True
    )
    
    history = []
    angles = init_angles
    iters = 5

    for i in range(iters):
        angles = iteration(fixed_qc, angles, get_beta_T(i), history)
        
    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit

        
eta = 1
eps = 0.15

delta_b_fixed = 0.63
delta_g_fixed = 0.16
max_beta_T =  0.15
alpha = 1.0

# init_angles = np.pi/2 * np.ones((num_qubits,))
prob = 1 / (2 * N)
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles = theta * np.ones((num_qubits,))


# rescaling = np.logspace(-0.5, 0.2, 8, base=10)
# ps = sorted(set([int(x) for x in np.logspace(0, 1.5, 3, base=10)]))
rescaling = np.array([1,])
ps = [1, 3, 5]


# MAIN
energies = np.zeros((len(ps), len(rescaling)))
samples_dict: dict[tuple[int, float], list[list[str]]] = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], np.round(rescaling[j],3))] = samples
    
to_save=dict(energies=energies,  delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/nonvariational.{filename}.db{delta_b_fixed}.dg{delta_g_fixed}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'wb') as f:
    pickle.dump(to_save, f)