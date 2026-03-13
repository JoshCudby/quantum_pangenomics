
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
from qubo_qaoa.utils.iterative_qaoa_utils import IterativeQAOAData, iteration, get_beta_T

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
parser.add_argument('-v', '--vertices', type=str)
parser.add_argument('-n', '--shots', type=int)
args = parser.parse_args()

shots: int = args.shots

rng = np.random.default_rng()

data_file = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational/phylogeny/{args.vertices}v_pauli.pickle'
with open(data_file, 'rb') as f:
    hamiltonian = pickle.load(f)
num_qubits: int = hamiltonian.num_qubits


def warm_start(
    p: int, 
    delta_b: float, 
    delta_g: float, 
    circ: Optional[QuantumCircuit]=None
) -> tuple[float, list[list[str]], QuantumCircuit]:
    phis = ParameterVector('ϕ', num_qubits)
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=phis, measure=True
    )
    print(f'Circuit 2Q depth: {fixed_qc.depth(lambda instr: len(instr.qubits) > 1)}, ops: {fixed_qc.count_ops()}')
    
    history = []
    angles = init_angles
    iters = 5
    
    for i in range(iters):
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T), data, history, T=None)
        
    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit

delta_b_fixed = 1.0
delta_g_fixed = 1.0
        
eta = 1
eps = 0.15
max_beta_T =  0.15
alpha = 1
ising_offset= 726/3 if int(args.vertices) == 64 else 812/3
print(ising_offset)
data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    ising_offset=ising_offset,
    eta=eta,
    eps=eps,
    alpha=alpha
)

# init_angles = np.pi/2 * np.ones((num_qubits,))
prob = 1 / 2
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles: npt.NDArray = theta * np.ones((num_qubits,))


# rescaling = np.logspace(-0.5, 0.2, 8, base=10)
# ps = sorted(set([int(x) for x in np.logspace(0, 1.5, 3, base=10)]))
rescaling = np.linspace(0.1, 1, 10)
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
    
to_save=dict(energies=energies, delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/phylogeny/iter_qaoa.{args.vertices}v.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'wb') as f:
    pickle.dump(to_save, f)