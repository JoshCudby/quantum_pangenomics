
import numpy as np
import pickle
import argparse
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector, Parameter

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions, DynamicalDecouplingOptions

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.parameterise_circuit import parameterise_circuit
from hubo_qaoa.utils.lr_qaoa import get_hardware_LR_qaoa_circuit
from hubo_qaoa.utils.iterative_qaoa_utils import IterativeQAOAData, iteration, get_beta_T

from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
parser.add_argument('--simulation', action='store_true')
parser.add_argument('--error-mitigation', action='store_true')
# parser.add_argument('--heavy-hex', action='store_true')

args = parser.parse_args()

filename: str = args.filename
shots: int = args.shots
error_mitigation = args.error_mitigation
simulation = args.simulation

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


rng = np.random.default_rng()

data_file = '/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.couplingheavy-hex.precompute.60.pkl'
with open(data_file, 'rb') as f:
    res = pickle.load(f)


cost_circuit = res[filename]['rzz']['circuit']
if cost_circuit.count_ops().get('u', 0) > 0:
    # Workaround for u gate failing to compile
    from qiskit.circuit.library import UGate
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    theta = Parameter('t')
    phi = Parameter('p')
    lamda = Parameter('l')
    qc = QuantumCircuit(1)
    qc.p(lamda, 0)
    qc.ry(theta, 0)
    qc.p(phi, 0)
    UGate(theta, phi, lamda).add_decomposition(qc)
cost_circuit = transpile(cost_circuit, basis_gates=['id', 'rz', 'rz', 'ry', 'cx', 'rzz', 'swap', 'cz', 'p'])
print(cost_circuit.count_ops())
cost_circuit = parameterise_circuit(cost_circuit, parameter=Parameter('γ'))
layout = res[filename]['rzz']['layout']

    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
hamiltonian = hamiltonian * norm
num_virtual_qubits: int = hamiltonian.num_qubits


def warm_start(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]=None):
    phis = ParameterVector('ϕ', num_virtual_qubits)
    fixed_qc, circuit = get_hardware_LR_qaoa_circuit(p, delta_b, delta_g, num_virtual_qubits, cost_circuit, layout, backend, circ, phis)
    history = []
    angles_history = [init_angles]
    angles = init_angles
    iters = 5

    for i in range(iters):
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T, max_iterations=iters), data, history)
        angles_history.append(angles)
        
        
    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit, angles_history
        
        
# delta_b_fixed, delta_g_fixed = 0.45, 0.26
# delta_b_fixed, delta_g_fixed = 0.33, 0.19
delta_b_fixed, delta_g_fixed = 0.75, 0.30

eta = 1
eps = 0.25
max_beta_T = 0.25
alpha = 0.05

probs = 1 / 2 * np.ones((num_virtual_qubits,))
#experimental
probs[0] = eps 
thetas = 2 * np.arcsin(np.sqrt(probs))
init_angles = thetas

data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    eta=eta,
    eps=eps,
    alpha=alpha
)

rescaling= [1,]
ps = [1,]

energies = np.zeros((len(ps), len(rescaling)))
samples_dict = {}
angles_dict = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit, angles_history = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], rescaling[j])] = samples
    angles_dict[(ps[i], rescaling[j])] = angles_history
    
    
to_save=dict(energies=energies, delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict,angles_dict=angles_dict)    
append_str = f'.{filename}{".error_mit" if error_mitigation else ""}{".simulation" if simulation else ""}.backend{backend.name}.db{delta_b_fixed}.dg{delta_g_fixed}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}'
with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/hardware/nonvariational.hardware{append_str}.pkl', 'wb') as f:
    pickle.dump(to_save, f)