import numpy as np
from typing import Optional
import pickle
import argparse
from itertools import product

from qiskit import QuantumCircuit


from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler


from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit

from qiskit_qaoa.utils.hamiltonian_utils import get_normalised_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples_all
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('--measure', action='store_true', default=False)
parser.add_argument('-n', '--shots', default=4000, type=int)
args = parser.parse_args()
logger.info(args)
measure = args.measure
filename: str = args.filename

if measure:
    backend_options = dict(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension='32', 
        # method='statevector',
        device='CPU',
        precision='single',
        basis_gates = ['rx', 'ry', 'rz', 'cx']
    )
else:
    backend_options = dict(
        method='statevector',
        device='GPU',
        precision='single',
        basis_gates = ['rx', 'ry', 'rz', 'cx']
    ) 
# fake_fez = FakeFez()
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)


rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

_, hamiltonian, _, ising_offset, _, ham_norm = get_normalised_Q_and_hamiltonian(data_file)
hamiltonian = hamiltonian * ham_norm
ising_offset = ising_offset * ham_norm
num_qubits: int = hamiltonian.num_qubits

evals = evaluate_sparse_pauli_samples_all(hamiltonian) + ising_offset
opt_evals = np.nonzero(evals < 1e-5)
logger.info(f'Opt evals: {opt_evals}')
    

def get_energy_and_p_opt(qc):
    job = sampler.run([qc], shots=args.shots)
    sampler_result = job.result()
    counts = sampler_result[0].data.meas.get_counts()
    samples, energies = [], []
    for sample, count in counts.items():
        samples.extend(count * [sample])
        energies.extend(count * [evals[int(sample, 2)]])
    energies = np.array(energies)
    energy = np.mean(energies)
    p_opt = np.flatnonzero(energies < 1e-5).shape[0] / args.shots
    return energy, p_opt


def get_energy_and_p_opt_sv(qc):
    result = backend.run([qc],shots=1).result()
    data = result.results[0].data
    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    p_opt = np.sum(np.abs(sv[opt_evals]) ** 2)
    return energy, p_opt


def LR_QAOA(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]):
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=None, measure=measure
    )

    if measure:
        _, p_opt = get_energy_and_p_opt(fixed_qc)
    else:
        _, p_opt = get_energy_and_p_opt_sv(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, probability:{np.round(p_opt, 4)}')
    return p_opt, circuit
    
eps = 1e-2
  
delta_b_fixed = 0.63
delta_g_fixed = 0.16

rescaling = 10 ** np.linspace(-1, 0.5, 31)
ps = sorted(set([int(p) for p in np.logspace(0, 2.5, 51, base=10)]))

probabilities = np.zeros((len(ps), len(rescaling)))
circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    p, circuit = LR_QAOA(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    probabilities[i, j] = p

to_save_name = f"/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/param_exploration/LR_equal.performance.{filename}.db{delta_b_fixed}.dg{delta_g_fixed}.rescaling{np.round(rescaling[-1],2)}.p{ps[-1]}.pkl"
ret = dict(delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, probabilities=probabilities)
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)