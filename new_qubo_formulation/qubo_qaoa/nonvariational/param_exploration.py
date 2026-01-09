
import numpy as np
import pickle
import argparse
from itertools import product

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit
from qubo_qaoa.utils.str_utils import genbin

from qiskit_qaoa.utils.hamiltonian_utils import get_normalised_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

backend_options = dict(
    method='matrix_product_state',
    matrix_product_state_max_bond_dimension='32', 
    # method='statevector',
    device='CPU',
    precision='single',
    basis_gates = ['rx', 'ry', 'rz', 'cx']
)
# fake_fez = FakeFez()
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-n', '--shots', default=4000, type=int)
args = parser.parse_args()
logger.info(args)
filename: str = args.filename

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

_, hamiltonian, _, ising_offset, _, ham_norm = get_normalised_Q_and_hamiltonian(data_file)
hamiltonian = hamiltonian * ham_norm
ising_offset = ising_offset * ham_norm
num_qubits: int = hamiltonian.num_qubits

keys = list(genbin(num_qubits))
evals = evaluate_sparse_pauli_samples(keys, hamiltonian) + ising_offset
opt_evals = np.nonzero(evals < 1e-5)
print(f'Opt evals: {opt_evals}')


def get_energy_and_p_opt(qc):
    job = sampler.run([qc], shots=args.shots)
    sampler_result = job.result()
    counts = sampler_result[0].data.meas.get_counts()
    evals = evaluate_sparse_pauli_samples(counts.keys(), hamiltonian) + ising_offset
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
    energies = np.array(energies)
    energy = np.mean(energies)
    p_opt = np.flatnonzero(energies < 1e-5).shape[0] / args.shots
    return energy, p_opt


def LR_QAOA(p, delta_b, delta_g, circ):    
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=None, measure=True
    )

    energy, p_opt = get_energy_and_p_opt(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}, p_opt: {np.round(p_opt, 4)}')
    return energy, p_opt, circuit
        
        

eps = 1e-2
delta_bs = np.logspace(-1.0, 0.0, 11, base=10)
delta_gs = np.logspace(-1.0, -0.5, 11, base=10)
ps = sorted(set([int(x) for x in np.logspace(0, 2, 5, base=10)]))


energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
p_opts = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, p_opt, circuit = LR_QAOA(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e
    p_opts[i, j, k] = p_opt

to_save_name = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/param_exploration/LR_unequal.{filename}.db{np.round(delta_bs[-1],2)}.dg{np.round(delta_gs[-1],2)}.p{ps[-1]}.pkl'
ret = dict(delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, energies=energies, p_opts=p_opts)
    
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)
