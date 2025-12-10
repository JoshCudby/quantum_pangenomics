
import numpy as np
import pickle
import argparse
from itertools import product


from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler


from qiskit_qaoa.utils.circuit_graph_utils import circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

backend_options = dict(
    # method='matrix_product_state',
    # matrix_product_state_max_bond_dimension='20', 
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates = ['rx', 'ry', 'rz', 'cx']
)
# fake_fez = FakeFez()
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
args = parser.parse_args()

filename: str = args.filename

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits

    
swap_strat = SwapStrategy.from_line(list(range(num_qubits)))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}

singles = hamiltonian[hamiltonian.paulis.z.sum(axis=-1) == 1]
doubles = hamiltonian[hamiltonian.paulis.z.sum(axis=-1) == 2]

keys = [np.binary_repr(x, num_qubits) for x in range(2**num_qubits)]
evals = evaluate_sparse_pauli_samples(keys, hamiltonian) + ising_offset


def get_energy(qc):
    job = backend.run([qc],shots=1)
    sampler_result = job.result()
    data = sampler_result.results[0].data

    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    return energy


def LR_QAOA(p, delta_b, delta_g, circ):
    betas = [(1-k/p) * delta_b for k in range(p)]
    gammas = [(k+1) / p * delta_g for k in range(p)]
    fixed_params = betas + gammas
    
    if circ is None:
        circ_dict = circuit_construction(singles, doubles, None, swap_strat, edge_coloring, {}, p, None, None)
        circuit = circ_dict["circuit_to_sample"]
        circuit.remove_final_measurements()
        circuit.save_statevector()
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}')
    else:
        circuit = circ

    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)

    energy = get_energy(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, circuit
        

eps = 1e-2
delta_bs = np.linspace(0 + eps, np.pi - eps, 50)
delta_gs = np.linspace(0 + eps, 2*np.pi - eps, 100)
ps = range(1, 20)


energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, circuit = LR_QAOA(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e

to_save_name = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/LR_param_exploration.no_shot_noise.{filename}.db{np.round(delta_bs[-1],2)}.dg{np.round(delta_gs[-1],2)}.p{ps[-1]}.pkl'
ret = dict(delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, energies=energies)
    
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)