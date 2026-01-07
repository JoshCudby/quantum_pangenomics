import numpy as np
from typing import Optional
import pickle
import argparse
from itertools import product

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, PauliEvolutionGate
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qiskit.circuit import Parameter, ParameterVector

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from hubo_qaoa.utils.get_swap_strategy import get_swap_strategy

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.str_utils import genbin
from hubo_qaoa.utils.parameterise_circuit import parameterise_circuit
from hubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit

from qiskit_qaoa.utils.transpiler_passes import FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_precompute_rzz import CommutingGateRouterPrecomputeRzz
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
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
parser.add_argument('--normalise', action='store_true', default=False)
args = parser.parse_args()


filename: str = args.filename

filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
normalised_hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
hamiltonian = normalised_hamiltonian * norm

circuit_hamiltonian = normalised_hamiltonian if args.normalise else hamiltonian

extended_swap_strat = get_swap_strategy('all', n, T)
num_physical_qubits = extended_swap_strat._num_vertices

donor_qc = QuantumCircuit(num_physical_qubits)
pm_rzz = PassManager(
    [
        FindCommutingPauliEvolutionsMulti(), 
        CommutingGateRouterPrecomputeRzz(
            extended_swap_strat,
            max_layers=0,
            perform_extra_swaps=True
        ),
        SwapToFinalMapping(),
        InverseCancellation(gates_to_cancel=[CXGate()]),
        CommutativeCancellation(basis_gates=["cx", "swap", "rz", "rzz"]),
        InverseCancellation(gates_to_cancel=[CXGate()]),
    ]
)
layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})
qc = QuantumCircuit(num_physical_qubits)
qc.append(PauliEvolutionGate(circuit_hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])     
    
logger.info('Compiling with precompute Rzz')
cost_circuit = pm_rzz.run(qc)   
cost_circuit = parameterise_circuit(cost_circuit, parameter=Parameter('γ'))


num_qubits: int = cost_circuit.num_qubits    
    
keys = list(genbin(num_qubits))
evals = evaluate_sparse_pauli_samples(keys, hamiltonian)
opt_evals = np.nonzero(evals < 1e-5)
print(len(opt_evals))

def get_energy(qc) -> float:
    job = backend.run([qc],shots=1)
    sampler_result = job.result()
    data = sampler_result.results[0].data

    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    return energy

def get_p_opt(qc) -> float:
    job = backend.run([qc],shots=1)
    sampler_result = job.result()
    data = sampler_result.results[0].data

    sv = np.asarray(data.statevector)
    p_opt = np.sum(np.abs(sv[opt_evals]) ** 2)
    return p_opt


def LR_QAOA(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]):
    fixed_qc, circuit = get_LR_qaoa_circuit(p, delta_b, delta_g, num_qubits, cost_circuit, circ)

    energy = get_energy(fixed_qc)
    p_opt = get_p_opt(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}, p_opt: {np.round(p_opt, 4)}')
    return energy, p_opt, circuit
    
eps = 1e-2
  

delta_bs = np.logspace(-1, 0, 11, base=10)
delta_gs = np.logspace(-1.0, -0.5, 11, base=10)
ps = [int(x) for x in np.logspace(0, 2.5, 6, base=10)]


energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
p_opts = np.zeros((len(ps), len(delta_bs), len(delta_gs)))

circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, p_opt, circuit = LR_QAOA(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e
    p_opts[i, j, k] = p_opt

to_save_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/LR_unequal.{filename}.db{np.round(delta_bs[-1],2)}.dg{np.round(delta_gs[-1],2)}.p{ps[-1]}.pkl'
ret = dict(delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, energies=energies, p_opts=p_opts)
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)

# delta_bs = np.linspace(0, 1, 200)
# ps = range(1, 1002, 10)

# energies = np.zeros((len(ps), len(delta_bs)))
# circuit = None
# for i, j in product(range(len(ps)), range(len(delta_bs))):
#     if j == 0:
#         circuit = None
#     e, circuit = LR_QAOA(ps[i], delta_bs[j], delta_bs[j], circuit)
#     energies[i, j] = e

# to_save_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/LR_equal.{filename}.d{np.round(delta_bs[-1],2)}.p{ps[-1]}.normalise{args.normalise}.pkl'
# ret = dict(deltas=delta_bs, ps=ps, energies=energies)
    
# with open(to_save_name, 'wb') as f:
#     pickle.dump(ret, f)