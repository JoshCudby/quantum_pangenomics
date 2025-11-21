import numpy as np
import pickle
import argparse
from collections import Counter
from sys import maxsize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate, CXGate
from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.quantum_info import SparsePauliOp

from typing import TypedDict

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.get_swap_strategy import get_swap_strategy


from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_precompute_rzz import CommutingGateRouterPrecomputeRzz
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.logging import get_logger


def two_qubit_count(qc: QuantumCircuit):
    ops = qc.count_ops()
    return ops.get("cz", 0) + ops.get("rzz", 0) + ops.get("cx", 0) + ops.get("swap", 0)
   
    
def print_circuit_info(qc: QuantumCircuit, circuit_name: str):
    logger.info(f'{circuit_name} has {two_qubit_count(qc)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


Best = TypedDict('Best', {'layout': Layout, 'depth': int, 'count': int, 'layers': int, 'circuit': QuantumCircuit})    


def sweep_swap_depths(layers: list[int], qubits: list, best_rzz: Best, swap_strategy: ExtendedSwapStrategy):
    for layer in layers:
        pm_rzz = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterPrecomputeRzz(
                    swap_strategy,
                    max_layers=layer,
                    perform_extra_swaps=True
                ),
                SwapToFinalMapping(),
                InverseCancellation(gates_to_cancel=[CXGate()]),
                CommutativeCancellation(basis_gates=["cx", "swap", "rz", "rzz"]),
                InverseCancellation(gates_to_cancel=[CXGate()]),
            ]
        )

        if args.timeout == 0:
            logger.info('Using trivial layout')
            layout = Layout({qubits[i]: i for i in range(num_physical_qubits)})
        else:
            sat_results = mapper.hubo_max_sat(
                num_physical_qubits, program_interactions, swap_strategy, layer
            )
            if sat_results is None:
                logger.info('No results')
                continue

            mapping = sat_results[layer][1]
            edge_map = dict(mapping)
            
            layout = Layout({qubits[key]: val for key, val in edge_map.items()})

        qc = QuantumCircuit(num_physical_qubits)
        qc.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[qubits[i]] for i in range(num_physical_qubits)])

        logger.info('Compiling with precompute Rz')
        
            
        logger.info('Compiling with precompute Rzz')
        tqc_rzz = pm_rzz.run(qc)   
        
        rzz_depth = tqc_rzz.depth(lambda instr: len(instr.qubits) > 1)
        if rzz_depth < best_rzz['depth']:
            best_rzz['depth'] = rzz_depth
            best_rzz['count'] = tqc_rzz.num_nonlocal_gates()
            best_rzz['layers'] = layer
            best_rzz['layout'] = layout 
            best_rzz['circuit'] = tqc_rzz
            logger.info(f'New best circuit. Depth {rzz_depth} with {layer} SWAP layers ')
    return
    

logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('--times-to-keep', help='delimited list input', 
    type=lambda s: tuple([int(item) for item in s.split(',') if len(item)]))
parser.add_argument('-t', '--timeout', type=int)
parser.add_argument('-C', '--coupling-map', choices=['line', 'grid', 'heavy-hex', 'all'])
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])

args = parser.parse_args()
logger.info(args)

    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{args.filename}.gfa'
graph, n, V, total_weight = gfa_file_to_graph(filepath, args.copy_numbers)
# TODO: search over T near to total_weight
T = total_weight
num_qubits = n * total_weight
logger.info(f'Virtual qubits: {num_qubits}')

mapper = HigherOrderSatMapper(timeout=args.timeout)


extended_swap_strat = get_swap_strategy(args.coupling_map, n, T)

num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map
donor_qc = QuantumCircuit(num_physical_qubits)


logger.info(f'Physical qubits: {num_physical_qubits}')

basis_gates=["sx", "x", "rz", "rx", "rzz", "cz", "id"]

backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates=basis_gates
)


config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = num_physical_qubits
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map, **backend_options)
backend.set_option("n_qubits", num_physical_qubits)
logger.info(f'Qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')

full_hamiltonian = graph_to_hubo_hamiltonian(graph, n, total_weight, lamda=10, constraint_terms=1.0)
hamiltonian = graph_to_hubo_hamiltonian(graph, n, total_weight, lamda=10, constraint_terms=args.times_to_keep)

logger.info(f'Number of hamiltonian terms: {len(hamiltonian)}')

logger.info('------------------------------------')
logger.info('------------------------------------')

qaoa_cost_op = QAOAAnsatz(
    hamiltonian,
    mixer_operator=QuantumCircuit(num_qubits),
    initial_state=QuantumCircuit(num_qubits)
)
backend_tqaoa = transpile(qaoa_cost_op, optimization_level=3, backend=backend, basis_gates=basis_gates)

print_circuit_info(backend_tqaoa, 'Default qaoa cost layer on backend')
logger.info(backend_tqaoa.count_ops())

logger.info('------------------------------------')
logger.info('------------------------------------')


all_pauli_z = np.array(
    [i.paulis[0].z for i in hamiltonian]
)
logger.info(f'Hamiltonian: {len(hamiltonian)}')
logger.info(f'Orders: {Counter(np.sum(all_pauli_z, axis=1))}')

program_interactions = hamiltonian_to_interactions(hamiltonian, args.fraction_four, args.fraction_six)
lengths = Counter([len(interaction) for interaction in program_interactions])

logger.info(f'Program interactions: {len(program_interactions)}')
logger.info(f'Orders: {Counter([len(interaction) for interaction in program_interactions])}')



layers = sorted(list(set([int(x) for x in np.linspace(0, len(extended_swap_strat._swap_layers), 10)])))

best_rzz = Best(
    count=maxsize, depth=maxsize, layers=0, 
    layout=Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)}),
    circuit=QuantumCircuit(num_physical_qubits)
)

sweep_swap_depths(layers, donor_qc.qubits, best_rzz, extended_swap_strat)


best_rzz_index = layers.index(best_rzz['layers'])
rzz_fine_layers = sorted(list(
    set([
        int(x) for x in np.linspace(layers[max(best_rzz_index - 1, 0)]+1, layers[min(best_rzz_index + 1, len(layers)-1)]-1, 5)
    ]).difference(layers)
))

logger.info(f'Best rzz layers: {best_rzz["layers"]}. Fine search over {rzz_fine_layers}.')
sweep_swap_depths(rzz_fine_layers, donor_qc.qubits, best_rzz, extended_swap_strat)    


results: dict[str, SparsePauliOp | Layout | Best] = {
    'full_hamiltonian': full_hamiltonian,
    'compiled_hamiltonian': hamiltonian,
    'best_rzz': best_rzz
}

basepath = '/lustre/scratch127/qpg/jc59/new_hubo_formulation/'
filename = 'compilation.{}.coupling{}.extra{}.times{}.four{}.six{}'.format(
    args.filename,
    args.coupling_map,
    args.extra,
    ''.join([str(t) for t in args.times_to_keep]),
    args.fraction_four,
    args.fraction_six
)
dump_file = basepath + filename + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(results, f)
    
