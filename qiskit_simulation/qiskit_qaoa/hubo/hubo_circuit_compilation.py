"""Full HUBO compilation pipeline for hardware-targeted execution.

This script converts a GFA pangenome graph into a compiled QAOA cost-layer
circuit suitable for execution on real quantum hardware (IBM heavy-hex
topology).  The pipeline proceeds as follows:

1. **Hamiltonian construction** — calls ``graph_to_hubo_hamiltonian`` to
   produce both a full and a (optionally subsampled) compiled SparsePauliOp
   over n*T qubits.

2. **Topology selection** — chooses the smallest heavy-hex patch that can
   accommodate the n*T virtual qubits and wraps it in an
   ``ExtendedSwapStrategy``.

3. **Interaction extraction** — ``hamiltonian_to_interactions`` decomposes the
   Hamiltonian into a list of multi-qubit interaction sets, optionally
   restricting 4-body and 6-body terms via ``--fraction-four`` and
   ``--fraction-six``.

4. **SAT layout** — ``HigherOrderSatMapper.hubo_max_sat`` finds the qubit
   mapping that maximises the fraction of interactions satisfiable within a
   given number of SWAP layers, looping over 10 evenly-spaced SWAP-layer
   budgets.

5. **Circuit compilation** — ``get_hubo_pass_manager`` applies
   ``FindCommutingPauliEvolutionsMulti`` followed by
   ``CommutingGateRouterPrecomputeRzz`` and cancellation passes to produce the
   compiled cost layer.

6. **Serialisation** — results are pickled to::

       <basepath>/compilation.<filename>.extra<e>.times<t>.four<f>.six<s>.pkl

   The pickle contains::

       {
           'full_hamiltonian':     SparsePauliOp,   # all constraint terms
           'compiled_hamiltonian': SparsePauliOp,   # subsampled constraint
           <num_layers: int>:      Layout,           # best layout per SWAP budget
       }

CLI arguments:
    -f / --filename:       GFA file stem (resolved relative to a hard-coded
                           data directory).
    -e / --extra:          Extra SWAP layers passed to the pass manager
                           (default 1).
    --fraction-four:       Fraction of 4-body interaction terms to retain.
    --fraction-six:        Fraction of 6-body interaction terms to retain.
    --times-to-keep:       Comma-separated list of timestep-transition indices
                           t for which the (t, t+1) constraint term is kept.
    -t / --timeout:        SAT solver timeout in seconds.
    -c / --copy-numbers:   Comma-separated node copy numbers (overrides GFA
                           weights).
"""
import numpy as np
import networkx as nx
import pickle
import argparse
from itertools import combinations
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz,  PauliEvolutionGate
from qiskit.transpiler import Layout

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.quantum_info import SparsePauliOp

from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.pass_managers import get_hubo_pass_manager
from qiskit_qaoa.utils.logging import get_logger


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
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])

args = parser.parse_args()
logger.info(args)

def two_qubit_count(qc: QuantumCircuit):
    """Count the total number of 2-qubit gates in a circuit.

    Counts CZ, RZZ, CX, and SWAP gates.

    Args:
        qc: The quantum circuit to inspect.

    Returns:
        Total number of 2-qubit gates as an integer.
    """
    ops: dict[str, int] = qc.count_ops()
    return ops.get("cz", 0) + ops.get("rzz", 0) + ops.get("cx", 0) + ops.get("swap", 0)


def print_circuit_info(qc: QuantumCircuit, circuit_name: str):
    """Log the 2-qubit gate count and 2-qubit gate depth of a circuit.

    Args:
        qc: The quantum circuit to summarise.
        circuit_name: A human-readable label included in the log message.
    """
    logger.info(f'{circuit_name} has {two_qubit_count(qc)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    
    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{args.filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
num_qubits = n * T
logger.info(f'Virtual qubits: {num_qubits}')


rows, cols = 1, 1
while 2 * (rows + cols + rows * cols) < num_qubits:
    if rows < cols:
        rows += 1
    else:
        cols += 1
logger.info(f'Min size to support virtual qubits: {(rows, cols)}, ')

extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(rows, cols)
num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map
    
physical_qubits = list(coupling_map.physical_qubits)


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

full_hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=args.times_to_keep)

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


mapper = HigherOrderSatMapper(timeout=args.timeout)
results: dict[str | int, SparsePauliOp | Layout] = {
    'full_hamiltonian': full_hamiltonian,
    'compiled_hamiltonian': hamiltonian
}

layers = sorted(list(set([int(x) for x in np.linspace(0, len(extended_swap_strat._swap_layers), 10)])))
for num_layers in layers:
    logger.info('--------------------------------------------------')
    sat_results = mapper.hubo_max_sat(
        num_qubits, program_interactions, extended_swap_strat, num_layers
    )
    if sat_results is None:
        logger.info('No results')
        continue
    mapping = sat_results[num_layers][1]
    edge_map = dict(mapping)
    unused_physical_qubits = list(set(range(num_physical_qubits)) - set(edge_map.values()))
    # logger.info(edge_map)
    # logger.info(unused_physical_qubits)
    # for i in range(num_qubits, num_physical_qubits):
    #     edge_map[i] = unused_physical_qubits[i - num_qubits]
    donor_qc = QuantumCircuit(num_physical_qubits)
    layout = Layout({donor_qc.qubits[key]: val for key, val in edge_map.items()})

    logger.info(f'Cost: {sat_results[num_layers][0]}')
    logger.info(edge_map)

    pm = get_hubo_pass_manager(extended_swap_strat, num_layers, args.extra)

    new_cost_circ = QuantumCircuit(num_physical_qubits)
    new_cost_circ.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_qubits)])
    new_tcost = pm.run(new_cost_circ)
    
    print_circuit_info(new_tcost, 'Remapped, commuting gate routed circuit')
    print(new_tcost.count_ops())
    
    results[num_layers] = layout
    
    
    
basepath = '/lustre/scratch127/qpg/jc59/hubo_hardware/'
filename = 'compilation.{}.extra{}.times{}.four{}.six{}'.format(
    args.filename,
    args.extra,
    ''.join([str(t) for t in args.times_to_keep]),
    args.fraction_four,
    args.fraction_six
)
dump_file = basepath + filename + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(results, f)
    
