import numpy as np
import networkx as nx
from fnmatch import fnmatch
from gfapy import Gfa
import pickle
from scipy.sparse import coo_array
import argparse

from qiskit import QuantumCircuit, transpile

from qiskit_aer import AerSimulator

from qiskit_prog_qaoa.utils.circuit_utils import state_prep, get_constraint_circuit, get_objective_circuit
from qiskit_prog_qaoa.utils.logging import get_logger

def print_circuit_info(qc: QuantumCircuit, circuit_name):
    logger.info(f'{circuit_name} has {qc.num_qubits} qubits')
    logger.info(f'{circuit_name} has {qc.num_nonlocal_gates()} non-local gates and {qc.depth(lambda instr: len(instr.qubits) > 1)} non-local depth')
    logger.info(f'{circuit_name} contains {list(qc.count_ops().keys())} gates.')
    logger.info(f'{circuit_name} has phase {qc.global_phase}')


logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--prepare', default=None)
parser.add_argument('-m', '--memory', type=int, default=4000)
parser.add_argument('--constraint', action='store_true', default=False)
parser.add_argument('--objective', action='store_true', default=False)
parser.add_argument('--obj-first', action='store_true', default=False)
args = parser.parse_args()

filename = args.filename
should_constraint = args.constraint
should_objective = args.objective
obj_first = args.obj_first
to_prepare = args.prepare

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    # fusion_enable=False,
    # matrix_product_state_max_bond_dimension=5,
    blocking_enable=True,
    blocking_qubits=28,
    # batched_shots_gpu_max_qubits=24,
    # batched_shots_gpu=noisy,
    precision='single'
)

backend = AerSimulator(**backend_options)

data_file = f'/lustre/scratch127/qpg/jc59/data/{filename}.gfa'
gfa = Gfa.from_file(data_file)

graph = nx.Graph()
for segment_line in gfa.segments:
    graph.add_node(segment_line.name, weight=segment_line.SC)

graph.add_node('end')
for segment_line in gfa.segments:
    graph.add_edges_from([(segment_line.name, 'end')])
for edge_line in gfa.edges:
    graph.add_edges_from([
        (edge_line.sid1.name, edge_line.sid2.name),
    ])

n = len(gfa.segments)
K = max(dict(graph.nodes(data="weight", default=0)).values()) # K should be more than max weight to allow for over-visiting a high weight node.
K = int(min(K, 5))
nodes_weights = list(graph.nodes(data="weight"))
total_weight = sum(x[1] if x[1] is not None else 0 for x in nodes_weights)
T = int(np.floor(total_weight * 1.2)) 
ceil_log_n2 = int(np.ceil(np.log2(n+2)))
logger.info(f'n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')


if to_prepare is None:
    logger.info('Full state prep')
    state_prep_circuit = state_prep(n, T)
    to_prepare= 'State'
else:
    logger.info(f'Preparing: {to_prepare}')
    state_prep_circuit = QuantumCircuit(T * ceil_log_n2)
    for i in range(len(to_prepare)):
        if to_prepare[::-1][i] == '1':
            state_prep_circuit.x(i)


circuit = QuantumCircuit((K+T)*ceil_log_n2+2)
circuit.global_phase = 0
circuit.append(state_prep_circuit, list(range(state_prep_circuit.num_qubits)))

constraint_circuit = get_constraint_circuit(n, K, T, graph, parameter=np.pi/20, state_prep_circuit=None)
objective_circuit = get_objective_circuit(n, K, T, graph, parameter=np.pi/64, state_prep_circuit=None)

if obj_first and should_constraint and should_objective:
    # Works
    logger.info(f'Objective circuit qubits: {objective_circuit.num_qubits}')
    circuit.append(objective_circuit, list(range(objective_circuit.num_qubits)))
    # circuit.save_statevector('after_objective')
    
    logger.info(f'Constraint circuit qubits: {constraint_circuit.num_qubits}')
    circuit.append(constraint_circuit, list(range(constraint_circuit.num_qubits)))
    # circuit.save_statevector('after_constraint')
else:
    # Breaks if both
    if should_constraint:
        logger.info(f'Constraint circuit qubits: {constraint_circuit.num_qubits}')
        circuit.append(constraint_circuit, list(range(constraint_circuit.num_qubits)))
        # circuit.save_statevector('after_constraint')
    if should_objective:
        logger.info(f'Objective circuit qubits: {objective_circuit.num_qubits}')
        circuit.append(objective_circuit, list(range(objective_circuit.num_qubits)))
        # circuit.save_statevector('after_objective')

circuit.save_statevector('after_phase')
# circuit.measure_all()

logger.error(f'To prepare: {to_prepare}')
logger.error(f'Constraint: {should_constraint}. Objective: {should_objective}')
logger.error(f'Objective circuit first: {obj_first}')

d_circuit: QuantumCircuit = circuit.decompose(gates_to_decompose=['state_prep', 'phase_operator', 'mixer_operator'] ,reps=1)
gtd = ['circuit*']
while any(fnmatch(key, p) for p in gtd for key in d_circuit.count_ops().keys()):
    d_circuit = d_circuit.decompose(gates_to_decompose=gtd)

print_circuit_info(d_circuit, 'Circuit')


t_circuit: QuantumCircuit = transpile(d_circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
print_circuit_info(t_circuit, 'Transpiled Circuit')

result = backend.run(t_circuit).result()
logger.error(result)

uniform_prob = (n+1) ** -T

try:
    with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/data.obj_first{obj_first}.{filename}.prepare{to_prepare}.constraint{should_constraint}.objective{should_objective}.pkl', 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    logger.error(e)
    data = {}
    
# savepoints = ['after_constraint_step_11']
savepoints= ['after_constraint', 'after_objective', 'after_phase']
for savepoint in savepoints:
    try:
        sv = result.data()[savepoint].data
        sv[np.abs(sv) ** 2 < uniform_prob / 100] = 0
        sv = coo_array(sv)
        data[savepoint] = sv
    except Exception:
        pass
    
data['global_phase'] = t_circuit.global_phase


with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/data.obj_first{obj_first}.{filename}.prepare{to_prepare}.constraint{should_constraint}.objective{should_objective}.pkl', 'wb') as f:
    pickle.dump(data, f)


# Something breaks only when constraint happens before objective...???
# See 6594, 6596 