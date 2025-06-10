import numpy as np
import networkx as nx
from fnmatch import fnmatch
from gfapy import Gfa
import pickle
from scipy.sparse import coo_array
import argparse

from qiskit import QuantumCircuit, transpile

from qiskit_aer import AerSimulator

from qiskit_prog_qaoa.utils.oriented_circuit_utils import state_prep, get_constraint_circuit, get_objective_circuit
from qiskit_prog_qaoa.utils.logging import get_logger

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

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
args = parser.parse_args()

filename = args.filename
to_prepare = args.prepare

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    blocking_enable=True,
    blocking_qubits=30,
    precision='single'
)

backend = AerSimulator(**backend_options)

# data_file = f'/lustre/scratch127/qpg/jc59/data/{filename}.gfa'
data_file = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
gfa = Gfa.from_file(data_file)

graph = nx.DiGraph()
for index, segment_line in enumerate(gfa.segments):
    graph.add_node(f'{segment_line.name}_+',  weight=segment_line.SC)
    graph.add_node(f'{segment_line.name}_-',  weight=segment_line.SC)
    
for edge_line in gfa.edges:
    v1 = edge_line.sid1
    v2 = edge_line.sid2
    graph.add_edges_from([
        (f'{v1.name}_{v1.orient}', f'{v2.name}_{v2.orient}'),
    ])
    v1.invert()
    v2.invert()
    graph.add_edges_from([
        (f'{v2.name}_{v2.orient}', f'{v1.name}_{v1.orient}'),
    ])

n = len(gfa.segments)
K = max(dict(graph.nodes(data="weight", default=0)).values()) # type: ignore # K should be more than max weight to allow for over-visiting a high weight node.
K = int(min(K, 5))
nodes_weights = list(graph.nodes(data="weight")) # type: ignore
total_weight = sum(x[1] if x[1] is not None else 0 for x in nodes_weights) / 2
T = int(np.floor(total_weight * 1.1)) 
ceil_log_n2 = int(np.ceil(np.log2(n+2)))
logger.info(f'n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')


if to_prepare == 'State':
    logger.info('Full state prep')
    state_prep_circuit = state_prep(n, T)
else:
    logger.info(f'Preparing: {to_prepare}')
    state_prep_circuit = QuantumCircuit(T * (ceil_log_n2+1))
    for i in range(len(to_prepare)):
        if to_prepare[::-1][i] == '1':
            logger.info(f'x on qubit {i}')
            state_prep_circuit.x(i)


circuit = QuantumCircuit((K+T)*(ceil_log_n2+1)+2)
circuit.global_phase = 0
circuit.append(state_prep_circuit, list(range(state_prep_circuit.num_qubits)))
circuit.save_statevector('after_prep') # type: ignore

constraint_circuit = get_constraint_circuit(n, K, T, graph, parameter=np.pi/20, state_prep_circuit=None) # type: ignore
objective_circuit = get_objective_circuit(n, K, T, graph, parameter=np.pi/64, state_prep_circuit=None) # type: ignore

logger.info(f'Objective circuit qubits: {objective_circuit.num_qubits}')
circuit.append(objective_circuit, list(range(objective_circuit.num_qubits)))
circuit.save_statevector('after_objective') # type: ignore

logger.info(f'Constraint circuit qubits: {constraint_circuit.num_qubits}')
circuit.append(constraint_circuit, list(range(constraint_circuit.num_qubits)))
circuit.save_statevector('after_constraint') # type: ignore


circuit.save_statevector('after_phase') # type: ignore
# circuit.measure_all()

logger.error(f'To prepare: {to_prepare}')

d_circuit: QuantumCircuit = circuit.decompose(gates_to_decompose=['state_prep', 'phase_operator', 'mixer_operator'] ,reps=1)
gtd = ['circuit*']
while any(fnmatch(key, p) for p in gtd for key in d_circuit.count_ops().keys()): # type: ignore
    d_circuit = d_circuit.decompose(gates_to_decompose=gtd)

print_circuit_info(d_circuit, 'Circuit')


t_circuit: QuantumCircuit = transpile(d_circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
print_circuit_info(t_circuit, 'Transpiled Circuit')

result = backend.run(t_circuit).result()
# logger.error(result)

uniform_prob = (2*(n+1)) ** -T

try:
    with open(f'/tmp/jc59/out/prog_qaoa/oriented/data.{filename}.prepare{to_prepare}.pkl', 'rb') as f:
    # with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/oriented/data.{filename}.prepare{to_prepare}.pkl', 'rb') as f:
        data = pickle.load(f)
except Exception as e:
    logger.error(e)
    data = {}
    

# for savepoint in result.data().keys():
savepoints= ['after_prep', 'after_constraint', 'after_objective', 'after_phase', 
             'after_next_nodes_n', 'after_compute_next_nodes_n1', 'after_next_nodes_n1', 'after_next_nodes_1_0',
             'after_compute_next_nodes_1_0'
             ]
for savepoint in savepoints:
    try:
        sv = result.data()[savepoint].data
        sv[np.abs(sv) ** 2 < uniform_prob / 100] = 0
        sv = coo_array(sv)
        data[savepoint] = sv
    except Exception:
        pass
    
data['global_phase'] = t_circuit.global_phase


# with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/oriented/data.{filename}.prepare{to_prepare}.pkl', 'wb') as f:
with open(f'/tmp/jc59/out/prog_qaoa/oriented/data.{filename}.prepare{to_prepare}.pkl', 'wb') as f:
    pickle.dump(data, f)

