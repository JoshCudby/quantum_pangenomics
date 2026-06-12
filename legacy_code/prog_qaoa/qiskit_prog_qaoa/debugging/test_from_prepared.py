import numpy as np
import pickle
import argparse
from gfapy import Gfa
import networkx as nx
from itertools import product
from scipy.sparse import coo_array

from qiskit import QuantumCircuit

from qiskit_prog_qaoa.utils.logging import get_logger


def print_circuit_info(qc: QuantumCircuit, circuit_name):
    logger.info(f'{circuit_name} has {qc.num_qubits} qubits')
    logger.info(f'{circuit_name} has {qc.num_nonlocal_gates()} non-local gates and {qc.depth(lambda instr: len(instr.qubits) > 1)} non-local depth')
    logger.info(f'{circuit_name} contains {list(qc.count_ops().keys())} gates.')


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--prepare', default=None)
parser.add_argument('--constraint', action='store_true', default=False)
parser.add_argument('--objective', action='store_true', default=False)
parser.add_argument('--obj-first', action='store_true', default=False)
args = parser.parse_args()

logger = get_logger(__name__)
filename = args.filename
prepare = args.prepare
should_constraint = args.constraint
should_objective = args.objective
obj_first = args.obj_first

if prepare is None:
    prepare = 'State'

logger.info(f'Filename: {filename}. Prepare: {prepare}. Obj first: {obj_first}. Constraint: {should_constraint}. Objective: {should_objective}')

with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/data.obj_first{obj_first}.{filename}.prepare{prepare}.constraint{should_constraint}.objective{should_objective}.pkl', 'rb') as f:
    data = pickle.load(f)

if prepare != 'State':
    for key in data.keys():
        if key == 'global_phase':
            logger.info(f'Phase: {data[key]}')
        else:
            logger.info(key)
            sv = data[key]
            logger.info(sv.coords)
            logger.info(np.binary_repr(sv.coords[0][0]).rjust(int(np.log2(sv.shape[0])), '0'))

            logger.info(sv.data)
else:
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
    uniform_prob = (n+1) ** -T
    allowed = ['0001','0010','0011','0100',
            '0101','0110','0111','1000',
            '1001','1010','1011']

    allowed_strings = list(product(allowed, repeat=7))
    allowed_binary_arrays = np.array(
        [[int(y) for y in ''.join(x)] for x in allowed_strings], dtype=int
    )
    expected_indexes = set(allowed_binary_arrays.dot(1 << np.arange(allowed_binary_arrays.shape[1])[::-1]))

    for key in data.keys():
        if key == 'global_phase':
            logger.info(f'Phase: {data[key]}')
        else:
            logger.info(key)
            sv: coo_array = data[key]
            coords = set(sv.coords[0])
            missing = sorted(expected_indexes - coords)
            unexpected = sorted(coords - expected_indexes)
            logger.info(f'Expected indexes missing: {len(missing)}')
            logger.info(f'Unexpected indexes present: {len(unexpected)}')
            if len(missing) or len(unexpected):
                with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/missing.obj_first{obj_first}.{filename}.prepare{prepare}.constraint{should_constraint}.objective{should_objective}.pkl', 'wb') as f:
                    pickle.dump({'missing': missing, 'unexpected': unexpected}, f)
