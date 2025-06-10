import numpy as np
import networkx as nx
from gfapy import Gfa
import pickle
from scipy.sparse import coo_array

from itertools import product

from qiskit import QuantumCircuit

from qiskit_prog_qaoa.utils.opt_utils import soln_to_path
from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger

def print_circuit_info(qc: QuantumCircuit, circuit_name):
    logger.info(f'{circuit_name} has {qc.num_qubits} qubits')
    logger.info(f'{circuit_name} has {qc.num_nonlocal_gates()} non-local gates and {qc.depth(lambda instr: len(instr.qubits) > 1)} non-local depth')
    logger.info(f'{circuit_name} contains {list(qc.count_ops().keys())} gates.')


logger = get_logger(__name__)
parser = get_parser()
args = parser.parse_args()

filename = args.filename
p = 1
lamda = args.lamda

with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/data.{filename}.p{p}.pkl', 'rb') as f:
    data = pickle.load(f)

data = coo_array.todense(data)
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
logger.info(f'p={p}, n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')
uniform_prob = (n+1) ** -T


data[np.abs(data) ** 2 < uniform_prob / 100] = 0
data_nz = np.transpose(np.nonzero(data))
for nz in data_nz:
    binary_rep = np.binary_repr(nz[0])
    # if len(binary_rep) > T * ceil_log_n2:
    #     logger.error('Unexpectedly large non-zero amplitude.')
    #     logger.error(f'Rep: {binary_rep}. Len: {len(binary_rep)}')
    if len(binary_rep) < np.log2(len(data)):
        binary_rep = binary_rep.rjust(int(np.log2(len(data))), '0')
    slices = [binary_rep[-ceil_log_n2:]]
    for t in range(1, T):
        slices.append(binary_rep[-ceil_log_n2*(t+1):-ceil_log_n2*t])

    if any(slice in ['0000','1100','1101', '1110', '1111'] for slice in slices):
        logger.error(f'Nonzero amplitude of: {binary_rep}. Amplitude: {np.abs(data[nz[0]]) ** 2}')

allowed = ['0001','0010','0011','0100',
            '0101','0110','0111','1000',
            '1001','1010','1011']
for rep in product(allowed, repeat=7):
# for rep in ['0001000100010001000100010001', '0001000100010001000100010011', '0001000100010001001100010011', '0001000100010001001101010011']:
    rep_str = ''.join(rep)
    # rep_str = rep_str[::-1]
    index = sum(2**i * int(rep_str[i]) for i in range(len(rep_str)))
    amplitude = np.abs(data[index]) ** 2
    if amplitude < uniform_prob / 100:
        logger.error(f'Rep: {rep_str}. Index: {index}. Path: {soln_to_path(rep_str, n, T, graph)}. Amplitude: {amplitude}')