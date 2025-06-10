import numpy as np
import pickle
import networkx as nx
from scipy.optimize import minimize
from collections import Counter
from fnmatch import fnmatch
from gfapy import Gfa

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qiskit_prog_qaoa.utils.opt_utils import objective, callback, soln_to_path, cost_function
from qiskit_prog_qaoa.utils.circuit_utils import get_prog_qaoa_circuit
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
p: int = args.reps
shots = args.shots
init_type = args.init
max_iter = args.maxiter
method = args.method
lamda = args.lamda

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    blocking_enable=True,
    blocking_qubits=30,
    # batched_shots_gpu_max_qubits=24,
    # batched_shots_gpu=noisy,
    precision='single'
)


backend = AerSimulator(**backend_options)
sampler = Sampler(options=dict(backend_options=backend_options))


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
K = max(dict(graph.nodes(data="weight", default=0)).values())
K = int(min(K, 5))
nodes_weights = list(graph.nodes(data="weight"))
total_weight = sum(x[1] if x[1] is not None else 0 for x in nodes_weights)
T = int(np.floor(total_weight * 1.2)) 
ceil_log_n2 = int(np.ceil(np.log2(n+2)))
logger.info(f'p={p}, n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')
logger.info(f'shots={shots}, iter={max_iter}')

circuit = get_prog_qaoa_circuit(p=p, n=n, K=K, T=T, graph=graph, lamda=lamda)
circuit.measure(list(range(T * ceil_log_n2)), list(range(T * ceil_log_n2)))

d_circuit = circuit.decompose(gates_to_decompose=['state_prep', 'phase_operator', 'mixer_operator'] ,reps=1)
gtd = ['circuit*', 'unitary', '*add-1', '*minus-1']
while any(fnmatch(key, p) for p in gtd for key in d_circuit.count_ops().keys()):
    d_circuit = d_circuit.decompose(gates_to_decompose=gtd)

print_circuit_info(d_circuit, 'Circuit')


t_circuit = transpile(d_circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
print_circuit_info(t_circuit, 'Transpiled Circuit')

if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = np.array(betas.tolist() + gammas.tolist())
else:
    init_params = rng.uniform(0.05, 0.95, p).tolist() + rng.uniform(0.05, 0.95, p).tolist()
logger.info(f'Init: {init_params}')

history = []

obj = objective(init_params, n, T, graph, lamda, shots, history, t_circuit, sampler)