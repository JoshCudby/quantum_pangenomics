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

# from qiskit_algorithms.optimizers import SPSA # Breaks because QNSPSA wants base sampler V1, deprecated


from qiskit_prog_qaoa.utils.opt_utils import oriented_objective, callback, callback_cobyla, oriented_soln_to_path, oriented_cost_function
from qiskit_prog_qaoa.utils.oriented_circuit_utils import get_prog_qaoa_circuit
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
method: str = args.method
lamda = args.lamda
blocking = args.blocking

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    blocking_enable=False,
    blocking_qubits=blocking,
    precision='single'
)

backend = AerSimulator(**backend_options)
sampler = Sampler(options=dict(backend_options=backend_options))


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
nodes_weights = list(graph.nodes(data="weight", default=0)) # type: ignore

K = max(x[1] for x in nodes_weights)  # K should be more than max weight to allow for over-visiting a high weight node.
K = int(min(K, 5))
total_weight = int(sum(x[1] for x in nodes_weights) / 2)
T = int(np.floor(total_weight * 1.1)) 

ceil_log_n2 = int(np.ceil(np.log2(n+2)))
logger.info(f'p={p}, n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')
logger.info(f'shots={shots}, iter={max_iter}, lamda={lamda}')

circuit = get_prog_qaoa_circuit(p=p, n=n, K=K, T=T, graph=graph, lamda=lamda)

d_circuit = circuit.decompose(gates_to_decompose=['state_prep', 'phase_operator', 'mixer_operator'] ,reps=1)
gtd = ['circuit*']
while any(fnmatch(key, p) for p in gtd for key in d_circuit.count_ops().keys()):
    d_circuit = d_circuit.decompose(gates_to_decompose=gtd)

print_circuit_info(d_circuit, 'Circuit')


t_circuit = transpile(d_circuit, backend=backend, optimization_level=3, seed_transpiler=seed)
print_circuit_info(t_circuit, 'Transpiled Circuit')
t_circuit.measure_all()


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

logger.info(f'Opt method: {method}')

if method == 'none':
    exit(0)
elif method == 'spsa':
    raise Exception('SPSA algorithm from qiskit_algorithms does not support Qiskit 2.0')
    # spsa = SPSA(maxiter=max_iter, termination_checker=TerminationChecker())
    # result = spsa.minimize(objective, x0=init_params)
    # print(f'SPSA completed after {result.nit} iterations')
else:
    result = minimize(
        oriented_objective, 
        x0=init_params,
        args=(n, T, graph, lamda, shots, history, t_circuit, sampler), 
        method=method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": max_iter, "maxfev": max_iter, "rhobeg": 0.1},  # , "ftol": 1e-7
        callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )

logger.info(result)

obj_to_dump = dict(
    result=result, history=history, init_params=init_params, circuit=circuit, graph=graph, n=n, T=T, K=K, lamda=lamda
)
# with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/oriented/{filename}.p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}.pkl', 'wb') as f:
with open(f'/tmp/jc59/out/prog_qaoa/oriented/{filename}.p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)

if len(history):
    last_run_data = history[-1]
    counts = Counter(last_run_data[3])
    most_common = counts.most_common(100)
    for e in most_common:
        logger.info(f'soln: {e[0]}. path: {oriented_soln_to_path(e[0], n, T, graph)}. cost: {oriented_cost_function(e[0], n, T, graph, lamda)}. count: {e[1]}')

exit(0)