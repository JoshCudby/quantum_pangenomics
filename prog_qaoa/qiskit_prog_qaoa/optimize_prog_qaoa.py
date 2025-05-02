import numpy as np
import pickle
import networkx as nx
from scipy.optimize import minimize
from collections import Counter
from fnmatch import fnmatch

from gfapy import Gfa

from qiskit import QuantumCircuit, transpile
# from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler #, EstimatorV2 as Estimator

# from qiskit_algorithms.optimizers import SPSA # Breaks because QNSPSA wants base sampler V1, deprecated

from qiskit_ibm_runtime.fake_provider import FakeFez, FakeHanoiV2

# from qopt_best_practices.sat_mapping import SATMapper

from qiskit_prog_qaoa.utils.opt_utils import objective, callback, soln_to_path, cost_function
from qiskit_prog_qaoa.utils.circuit_utils import get_prog_qaoa_circuit
from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger

# import tracemalloc
# tracemalloc.start(25)


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

# fake_fez = FakeFez()
# backend = AerSimulator.from_backend(fake_fez, **backend_options)

# fake_hanoi = FakeHanoiV2()
# backend = AerSimulator.from_backend(fake_hanoi, **backend_options)

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

# snapshot3 = tracemalloc.take_snapshot()
# top_stats = snapshot3.compare_to(snapshot2, 'lineno')

# logger.info("[ Top 10 differences after transpile circuit]")
# for stat in top_stats[:10]:
#     logger.info(stat)


# TODO: can we do a swap strategy mapping? It relies on commuting gates I think...
# graph = circuit_to_graph(circuit, circuit.parameters[p]) 

# swap_strat = SwapStrategy.from_line(range(graph.order()))
# edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(graph.order())}

# remapped_g, sat_map, min_sat_layers = SATMapper(timeout=60).remap_graph_with_sat(
#     graph=graph, swap_strategy=swap_strat
# )

# cost_op = graph_to_operator(remapped_g)
# singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
# doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]

# circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p)

if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
else:
    init_params = rng.uniform(0.05, 0.95, p).tolist() + rng.uniform(0.05, 0.95, p).tolist()
logger.info(f'Init: {init_params}')

history = []


################################################
###### CHECK FOR INVALID NODES
################################################
# to_run = t_circuit.assign_parameters(init_params, inplace=False)
# result = backend.run(to_run).result()

# savepoints = ['after_prep'] + [f'after_phase_{i}' for i in range(p)] + [f'after_mixer_{i}' for i in range(p)]
# for savepoint in savepoints:
#     data = result.data()[savepoint].data
#     data[np.abs(data) < 1e-8] = 0
#     data_nz = np.transpose(np.nonzero(data))
#     logger.error(f'Checking: {savepoint}')
#     for nz in data_nz:
#         binary_rep = np.binary_repr(nz[0], T * ceil_log_n2)
#         for t in range(T):
#             slice = binary_rep[ceil_log_n2*t:ceil_log_n2*(t+1)]
#             if slice in ['000', '110', '111']:
#                 logger.error(f'Nonzero amplitude of: {binary_rep}. Amplitude: {data[nz[0]]}')
################################################



logger.info(f'Opt method: {method}')

if method == 'spsa':
    raise Exception('SPSA algorithm from qiskit_algorithms does not support Qiskit 2.0')
    # spsa = SPSA(maxiter=max_iter, termination_checker=TerminationChecker())
    # result = spsa.minimize(objective, x0=init_params)
    # print(f'SPSA completed after {result.nit} iterations')
else:
    result = minimize(
        objective, 
        x0=init_params,
        args=(n, T, graph, lamda, shots, history, t_circuit, sampler), 
        method=method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": max_iter, },  # "rhobeg": 0.01
        callback=callback
    )

logger.info(result)

obj_to_dump = dict(
    result=result, history=history, init_params=init_params, circuit=circuit, graph=graph, n=n, T=T, K=K
)
with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/{filename}.p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)

if len(history):
    last_run_data = history[-1]
    counts = Counter(last_run_data[3])
    most_common = counts.most_common(100)
    for e in most_common:
        logger.info(f'soln: {e[0]}. path: {soln_to_path(e[0], n, T, graph)}. cost: {cost_function(e[0], n, T, graph)}. count: {e[1]}')

            