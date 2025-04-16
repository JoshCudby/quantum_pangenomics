import numpy as np
from time import time
import pickle
import networkx as nx
from scipy.optimize import minimize

from gfapy import Gfa

from qiskit import QuantumCircuit, transpile
# from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler #, EstimatorV2 as Estimator

from qiskit_ibm_runtime.fake_provider import FakeFez, FakeHanoiV2

# from qopt_best_practices.sat_mapping import SATMapper

from qiskit_prog_qaoa.utils.circuit_utils import get_prog_qaoa_circuit
# from qiskit_prog_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = get_parser()
args = parser.parse_args()

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    blocking_enable=True,
    blocking_qubits=24,
    # batched_shots_gpu_max_qubits=24,
    # batched_shots_gpu=noisy,
    precision='single'
)

fake_fez = FakeFez()
backend = AerSimulator.from_backend(fake_fez, **backend_options)

# fake_hanoi = FakeHanoiV2()
# backend = AerSimulator.from_backend(fake_hanoi, **backend_options)

# backend = AerSimulator(**backend_options)


# gfa = Gfa("H	VN:Z:1.0\n\
# S	u0	TAAC	LN:i:4	SC:f:1.0\n\
# S	u1	CCCG	LN:i:4	SC:f:1.0\n\
# L	u0	+	u1	+	0M	EC:i:1\n\
# L	u1	-	u0	-	0M	EC:i:1")

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
K = 2 # TODO: increase to 5/10
nodes_weights = list(graph.nodes(data="weight"))
total_weight = sum(x[1] if x[1] is not None else 0 for x in nodes_weights)
T = int(np.floor(total_weight * 1.2)) 
ceil_log2_K1 = int(np.ceil(np.log2(K+1)))
ceil_log_n2 = int(np.ceil(np.log2(n+2)))
logger.info(f'n={n}, K={K}, T={T}, ceil_log_n2={ceil_log_n2}')

circuit = get_prog_qaoa_circuit(p=p, n=n, K=K, T=T, graph=graph)
circuit.measure(list(range(T * ceil_log_n2)), list(range(T * ceil_log_n2)))
t_circuit = transpile(circuit, backend, optimization_level=3, seed_transpiler=seed)

def print_circuit_info(qc: QuantumCircuit, circuit_name):
    logger.info(f'{circuit_name} has {qc.num_qubits} qubits')
    qc.count_ops()
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


print_circuit_info(circuit, 'Circuit')
print_circuit_info(t_circuit, '(Transpiled) Circuit')


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
    init_params = rng.uniform(0, 0.9 * np.pi, p).tolist() + rng.uniform(0, 0.5 * np.pi, p).tolist()
logger.info(f'Init: {init_params}')

history = []
sampler = Sampler(seed=seed, options=dict(backend_options=backend_options))
nodes = list(graph.nodes)


def cost_function(sample: str):
    cost = 0
    x = []
    counts = {}
    for t in range(T):
        x_bin = sample[t * ceil_log_n2: (t+1) * ceil_log_n2]
        x_int = sum(2 ** (ceil_log_n2-i-1) * int(x_bin[i]) for i in range(ceil_log_n2))
        x.append(x_int)
        counts[x_int] = counts.get(x_int, 0) + 1
    for t in range(T-1):
        if x[t] > n+1:
            cost += 5
        elif x[t] == n+1:
            if not x[t+1] == n+1:
                cost += 5
        else:
            if x[t+1] > n+1:
                pass
            elif not x[t+1] == n+1 and (nodes[x[t]-1], nodes[x[t+1]-1]) not in graph.edges:
                cost += 5
    for i in range(1, n+1):
        cost += (counts.get(i, 0) - graph.nodes[nodes[i-1]]["weight"]) ** 2
    return cost
    

def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(max(alpha,1) * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
    start = time()
    assigned_circuit = t_circuit.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = [cost_function(key) for key in counts.keys()]
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, 0.05)

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy

result = minimize(
    objective, x0=init_params, method="COBYLA", options={"maxiter": 100, "rhobeg": 0.01}
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, init_params=init_params, circuit=circuit, graph=graph
)
with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/{filename}.p{p}.shots{shots}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)


last_run_data = history[-1]
counts = last_run_data[3]
keys = sorted(list(counts.keys()))
for key in keys:
    logger.info(f'key: {key}. cost: {cost_function(key)}. count: {counts[key]}')


